import copy
import logging

import ray
import requests
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from orbit.backends.sglang_utils.sglang_engine import SGLangEngine
from orbit.backends.megatron_utils.actor import get_rollout_data
# NOTE：
# I reuse _allocate_rollout_engine_addr_and_ports_normal for convience.
# TODO:
# We should move _allocate_rollout_engine_addr_and_ports_normal to a more general utils module instead of rollout.py, since it's also used by TeacherManager which is not rollout-specific.
from orbit.ray.rollout import _allocate_rollout_engine_addr_and_ports_normal
from orbit.utils.logging_utils import configure_logger

from .utils import build_noset_visible_devices_env_vars

logger = logging.getLogger(__name__)


def _teacher_args(args):
    """Shallow-copy args with hf_checkpoint swapped to the teacher checkpoint.
    """
    teacher_args = copy.copy(args)
    teacher_args.hf_checkpoint = args.opd_teacher_model_path
    if getattr(args, "colocate", False):
        teacher_args.sglang_mem_fraction_static = getattr(
            args, "opd_teacher_mem_fraction_static", 0.25
        )
    return teacher_args


@ray.remote
class TeacherManager:
    """
    Ray actor wrapping SGLangEngine instances for frozen teacher inference.
    One TeacherManager instance handles a single teacher model.

    Public API:
        score(token_ids, attention_mask) -> dict
        offload() / onload()             -> GPU memory management
    """

    def __init__(self, args, pg):
        configure_logger()

        self.args = args
        self.pg = pg
        # self.loss_type = getattr(args, "opd_loss_type", "sampled_token")
        # self.topk_k = getattr(args, "opd_topk_k", 32)

        pg_obj, bundle_indices, gpu_ids = pg
        tp_size = args.opd_teacher_tp_size
        total_gpus = len(gpu_ids)
        num_engines = total_gpus // tp_size

        teacher_args = _teacher_args(args)

        # --- Create one SGLangEngine sub-actor per engine slot ---
        self._engines = []
        for i in range(num_engines):
            gpu_index = i * tp_size
            base_gpu_id = int(gpu_ids[gpu_index])
            bundle_index = bundle_indices[gpu_index]

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg_obj,
                placement_group_bundle_index=bundle_index,
            )
            engine = ray.remote(SGLangEngine).options(
                num_cpus=0.2,
                num_gpus=0.2,
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": build_noset_visible_devices_env_vars()},
            ).remote(
                teacher_args,
                rank=i,
                worker_type="regular",
                base_gpu_id=base_gpu_id,
                sglang_overrides={},
                num_gpus_per_engine=tp_size,
            )
            self._engines.append(engine)

        rollout_engines = list(enumerate(self._engines))

        addr_and_ports, _ = _allocate_rollout_engine_addr_and_ports_normal(
            args=teacher_args,
            rollout_engines=rollout_engines,
            worker_type="regular",
            num_gpus_per_engine=tp_size,
            rank_offset=0,
            base_port=25000,  # offset from rollout's default 15000 to avoid collisions
        )

        init_handles = [
            engine.init.remote(**addr_and_ports[rank])
            for rank, engine in rollout_engines
        ]
        ray.get(init_handles)

        # Cache all engine host/port pairs for round-robin dispatch in score().
        self._engine_addrs = [
            (addr_and_ports[rank]["host"], addr_and_ports[rank]["port"])
            for rank in range(num_engines)
        ]
        self._next_engine = 0  # round-robin cursor

        logger.info(
            f"TeacherManager ready: model={args.opd_teacher_model_path}, "
            f"tp_size={tp_size}, num_engines={num_engines}, "
            # f"loss_type={self.loss_type}, addr={self._server_host}:{self._server_port}"
        )

    def score(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """
        Score a batch of student rollout sequences with the teacher model.

        Args:
            token_ids:      LongTensor [B, T] -- may include right-padding
            attention_mask: LongTensor [B, T] -- 1 for real tokens, 0 for padding

        Returns:
            dict: {"teacher_log_probs": Tensor [B, T]}. Position 0 of every
            sequence is nan (no preceding context). Padded positions (where
            attention_mask == 0) are ALSO set to nan, since SGLang is only
            sent each sample's real (unpadded) tokens -- it has no concept of
            attention_mask itself. Callers (train_student) must mask using
            attention_mask before these values reach the loss, the same way
            position-0 nan is already handled.
        """
        with torch.no_grad():
            seq_lens = attention_mask.sum(dim=1).tolist()
            input_ids_list = [
                token_ids[i, : seq_lens[i]].cpu().tolist()
                for i in range(token_ids.shape[0])
            ]

            payload = {
                "input_ids": input_ids_list,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 0,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
            }

            host, port = self._engine_addrs[self._next_engine]
            self._next_engine = (self._next_engine + 1) % len(self._engine_addrs)
            url = f"http://{host}:{port}/generate"

            try:
                response = requests.post(url, json=payload, timeout=30.0)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Teacher score HTTP request failed: {e}")
                raise

            if not isinstance(result, list):
                result = [result]

            max_len = token_ids.shape[1]
            teacher_log_probs_list = []
            for i, r in enumerate(result):
                meta = r.get("meta_info", {})
                raw_entries = meta.get("input_token_logprobs", None)
                if raw_entries is None:
                    raise ValueError(f"Missing input_token_logprobs in {meta.keys()}")

                log_probs = [
                    entry[0] if entry[0] is not None else float("nan")
                    for entry in raw_entries
                ]
                log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32)

                # Pad back up to max_len with nan, matching the original
                # (unpadded) length we sent for this sample, seq_lens[i].
                if log_probs_tensor.shape[0] < max_len:
                    pad = torch.full(
                        (max_len - log_probs_tensor.shape[0],),
                        float("nan"),
                        dtype=torch.float32,
                    )
                    log_probs_tensor = torch.cat([log_probs_tensor, pad])

                teacher_log_probs_list.append(log_probs_tensor)

            teacher_log_probs_tensor = torch.stack(teacher_log_probs_list)
            return {"teacher_log_probs": teacher_log_probs_tensor}
 
    def _extract_logprobs_from_logprobs_field(self, logprobs_list):
        """
        Fallback parser for older SGLang response format:
            [[{"token_id": 101, "logprob": -0.1}, ...], ...]
        """
        result = []
        for token_logprobs in logprobs_list:
            batch_logprobs = []
            for token_info in token_logprobs:
                if isinstance(token_info, dict):
                    batch_logprobs.append(token_info.get("logprob", 0.0))
                else:
                    batch_logprobs.append(float(token_info))
            result.append(batch_logprobs)
        return result
 
    def offload(self):
        """Offload teacher weights to CPU to free GPU memory during student training."""
        ray.get([engine.release_memory_occupation.remote() for engine in self._engines])
 
    def onload(self):
        """Reload teacher weights to GPU before scoring."""
        ray.get([engine.resume_memory_occupation.remote() for engine in self._engines])
 
 
def create_teacher_manager(args, pg) -> "ray.actor.ActorHandle":
    """
    Instantiate TeacherManager on the teacher placement group.
 
    Called from train_opd.py after create_opd_placement_groups(). The
    returned handle is a Ray actor ref; call .score.remote() on it.
 
    Args:
        args: parsed arguments (must include opd_teacher_* fields)
        pg:   teacher placement group, as (pg_obj, bundle_indices, gpu_ids)
    """
    if pg is None:
        raise ValueError("No 'teacher' placement group found.")
 
    pg_obj, bundle_indices, _ = pg
 
    server = TeacherManager.options(
        num_cpus=0.1,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_bundle_index=bundle_indices[0],
        ),
    ).remote(args, pg)
 
    logger.info(
        f"TeacherManager actor scheduled on teacher placement group "
        f"({args.opd_teacher_tp_size} GPU(s))."
    )
    return server
 
 
def merge_teacher_signal(rollout_data: dict, teacher_output: dict) -> dict:
    """
    Merge teacher scoring output into rollout_data in-place.
 
    rollout_data must already be materialized (not a Ray object ref).
    The merged dict is passed to actor_model.train() via ray.put().
    """
    rollout_data.update(teacher_output)
    return rollout_data