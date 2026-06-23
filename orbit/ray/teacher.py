import copy
import logging

import ray
import requests
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from orbit.backends.sglang_utils.sglang_engine import SGLangEngine
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

    SGLangEngine's _compute_server_args always reads args.hf_checkpoint as
    model_path -- there is no teacher-specific field it understands. This
    is the smallest change that lets TeacherManager reuse SGLangEngine
    unmodified.
    """
    teacher_args = copy.copy(args)
    teacher_args.hf_checkpoint = args.opd_teacher_model_path
    return teacher_args


@ray.remote
class TeacherManager:
    """
    Ray actor wrapping SGLangEngine instances for frozen teacher inference.

    One TeacherManager instance handles a single teacher model. For MOPD
    with multiple teachers, instantiate one TeacherManager per teacher
    model and route samples from train_opd.py.

    Public API:
        score(token_ids, attention_mask) -> dict
        offload() / onload()             -> GPU memory management
        num_engines()                    -> int, for tests/health checks
    """

    def __init__(self, args, pg):
        configure_logger()

        self.args = args
        self.pg = pg
        self.loss_type = getattr(args, "opd_loss_type", "sampled_token")
        self.topk_k = getattr(args, "opd_topk_k", 32)

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
                placement_group_capture_child_tasks=True,
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
                worker_type="teacher",
                base_gpu_id=base_gpu_id,
                sglang_overrides={},
                num_gpus_per_engine=tp_size,
            )
            self._engines.append(engine)

        # --- Allocate dist_init_addr / port / nccl_port for every engine ---
        # engine.init() requires these as explicit arguments; they are not
        # read off `args`. rollout_engines must be a list of (rank, engine)
        # tuples, matching what _allocate_rollout_engine_addr_and_ports_normal
        # and ServerGroup.start_engines() both expect.
        rollout_engines = list(enumerate(self._engines))

        addr_and_ports, _ = _allocate_rollout_engine_addr_and_ports_normal(
            args=teacher_args,
            rollout_engines=rollout_engines,
            worker_type="teacher",
            num_gpus_per_engine=tp_size,
            rank_offset=0,
            base_port=25000,  # offset from rollout's default 15000 to avoid collisions
        )

        # --- Initialize engines (blocking, so teacher is ready before training starts) ---
        # No router: teacher has no router_ip/router_port, so init() will not
        # attempt to register with a router (see SGLangEngine._init_normal).
        init_handles = [
            engine.init.remote(**addr_and_ports[rank])
            for rank, engine in rollout_engines
        ]
        ray.get(init_handles)

        # Cache rank-0's host/port for score() -- SGLangEngine does not expose
        # a getter for these, and they cannot be read off the actor handle
        # directly (Ray actor attributes are not externally readable).
        # We already allocated them above, so record them here instead of
        # querying the engine.
        self._server_host = addr_and_ports[0]["host"]
        self._server_port = addr_and_ports[0]["port"]

        logger.info(
            f"TeacherManager ready: model={args.opd_teacher_model_path}, "
            f"tp_size={tp_size}, num_engines={num_engines}, "
            f"loss_type={self.loss_type}, addr={self._server_host}:{self._server_port}"
        )

    def num_engines(self) -> int:
        """Expose engine count for tests/health checks."""
        return len(self._engines)

    def score(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """
        Score student rollout tokens with the teacher model via SGLang's
        HTTP /generate endpoint.
 
        Args:
            token_ids:      LongTensor [B, T]  -- student-generated token ids
            attention_mask: BoolTensor [B, T]  -- 1 for real tokens, 0 for pad
 
        Returns:
            dict: {"teacher_log_probs": Tensor [B, T]}
 
        NOTE: max_new_tokens=0 + return_logprob=True is assumed to make
        SGLang score the input sequence without generating new tokens.
        This assumption has not been verified against a running server --
        confirm the response shape/fields the first time this is run.
        """
        with torch.no_grad():
            input_ids_list = token_ids.cpu().tolist()  # [B, T]
 
            # NOTE: return_logprob belongs at the TOP LEVEL of the SGLang
            # /generate request, not inside sampling_params -- SamplingParams
            # does not accept a return_logprob keyword and will raise
            # TypeError if it is passed there (confirmed via a real 500 error
            # from a running server).
            # NOTE: return_logprob alone is not enough -- by default SGLang's
            # logprob_start_len=-1 means "don't return input logprobs at all".
            # Setting logprob_start_len=0 is what makes it return a logprob
            # for every input token (confirmed via SGLang's own CLI arg help
            # text: "-1 means no input logprobs, 0 means all"). Without this,
            # input_token_logprobs comes back with only one entry regardless
            # of prompt length (confirmed empirically against a running
            # server: a 5-token prompt returned only 1 entry before this fix).
            payload = {
                "input_ids": input_ids_list,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 0,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
            }
 
            url = f"http://{self._server_host}:{self._server_port}/generate"
 
            try:
                response = requests.post(url, json=payload, timeout=30.0)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Teacher score HTTP request failed: {e}")
                raise
 
            # NOTE: SGLang's /generate endpoint returns a LIST of per-request
            # results (one dict per input sequence), even for a single prompt
            # -- e.g. [{"input_token_logprobs": [...], ...}]. Unwrap it before
            # looking for the logprob fields. Confirmed via a real 200 OK
            # response from a running server; the exact inner schema is
            # logged below the first time this runs so it can be inspected.
            if isinstance(result, list):
                logger.info(f"score() raw response (first item): {result[0] if result else result}")
                result = result[0]
            else:
                logger.info(f"score() raw response: {result}")
 
            # SGLang's input_token_logprobs entries are 3-element
            # [logprob, token_id, decoded_text] tuples, e.g.
            #   [[None, 785, 'The'], [-2.1, 6722, ' capital'], ...]
            # The first token's logprob is always None (no preceding
            # context to condition on), which is expected, not an error.
            # Confirmed via a real response: meta_info.input_token_logprobs
            # is where SGLang puts these for this version/config, not the
            # top-level fields this code originally checked first.
            if "meta_info" in result and "input_token_logprobs" in result["meta_info"]:
                raw_entries = result["meta_info"]["input_token_logprobs"]
                teacher_log_probs_list = [
                    entry[0] if entry[0] is not None else float("nan")
                    for entry in raw_entries
                ]
            elif "input_token_logprobs" in result:
                raw_entries = result["input_token_logprobs"]
                teacher_log_probs_list = [
                    entry[0] if isinstance(entry, (list, tuple)) else entry
                    for entry in raw_entries
                ]
            elif "logprobs" in result:
                teacher_log_probs_list = self._extract_logprobs_from_logprobs_field(
                    result["logprobs"]
                )
            else:
                raise ValueError(f"Unexpected SGLang response format: {result.keys()}")
 
            teacher_log_probs = torch.tensor(teacher_log_probs_list, dtype=torch.float32)
            return {"teacher_log_probs": teacher_log_probs}
 
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
        num_cpus=1,
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