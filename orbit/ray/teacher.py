import copy
import json
import logging
from collections import defaultdict

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

    def __init__(self, args, pg, base_port: int = 25000):
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
            base_port=base_port,  # offset from rollout's default 15000; unique per teacher in MOPD
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
            token_ids:      LongTensor [B, T] -- 0 for right-padding
            attention_mask: LongTensor [B, T] -- 1 for real tokens, 0 for padding

        Returns:
            dict: {"teacher_log_probs": Tensor [B, T]}. Position 0 of every
            sequence is nan (no preceding context). Padded positions (where
            attention_mask == 0) are ALSO set to nan.
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
                result = response.json() # result is list of dicts.
            except requests.exceptions.RequestException as e:
                logger.error(f"Teacher score HTTP request failed: {e}")
                raise
            
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

                # Pad back up to max_len with nan.
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
 
    def offload(self, tags: list[str] | None = None):
        """Offload teacher weights to CPU to free GPU memory during student training."""
        ray.get([engine.release_memory_occupation.remote(tags=tags) for engine in self._engines])

    def onload(self, tags: list[str] | None = None):
        """Reload teacher weights to GPU before scoring."""
        ray.get([engine.resume_memory_occupation.remote(tags=tags) for engine in self._engines])
 
 
# def create_teacher_manager(args, pg) -> "ray.actor.ActorHandle":
#     """
#     Instantiate TeacherManager on the teacher placement group.
 
#     Called from train_opd.py after create_opd_placement_groups(). The
#     returned handle is a Ray actor ref; call .score.remote() on it.
 
#     Args:
#         args: parsed arguments (must include opd_teacher_* fields)
#         pg:   teacher placement group, as (pg_obj, bundle_indices, gpu_ids)
#     """
#     if pg is None:
#         raise ValueError("No 'teacher' placement group found.")
 
#     pg_obj, bundle_indices, _ = pg
 
#     server = TeacherManager.options(
#         num_cpus=0.1,
#         num_gpus=0,
#         scheduling_strategy=PlacementGroupSchedulingStrategy(
#             placement_group=pg_obj,
#             placement_group_bundle_index=bundle_indices[0],
#         ),
#     ).remote(args, pg)
 
#     logger.info(
#         f"TeacherManager actor scheduled on teacher placement group "
#         f"({args.opd_teacher_tp_size} GPU(s))."
#     )
#     return server
 
 
def merge_teacher_signal(rollout_data: dict, teacher_output: dict) -> dict:
    """
    Merge teacher scoring output into rollout_data in-place.
    """
    rollout_data.update(teacher_output)
    return rollout_data


# ─────────────────────────────────────────────────────────────────────────────
# MOPD (Multi-Teacher On-Policy Distillation) 
# ─────────────────────────────────────────────────────────────────────────────


def _build_teacher_args_for_model(args, model_path: str, mem_fraction_static: float | None = None):
    """Shallow-copy args, overriding the teacher checkpoint path.

    Sets both hf_checkpoint and opd_teacher_model_path so that
    TeacherManager.__init__ (which calls _teacher_args internally) resolves
    to model_path regardless of the original args value.
    """
    teacher_args = copy.copy(args)
    teacher_args.hf_checkpoint = model_path
    teacher_args.opd_teacher_model_path = model_path
    if getattr(args, "colocate", False) and mem_fraction_static is not None:
        teacher_args.sglang_mem_fraction_static = mem_fraction_static
    return teacher_args


def _create_single_teacher_manager(
    args,
    pg,
    model_path: str,
    mem_fraction_static: float | None = None,
    base_port: int = 25000,
) -> "ray.actor.ActorHandle":
    """Instantiate one TeacherManager on *pg* for *model_path*.

    Extracted from create_teacher_manager() so that create_mopd_teachers() can
    call it once per teacher without duplicating the placement-group scheduling
    boilerplate.

    Each teacher must receive a unique base_port so their SGLang server
    processes do not collide on the same node.
    """
    if pg is None:
        raise ValueError("No placement group provided for TeacherManager.")
    pg_obj, bundle_indices, _ = pg
    teacher_args = _build_teacher_args_for_model(args, model_path, mem_fraction_static)
    handle = TeacherManager.options(
        num_cpus=0.1,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_bundle_index=bundle_indices[0],
        ),
    ).remote(teacher_args, pg, base_port)
    return handle


@ray.remote
class MopdRouter:
    """Route per-sample teacher-scoring requests to the right TeacherManager.

    Routing is *static*: each domain string maps to a named teacher baked in
    at construction time.  Samples whose domain is not in the routing table fall
    back to the general teacher (always required).

    Usage (from train_opd.py)::

        domains = rd.get("domain") or [rd.get("rm_type", "general")] * B
        output  = ray.get(mopd_router.score.remote(token_ids, mask, domains))
        # output == {"teacher_log_probs": Tensor[B, T]}  -- same as TeacherManager.score()

    All teacher sub-calls within one batch are fired concurrently before any
    result is awaited, so multi-teacher latency is bounded by the slowest
    teacher, not their sum.
    """

    def __init__(
        self,
        teachers: dict,           # {name: TeacherManager Ray handle}
        domain_to_teacher: dict,  # {domain_str: teacher_name}
        fallback: str = "general",
    ):
        if fallback not in teachers:
            raise ValueError(
                f"Fallback teacher '{fallback}' not found in teachers={list(teachers.keys())}"
            )
        self._teachers = teachers
        self._domain_to_teacher = domain_to_teacher
        self._fallback = fallback
        logger.info(
            f"MopdRouter ready: teachers={list(teachers.keys())}, "
            f"domain_map={domain_to_teacher}, fallback='{fallback}'"
        )

    def score(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        domains: list,
    ) -> dict:
        """Score a batch, routing each sample to its designated teacher.

        Args:
            token_ids:      LongTensor [B, T]
            attention_mask: LongTensor [B, T]
            domains:        list[str] of length B — one domain label per sample
                            (e.g. ``rd.get("domain")`` or ``["math"] * B``)

        Returns:
            dict: {"teacher_log_probs": Tensor [B, T]}
        """
        # ── 1. Group sample indices by resolved teacher name ──────────────
        groups: dict[str, list[int]] = defaultdict(list)
        for i, domain in enumerate(domains):
            teacher_name = self._domain_to_teacher.get(domain, self._fallback)
            groups[teacher_name].append(i)

        # ── 2. Fire all teacher sub-calls concurrently ────────────────────
        pending: dict[str, tuple[list[int], ray.ObjectRef]] = {}
        for teacher_name, indices in groups.items():
            idx_t = torch.tensor(indices, dtype=torch.long)
            future = self._teachers[teacher_name].score.remote(
                token_ids[idx_t], attention_mask[idx_t]
            )
            pending[teacher_name] = (indices, future)

        # ── 3. Collect and reassemble in original sample order ────────────
        B, T = token_ids.shape
        out_logprobs = torch.full((B, T), float("nan"), dtype=torch.float32)
        for teacher_name, (indices, future) in pending.items():
            sub_out = ray.get(future)
            sub_logprobs = sub_out["teacher_log_probs"]  # [len(indices), T]
            for j, orig_idx in enumerate(indices):
                out_logprobs[orig_idx] = sub_logprobs[j]

        return {"teacher_log_probs": out_logprobs}

    def offload(self, tags: list | None = None):
        """Offload all teacher weights to CPU (free GPU memory during student training)."""
        ray.get([t.offload.remote(tags=tags) for t in self._teachers.values()])

    def onload(self, tags: list | None = None):
        """Reload all teacher weights to GPU before the next scoring phase."""
        ray.get([t.onload.remote(tags=tags) for t in self._teachers.values()])


def create_mopd_teachers(args, teacher_pgs: dict) -> "ray.actor.ActorHandle":
    """Build a MopdRouter from ``--mopd-teacher-configs`` + ``--opd-teacher-model-path``.

    Args:
        args: parsed arguments.  Must include:
            - ``opd_teacher_model_path`` — path for the general (fallback) teacher.
            - ``mopd_teacher_configs``   — optional JSON string listing specialised
                teachers.
        teacher_pgs: dict mapping teacher name → placement-group tuple
                    ``(pg_obj, bundle_indices, gpu_ids)``.
                    Must contain at least ``{"general": pg}``.

    Returns:
        MopdRouter Ray actor handle.
    """
    # ── 0. Debug-mode pass-through ────────────────────────────────────────
    # create_opd_placement_groups() returns teachers=None in debug_* modes. 
    if teacher_pgs is None:
        raise ValueError(
            "No teacher placement groups found (teacher_pgs is None). "
        )

    # ── 1. Parse specialised-teacher config list ──────────────────────────
    configs_raw = getattr(args, "mopd_teacher_configs", None)
    teacher_configs: list[dict] = json.loads(configs_raw) if configs_raw else []

    # ── 2. Always register the general (fallback) teacher ─────────────────
    general_pg = teacher_pgs.get("general")
    if general_pg is None:
        raise ValueError(
            f"teacher_pgs must contain a 'general' key. Got: {list(teacher_pgs.keys())}"
        )
    mem_frac = getattr(args, "opd_teacher_mem_fraction_static", 0.25)
    # Each teacher gets its own base_port offset by 1000 to avoid SGLang port
    # conflicts when multiple teachers run on the same node.
    _next_port = 25000
    teachers: dict[str, ray.actor.ActorHandle] = {
        "general": _create_single_teacher_manager(
            args, general_pg, args.opd_teacher_model_path, mem_frac, base_port=_next_port
        ),
    }
    _next_port += 1000
    domain_to_teacher: dict[str, str] = {}

    # ── 3. Register specialised teachers ──────────────────────────────────
    for cfg in teacher_configs:
        name = cfg["name"]
        path = cfg["path"]
        domains = cfg.get("domains", [])
        pg = teacher_pgs.get(name)
        if pg is None:
            raise ValueError(
                f"No placement group for teacher '{name}'. Available pgs: {list(teacher_pgs.keys())}"
            )
        teachers[name] = _create_single_teacher_manager(args, pg, path, mem_frac, base_port=_next_port)
        _next_port += 1000
        for domain in domains:
            if domain in domain_to_teacher:
                logger.warning(
                    f"Domain '{domain}' already mapped to '{domain_to_teacher[domain]}'; "
                    f"overriding with '{name}'."
                )
            domain_to_teacher[domain] = name
        logger.info(f"MOPD teacher '{name}': model={path}, domains={domains}")

    logger.info(f"MOPD domain routing table: {domain_to_teacher} (fallback → 'general')")

    # ── 4. Wrap all teachers in MopdRouter ────────────────────────────────
    # Schedule the router itself on the general teacher's placement group
    general_pg_obj, general_bundle_indices, _ = general_pg
    router = MopdRouter.options(
        num_cpus=0.1,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=general_pg_obj,
            placement_group_bundle_index=general_bundle_indices[0],
        ),
    ).remote(teachers, domain_to_teacher, fallback="general")

    return router