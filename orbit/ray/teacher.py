"""
Teacher server for On-Policy Distillation (OPD).

Follows the same Ray-actor pattern as RolloutManager in rollout.py:
  - TeacherManager is a @ray.remote class allocated on dedicated GPUs.
  - create_teacher_server() is the factory function called from train_opd.py.

The teacher is frozen throughout training; it only runs forward passes to
score student-generated rollouts.

Loss types (--opd-loss-type):
  sampled_token  -- log-prob of the student's sampled token (Nemotron approach, default)
  topk           -- top-k logits and indices for distribution-level distillation
  full_vocab     -- full vocabulary logits (ablation only, expensive)
"""

import logging

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from orbit.backends.sglang_utils.sglang_engine import SGLangEngine
from orbit.utils.logging_utils import configure_logger

from .utils import build_noset_visible_devices_env_vars

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TeacherManager
# ---------------------------------------------------------------------------


@ray.remote
class TeacherManager:
    """
    Ray actor wrapping a SGLang engine for frozen teacher inference.

    One TeacherManager instance handles a single teacher model.
    For MOPD with multiple teachers, instantiate one TeacherManager per
    teacher model and route samples from train_opd.py.

    Public API:
        score(token_ids, attention_mask) -> dict
        offload() / onload()             -> GPU memory management
    """

    def __init__(self, args, pg):
        configure_logger()

        self.args = args
        self.loss_type = args.opd_loss_type
        self.topk_k = args.opd_topk_k

        pg_obj, bundle_indices, gpu_ids = pg
        base_gpu_id = int(gpu_ids[0])

        # Reuse SGLangEngine so we stay consistent with the rollout engine stack.
        # tp_size engines share the same pg; each takes one bundle slot.
        self._engines = []
        for i in range(args.opd_teacher_tp_size):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg_obj,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_indices[i],
            )
            engine = ray.remote(SGLangEngine).options(
                num_cpus=0.2,
                num_gpus=0.2,
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": build_noset_visible_devices_env_vars()},
            ).remote(
                args,
                rank=i,
                worker_type="teacher",
                base_gpu_id=int(gpu_ids[i]),
                sglang_overrides={},
                num_gpus_per_engine=args.opd_teacher_tp_size,
            )
            self._engines.append(engine)

        # Initialize engines (blocking so teacher is ready before training starts)
        ray.get([engine.init.remote() for engine in self._engines])
        logger.info(
            f"TeacherManager ready: model={args.opd_teacher_model_path}, "
            f"tp_size={args.opd_teacher_tp_size}, loss_type={self.loss_type}"
        )

    def score(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """
        Score student rollout tokens with the teacher model.

        Args:
            token_ids:      LongTensor [B, T]  -- student-generated token ids
            attention_mask: BoolTensor [B, T]  -- 1 for real tokens, 0 for pad

        Returns dict whose keys depend on loss_type:
            sampled_token -> {"teacher_log_probs": FloatTensor [B, T]}
            topk          -> {"topk_logits": [B, T, k], "topk_indices": [B, T, k]}
            full_vocab    -> {"teacher_logits": [B, T, V]}

        MOPD extension point:
            A router in train_opd.py calls score() on multiple TeacherManager
            instances in parallel and merges results before actor_model.train().
        """
        with torch.no_grad():
            # Use the first (rank-0) engine to run the forward pass.
            # For TP > 1, SGLangEngine handles internal tensor-parallel
            # communication across the engine group.
            # NOTE: replace get_logits() with the actual SGLang logprob API
            # once orbit's SGLang version exposes it.
            logits = ray.get(
                self._engines[0].get_logits.remote(token_ids, attention_mask)
            )  # [B, T, V]

            if self.loss_type == "sampled_token":
                log_probs = torch.log_softmax(logits, dim=-1)
                # Gather the log-prob of the token the student actually sampled.
                teacher_log_probs = log_probs.gather(
                    dim=-1,
                    index=token_ids.unsqueeze(-1),
                ).squeeze(-1)  # [B, T]
                return {"teacher_log_probs": teacher_log_probs.cpu()}

            elif self.loss_type == "topk":
                topk_logits, topk_indices = torch.topk(logits, self.topk_k, dim=-1)
                return {
                    "topk_logits": topk_logits.cpu(),
                    "topk_indices": topk_indices.cpu(),
                }

            else:  # full_vocab
                return {"teacher_logits": logits.cpu()}

    def offload(self):
        """Offload teacher weights to CPU to free GPU memory during student training."""
        ray.get([engine.release_memory_occupation.remote() for engine in self._engines])

    def onload(self):
        """Reload teacher weights to GPU before scoring."""
        ray.get([engine.resume_memory_occupation.remote() for engine in self._engines])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_teacher_manager(args, pg) -> TeacherManager:
    """
    Instantiate TeacherManager on the teacher placement group.

    Called from train_opd.py after create_opd_placement_groups().
    The returned handle is a Ray actor ref; call .score.remote() on it.

    Args:
        args: parsed arguments (must include opd_teacher_* fields)
        pg:  teacher placement group
    """
    if pg is None:
        raise ValueError(
            "No 'teacher' placement group found. "
        )

    pg_obj, bundle_indices, _ = pg

    # Pin the TeacherManager head actor to the first bundle of the teacher group.
    # The actor itself spawns per-rank engine sub-actors inside __init__.
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


# ---------------------------------------------------------------------------
# Data merge helper
# ---------------------------------------------------------------------------


def merge_teacher_signal(rollout_data: dict, teacher_output: dict) -> dict:
    """
    Merge teacher scoring output into rollout_data in-place.

    rollout_data must already be materialized (not a Ray object ref).
    The merged dict is passed to actor_model.train() via ray.put().

    Keys added to rollout_data depend on TeacherManager.score() loss_type:
        sampled_token -> "teacher_log_probs"
        topk          -> "topk_logits", "topk_indices"
        full_vocab    -> "teacher_logits"
    """
    rollout_data.update(teacher_output)
    return rollout_data