import asyncio
import contextlib
import logging
import time

import ray
from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from tqdm.auto import tqdm

from orbit.ray.placement_group import (
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from orbit.utils import tracking_utils
from orbit.utils.arguments import parse_args
from orbit.utils.logging_utils import configure_logger
from orbit.utils.metric_utils import compute_rollout_step
from orbit.utils.misc import should_run_periodic_action
from orbit.utils.tracking_utils import init_tracking
from orbit.utils.training_eta import TrainingETA, format_duration

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def _timed_phase(prefix: str, name: str, *, timing_raw: dict | None = None, start_extra: str = ""):
    if start_extra:
        logger.info("%s: %s start %s", prefix, name, start_extra)
    else:
        logger.info("%s: %s start", prefix, name)
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        logger.info("%s: %s done elapsed=%.2fs", prefix, name, elapsed)
        if timing_raw is not None:
            timing_raw[name] = timing_raw.get(name, 0.0) + elapsed


@contextlib.contextmanager
def _timed_block(prefix: str, name: str, *, timing_raw: dict | None = None, start_extra: str = ""):
    if start_extra:
        logger.info("%s: %s start %s", prefix, name, start_extra)
    else:
        logger.info("%s: %s start", prefix, name)
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        logger.info("%s: %s done elapsed=%.2fs", prefix, name, elapsed)
        if timing_raw is not None:
            timing_raw[name] = timing_raw.get(name, 0.0) + elapsed

# ============================================================
# Teacher Server
# ============================================================

@ray.remote
class TeacherServer:
    """
    Thin Ray actor wrapping a SGLang engine for teacher inference.

    Responsibilities:
      - Load teacher weights once at startup (frozen throughout training).
      - Accept student rollout token ids, return top-k logits.
      - Expose offload/onload hooks so GPU memory can be reclaimed
        when the student trainer needs full GPU headroom.

    Note: top-k compression (default k=32) follows the Nemotron approach.
    Full-vocab logits can be enabled by setting topk_k=-1, but requires
    significantly more GPU memory and inter-node bandwidth.
    """

    def __init__(self, model_path: str, topk_k: int = 32, tp_size: int = 1):
        import torch
        from sglang import Engine

        self.topk_k = topk_k
        self.device = "cuda"

        # SGLang engine for teacher inference (generation disabled, logprob-only mode).
        # tp_size should match the number of GPUs allocated to teacher placement group.
        self.engine = Engine(
            model_path=model_path,
            tp_size=tp_size,
            # Disable KV cache for pure logprob scoring (no generation needed)
            max_running_requests=1,
        )

    def score(self, token_ids, attention_mask):
        """
        Compute per-token top-k logits for a batch of student rollouts.

        Args:
            token_ids:      LongTensor [B, T]  -- student-generated token ids
            attention_mask: BoolTensor [B, T]  -- 1 for real tokens, 0 for pad

        Returns:
            dict with:
                topk_logits:  FloatTensor [B, T, k]
                topk_indices: LongTensor  [B, T, k]

        This interface is designed to be MOPD-extensible: a router layer can
        call multiple TeacherServer.score() in parallel and merge results.
        """
        import torch

        with torch.no_grad():
            # SGLang forward pass returning full logits
            # NOTE: replace with engine.forward() or equivalent logprob API
            # once Orbit's SGLang version exposes it.
            logits = self.engine.get_logits(token_ids, attention_mask)  # [B, T, V]

            if self.topk_k > 0:
                topk_logits, topk_indices = torch.topk(logits, self.topk_k, dim=-1)
            else:
                # full vocab -- expensive, only for ablation
                topk_logits = logits
                topk_indices = torch.arange(logits.shape[-1]).expand_as(logits)

        return {
            "topk_logits": topk_logits.cpu(),
            "topk_indices": topk_indices.cpu(),
        }

    def offload(self):
        """Offload teacher weights to CPU to free GPU memory during student training."""
        self.engine.offload_weights()

    def onload(self):
        """Reload teacher weights to GPU before scoring."""
        self.engine.onload_weights()


# ============================================================
# Teacher placement group helper
# ============================================================

def create_teacher_server(args, pgs):
    """
    Instantiate TeacherServer on the teacher placement group.

    args expected fields (add to parse_args / config):
        args.teacher_model_path  -- HF or local path to teacher weights
        args.teacher_topk_k      -- top-k for logit compression (default 32)
        args.teacher_tp_size     -- tensor parallel size for teacher (default 1)
    """
    teacher_pg = pgs.get("teacher")
    if teacher_pg is None:
        raise ValueError(
            "No 'teacher' placement group found. "
            "Add teacher GPU allocation to create_placement_groups()."
        )

    server = TeacherServer.options(
        placement_group=teacher_pg,
        num_gpus=args.teacher_tp_size,
    ).remote(
        model_path=args.teacher_model_path,
        topk_k=args.teacher_topk_k,
        tp_size=args.teacher_tp_size,
    )
    return server


# ============================================================
# OPD loss helper (student side)
# ============================================================

def compute_opd_loss(rollout_data, teacher_topk):
    """
    Merge teacher top-k logits into rollout_data so actor_model.train()
    can compute the per-token reverse-KL loss.

    This is a thin data-plumbing function -- the actual KL computation
    lives inside actor_model (Megatron side) where it has access to
    the student logits during the forward pass.

    Args:
        rollout_data:  ref to existing rollout data dict (Ray object ref)
        teacher_topk:  dict { topk_logits, topk_indices }

    Returns:
        augmented rollout_data dict (in-memory, not a Ray ref)
    """
    data = ray.get(rollout_data)
    data["teacher_topk_logits"] = teacher_topk["topk_logits"]
    data["teacher_topk_indices"] = teacher_topk["topk_indices"]
    return data


# ============================================================
# Main training loop
# ============================================================

async def train(args):
    configure_logger()
    startup_timing: dict[str, float] = {}

    # GPU allocation
    with _timed_block("startup", "placement groups", timing_raw=startup_timing):
        pgs = create_placement_groups(args)
        # NOTE: create_placement_groups needs a "teacher" group added upstream.
        # Expected args: args.teacher_num_gpus for teacher GPU count.

    with _timed_block("startup", "init tracking", timing_raw=startup_timing):
        init_tracking(args)

    # Student rollout engine (SGLang)
    with _timed_block("startup", "create rollout manager", timing_raw=startup_timing):
        rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Student training model (Megatron actor)
    async with _timed_phase("startup", "create training models", timing_raw=startup_timing):
        actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    # Teacher inference server (frozen, separate GPUs)
    with _timed_block("startup", "create teacher server", timing_raw=startup_timing):
        teacher_server = create_teacher_server(args, pgs)

    if args.offload_rollout:
        async with _timed_phase("startup", "onload rollout weights", timing_raw=startup_timing):
            await rollout_manager.onload_weights.remote()

    # Sync student weights to SGLang before first rollout
    async with _timed_phase("startup", "actor update_weights", timing_raw=startup_timing):
        await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(action="compare")

    if args.offload_rollout:
        async with _timed_phase("startup", "onload rollout kv", timing_raw=startup_timing):
            await rollout_manager.onload_kv.remote()

    if startup_timing:
        startup_metrics = {f"timing_s_startup/{k}": v for k, v in startup_timing.items()}
        startup_metrics["timing_s_startup/total"] = sum(startup_timing.values())
        startup_metrics["rollout/step"] = compute_rollout_step(args, args.start_rollout_id)
        tracking_utils.log(args, startup_metrics, step_key="rollout/step")

    # ----------------------------------------
    # Helpers
    # ----------------------------------------

    async def offload_train():
        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def save(rollout_id):
        await actor_model.save_model(
            rollout_id,
            force_sync=(rollout_id == args.num_rollout - 1),
        )
        if args.rollout_global_dataset:
            await rollout_manager.save.remote(rollout_id)

    # ----------------------------------------
    # OPD train loop
    # ----------------------------------------
    eta = TrainingETA(start_rollout_id=args.start_rollout_id, num_rollout=args.num_rollout)
    rollout_pbar = tqdm(
        total=max(args.num_rollout - args.start_rollout_id, 0),
        desc="OPD training",
        unit="rollout",
        initial=0,
        dynamic_ncols=True,
        smoothing=0.0,
        mininterval=1.0,
    )

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        eta.mark_rollout_start(rollout_id)
        timing_raw: dict[str, float] = {}
        prefix = f"rollout {rollout_id}"

        # Optional eval before first training step
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            async with _timed_phase(prefix, "eval-before-train", timing_raw=timing_raw):
                await rollout_manager.eval.remote(rollout_id)

        # --- Step 1: Student generates rollouts ---
        async with _timed_phase(prefix, "generate", timing_raw=timing_raw):
            rollout_data_ref = await rollout_manager.generate.remote(rollout_id)

        # Offload rollout engine if configured (frees GPU for teacher scoring)
        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            async with _timed_phase(prefix, "offload rollout", timing_raw=timing_raw):
                await rollout_manager.offload.remote(tags=offload_tags)

        # --- Step 2: Teacher scores student tokens ---
        #
        # Design note: teacher scoring happens AFTER rollout and BEFORE training.
        # This is the natural insertion point in the synchronous pipeline.
        # For async OPD, teacher scoring of rollout N could overlap with
        # student training on rollout N-1 -- left as future work (see train_opd_async.py).
        #
        # MOPD extension point: replace single teacher_server.score.remote()
        # with a routing layer that dispatches to multiple teachers by domain/task.
        async with _timed_phase(prefix, "teacher score", timing_raw=timing_raw):
            rollout_data = ray.get(rollout_data_ref)
            teacher_future = teacher_server.score.remote(
                rollout_data["token_ids"],
                rollout_data["attention_mask"],
            )
            teacher_topk = ray.get(teacher_future)

        # Merge teacher logits into rollout data for student training
        with _timed_block(prefix, "merge teacher logits", timing_raw=timing_raw):
            opd_data = compute_opd_loss(rollout_data_ref, teacher_topk)
            opd_data_ref = ray.put(opd_data)

        # --- Step 3: Student trains on (rollout + teacher logits) ---
        #
        # actor_model.train() is expected to detect teacher_topk_logits in the
        # data dict and switch loss from GRPO advantage to per-token reverse KL.
        # This requires a small modification in orbit/backends/megatron_utils/actor.py.
        async with _timed_phase(prefix, "actor train", timing_raw=timing_raw):
            await actor_model.train(rollout_id, opd_data_ref)

        # Checkpointing
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            async with _timed_phase(prefix, "save", timing_raw=timing_raw):
                await save(rollout_id)

        # Offload student trainer, reload rollout engine
        async with _timed_phase(prefix, "offload/clear train", timing_raw=timing_raw):
            await offload_train()

        if args.offload_rollout:
            async with _timed_phase(prefix, "onload rollout weights", timing_raw=timing_raw):
                await rollout_manager.onload_weights.remote()

        # --- Step 4: Sync updated student weights to SGLang ---
        async with _timed_phase(prefix, "actor update_weights", timing_raw=timing_raw):
            await actor_model.update_weights()

        if args.offload_rollout:
            async with _timed_phase(prefix, "onload rollout kv", timing_raw=timing_raw):
                await rollout_manager.onload_kv.remote()

        # Periodic eval
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            async with _timed_phase(prefix, "eval", timing_raw=timing_raw):
                await rollout_manager.eval.remote(rollout_id)

        # Progress tracking
        eta_report = eta.mark_rollout_done(rollout_id)
        logger.info("progress %s", eta_report.to_log_message())
        rollout_pbar.set_postfix(
            last=format_duration(eta_report.last_rollout_seconds),
            avg=format_duration(eta_report.avg_rollout_seconds),
            eta=format_duration(eta_report.eta_seconds),
            refresh=False,
        )
        rollout_pbar.update(1)

        rollout_step = compute_rollout_step(args, rollout_id)
        progress_metrics = eta_report.to_metrics(step=rollout_step)
        for phase_name, elapsed in timing_raw.items():
            progress_metrics[f"timing_s/{phase_name.replace(' ', '_')}"] = elapsed
        tracking_utils.log(args, progress_metrics, step_key="rollout/step")

    rollout_pbar.close()
    async with _timed_phase("shutdown", "dispose rollout"):
        await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(train(args))