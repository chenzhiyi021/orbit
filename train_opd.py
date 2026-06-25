import asyncio
import contextlib
import logging
import time

import torch
import ray
from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from tqdm.auto import tqdm

from orbit.ray.placement_group import (
    create_opd_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from orbit.ray.teacher import (
    TeacherManager,
    merge_teacher_signal,
    create_teacher_manager,
)

from orbit.utils import tracking_utils
from orbit.utils.arguments import parse_args
from orbit.utils.logging_utils import configure_logger
from orbit.utils.metric_utils import compute_rollout_step
from orbit.utils.misc import should_run_periodic_action
from orbit.utils.tracking_utils import init_tracking
from orbit.utils.training_eta import TrainingETA, format_duration
from orbit.utils.ray_utils import Box

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


async def train(args):
    configure_logger()
    startup_timing: dict[str, float] = {}

    # GPU allocation
    with _timed_block("startup", "placement groups", timing_raw=startup_timing):
        pgs = create_opd_placement_groups(args)

    with _timed_block("startup", "init tracking", timing_raw=startup_timing):
        init_tracking(args)

    # Student rollout engine (SGLang)
    with _timed_block("startup", "create rollout manager", timing_raw=startup_timing):
        rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Student training model (Megatron actor)
    async with _timed_phase("startup", "create training models", timing_raw=startup_timing):
        actor_model, _ = await create_training_models(args, pgs, rollout_manager)

    # Teacher inference server
    with _timed_block("startup", "create teacher manager", timing_raw=startup_timing):
        teacher_server = create_teacher_manager(args, pgs["teacher"])
        try:
            num_engines = ray.get(teacher_server.num_engines.remote(), timeout=600)
            logger.info(f"Teacher manager ready with {num_engines} engine(s).")
        except ray.exceptions.GetTimeoutError:
            logger.error("Teacher manager failed to initialize within 600s.")
            raise

    if args.offload_rollout:
        async with _timed_phase("startup", "onload rollout weights", timing_raw=startup_timing):
            await rollout_manager.onload_weights.remote()

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

        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            async with _timed_phase(prefix, "eval-before-train", timing_raw=timing_raw):
                await rollout_manager.eval.remote(rollout_id)

        # --- Step 1: Student generates rollouts ---
        async with _timed_phase(prefix, "generate", timing_raw=timing_raw):
            rollout_data_ref = await rollout_manager.generate.remote(rollout_id)

        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            async with _timed_phase(prefix, "offload rollout", timing_raw=timing_raw):
                await rollout_manager.offload.remote(tags=offload_tags)

        # --- Step 2: Teacher scores student tokens ---
        # NOTE: rollout_data["tokens"] is a ragged list of list[int] (samples
        # are NOT pre-padded to a common length -- confirmed via
        # RolloutManager._convert_samples_to_train_data, which builds
        # "tokens" as [sample.tokens for sample in samples] without padding).
        # score() accepts this directly: SGLang's /generate endpoint takes
        # input_ids as a list of lists and does not require a rectangular
        # batch, so there is no separate "attention_mask" concept here
        # (unlike padded-tensor APIs) -- each sequence's real length is just
        # its own list length.
        #
        # MOPD extension point: replace single score.remote() with a routing
        # layer that dispatches to multiple teachers by domain/task.
        async with _timed_phase(prefix, "teacher score", timing_raw=timing_raw):
            # teacher_output = await teacher_server.score.remote(rollout_data_ref)
            opd_data_refs = []
            for ref in rollout_data_ref:
                rd = ray.get(ref.inner)
                raw_tokens = rd["tokens"]
                max_len = max(len(t) for t in raw_tokens)
                pad_token_id = 0  # TODO
                token_ids = torch.tensor([t + [pad_token_id]*(max_len-len(t)) for t in raw_tokens])
                attention_mask = torch.tensor([[1]*len(t) + [0]*(max_len-len(t)) for t in raw_tokens])
                teacher_output = ray.get(teacher_server.score.remote(token_ids, attention_mask))
                rd = merge_teacher_signal(rd, teacher_output)
                opd_data_refs.append(Box(ray.put(rd)))
            opd_data_ref = opd_data_refs

        # # Merge teacher signal into rollout data
        # with _timed_block(prefix, "merge teacher signal", timing_raw=timing_raw):
        #     opd_data = merge_teacher_signal(rollout_data, teacher_output)
        #     opd_data_ref = ray.put(opd_data)

        # --- Step 3: Student trains on (rollout + teacher signal) ---
        #
        # actor_model.train() detects teacher keys in the data dict (see
        # _is_opd_batch in orbit/backends/megatron_utils/actor.py) and
        # routes to train_student(), which masks out the nan at position 0
        # of each teacher_log_probs entry before it reaches the loss.
        async with _timed_phase(prefix, "actor train", timing_raw=timing_raw):
            await actor_model.train(rollout_id, opd_data_ref)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            async with _timed_phase(prefix, "save", timing_raw=timing_raw):
                await save(rollout_id)

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

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            async with _timed_phase(prefix, "eval", timing_raw=timing_raw):
                await rollout_manager.eval.remote(rollout_id)

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