"""Small explicit integration points in Orbit's existing Megatron execution path."""

from __future__ import annotations
import json
import math
from pathlib import Path
import torch
import torch.distributed as dist
from .protocol import validate, applied_lr
from .factors import fused_modules, initial_factors, registry
from .constraints import _iter_optimizer_leaves, install_lora_factor_constraint


def active(args):
    return getattr(args, "posttrain_protocol", None) is not None


def event(args, name, **values):
    if not getattr(args, "posttrain_observe", True):
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    target = Path(args.posttrain_run_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / f"events_rank{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": name, "rank": rank, **values}, allow_nan=False) + "\n")


def configure(args):
    p = args.posttrain_protocol
    validate(p)
    from .launch import source_digest

    if not getattr(args, "posttrain_execution_ready", False):
        raise ValueError("This is a plan, not a validated execution; use launch --execute")
    if args.posttrain_source_hash != source_digest():
        raise ValueError("Research source changed after launch preflight")
    import orbit

    if Path(orbit.__file__).resolve().parent.parent != Path(args.posttrain_source_root).resolve():
        raise ValueError("Worker imported Orbit from another checkout")
    for field in ["use_tis", "normalize_advantages", "get_mismatch_metrics", "use_opsm", "use_kl_loss"]:
        if getattr(args, field, False):
            raise ValueError(f"{field} is not certified for this profile")
    if args.global_batch_size != p["batch"]["global_sequences"] or args.num_rollout != p["steps"]:
        raise ValueError("Runtime batch/steps changed after protocol resolution")
    if getattr(args, "no_save_optim", False) or getattr(args, "no_save_rng", False):
        raise ValueError("Research profiles require optimizer and RNG checkpoints")
    args.use_distributed_optimizer = p["optimizer"]["distributed"]
    args.lr_decay_iters = p["steps"]
    args.check_for_nan_in_loss_and_grad = True
    if p["kind"] == "m6":
        args.overlap_param_gather = False
        args.overlap_grad_reduce = False
        args.lora_A_init_method = "kaiming"
        args.lora_B_init_method = "zero"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif p["adapter"]["method"] == "lora":
        args.lora_A_init_method = "kaiming" if p["adapter"]["initialization"] == "name_seeded_kaiming" else "xavier"
        args.lora_B_init_method = "zero"
    args._posttrain_completed = 0


def initialized(args, model, optimizer, scheduler, iteration):
    p = args.posttrain_protocol
    rank = dist.get_rank() if dist.is_initialized() else 0
    from .checkpoint import _master_pairs, load_state

    pairs = _master_pairs(model, optimizer)
    rows = registry(model)
    if p["adapter"]["method"] == "lora":
        for row in rows:
            if row["requires_grad"] and ".adapter.linear_" not in row["name"]:
                raise ValueError(f"Unexpected trainable base/auxiliary parameter: {row['name']}")
        modules = fused_modules(model)
        if len(modules) != 112:
            raise ValueError("Expected exactly 112 Qwen3-1.7B fused LoRA modules")
        # Count unique global parameters, excluding tensor-parallel replicas.
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate

        count = sum(param.numel() for chunk in model for param in chunk.parameters() if param.requires_grad and param_is_not_tensor_parallel_duplicate(param))
        count = torch.tensor(count, dtype=torch.int64, device=next(model[0].parameters()).device)
        if dist.is_initialized():
            dist.all_reduce(count, group=parallel_state.get_tensor_model_parallel_group())
        if int(count) != p["adapter"]["expected_parameters"]:
            raise ValueError(f"Independent parameter registry mismatch: {int(count)}")
        if p["adapter"]["initialization"] == "name_seeded_kaiming" and not args.posttrain_resume:
            masters = {id(parameter): master for parameter, master in pairs.values()}
            with torch.no_grad():
                for name, module in modules.items():
                    a, b = module.adapter.linear_in.weight, module.adapter.linear_out.weight
                    a0, b0 = initial_factors((b.shape[0], a.shape[1]), p["adapter"]["rank"], name, device=a.device, initialization_seed=p["init_seed"])
                    for parameter, value in [(a, a0), (b, b0)]:
                        master = masters[id(parameter)]
                        if master.dtype != torch.float32:
                            raise ValueError("M6 requires FP32 master parameters")
                        master.copy_(value)
                        parameter.copy_(value.to(parameter.dtype))
    if args.posttrain_resume:
        restored = load_state(args.posttrain_resume, model, optimizer, scheduler, protocol=p, data_hash=args.posttrain_data_manifest["data"]["sha256"], rank=rank, freeze_factor=args.posttrain_freeze_factor)
        args._posttrain_completed = restored["completed_updates"]
        args._posttrain_pending_rng = restored["rng"]
        # Orbit checkpoints use zero-based rollout IDs; outer train starts at +1.
        iteration = restored["completed_updates"] - 1
        event(args, "resume_loaded", completed_updates=restored["completed_updates"], policy_version=restored["policy_version"], next_prompt_index=restored["next_prompt_index"])
    elif iteration != 0:
        raise ValueError("Reference profiles must start fresh or use explicit complete resume")
    else:
        iteration = -1
    freeze = getattr(args, "posttrain_freeze_factor", "none")
    if freeze != "none":
        if p["adapter"]["method"] != "lora":
            raise ValueError("Factor freezing is only defined for LoRA")
        install_lora_factor_constraint(model, optimizer, factor=freeze)
    target = Path(args.posttrain_run_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / f"parameter_registry_rank{rank}.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rank == 0:
        effective = {key: value for key, value in vars(args).items() if not key.startswith("_")}
        (target / "effective_protocol.json").write_text(json.dumps({"profile": p, "runtime_args": effective}, indent=2, default=str) + "\n")
    args._posttrain_frozen = {f"chunk{i}.{name}": value.detach().clone() for i, chunk in enumerate(model) for name, value in chunk.named_parameters() if not value.requires_grad}
    optimizer._posttrain_context = (args, model)
    if not args.posttrain_resume:
        from .checkpoint import save_state

        save_state(Path(args.save) / "initial_state", model, optimizer, scheduler, completed_updates=0, protocol=p, data_hash=args.posttrain_data_manifest["data"]["sha256"], rank=rank)
    geometry_snapshot(args, optimizer)
    return iteration


def before_step(args, optimizer, scheduler, rollout_id, step_id, num_microbatches):
    completed = args._posttrain_completed
    if step_id != 0 or rollout_id != completed:
        raise ValueError("Reference profile received an extra update or a discontinuous rollout")
    if args.posttrain_protocol["kind"] == "m6" and num_microbatches != 32:
        raise ValueError("M6 reference requires DP8 x micro1 x accumulation32")
    if hasattr(args, "_posttrain_pending_rng"):
        from .checkpoint import restore_rng

        restore_rng(args._posttrain_pending_rng)
        del args._posttrain_pending_rng
    if getattr(args, "posttrain_observe", True):
        from .checkpoint import _master_pairs

        _, model = optimizer._posttrain_context
        args._posttrain_previous_masters = {name: master.detach().clone() for name, (_, master) in _master_pairs(model, optimizer).items()}
        if args.posttrain_protocol["parallel"]["tp"] == 1 and args.posttrain_protocol["adapter"]["method"] == "lora":
            args._posttrain_previous_effective = {name: (module.adapter.linear_in.weight.detach().clone(), module.adapter.linear_out.weight.detach().clone()) for name, module in fused_modules(model).items()}
    expected = applied_lr(args.posttrain_protocol, completed)
    actual = [float(group["lr"]) for leaf in _iter_optimizer_leaves(optimizer) for group in leaf.optimizer.param_groups]
    if not actual or any(not math.isclose(lr, expected, rel_tol=1e-10, abs_tol=1e-14) for lr in actual):
        raise ValueError(f"Applied LR mismatch at step {completed + 1}: expected {expected}, got {actual}")
    event(args, "before_optimizer_step", completed_updates=completed, applied_lrs=actual, num_microbatches=num_microbatches)


def after_step(args, optimizer, grad_norm):
    if not math.isfinite(float(grad_norm)):
        raise ValueError("Non-finite gradient norm")
    args._posttrain_completed += 1
    if args._posttrain_completed in args.posttrain_protocol["checkpoint"]["complete_steps"]:
        _, model = optimizer._posttrain_context
        for i, chunk in enumerate(model):
            for name, value in chunk.named_parameters():
                key = f"chunk{i}.{name}"
                if key in args._posttrain_frozen and not torch.equal(value, args._posttrain_frozen[key]):
                    raise ValueError(f"Frozen base tensor changed: {key}")
    motion = {}
    if hasattr(args, "_posttrain_previous_masters"):
        from .checkpoint import _master_pairs

        _, model = optimizer._posttrain_context
        pairs = _master_pairs(model, optimizer)
        squared = sum((master.detach().double() - args._posttrain_previous_masters[name].double()).square().sum() for name, (_, master) in pairs.items())
        motion["master_parameter_motion_l2_local"] = float(squared.sqrt())
        del args._posttrain_previous_masters
        if hasattr(args, "_posttrain_previous_effective"):
            from .factors import effective_motion_squared

            squares = [effective_motion_squared(*args._posttrain_previous_effective[name], module.adapter.linear_in.weight, module.adapter.linear_out.weight) * (module.adapter.alpha / module.adapter.dim) ** 2 for name, module in fused_modules(model).items()]
            motion["effective_delta_motion_l2"] = float(sum(squares).sqrt())
            del args._posttrain_previous_effective
    event(args, "optimizer_step", completed_updates=args._posttrain_completed, rollout_id=args._posttrain_completed - 1, grad_norm=float(grad_norm), **motion)
    geometry_snapshot(args, optimizer)


def check_m6_advantages(args, data):
    from .objectives import m6_advantages

    stats = []
    values = []
    for old, teacher, behavior, length in zip(data["log_probs"], data["teacher_log_probs"], data["rollout_log_probs"], data["response_lengths"], strict=True):
        teacher = teacher.to(old.device)[-length:]
        values.append(m6_advantages(old, teacher, args.posttrain_protocol["reward_scale"]))
        stats.append((old - behavior.to(old.device)).detach().double().abs())
    errors = torch.cat(stats)
    totals = torch.stack((errors.sum(), errors.square().sum(), errors.new_tensor(errors.numel())))
    if dist.is_initialized():
        dist.all_reduce(totals)
    mean, rms = float(totals[0] / totals[2]), float((totals[1] / totals[2]).sqrt())
    if mean > 0.02 or rms > 0.05:
        raise ValueError(f"M6 sampled-logprob parity gate failed: mean={mean}, rms={rms}")
    event(args, "sampled_logprob_parity", mean_abs=mean, rms=rms, tokens=int(totals[2]))
    data["advantages"] = values
    data["returns"] = values


def should_save(args, rollout_id):
    steps = args.posttrain_protocol["checkpoint"]["complete_steps"]
    return rollout_id + 1 in steps


def geometry_snapshot(args, optimizer):
    step = args._posttrain_completed
    if step not in args.posttrain_protocol["checkpoint"]["geometry_steps"]:
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    # Geometry sources for distributed/TP layouts require their collective HF export.
    if args.posttrain_protocol["parallel"]["tp"] != 1:
        return
    from .checkpoint import _master_pairs

    _, model = optimizer._posttrain_context
    pairs = _master_pairs(model, optimizer)
    path = Path(args.posttrain_run_dir) / "geometry"
    path.mkdir(exist_ok=True)
    torch.save({"completed_optimizer_transactions": step, "direction_defined": step >= 2, "master": {name: master.detach().cpu() for name, (_, master) in pairs.items()}}, path / f"step_{step:06d}.pt")
