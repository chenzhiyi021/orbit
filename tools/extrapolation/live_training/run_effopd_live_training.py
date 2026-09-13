#!/usr/bin/env python3
"""Live EffOPD: actually shortens a real orbit OPD training run.

Everything in `tools/extrapolation/*.py` (the parent directory) is post-hoc:
it reads checkpoints a training run already produced and never touches the
trainer. This script is the opposite -- it drives real
`examples/on_policy_distillation/unified_300step_constant/run-m6-opd-st-non-thinking-full.sh`
invocations end to end, so the model each segment actually trains from *is*
the (possibly extrapolated) accepted checkpoint, matching the paper's
Section 4.1 definition where later steps build on W^EffOPD, not on a
parallel copy of it.

Segment plan for --total-steps 20 (paper schedule t=2^n, n=0..4, all <=20):

    n=0: train step        1  (NUM_ROLLOUT=1)  -> extrapolation search at t=1
    n=1: train step        2  (NUM_ROLLOUT=1)  -> extrapolation search at t=2
    n=2: train steps     3-4  (NUM_ROLLOUT=2)  -> extrapolation search at t=4
    n=3: train steps     5-8  (NUM_ROLLOUT=4)  -> extrapolation search at t=8
    n=4: train steps    9-16  (NUM_ROLLOUT=8)  -> extrapolation search at t=16
         train steps   17-20  (NUM_ROLLOUT=4)  -> finish; no trigger (2^5=32>20)

1+1+2+4+8+4 = 20 real optimizer steps total, same budget as the existing
lr sweep this study has been comparing against.

At each trigger t, the anchor for Delta_n is the PREVIOUS segment's accepted
checkpoint (real, or extrapolated if a candidate won that round) -- not
necessarily a plain-trained checkpoint. This is what makes it "live" instead
of the oracle-style post-hoc convention `effopd_extrapolate.py` uses
(deliberately different; see that script's docstring for why the post-hoc
tool always diffs real checkpoints instead).

*** READ `live_training/README.md`'s "What is genuinely unverified" section
*** before running this. It launches real multi-GPU training jobs and stages
*** checkpoints through an HF->Megatron path (`megatron_checkpoint_stage.py`)
*** whose iteration-numbering behavior could not be confirmed without a
*** cluster to test on. Segment 1 is cheap (one real step); let it finish
*** and inspect its logs/wandb step count before trusting the rest of a run.

Example (every path arg now defaults to a confirmed location under this
cluster's /mnt/L202500431/..., so only --validator-cmd is required):

    python tools/extrapolation/live_training/run_effopd_live_training.py \\
        --validator-cmd "python tools/extrapolation/validate_checkpoint.py \\
            --evalchemy-root /mnt/L202500431/third_party/evalchemy \\
            --task aime24 --num-samples 50 --num-gpus 2 --eval-tp-size 1" \\
        --live-root live_training_runs/effopd_lr_2e-6 \\
        --total-steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/extrapolation, for checkpoint_io

from checkpoint_io import (  # noqa: E402
    CheckpointTensorStore,
    build_candidate,
    convert_iter_to_hf,
    resolve_run_iterations,
    write_hf_checkpoint,
)
from megatron_checkpoint_stage import materialize_accepted_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/mnt/L202500431/models/qwen3-1.7b"),
        help="HF checkpoint dir (W_0 and this run's HF_CKPT/tokenizer source). "
        "Default confirmed against this cluster's ../models layout.",
    )
    parser.add_argument(
        "--megatron-base",
        type=Path,
        default=Path("/mnt/L202500431/models/megatron_ckpt/qwen3-1.7b"),
        help="Megatron-format version of --base-model, used as the first segment's MEGATRON_LOAD. "
        "Default confirmed via `ls ../models/megatron_ckpt` (qwen3-1.7b subdir present).",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("/mnt/L202500431/datasets/openreasoning_mixed_100k/train.parquet"),
        help="Rollout prompt data (the launcher's TRAIN_JSONL). Default confirmed via `ls ../datasets`.",
    )
    parser.add_argument(
        "--teacher-hf-ckpt",
        type=Path,
        default=Path("/mnt/L202500431/models/qwen3-4b-instruct-2507"),
        help="Teacher checkpoint (OPD_TEACHER_CKPT). Default confirmed via `ls ../models`.",
    )
    parser.add_argument(
        "--evalchemy-root",
        type=Path,
        default=Path("/mnt/L202500431/third_party/evalchemy"),
        help="Forwarded to the launcher's eval stage (unused; RUN_EVAL stays 0) and available for "
        "--validator-cmd to reference. Default is what you've been passing to eval-math-evalchemy.sh.",
    )
    parser.add_argument("--lr", default="2e-6", help="Constant LR for every segment (default matches this study's lr=2e-6 arm).")
    parser.add_argument("--total-steps", type=int, default=20, help="Total real optimizer steps across the whole live run.")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 8.0, 10.0],
        help="Candidate magnitudes 2k for k=1..5, tried in order (paper default: 2 4 6 8 10).",
    )
    parser.add_argument(
        "--validator-cmd",
        required=True,
        help="Same contract as effopd_extrapolate.py: scores one HF checkpoint dir (appended as the final "
        "argument), must print a final stdout line that is JSON {'acc': <float>} or a bare float.",
    )
    parser.add_argument(
        "--live-root",
        type=Path,
        default=Path("live_training_runs/effopd_lr_2e-6"),
        help="Where this run's segment checkpoints, staged Megatron checkpoints, and manifest are written.",
    )
    parser.add_argument(
        "--launcher-script",
        type=Path,
        default=Path("examples/on_policy_distillation/unified_300step_constant/run-m6-opd-st-non-thinking-full.sh"),
        help="Orbit launcher invoked once per segment (relative to --orbit-root unless absolute).",
    )
    parser.add_argument("--gpus-per-node", type=int, default=2, help="GPUS_PER_NODE forwarded to every segment.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra env var forwarded to every segment (repeatable), e.g. --env OPD_TEACHER_NUM_GPUS=2.",
    )
    parser.add_argument("--orbit-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def parse_validator_score(stdout: str) -> float:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Validator produced no output")
    last = lines[-1]
    try:
        parsed = json.loads(last)
        if isinstance(parsed, dict):
            return float(parsed["acc"])
        return float(parsed)
    except json.JSONDecodeError:
        return float(last)


def validate(validator_cmd: str, checkpoint_dir: Path) -> float:
    result = subprocess.run(shlex.split(validator_cmd) + [str(checkpoint_dir)], check=True, capture_output=True, text=True)
    return parse_validator_score(result.stdout)


def build_trigger_schedule(total_steps: int) -> list[int]:
    triggers = []
    n = 0
    while 2**n <= total_steps:
        triggers.append(2**n)
        n += 1
    return triggers


def run_segment(
    *,
    args: argparse.Namespace,
    segment_dir: Path,
    megatron_load: Path,
    num_rollout: int,
    expected_end_iteration: int,
    wandb_group: str,
) -> Path:
    """Run one real training segment via the existing launcher script.

    Returns the resolved `iter_{expected_end_iteration:07d}` checkpoint dir
    under `segment_dir`, after verifying that iteration number is actually
    what got saved -- fail loudly here rather than silently drift the step
    budget if the MEGATRON_LOAD resume didn't continue counting the way this
    script assumes (see megatron_checkpoint_stage.py's docstring).
    """
    launcher = args.launcher_script if args.launcher_script.is_absolute() else args.orbit_root / args.launcher_script
    env = os.environ.copy()
    env.update(
        {
            "HF_CKPT": str(args.base_model),
            "MEGATRON_LOAD": str(megatron_load),
            "SAVE_DIR": str(segment_dir),
            "OPD_TEACHER_CKPT": str(args.teacher_hf_ckpt),
            "TRAIN_JSONL": str(args.train_jsonl),
            "EVALCHEMY_ROOT": str(args.evalchemy_root),
            "LR": str(args.lr),
            "NUM_ROLLOUT": str(num_rollout),
            "GPUS_PER_NODE": str(args.gpus_per_node),
            "WANDB_GROUP": wandb_group,
            "RUN_TRAIN": "1",
            "RUN_EVAL": "0",
        }
    )
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value

    print(f"[live-effopd] segment: MEGATRON_LOAD={megatron_load} NUM_ROLLOUT={num_rollout} -> SAVE_DIR={segment_dir}", flush=True)
    subprocess.run(["bash", str(launcher)], env=env, check=True)

    produced = resolve_run_iterations(segment_dir)
    if max(produced) != expected_end_iteration:
        raise RuntimeError(
            f"Segment produced iterations {produced} under {segment_dir}, expected max="
            f"{expected_end_iteration}. This means MEGATRON_LOAD={megatron_load} did not resume "
            "iteration counting the way this script assumes -- STOP and read "
            "live_training/README.md's 'What is genuinely unverified' section before continuing; "
            "do not just bump --total-steps or re-run to paper over this."
        )
    return segment_dir / f"iter_{expected_end_iteration:07d}"


def extrapolate_at_trigger(
    *,
    args: argparse.Namespace,
    n: int,
    t: int,
    current_iter_dir: Path,
    anchor_hf_dir: Path,
    hf_cache_dir: Path,
    output_root: Path,
) -> dict:
    """Same accept-while-improving search as effopd_extrapolate.py, but the
    anchor is whatever this live run actually accepted last round (possibly
    itself extrapolated), not necessarily a plain-trained checkpoint."""
    current_hf = convert_iter_to_hf(current_iter_dir, args.base_model, hf_cache_dir / f"real_iter{t:07d}", python_bin=args.python_bin)

    n_dir = output_root / f"n{n}_t{t}"
    print(f"[live-effopd] n={n} t={t}: scoring baseline (unmodified real checkpoint)", flush=True)
    v_acc = validate(args.validator_cmd, current_hf)
    accepted_dir = current_hf
    accepted_alpha = 0.0
    accepted_score = v_acc
    log = [{"k": 0, "alpha": 0.0, "score": v_acc, "accepted": True, "candidate": str(current_hf)}]

    with ExitStack() as stack:
        current_store = CheckpointTensorStore(current_hf, stack)
        anchor_store = CheckpointTensorStore(anchor_hf_dir, stack)
        for k, alpha in enumerate(args.alphas, start=1):
            candidate_dir = n_dir / f"k{k}_alpha{alpha:g}"
            tensors, stats = build_candidate(current_store, anchor_store, alpha, keyword_filter=True)
            metadata = {
                "created_at": datetime.now(UTC).isoformat(),
                "method": "EffOPD directional extrapolation (live)",
                "n": n,
                "t": t,
                "k": k,
                "anchor_hf_dir": str(anchor_hf_dir),
                "current_hf_dir": str(current_hf),
                **stats,
            }
            write_hf_checkpoint(tensors, current_hf, candidate_dir, metadata)
            score = validate(args.validator_cmd, candidate_dir)
            improved = score >= v_acc
            log.append({"k": k, "alpha": alpha, "score": score, "accepted": improved, "candidate": str(candidate_dir)})
            print(f"[live-effopd] n={n} t={t} k={k} alpha={alpha:g}: score={score:.4f} (v_acc={v_acc:.4f}) -> {'accept' if improved else 'STOP'}", flush=True)
            if improved:
                v_acc, accepted_dir, accepted_alpha, accepted_score = score, candidate_dir, alpha, score
            else:
                break

    result = {
        "n": n,
        "t": t,
        "accepted_alpha": accepted_alpha,
        "accepted_hf_dir": str(accepted_dir),
        "accepted_score": accepted_score,
        "search_log": log,
    }
    n_dir.mkdir(parents=True, exist_ok=True)
    (n_dir / "search_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"[live-effopd] n={n} t={t}: accepted alpha={accepted_alpha:g} score={accepted_score:.4f}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    live_root = args.live_root.expanduser().resolve()
    hf_cache_dir = live_root / "_hf_cache"
    extrapolation_root = live_root / "extrapolation"
    accepted_megatron_root = live_root / "accepted_megatron"
    real_segments_root = live_root / "segments"
    for path in (hf_cache_dir, extrapolation_root, accepted_megatron_root, real_segments_root):
        path.mkdir(parents=True, exist_ok=True)

    wandb_group = f"effopd-live-lr{args.lr}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    triggers = build_trigger_schedule(args.total_steps)
    print(f"[live-effopd] total_steps={args.total_steps} triggers(t)={triggers} wandb_group={wandb_group}", flush=True)

    accepted_hf_dir = args.base_model.expanduser().resolve()
    accepted_megatron_load = args.megatron_base.expanduser().resolve()
    accepted_iteration = 0
    manifest_segments = []
    manifest_extrapolations = []

    for n, t in enumerate(triggers):
        num_rollout = t - accepted_iteration
        segment_dir = real_segments_root / f"n{n}_t{t}"
        end_dir = run_segment(
            args=args,
            segment_dir=segment_dir,
            megatron_load=accepted_megatron_load,
            num_rollout=num_rollout,
            expected_end_iteration=t,
            wandb_group=wandb_group,
        )
        manifest_segments.append({"n": n, "start_iteration": accepted_iteration, "end_iteration": t, "num_rollout": num_rollout, "save_dir": str(segment_dir)})

        result = extrapolate_at_trigger(
            args=args,
            n=n,
            t=t,
            current_iter_dir=end_dir,
            anchor_hf_dir=accepted_hf_dir,
            hf_cache_dir=hf_cache_dir,
            output_root=extrapolation_root,
        )
        manifest_extrapolations.append(result)

        accepted_hf_dir = Path(result["accepted_hf_dir"])
        accepted_megatron_load = materialize_accepted_checkpoint(
            hf_dir=accepted_hf_dir,
            target_iteration=t,
            accepted_root=accepted_megatron_root,
            python_bin=args.python_bin,
            orbit_root=args.orbit_root,
        )
        accepted_iteration = t

    if accepted_iteration < args.total_steps:
        num_rollout = args.total_steps - accepted_iteration
        segment_dir = real_segments_root / f"final_{accepted_iteration}_to_{args.total_steps}"
        end_dir = run_segment(
            args=args,
            segment_dir=segment_dir,
            megatron_load=accepted_megatron_load,
            num_rollout=num_rollout,
            expected_end_iteration=args.total_steps,
            wandb_group=wandb_group,
        )
        manifest_segments.append(
            {"n": None, "start_iteration": accepted_iteration, "end_iteration": args.total_steps, "num_rollout": num_rollout, "save_dir": str(segment_dir)}
        )
        final_hf = convert_iter_to_hf(end_dir, args.base_model, hf_cache_dir / f"real_iter{args.total_steps:07d}", python_bin=args.python_bin)
    else:
        final_hf = accepted_hf_dir

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "total_steps": args.total_steps,
        "triggers": triggers,
        "lr": args.lr,
        "base_model": str(args.base_model),
        "megatron_base": str(args.megatron_base),
        "wandb_group": wandb_group,
        "segments": manifest_segments,
        "extrapolations": manifest_extrapolations,
        "final_checkpoint_hf_dir": str(final_hf),
    }
    manifest_path = live_root / "live_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[live-effopd] done. final checkpoint: {final_hf}\n[live-effopd] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
