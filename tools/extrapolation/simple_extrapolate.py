#!/usr/bin/env python3
"""Simple (fixed-alpha, no validation gating) checkpoint extrapolation.

The "naive" baseline this package compares EffOPD against: given two already
-trained checkpoints (an anchor B and a later checkpoint A from the same
run), materialize `A + alpha * (A - B)` for one or more fixed alphas, with no
adaptive accept/reject search. This is the "fixed extrapolation strategy"
family the EffOPD paper contrasts itself with (Section 4.2: "Unlike AlphaOPD
and ExOPD, which use fixed extrapolation strategies, EffOPD adaptively
selects the extrapolation magnitude via validation feedback") -- useful as a
sanity baseline: if a fixed alpha does about as well as EffOPD's gated
search, the validation step in effopd_extrapolate.py isn't buying much on
this run; if fixed alphas are unstable/degrade while EffOPD stays flat, that
is the acceleration story from the paper reproducing here.

By default this extrapolates every parameter (unlike EffOPD, which restricts
to the attention/MLP substring match in `checkpoint_io.EFFOPD_KEYWORDS`).
Pass --keyword-only to restrict to the same subset, for an apples-to-apples
comparison against effopd_extrapolate.py isolating the validation-gating
effect from the parameter-subset effect.

Example (--run-dir and --base-model default to the lr=2e-6 run and this
cluster's base Qwen3-1.7B; run from the orbit repo root, in the
orbit_env_v2 environment):

    python tools/extrapolation/simple_extrapolate.py \\
        --anchor-iter 17 --current-iter 19 \\
        --alphas 2 4 6 8 10 \\
        --output-root extrapolation_results/simple/lr_2e-6
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from checkpoint_io import (
    CheckpointTensorStore,
    build_candidate,
    convert_iter_to_hf,
    resolve_run_iterations,
    write_hf_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("orbit_ckpts/m6_st_full_non_thinking_lr_2e-6"),
        help="orbit_ckpts/<run> directory holding iter_NNNNNNN DCP checkpoints.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/mnt/L202500431/models/qwen3-1.7b"),
        help="HF checkpoint dir the run was trained from (the launcher's HF_CKPT). "
        "Used as the tokenizer/config source and, when --anchor-iter base is passed, as the anchor. "
        "Default: this cluster's base Qwen3-1.7B under ../models/qwen3-1.7b.",
    )
    parser.add_argument(
        "--anchor-iter",
        default="base",
        help="Iteration number for the anchor B, or the literal 'base' to use --base-model as B "
        "(default: 'base', i.e. extrapolate the whole run's displacement from initialization).",
    )
    parser.add_argument(
        "--current-iter",
        type=int,
        default=None,
        help="Iteration number for the current checkpoint A (default: the run's latest saved iteration).",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 8.0, 10.0],
        help="Extrapolation magnitudes to sweep (candidate = A + alpha*(A-B) per alpha). "
        "Default mirrors EffOPD's released alpha set (2,4,6,8,10) for comparability.",
    )
    parser.add_argument(
        "--keyword-only",
        action="store_true",
        help="Restrict to the same attention/MLP substring subset EffOPD uses, instead of the whole model.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Where converted (Megatron->HF) anchor/current checkpoints are cached "
        "(default: <run-dir>/../_hf_cache/<run-name>).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where candidate checkpoints are written, one subdir per alpha "
        "(default: extrapolation_results/simple/<run-name>).",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Interpreter used for the torch_dist->HF conversion subprocess.")
    return parser.parse_args()


def resolve_checkpoint_hf_dir(
    label: str,
    iteration: int | None,
    run_dir: Path,
    base_model: Path,
    hf_cache_dir: Path,
    python_bin: str,
) -> Path:
    if iteration is None:
        return base_model
    checkpoint_dir = run_dir / f"iter_{iteration:07d}"
    if not (checkpoint_dir / ".metadata").is_file():
        raise FileNotFoundError(f"{label}: no DCP checkpoint metadata at {checkpoint_dir}")
    output_dir = hf_cache_dir / checkpoint_dir.name
    print(f"[simple-extrapolate] converting {label} ({checkpoint_dir.name}) -> {output_dir}", flush=True)
    return convert_iter_to_hf(checkpoint_dir, base_model, output_dir, python_bin=python_bin)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    base_model = args.base_model.expanduser().resolve()
    run_name = run_dir.name

    available = resolve_run_iterations(run_dir)
    current_iter = args.current_iter if args.current_iter is not None else available[-1]
    if current_iter not in available:
        raise ValueError(f"--current-iter {current_iter} not among saved iterations {available}")
    anchor_iter = None if args.anchor_iter == "base" else int(args.anchor_iter)
    if anchor_iter is not None and anchor_iter not in available:
        raise ValueError(f"--anchor-iter {anchor_iter} not among saved iterations {available}")

    hf_cache_dir = args.hf_cache_dir or (run_dir.parent / "_hf_cache" / run_name)
    output_root = args.output_root or (Path("extrapolation_results") / "simple" / run_name)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    anchor_hf = resolve_checkpoint_hf_dir(
        "anchor", anchor_iter, run_dir, base_model, hf_cache_dir, args.python_bin
    )
    current_hf = resolve_checkpoint_hf_dir(
        "current", current_iter, run_dir, base_model, hf_cache_dir, args.python_bin
    )

    results = []
    with ExitStack() as stack:
        current_store = CheckpointTensorStore(current_hf, stack)
        anchor_store = CheckpointTensorStore(anchor_hf, stack)
        for alpha in args.alphas:
            output_dir = output_root / (
                f"anchor_{'base' if anchor_iter is None else f'iter{anchor_iter}'}"
                f"_current_iter{current_iter}_alpha{alpha:g}"
            )
            tensors, stats = build_candidate(current_store, anchor_store, alpha, args.keyword_only)
            metadata = {
                "created_at": datetime.now(UTC).isoformat(),
                "method": "simple fixed-alpha directional extrapolation (no validation gating)",
                "definition": "candidate = current + alpha * (current - anchor)",
                "run_dir": str(run_dir),
                "base_model": str(base_model),
                "anchor": "base" if anchor_iter is None else anchor_iter,
                "anchor_hf_dir": str(anchor_hf),
                "current_iter": current_iter,
                "current_hf_dir": str(current_hf),
                **stats,
            }
            print(f"[simple-extrapolate] writing alpha={alpha:g} -> {output_dir}", flush=True)
            write_hf_checkpoint(tensors, current_hf, output_dir, metadata)
            results.append({"alpha": alpha, "output_dir": str(output_dir), **stats})

    manifest = {
        "run_dir": str(run_dir),
        "anchor": "base" if anchor_iter is None else anchor_iter,
        "current_iter": current_iter,
        "keyword_only": args.keyword_only,
        "candidates": results,
    }
    manifest_path = output_root / f"manifest_anchor_{'base' if anchor_iter is None else anchor_iter}_current{current_iter}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[simple-extrapolate] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
