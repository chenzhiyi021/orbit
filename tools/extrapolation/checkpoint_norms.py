#!/usr/bin/env python3
"""Reference L2-norm table for `orbit_ckpts/<run>` checkpoints.

Prints, for every saved checkpoint (default: all of them), the Frobenius
norm of its displacement from the base model (``||W_t - W_0||``, the "total
so-far" displacement) and from the immediately preceding saved checkpoint
(``||W_t - W_{t-1}||``, the per-save-interval step delta) -- both
whole-model and restricted to EffOPD's attention/MLP keyword subset
(`checkpoint_io.EFFOPD_KEYWORDS`), plus each delta as a fraction of the
current weight norm for scale.

Meant to be run *before* `simple_extrapolate.py` / `effopd_extrapolate.py`:
a candidate is ``current + alpha * delta``, so alpha=10 turns whatever
``||delta||`` this prints into an 11x-sized perturbation. Knowing that
number up front is cheaper than discovering an alpha was wildly too
aggressive after a full serve+eval cycle.

Example (defaults match this study's lr=2e-6 run):

    python tools/extrapolation/checkpoint_norms.py \\
        --output extrapolation_results/norms/lr_2e-6.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path

from checkpoint_io import (
    CheckpointTensorStore,
    convert_iter_to_hf,
    resolve_run_iterations,
    should_modify,
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
        help="HF checkpoint dir the run was trained from (W_0). Default: this cluster's base Qwen3-1.7B.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=None,
        help="Iteration numbers to report (default: every saved iteration under --run-dir).",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Where converted (Megatron->HF) checkpoints are cached (default: <run-dir>/../_hf_cache/<run-name>, "
        "shared with simple_extrapolate.py/effopd_extrapolate.py so this doesn't reconvert what they already did).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the full table as JSON.")
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def squared_norms(current: CheckpointTensorStore, previous: CheckpointTensorStore) -> dict[str, float]:
    """Sum-of-squares for (current - previous), split into the EffOPD keyword
    subset vs. everything else, plus ||current|| itself, all in fp32."""
    keyword_delta_sq = 0.0
    rest_delta_sq = 0.0
    keyword_weight_sq = 0.0
    rest_weight_sq = 0.0
    missing = sorted(current.keys - previous.keys)
    if missing:
        raise ValueError(f"current has tensors absent from previous: {missing[:10]}")
    for name in sorted(current.keys):
        current_tensor = current.get_tensor(name).float()
        previous_tensor = previous.get_tensor(name).float()
        if current_tensor.shape != previous_tensor.shape:
            raise ValueError(f"Shape mismatch for {name}")
        delta_sq = float((current_tensor - previous_tensor).square().sum())
        weight_sq = float(current_tensor.square().sum())
        if should_modify(name):
            keyword_delta_sq += delta_sq
            keyword_weight_sq += weight_sq
        else:
            rest_delta_sq += delta_sq
            rest_weight_sq += weight_sq
    return {
        "delta_norm_keyword_subset": math.sqrt(keyword_delta_sq),
        "delta_norm_rest": math.sqrt(rest_delta_sq),
        "delta_norm_whole_model": math.sqrt(keyword_delta_sq + rest_delta_sq),
        "weight_norm_keyword_subset": math.sqrt(keyword_weight_sq),
        "weight_norm_rest": math.sqrt(rest_weight_sq),
        "weight_norm_whole_model": math.sqrt(keyword_weight_sq + rest_weight_sq),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    base_model = args.base_model.expanduser().resolve()
    run_name = run_dir.name
    hf_cache_dir = args.hf_cache_dir or (run_dir.parent / "_hf_cache" / run_name)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    available = resolve_run_iterations(run_dir)
    iters = sorted(args.iters) if args.iters else available
    unknown = [i for i in iters if i not in available]
    if unknown:
        raise ValueError(f"--iters {unknown} not among saved iterations {available}")

    def hf_dir_for(iteration: int) -> Path:
        checkpoint_dir = run_dir / f"iter_{iteration:07d}"
        output_dir = hf_cache_dir / checkpoint_dir.name
        print(f"[norms] converting iter_{iteration:07d} -> {output_dir}", flush=True)
        return convert_iter_to_hf(checkpoint_dir, base_model, output_dir, python_bin=args.python_bin)

    rows = []
    previous_iter: int | None = None
    with ExitStack() as stack:
        base_store = CheckpointTensorStore(base_model, stack)
        previous_store = base_store
        for iteration in iters:
            current_store = CheckpointTensorStore(hf_dir_for(iteration), stack)

            from_base = squared_norms(current_store, base_store)
            from_previous = squared_norms(current_store, previous_store)
            row = {
                "iteration": iteration,
                "previous_iteration": previous_iter,  # None means base model
                "from_base": {
                    "whole_model": from_base["delta_norm_whole_model"],
                    "keyword_subset": from_base["delta_norm_keyword_subset"],
                    "keyword_fraction_of_weight_norm": (
                        from_base["delta_norm_keyword_subset"] / from_base["weight_norm_keyword_subset"]
                        if from_base["weight_norm_keyword_subset"]
                        else float("nan")
                    ),
                },
                "from_previous_saved_checkpoint": {
                    "whole_model": from_previous["delta_norm_whole_model"],
                    "keyword_subset": from_previous["delta_norm_keyword_subset"],
                },
                "weight_norm_whole_model": from_base["weight_norm_whole_model"],
                "weight_norm_keyword_subset": from_base["weight_norm_keyword_subset"],
            }
            rows.append(row)
            print(
                f"[norms] iter_{iteration:07d}: "
                f"||W-W0||(keyword)={row['from_base']['keyword_subset']:.3f}  "
                f"||W-W0||(whole)={row['from_base']['whole_model']:.3f}  "
                f"||W-Wprev||(keyword)={row['from_previous_saved_checkpoint']['keyword_subset']:.3f}  "
                f"||W-Wprev||(whole)={row['from_previous_saved_checkpoint']['whole_model']:.3f}",
                flush=True,
            )
            previous_store = current_store
            previous_iter = iteration

    print("\n[norms] summary (keyword_subset = attention+MLP tensors EffOPD actually modifies):")
    header = f"{'iter':>6}  {'||W-W0|| kw':>14}  {'||W-W0|| all':>14}  {'||W-Wprev|| kw':>16}  {'||W-Wprev|| all':>16}"
    print(header)
    for row in rows:
        print(
            f"{row['iteration']:>6}  "
            f"{row['from_base']['keyword_subset']:>14.3f}  "
            f"{row['from_base']['whole_model']:>14.3f}  "
            f"{row['from_previous_saved_checkpoint']['keyword_subset']:>16.3f}  "
            f"{row['from_previous_saved_checkpoint']['whole_model']:>16.3f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"run_dir": str(run_dir), "base_model": str(base_model), "rows": rows}, indent=2) + "\n")
        print(f"\n[norms] wrote {args.output}")


if __name__ == "__main__":
    main()
