#!/usr/bin/env python3
"""Compare intermediate Megatron torch-dist checkpoints against a run's final checkpoint.

For each ``--run`` spec, loads the raw (unconverted) Megatron distributed
state dict at a series of iterations and computes the global cosine
similarity of each intermediate checkpoint's *update relative to the base
model* (``delta_iter = W_iter - W_base``) against the final checkpoint's
update (``delta_final = W_final - W_base``). All runs are plotted on one
figure so their similarity-to-final trajectories can be compared directly.

Comparing deltas rather than raw weights matters: raw weights are dominated
by the shared base-model component (``W_t = W_base + delta_t``, with
``delta_t`` tiny next to ``W_base``), so raw-weight cosine similarity sits
at ~1.0 regardless of how much training actually changed the model. Deltas
isolate the actual training-induced update.

This intentionally skips HF conversion (no layer/expert unrolling, no
architecture-specific renaming) since we only need raw tensors to compare
checkpoints within the same run/architecture against each other.

See also ``tools/analyze_checkpoint_subspace_overlap.py``, which extends
this same delta-comparison idea with per-tensor top-rank singular-subspace
overlap and an interval (step-to-step) delta view alongside cumulative.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _megatron_dcp_common import (
    compute_delta,
    global_cosine_similarity,
    load_weight_state_dict,
    parse_run_spec,
    resolve_checkpoint_dir,
    resolve_iter_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        dest="runs",
        help=(
            "label:checkpoint_root:iter1,iter2,...:final_iter -- repeatable, "
            "one per run to plot on the same figure"
        ),
    )
    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help=(
            "MEGATRON_LOAD-style path to the base/pretrained-init Megatron dist "
            "checkpoint (same directory you pass as --load when launching "
            "training), shared across all --run entries. Resolved the same way "
            "Megatron resolves --load: latest_checkpointed_iteration.txt or the "
            "highest iter_* subdirectory. All comparisons use "
            "delta = weights - base_weights."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("ckpt_final_similarity.png"))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    runs = [parse_run_spec(spec) for spec in args.runs]

    base_dir = resolve_checkpoint_dir(args.base)
    print(f"loading base checkpoint from {base_dir}")
    base_state = load_weight_state_dict(base_dir)

    figure, axis = plt.subplots(figsize=(7, 5))
    csv_rows: list[tuple[str, int, float]] = []
    for run in runs:
        final_dir = resolve_iter_dir(run["root"], run["final_iter"])
        print(f"[{run['label']}] loading final checkpoint iter_{run['final_iter']:07d}")
        final_state = load_weight_state_dict(final_dir)
        delta_final, final_skipped = compute_delta(final_state, base_state)
        if final_skipped:
            print(f"[{run['label']}] final iter_{run['final_iter']:07d}: {len(final_skipped)} keys missing from base")
        del final_state

        steps: list[int] = []
        sims: list[float] = []
        for iteration in run["iters"]:
            iter_dir = resolve_iter_dir(run["root"], iteration)
            print(f"[{run['label']}] loading iter_{iteration:07d}")
            state = load_weight_state_dict(iter_dir)
            delta_iter, iter_skipped = compute_delta(state, base_state)
            del state
            if iter_skipped:
                print(f"[{run['label']}] iter_{iteration:07d}: {len(iter_skipped)} keys missing from base")

            similarity, skipped = global_cosine_similarity(delta_iter, delta_final)
            if skipped:
                print(f"[{run['label']}] iter_{iteration:07d}: {len(skipped)} skipped/mismatched delta keys")
            steps.append(iteration)
            sims.append(similarity)
            csv_rows.append((run["label"], iteration, similarity))
            del delta_iter

        axis.plot(steps, sims, marker="o", label=run["label"])
        del delta_final

    axis.set_xlabel("Iteration")
    axis.set_ylabel("Cosine similarity of ΔW_iter to ΔW_final")
    axis.set_title("Checkpoint update similarity to final update (ΔW vs. base)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output, dpi=150)
    print(f"Saved figure to {args.output}")

    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run", "iteration", "delta_cosine_similarity_to_final"])
            writer.writerows(csv_rows)
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
