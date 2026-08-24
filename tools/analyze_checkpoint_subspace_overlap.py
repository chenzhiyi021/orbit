#!/usr/bin/env python3
"""Top-rank singular-subspace overlap between checkpoints, plus a cumulative/interval
cosine-similarity summary -- the Megatron torch-dist analogue of trl's
``scripts/compute_opd_trajectory_relations.py`` / ``opd_posthoc_common.subspace_overlap``.

Two delta kinds, both measured against the run's final checkpoint:

  cumulative[step] = W_step - W_base     (total update since the base model)
  interval[step]   = W_step - W_prev     (update since the *previous requested step*)

For each 2-D (or reshape-to-2-D) weight matrix, the top-``rank`` left/right
singular subspaces of ``delta[step]`` are compared against the final
checkpoint's subspaces via ``subspace_overlap`` (mean squared cosine of
principal angles -> 1.0 means the same subspace, ~rank/min(rows,cols) means
unrelated random subspaces). This tells you *where* (in which directions)
the update landed, which plain cosine similarity on the full delta can miss
if the update rotates within a low-rank subspace.

Outputs:
  --summary-csv   one row per (run, delta_kind, step): whole-model cosine to
                   final, plus the per-rank subspace overlap to final
                   averaged across all matrices (both a plain mean and one
                   weighted by the final delta's matrix Frobenius energy).
  --detail-csv    one row per (run, delta_kind, step, tensor, rank, side):
                   the raw per-matrix overlap (gzip'd -- this is large).
  --output        PNG with two panels (cosine, and rank-64 mean subspace
                   overlap) x (cumulative, interval), one line per run.

Memory note: this keeps the base state, the final checkpoint's cumulative
and interval deltas, and (transiently) each step's own state/deltas
resident in fp32 at once -- roughly 6-7x the model's parameter count. Run
it on a node with enough host RAM (or pass --device cuda if VRAM allows).
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _megatron_dcp_common import (
    global_cosine_similarity,
    load_weight_state_dict,
    parse_run_spec,
    resolve_checkpoint_dir,
    resolve_iter_dir,
)


DEFAULT_RANKS = (8, 16, 32, 64)


def parse_ranks(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(x) for x in value.split(",") if x.strip()}))


def as_matrix(tensor: torch.Tensor) -> torch.Tensor | None:
    """Reshape a weight tensor to 2-D for SVD, or None if it isn't matrix-like.

    Megatron sometimes stores a projection stacked across all layers as one
    leading-dimension tensor (shape[0] == num_layers) rather than one tensor
    per layer; those are treated here as a single flattened matrix
    (shape[0] rows, everything else collapsed into columns) rather than
    unstacked per layer, since that unstacking needs model config this
    script doesn't load. 1-D tensors (norm weights, biases) are skipped --
    they still count toward the whole-model cosine similarity, just not
    subspace overlap.
    """
    if tensor.dim() < 2:
        return None
    matrix = tensor.reshape(tensor.shape[0], -1) if tensor.dim() > 2 else tensor
    if min(matrix.shape) < 2:
        return None
    return matrix


def matrix_spectrum_and_basis(matrix: torch.Tensor, basis_rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the leading left/right singular bases (columns orthonormal)."""
    rows, cols = matrix.shape
    basis_rank = min(basis_rank, rows, cols)
    if rows >= cols:
        gram = matrix.mT @ matrix
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        singular = eigenvalues.clamp_min(0).sqrt().flip(0)
        right = eigenvectors.flip(1)[:, :basis_rank]
        scale = singular[:basis_rank].clamp_min(torch.finfo(matrix.dtype).eps)
        left = matrix @ right / scale.unsqueeze(0)
    else:
        gram = matrix @ matrix.mT
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        singular = eigenvalues.clamp_min(0).sqrt().flip(0)
        left = eigenvectors.flip(1)[:, :basis_rank]
        scale = singular[:basis_rank].clamp_min(torch.finfo(matrix.dtype).eps)
        right = matrix.mT @ left / scale.unsqueeze(0)

    left = torch.linalg.qr(left.float(), mode="reduced").Q
    right = torch.linalg.qr(right.float(), mode="reduced").Q
    return left.cpu().numpy(), right.cpu().numpy()


def subspace_overlap(left: np.ndarray, right: np.ndarray, rank: int) -> float:
    rank = min(rank, left.shape[1], right.shape[1])
    if rank == 0:
        return 0.0
    cross = left[:, :rank].T @ right[:, :rank]
    return float(np.square(cross).sum() / rank)


def build_bases(
    delta: dict[str, torch.Tensor], ranks: tuple[int, ...], device: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """SVD every matrix-like tensor in delta once, at the max requested rank."""
    max_rank = max(ranks)
    bases: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, tensor in delta.items():
        matrix = as_matrix(tensor)
        if matrix is None:
            continue
        bases[name] = matrix_spectrum_and_basis(matrix.to(device), max_rank)
    return bases


def frobenius_energy(delta: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(torch.sum(tensor.float() ** 2)) for name, tensor in delta.items()}


def compare_delta_to_final(
    *,
    run_label: str,
    delta_kind: str,
    step: int,
    delta: dict[str, torch.Tensor],
    delta_final: dict[str, torch.Tensor],
    bases_final: dict[str, tuple[np.ndarray, np.ndarray]],
    final_energy: dict[str, float],
    ranks: tuple[int, ...],
    device: str,
    detail_writer: csv.writer | None,
) -> dict:
    cosine, skipped = global_cosine_similarity(delta, delta_final)
    if skipped:
        print(f"[{run_label}] {delta_kind} step={step}: {len(skipped)} skipped/mismatched keys for cosine")

    per_rank_overlaps: dict[int, list[float]] = {rank: [] for rank in ranks}
    per_rank_weight: dict[int, list[float]] = {rank: [] for rank in ranks}
    for name, (final_left, final_right) in bases_final.items():
        tensor = delta.get(name)
        if tensor is None:
            continue
        matrix = as_matrix(tensor)
        if matrix is None:
            continue
        left, right = matrix_spectrum_and_basis(matrix.to(device), max(ranks))
        weight = final_energy.get(name, 0.0)
        for rank in ranks:
            overlap_left = subspace_overlap(left, final_left, rank)
            overlap_right = subspace_overlap(right, final_right, rank)
            overlap = 0.5 * (overlap_left + overlap_right)
            per_rank_overlaps[rank].append(overlap)
            per_rank_weight[rank].append(weight)
            if detail_writer is not None:
                detail_writer.writerow(
                    [run_label, delta_kind, step, name, rank, "left", overlap_left]
                )
                detail_writer.writerow(
                    [run_label, delta_kind, step, name, rank, "right", overlap_right]
                )

    summary = {"run": run_label, "delta_kind": delta_kind, "step": step, "cosine_to_final": cosine}
    for rank in ranks:
        values = np.array(per_rank_overlaps[rank])
        weights = np.array(per_rank_weight[rank])
        mean_overlap = float(values.mean()) if values.size else float("nan")
        weighted_overlap = (
            float(np.average(values, weights=weights)) if values.size and weights.sum() > 0 else mean_overlap
        )
        summary[f"subspace_overlap_rank{rank}_mean"] = mean_overlap
        summary[f"subspace_overlap_rank{rank}_energy_weighted"] = weighted_overlap
    return summary


def plot_delta_kind_panels(
    runs: list[dict],
    summaries: list[dict],
    ranks: tuple[int, ...],
    delta_kind: str,
    output_path: Path,
) -> None:
    """One cosine panel + one panel per rank, side by side -- each point is
    ``delta[step]`` (cumulative: ``W_step - W_base``; interval: ``W_step - W_prev``)
    compared against the run's final-checkpoint delta of the same kind. Mirrors
    the "Adjacent interval global cosine / rank-N overlap" reference plot style.
    """
    metrics = ["cosine_to_final"] + [f"subspace_overlap_rank{rank}_mean" for rank in ranks]
    kind_label = "Adjacent interval" if delta_kind == "interval" else "Cumulative"
    titles = [f"{kind_label} global cosine"] + [f"{kind_label} rank-{rank} overlap" for rank in ranks]

    figure, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.5), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")

    for run_index, run in enumerate(runs):
        color = colors[run_index % len(colors)]
        marker = markers[run_index % len(markers)]
        rows = sorted(
            (s for s in summaries if s["run"] == run["label"] and s["delta_kind"] == delta_kind),
            key=lambda s: s["step"],
        )
        steps = [s["step"] for s in rows]
        for axis, metric in zip(axes, metrics):
            axis.plot(steps, [s[metric] for s in rows], marker=marker, color=color, label=run["label"])

    for axis, title in zip(axes, titles):
        axis.set_xlabel("Later step" if delta_kind == "interval" else "Iteration")
        axis.set_title(title)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        dest="runs",
        help="label:checkpoint_root:iter1,iter2,...:final_iter -- repeatable",
    )
    parser.add_argument("--base", required=True, type=Path, help="MEGATRON_LOAD-style base checkpoint path")
    parser.add_argument("--ranks", type=parse_ranks, default=DEFAULT_RANKS, help="comma-separated subspace ranks")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summary-csv", type=Path, default=Path("subspace_overlap_summary.csv"))
    parser.add_argument("--detail-csv", type=Path, default=Path("subspace_overlap_detail.csv.gz"))
    parser.add_argument(
        "--output-interval",
        type=Path,
        default=Path("interval_to_final_subspace.png"),
        help="Multi-panel plot (cosine + one panel per rank) for the interval delta kind: "
        "each point is W_step - W_prev compared against the final checkpoint's own "
        "W_final - W_prev, i.e. 'does this step's update look like the final step's update'.",
    )
    parser.add_argument(
        "--output-cumulative",
        type=Path,
        default=Path("cumulative_to_final_subspace.png"),
        help="Same multi-panel plot for the cumulative delta kind (W_step - W_base vs. W_final - W_base).",
    )
    args = parser.parse_args()

    runs = [parse_run_spec(spec) for spec in args.runs]
    ranks = args.ranks

    base_dir = resolve_checkpoint_dir(args.base)
    print(f"loading base checkpoint from {base_dir}")
    base_state = load_weight_state_dict(base_dir)

    all_summaries: list[dict] = []
    with gzip.open(args.detail_csv, "wt", newline="") as detail_handle:
        detail_writer = csv.writer(detail_handle)
        detail_writer.writerow(["run", "delta_kind", "step", "tensor_name", "rank", "side", "overlap"])

        for run in runs:
            steps = run["iters"]
            final_iter = run["final_iter"]
            if final_iter not in steps:
                steps = sorted({*steps, final_iter})

            # Pass A: build the final checkpoint's cumulative/interval deltas and bases once.
            final_index = steps.index(final_iter)
            prev_iter_for_final = steps[final_index - 1] if final_index > 0 else None

            print(f"[{run['label']}] loading final checkpoint iter_{final_iter:07d}")
            final_state = load_weight_state_dict(resolve_iter_dir(run["root"], final_iter))
            cumulative_final = {
                k: final_state[k].to(torch.float32) - base_state[k].to(torch.float32)
                for k in set(final_state) & set(base_state)
                if final_state[k].shape == base_state[k].shape
            }
            if prev_iter_for_final is not None:
                print(f"[{run['label']}] loading iter_{prev_iter_for_final:07d} (final's interval reference)")
                prev_state_for_final = load_weight_state_dict(resolve_iter_dir(run["root"], prev_iter_for_final))
            else:
                prev_state_for_final = base_state
            interval_final = {
                k: final_state[k].to(torch.float32) - prev_state_for_final[k].to(torch.float32)
                for k in set(final_state) & set(prev_state_for_final)
                if final_state[k].shape == prev_state_for_final[k].shape
            }
            if prev_state_for_final is not base_state:
                del prev_state_for_final
            del final_state

            print(f"[{run['label']}] building final-step subspace bases (ranks={ranks})")
            bases_cumulative_final = build_bases(cumulative_final, ranks, args.device)
            bases_interval_final = build_bases(interval_final, ranks, args.device)
            energy_cumulative_final = frobenius_energy(cumulative_final)
            energy_interval_final = frobenius_energy(interval_final)

            # Pass B: walk every requested step, comparing its own cumulative/interval
            # delta against the final references above, then discard.
            prev_state = base_state
            for step in steps:
                print(f"[{run['label']}] loading iter_{step:07d}")
                state = load_weight_state_dict(resolve_iter_dir(run["root"], step))

                cumulative_step = {
                    k: state[k].to(torch.float32) - base_state[k].to(torch.float32)
                    for k in set(state) & set(base_state)
                    if state[k].shape == base_state[k].shape
                }
                interval_step = {
                    k: state[k].to(torch.float32) - prev_state[k].to(torch.float32)
                    for k in set(state) & set(prev_state)
                    if state[k].shape == prev_state[k].shape
                }

                summary_cumulative = compare_delta_to_final(
                    run_label=run["label"],
                    delta_kind="cumulative",
                    step=step,
                    delta=cumulative_step,
                    delta_final=cumulative_final,
                    bases_final=bases_cumulative_final,
                    final_energy=energy_cumulative_final,
                    ranks=ranks,
                    device=args.device,
                    detail_writer=detail_writer,
                )
                summary_interval = compare_delta_to_final(
                    run_label=run["label"],
                    delta_kind="interval",
                    step=step,
                    delta=interval_step,
                    delta_final=interval_final,
                    bases_final=bases_interval_final,
                    final_energy=energy_interval_final,
                    ranks=ranks,
                    device=args.device,
                    detail_writer=detail_writer,
                )
                all_summaries.append(summary_cumulative)
                all_summaries.append(summary_interval)

                del cumulative_step, interval_step
                if prev_state is not base_state:
                    del prev_state
                prev_state = state

    summary_fields = (
        ["run", "delta_kind", "step", "cosine_to_final"]
        + [f"subspace_overlap_rank{rank}_mean" for rank in ranks]
        + [f"subspace_overlap_rank{rank}_energy_weighted" for rank in ranks]
    )
    with args.summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"Saved summary CSV to {args.summary_csv}")
    print(f"Saved detail CSV to {args.detail_csv}")

    plot_delta_kind_panels(runs, all_summaries, ranks, "interval", args.output_interval)
    plot_delta_kind_panels(runs, all_summaries, ranks, "cumulative", args.output_cumulative)


if __name__ == "__main__":
    main()
