"""Plot per-tensor-kind top-k subspace overlap from a saved
subspace_overlap_profile.py transcript.

Sibling of plot_attention_circuit_overlap.py, but for logs produced by
subspace_overlap_profile.py directly (raw weight-tensor deltas -- e.g. all
MLP gate/up/down projections across every layer), not
attention_circuit_overlap_profile.py's composed QK/OV circuits. The two
scripts' log formats differ (plain tensor-name headers here vs layer/head +
circuit headers there), hence a separate small parser instead of overloading
one regex set for both.

Groups tensors by "kind" -- the second-to-last dot-separated component of
the tensor name before ".weight" (e.g. "down_proj", "gate_proj") -- and
produces, per kind:
  - {kind}_subspace_overlap.png: one panel per side (left = output space,
    right = input space; their chance levels k/n differ). One line per model
    pair, x = top-k (log2 scale), y = median sim_k across every tensor of that
    kind found in the log (e.g. across all layers' down_proj), shaded band =
    25-75% across those tensors, dashed black = uniform-random chance k/n.
  - {kind}_{A}_vs_{B}_layers.png, per pair: layer x k heatmap of
    log10(observed/chance), so a single anomalous layer (hidden by the layer
    mean above) stands out. Skipped when the log has only one layer.
Summary lines ("mean_sim_k=...") in the log are ignored; the plots recompute
the layer statistics from the per-tensor lines.

Usage:
    python plot_tensor_subspace_overlap.py \\
        --log mlp_all_layers.log \\
        --output-dir mlp_subspace_figures
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


HEADER_RE = re.compile(r"^=== (\S+)\s+\(shape \(([^)]*)\)\) ===\s*$")
PAIR_RE = re.compile(r"^\s*-- (\S+) vs (\S+) --\s*$")
NUM = r"(nan|[-\d.]+)"
# Current logs (subspace_overlap_profile.format_sim_line): sim_k = ||.||_F^2 / k (LoRA phi),
# with the closed-form chance mean k/n. The optional "+/-std  z=..." part is from a short-lived
# variant that also printed the closed-form std; it is accepted and ignored.
ANALYTIC_RE = re.compile(
    rf"^\s*(left|right)\s+k=(\d+)\s+sim_k={NUM}\s+"
    rf"chance\(k/n\)={NUM}(?:\+/-[-\d.nan]+\s+z=[-\d.nan]+)?\s+"
    rf"observed/chance={NUM}x\s*$"
)
# Older logs with a sampled (--random-trials) chance baseline. "chance_asymptotic_sqrt(k/n)"
# marks sim_k = ||.||_F / sqrt(k); "chance_expected(k/n)" marks sim_k = ||.||_F^2 / k.
EMPIRICAL_LABEL_TO_METRIC = {"chance_asymptotic_sqrt": "sqrt_phi", "chance_expected": "phi"}
EMPIRICAL_RE = re.compile(
    rf"^\s*(left|right)\s+k=(\d+)\s+sim_k={NUM}\s+"
    rf"chance_empirical={NUM}\+/-{NUM}\s+"
    rf"(chance_asymptotic_sqrt|chance_expected)\(k/n\)={NUM}\s+"
    rf"observed/chance={NUM}x\s*$"
)


def parse_sim_line(line: str) -> dict | None:
    """Parse one per-k data line from any log generation; None if the line is not one."""
    data = ANALYTIC_RE.match(line)
    if data:
        side, k, sim_k, chance_mean, multiple = data.groups()
        metric, chance_source = "phi", "analytic"
    else:
        data = EMPIRICAL_RE.match(line)
        if not data:
            return None
        side, k, sim_k, chance_mean, _chance_std, chance_label, _closed_form, multiple = data.groups()
        metric, chance_source = EMPIRICAL_LABEL_TO_METRIC[chance_label], "empirical"
    return dict(
        side=side, k=int(k), sim_k=float(sim_k), metric=metric, chance_source=chance_source,
        chance_mean=float(chance_mean), observed_over_chance=float(multiple),
    )


def tensor_kind(tensor_name: str) -> str:
    parts = tensor_name.split(".")
    return parts[-2] if len(parts) >= 2 else tensor_name


def tensor_layer(tensor_name: str) -> int | None:
    parts = tensor_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def parse_log(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    tensor_name = kind = layer = None
    left = right = None
    for line in path.read_text().splitlines():
        header = HEADER_RE.match(line)
        if header:
            tensor_name = header.group(1)
            kind = tensor_kind(tensor_name)
            layer = tensor_layer(tensor_name)
            continue
        pair = PAIR_RE.match(line)
        if pair:
            left, right = pair.groups()
            continue
        data = parse_sim_line(line)
        if data:
            if tensor_name is None or left is None:
                raise ValueError(f"data line appeared before any tensor/pair header in {path}: {line!r}")
            rows.append(dict(tensor=tensor_name, kind=kind, layer=layer, left=left, right=right, **data))
    if not rows:
        raise ValueError(f"No data rows parsed from {path} -- is this a saved "
                          f"subspace_overlap_profile.py transcript (stdout captured with tee/> )?")
    return pd.DataFrame(rows)


SIDE_TITLES = {"left": "left singular vectors (output space)", "right": "right singular vectors (input space)"}


def plot_kind(frame: pd.DataFrame, kind: str, output_path: Path) -> None:
    """One panel per side (their chance levels differ): median sim_k over layers vs k, one line per
    pair, interquartile band across layers, and the uniform-random chance level k/n dashed.
    Median/IQR rather than mean/std: one anomalous layer inflates the std into a band that
    dips below 0 (sim_k cannot); single layers are what plot_layer_heatmap is for."""
    subset = frame[frame.kind == kind]
    pairs = list(subset[["left", "right"]].drop_duplicates().itertuples(index=False, name=None))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pair_color = {pair: colors[i % len(colors)] for i, pair in enumerate(pairs)}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    ks = sorted(subset.k.unique())
    for ax, side in zip(axes, ("left", "right")):
        side_rows = subset[subset.side == side]
        for pair in pairs:
            line = side_rows[(side_rows.left == pair[0]) & (side_rows.right == pair[1])]
            if line.empty:
                continue
            by_k = line.groupby("k")["sim_k"]
            median, q25, q75 = by_k.median(), by_k.quantile(0.25), by_k.quantile(0.75)
            color = pair_color[pair]
            ax.plot(median.index, median.values, marker="o", color=color, label=f"{pair[0]} vs {pair[1]}")
            ax.fill_between(median.index, q25.values, q75.values, color=color, alpha=0.15)
        if subset.chance_source.eq("analytic").all():
            chance = side_rows.groupby("k")["chance_mean"].mean().sort_index()
            ax.plot(chance.index, chance.values, linestyle="--", color="black", label="chance k/n")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Top-k singular vectors")
        ax.set_title(SIDE_TITLES[side])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    metric_label = "||Q^T Q'||_F^2 / k" if subset.metric.iloc[0] == "phi" else "||Q^T Q'||_F / sqrt(k)"
    axes[0].set_ylabel(f"Median sim_k = {metric_label} across layers")
    n_tensors = subset["tensor"].drop_duplicates().shape[0]
    fig.suptitle(f"{kind}: top singular subspace similarity (median, 25-75% band across {n_tensors} layers)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_layer_heatmap(frame: pd.DataFrame, kind: str, pair: tuple[str, str], output_path: Path) -> None:
    """Layer x k grid of log10(observed/chance) for one pair, one panel per side. The layer mean in
    plot_kind hides single anomalous layers; this is where they show up."""
    subset = frame[(frame.kind == kind) & (frame.left == pair[0]) & (frame.right == pair[1])]
    subset = subset.dropna(subset=["layer"])
    if subset.layer.nunique() < 2:
        return
    ratio = subset.observed_over_chance.clip(lower=1e-3)
    subset = subset.assign(log_ratio=np.log10(ratio))
    limit = max(float(subset.log_ratio.abs().max()), 1e-6)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.0, 0.22 * subset.layer.nunique() + 2)))
    for ax, side in zip(axes, ("left", "right")):
        grid = subset[subset.side == side].pivot_table(index="layer", columns="k", values="log_ratio")
        if grid.empty:
            ax.set_visible(False)
            continue
        image = ax.imshow(grid.values, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax.set_xticks(range(grid.shape[1]))
        ax.set_xticklabels([str(k) for k in grid.columns])
        ax.set_yticks(range(grid.shape[0]))
        ax.set_yticklabels([str(int(layer)) for layer in grid.index], fontsize=7)
        ax.set_xlabel("Top-k singular vectors")
        ax.set_ylabel("Layer")
        ax.set_title(SIDE_TITLES[side])
        fig.colorbar(image, ax=ax, label="log10(observed / chance)")
    fig.suptitle(f"{kind}: {pair[0]} vs {pair[1]} per layer (0 = chance, 1 = 10x, 2 = 100x)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path, action="append",
                         help="path to a saved subspace_overlap_profile.py transcript; repeat to merge multiple logs")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.concat([parse_log(p) for p in args.log], ignore_index=True)
    metrics = sorted(frame.metric.unique())
    if len(metrics) > 1:
        raise ValueError(f"Refusing to merge logs with different sim_k definitions {metrics}: "
                         f"legacy logs use ||.||_F/sqrt(k), current logs use ||.||_F^2/k. Re-run the old ones.")
    csv_path = args.output_dir / "parsed_tensor_subspace_overlap.csv"
    frame.to_csv(csv_path, index=False)
    print(f"parsed {len(frame)} rows from {len(args.log)} log file(s) -> {csv_path}; "
          f"kinds={sorted(frame.kind.unique())}")

    for kind in sorted(frame.kind.unique()):
        plot_kind(frame, kind, args.output_dir / f"{kind}_subspace_overlap.png")
        kind_pairs = frame[frame.kind == kind][["left", "right"]].drop_duplicates().itertuples(index=False, name=None)
        for pair in kind_pairs:
            plot_layer_heatmap(frame, kind, pair,
                               args.output_dir / f"{kind}_{pair[0]}_vs_{pair[1]}_layers.png")


if __name__ == "__main__":
    main()
