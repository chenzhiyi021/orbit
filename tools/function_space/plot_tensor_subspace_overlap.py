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
produces one figure per kind, analogous to one figure per circuit in the
attention script. Each figure: one line per (left, right, side) triple,
x = top-k (log2 scale), y = mean sim_k across every tensor of that kind
found in the log (e.g. across all layers' down_proj), shaded band = +/- 1
std across those tensors.

Does not modify subspace_overlap_profile.py.

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
import pandas as pd


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


def plot_kind(frame: pd.DataFrame, kind: str, output_path: Path) -> None:
    subset = frame[frame.kind == kind]
    pairs = list(subset[["left", "right"]].drop_duplicates().itertuples(index=False, name=None))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_index = 0
    for left, right in pairs:
        for side in ("left", "right"):
            line = subset[(subset.left == left) & (subset.right == right) & (subset.side == side)]
            if line.empty:
                continue
            grouped = line.groupby("k")["sim_k"].agg(["mean", "std"]).reset_index().sort_values("k")
            std = grouped["std"].fillna(0.0)
            color = colors[color_index % len(colors)]
            color_index += 1
            label = f"{left.capitalize()} vs {right.capitalize()} ({side})"
            ax.plot(grouped["k"], grouped["mean"], marker="o", label=label, color=color)
            ax.fill_between(grouped["k"], grouped["mean"] - std, grouped["mean"] + std, color=color, alpha=0.15)

    ax.set_xscale("log", base=2)
    ks = sorted(subset.k.unique())
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Top-k singular vectors")
    metric_label = "||Q^T Q'||_F^2 / k" if subset.metric.iloc[0] == "phi" else "||Q^T Q'||_F / sqrt(k)"
    ax.set_ylabel(f"Mean sim_k = {metric_label} across layers")
    n_tensors = subset["tensor"].drop_duplicates().shape[0]
    ax.set_title(f"{kind}: top singular subspace similarity across model pairs\n"
                 f"(mean ± 1 std across {n_tensors} layers)")
    ax.legend()
    ax.grid(alpha=0.3)
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


if __name__ == "__main__":
    main()
