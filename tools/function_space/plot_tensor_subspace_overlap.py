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
DATA_RE = re.compile(
    rf"^\s*(left|right)\s+k=(\d+)\s+sim_k={NUM}\s+"
    rf"chance_empirical={NUM}\+/-{NUM}\s+"
    rf"chance_asymptotic_sqrt\(k/n\)={NUM}\s+"
    rf"observed/chance={NUM}x\s*$"
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
        data = DATA_RE.match(line)
        if data:
            if tensor_name is None or left is None:
                raise ValueError(f"data line appeared before any tensor/pair header in {path}: {line!r}")
            side, k, sim_k, chance_mean, chance_std, chance_asym, multiple = data.groups()
            rows.append(dict(
                tensor=tensor_name, kind=kind, layer=layer, left=left, right=right,
                side=side, k=int(k), sim_k=float(sim_k),
                chance_empirical=float(chance_mean), chance_std=float(chance_std),
                chance_asymptotic=float(chance_asym), observed_over_chance=float(multiple),
            ))
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
    ax.set_ylabel("Mean sim_k across layers")
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
    csv_path = args.output_dir / "parsed_tensor_subspace_overlap.csv"
    frame.to_csv(csv_path, index=False)
    print(f"parsed {len(frame)} rows from {len(args.log)} log file(s) -> {csv_path}; "
          f"kinds={sorted(frame.kind.unique())}")

    for kind in sorted(frame.kind.unique()):
        plot_kind(frame, kind, args.output_dir / f"{kind}_subspace_overlap.png")


if __name__ == "__main__":
    main()
