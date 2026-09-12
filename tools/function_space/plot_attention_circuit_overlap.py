"""Plot attention-circuit top-k subspace overlap from a saved
attention_circuit_overlap_profile.py transcript.

Does not modify attention_circuit_overlap_profile.py -- reads its plain-text
stdout log (e.g. saved via `... | tee some.log`) with a regex parser, then
plots with matplotlib. Re-running the SVD analysis is not needed.

One figure per circuit ("qk", "ov") found in the log. Each figure has one
line per (left, right, side) triple (e.g. "Oracle vs Random (left)"),
x = top-k (log2-scaled, matching the profiler's --k values), y = mean sim_k
across every (layer, head) tensor seen at that k for that pair+side+circuit,
shaded band = +/- 1 std across those tensors (the "uncertainty" is
tensor-to-tensor spread across the analyzed layers/heads, not a bootstrap or
a training-seed variance -- see the caption baked into each figure title).

Also writes a tidy `parsed_attention_circuit_overlap.csv` with one row per
(layer, head, circuit, left, right, side, k) so you can slice it yourself
(e.g. per-layer trends) without re-parsing the log.

Usage:
    python plot_attention_circuit_overlap.py \\
        --log attention_circuit_all_layers.log \\
        --output-dir attention_circuit_figures

    # merge several logs (e.g. one per --heads run) into one plot:
    python plot_attention_circuit_overlap.py \\
        --log run1.log --log run2.log --output-dir attention_circuit_figures
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HEADER_RE = re.compile(r"^=== layer (\d+), head (\d+) \(kv_head (\d+)\) ===\s*$")
PAIR_RE = re.compile(r"^\s*-- (\S+) (\S+) vs (\S+) --\s*$")
NUM = r"(nan|[-\d.]+)"
DATA_RE = re.compile(
    rf"^\s*(left|right)\s+k=(\d+)\s+sim_k={NUM}\s+"
    rf"chance_empirical={NUM}\+/-{NUM}\s+"
    rf"chance_asymptotic_sqrt\(k/n\)={NUM}\s+"
    rf"observed/chance={NUM}x\s*$"
)


def parse_log(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    layer = head = kv_head = None
    circuit = left = right = None
    for line in path.read_text().splitlines():
        header = HEADER_RE.match(line)
        if header:
            layer, head, kv_head = (int(x) for x in header.groups())
            continue
        pair = PAIR_RE.match(line)
        if pair:
            circuit, left, right = pair.groups()
            continue
        data = DATA_RE.match(line)
        if data:
            if layer is None or circuit is None:
                raise ValueError(f"data line appeared before any layer/pair header in {path}: {line!r}")
            side, k, sim_k, chance_mean, chance_std, chance_asym, multiple = data.groups()
            rows.append(dict(
                layer=layer, head=head, kv_head=kv_head, circuit=circuit,
                left=left, right=right, side=side, k=int(k), sim_k=float(sim_k),
                chance_empirical=float(chance_mean), chance_std=float(chance_std),
                chance_asymptotic=float(chance_asym), observed_over_chance=float(multiple),
            ))
    if not rows:
        raise ValueError(f"No data rows parsed from {path} -- is this a saved "
                          f"attention_circuit_overlap_profile.py transcript (stdout captured with tee/> )?")
    return pd.DataFrame(rows)


def plot_circuit(frame: pd.DataFrame, circuit: str, output_path: Path) -> None:
    subset = frame[frame.circuit == circuit]
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
    ax.set_ylabel("Mean sim_k across (layer, head) circuits")
    n_tensors = subset[["layer", "head"]].drop_duplicates().shape[0]
    ax.set_title(f"{circuit.upper()} circuit: top singular subspace similarity across model pairs\n"
                 f"(mean ± 1 std across {n_tensors} (layer, head) tensors)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path, action="append",
                         help="path to a saved attention_circuit_overlap_profile.py transcript; repeat to merge multiple logs")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.concat([parse_log(p) for p in args.log], ignore_index=True)
    csv_path = args.output_dir / "parsed_attention_circuit_overlap.csv"
    frame.to_csv(csv_path, index=False)
    print(f"parsed {len(frame)} rows from {len(args.log)} log file(s) -> {csv_path}; "
          f"circuits={sorted(frame.circuit.unique())}")

    for circuit in sorted(frame.circuit.unique()):
        plot_circuit(frame, circuit, args.output_dir / f"{circuit}_subspace_overlap.png")


if __name__ == "__main__":
    main()
