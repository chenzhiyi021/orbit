"""Cumulative singular-value energy of a fine-tune's weight deltas, one curve per tensor kind.

For every canonical q/k/v/o/gate/up/down weight this computes the full spectrum of

    Delta W = W_finetuned - W_base

and the cumulative energy fraction

    E(k) = sum_{i<=k} s_i^2 / sum_i s_i^2 = sum_{i<=k} s_i^2 / ||Delta W||_F^2,

then plots E(k) at k = 1, 2, 4, ..., --max-k on a log2 axis: one line per kind (mean over
layers), a thick black line for the mean over every tensor, and a dashed grey reference for an
i.i.d. Gaussian matrix of the same shapes (no low-rank structure; averaged over tensors the same
way as the black line). The SVD runs once per --svd-dtype and each dtype gets its own figure,
so float32 and float64 can be compared on identical deltas (bf16 checkpoints upcast exactly to
either).

Outputs in --output-dir:
    energy_spectrum_<dtype>.png     one figure per --svd-dtype
    singular_values.npz             full spectra, keys "<dtype>/<tensor name>", for replotting
and a per-kind table of E(k) at a few k on stdout.

Usage:
    python plot_delta_energy_spectrum.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --finetuned /mnt/L202500431/models/zh_models/m6_st_full_non_thinking_hf_step300 \\
        --svd-dtypes float32,float64 --device cuda \\
        --output-dir output/energy_spectrum/m6_full_step300
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from svd_rank_profile import CANONICAL_RE, load_tensor, tensor_locations


SVD_DTYPES = {"float32": torch.float32, "float64": torch.float64}
KIND_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
REPORT_KS = (1, 4, 16, 64, 256, 1024, 2048)
ALL_LABEL = "all"


def tensor_kind(tensor_name: str) -> str:
    return tensor_name.split(".")[-2]


def cumulative_energy(singular: np.ndarray, max_k: int) -> np.ndarray:
    """E(0..max_k); spectra shorter than max_k (min dim < max_k) are held at 1.0 past their end."""
    energy = singular.astype(np.float64) ** 2
    cum = np.concatenate([[0.0], np.cumsum(energy) / max(energy.sum(), 1e-300)])
    if cum.size < max_k + 1:
        cum = np.concatenate([cum, np.full(max_k + 1 - cum.size, cum[-1])])
    return cum[: max_k + 1]


def random_baseline(shape: tuple[int, int], dtype: torch.dtype, device: torch.device, max_k: int,
                    seed: int = 0) -> np.ndarray:
    """E(k) of one i.i.d. Gaussian matrix of `shape` -- the no-structure reference."""
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return cumulative_energy(torch.linalg.svdvals(noise).double().cpu().numpy(), max_k)


def plot_dtype(curves: dict[str, list[np.ndarray]], random_curves: list[np.ndarray], dtype: str,
               max_k: int, title: str, output_path: Path) -> None:
    """E(k) at powers of two on a log2 axis: one line per kind (mean over layers), black = mean over
    every tensor, dashed grey = same-shape Gaussian reference."""
    ks = [2 ** p for p in range(int(np.log2(max_k)) + 1)]
    fig, ax = plt.subplots(figsize=(8, 6))
    for kind in [k for k in KIND_ORDER if k in curves] + sorted(set(curves) - set(KIND_ORDER)):
        stack = np.stack(curves[kind])
        ax.plot(ks, stack.mean(0)[ks], marker="o", linewidth=1.4, label=kind)
    everything = np.stack([c for kind_curves in curves.values() for c in kind_curves])
    ax.plot(ks, everything.mean(0)[ks], color="black", linewidth=2.6, label=f"{ALL_LABEL} (n={len(everything)})")
    ax.plot(ks, np.stack(random_curves).mean(0)[ks], color="grey", linestyle="--", linewidth=1.2,
            label="random Gaussian (same shapes)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("k")
    ax.set_ylabel("energy in top-k singular directions")
    ax.set_title(f"{title}\nSVD in {dtype}, mean over layers")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--finetuned", required=True, type=Path)
    parser.add_argument("--svd-dtypes", default="float32,float64",
                         help=f"comma-separated SVD precisions from {tuple(SVD_DTYPES)}; one figure each")
    parser.add_argument("--max-k", type=int, default=2048,
                         help="x-axis extent, a power of two (2048 = the largest min dim of Qwen3-1.7B's targets)")
    parser.add_argument("--device", default="cpu", help="where the deltas and SVDs are computed, e.g. 'cuda'")
    parser.add_argument("--title", default=None, help="figure title (default: the fine-tune's dir name)")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    dtypes = [d.strip() for d in args.svd_dtypes.split(",") if d.strip()]
    unknown = set(dtypes) - set(SVD_DTYPES)
    if unknown:
        raise ValueError(f"unknown --svd-dtypes {sorted(unknown)}; choose from {tuple(SVD_DTYPES)}")
    if args.max_k < 1 or args.max_k & (args.max_k - 1):
        raise ValueError(f"--max-k must be a power of two, got {args.max_k}")
    device = torch.device(args.device)
    title = args.title or args.finetuned.name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    base_locations = tensor_locations(args.base)
    tuned_locations = tensor_locations(args.finetuned)
    names = sorted((n for n in base_locations if CANONICAL_RE.match(n)),
                   key=lambda n: (int(n.split(".")[2]), n))
    missing = [n for n in names if n not in tuned_locations]
    if missing:
        raise KeyError(f"{len(missing)} tensors missing from --finetuned, e.g. {missing[:3]}")
    print(f"{len(names)} tensors, svd_dtypes={dtypes}, device={device}", flush=True)

    spectra: dict[str, np.ndarray] = {}
    curves: dict[str, dict[str, list[np.ndarray]]] = {d: defaultdict(list) for d in dtypes}
    random_by_shape: dict[tuple[str, tuple[int, int]], np.ndarray] = {}
    random_curves: dict[str, list[np.ndarray]] = {d: [] for d in dtypes}
    for i, name in enumerate(names, 1):
        base = load_tensor(args.base, base_locations, name)
        tuned = load_tensor(args.finetuned, tuned_locations, name)
        for dtype in dtypes:
            key = (dtype, tuple(base.shape))
            if key not in random_by_shape:
                random_by_shape[key] = random_baseline(key[1], SVD_DTYPES[dtype], device, args.max_k)
            random_curves[dtype].append(random_by_shape[key])
            delta = tuned.to(device, SVD_DTYPES[dtype]) - base.to(device, SVD_DTYPES[dtype])
            singular = torch.linalg.svdvals(delta).double().cpu().numpy()
            spectra[f"{dtype}/{name}"] = singular
            curves[dtype][tensor_kind(name)].append(cumulative_energy(singular, args.max_k))
            del delta
        print(f"  [{i}/{len(names)}] {name}  shape={tuple(base.shape)}  "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    np.savez_compressed(args.output_dir / "singular_values.npz", **spectra)
    print(f"wrote {args.output_dir / 'singular_values.npz'}")

    report_ks = [k for k in REPORT_KS if k <= args.max_k]
    for dtype in dtypes:
        print(f"\n[{dtype}] mean cumulative energy E(k) over layers")
        print(f"  {'kind':10s} " + " ".join(f"k={k:<6d}" for k in report_ks))
        rows = [(kind, np.stack(curves[dtype][kind])) for kind in KIND_ORDER if kind in curves[dtype]]
        rows.append((ALL_LABEL, np.stack([c for kc in curves[dtype].values() for c in kc])))
        rows.append(("random", np.stack(random_curves[dtype])))
        for kind, stack in rows:
            mean = stack.mean(0)
            print(f"  {kind:10s} " + " ".join(f"{mean[k]:<8.4f}" for k in report_ks))
        plot_dtype(curves[dtype], random_curves[dtype], dtype, args.max_k, title,
                   args.output_dir / f"energy_spectrum_{dtype}.png")

    if len(dtypes) > 1:
        ref = dtypes[0]
        for dtype in dtypes[1:]:
            worst = max(np.abs(np.stack(curves[dtype][kind]) - np.stack(curves[ref][kind])).max()
                        for kind in curves[ref])
            print(f"\nmax |E_{dtype}(k) - E_{ref}(k)| over all tensors and k: {worst:.3e}")
    print(f"\ndone in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
