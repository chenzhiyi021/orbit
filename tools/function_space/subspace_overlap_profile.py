"""Top-k singular-subspace overlap between two weight deltas, per tensor.

    sim_k(A, B) = ||Q_k(A)^T Q_k(B)||_F^2 / k  ∈  [0, 1]

This is the LoRA paper's (Hu et al. 2021, Sec. 7.2) normalized subspace
similarity φ(A, B, i, j) = ||U_A^i^T U_B^j||_F^2 / min(i, j) at i = j = k,
i.e. the mean cos^2 of the principal angles between the two subspaces.

1.0 = identical k-dim subspace, 0 = orthogonal subspaces.

Chance level, in closed form (no sampling): for two independent uniformly
random (Haar) k-subspaces of R^n, E[sim_k] = k / n . 

--compare-base additionally reports, per checkpoint, sim_k between its delta
and the base weight W's own top-k singular subspace (pair label "W_base";
the question LoRA Sec. 7.3 asks of Delta W vs W).

--cross-kind additionally compares, within each layer and each checkpoint, the deltas of two
different tensor kinds on every side where their dimensions match (pair label
"NAME.kind_a vs NAME.kind_b", pseudo-tensor "model.layers.L.cross.kind_a~kind_b.weight", so
plot_tensor_subspace_overlap.py plots each kind pair like a kind). This tests whether delta
structure is set by the layer's input activations: grad_W = E[delta x^T] puts the right
singular vectors of an SGD delta in the span of the inputs x (Adam only approximately), so
kinds that READ THE SAME INPUT should share right subspaces:
    q/k/v_proj      input_layernorm(h)            -- same input
    gate/up_proj    post_attention_layernorm(h)   -- same input
    q vs gate etc.  residual stream before/after attention -- related, not identical
    o_proj          attention output              -- different input (control)
Left sides that match in size are reported too; gate/up share neuron coordinates and
o/down both write to the residual stream, other size-matched left pairs (e.g. q vs o) live in
unrelated spaces and serve as chance controls. With --compare-base the same is done for the
base weights themselves ("W_base.kind_a vs W_base.kind_b").

CPU-only, no model reload -- reuses svd_rank_profile.py's tensor I/O.

Usage:
    python subspace_overlap_profile.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --checkpoint oracle=/mnt/.../m6_coord_mask_1p6pct_oracle_step300 \\
        --checkpoint random=/mnt/.../m6_coord_mask_1p6pct_random_step300 \\
        --k 4,8,16,32,64,128,256 [--compare-base] [--cross-kind]
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from svd_rank_profile import CANONICAL_RE, load_tensor, tensor_locations

# Pair label for --compare-base rows; reserved, so no --checkpoint may use it.
BASE_W_NAME = "W_base"

KIND_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def tensor_kind(tensor_name: str) -> str:
    """'model.layers.3.mlp.down_proj.weight' -> 'down_proj'"""
    parts = tensor_name.split(".")
    return parts[-2] if len(parts) >= 2 else tensor_name


def all_target_tensors(locations: dict[str, Path]) -> list[str]:
    """Every canonical q/k/v/o/gate/up/down weight, ordered by (layer, kind)."""
    names = [name for name in locations if CANONICAL_RE.match(name)]
    if not names:
        raise RuntimeError("No canonical q/k/v/o/gate/up/down tensors found -- pass --tensors explicitly")
    return sorted(names, key=lambda name: (int(CANONICAL_RE.match(name).group(1)),
                                           KIND_ORDER.index(tensor_kind(name))))


def resolve_targets(spec: str | None, locations: dict[str, Path]) -> tuple[list[str], set[str]]:
    """--tensors -> (tensor names to profile, kinds to average over layers).

    Each comma-separated entry is either a full tensor name (profiled alone, not averaged)
    or a kind like "down_proj" (every layer's tensor of that kind, plus its layer average).
    No --tensors = every kind.
    """
    canonical = all_target_tensors(locations)
    if not spec:
        return canonical, {tensor_kind(name) for name in canonical}
    names: list[str] = []
    summary_kinds: set[str] = set()
    for entry in (t.strip() for t in spec.split(",")):
        if not entry:
            continue
        if entry in locations:
            names.append(entry)
        elif entry in KIND_ORDER:
            names += [name for name in canonical if tensor_kind(name) == entry]
            summary_kinds.add(entry)
        else:
            raise ValueError(f"--tensors entry {entry!r} is neither a tensor in the base checkpoint "
                             f"nor a kind in {KIND_ORDER}")
    return list(dict.fromkeys(names)), summary_kinds


SVD_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def top_k_bases(delta: torch.Tensor, k_values: list[int],
                dtype: torch.dtype = torch.float32) -> dict[str, dict[int, torch.Tensor]]:
    """SVD in `dtype`, slice out top-k left/right singular vectors for every k."""
    u, _, vh = torch.linalg.svd(delta.to(dtype), full_matrices=False)
    v = vh.transpose(0, 1)
    max_k = min(u.shape[1], v.shape[1])
    return {
        "left": {k: u[:, : min(k, max_k)] for k in k_values},
        "right": {k: v[:, : min(k, max_k)] for k in k_values},
    }


def subspace_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """||A^T B||_F^2 / k for two (n, k) orthonormal-column matrices (LoRA Sec. 7.2 φ)."""
    k = a.shape[1]
    if k == 0:
        return float("nan")
    return float(torch.linalg.norm(a.transpose(0, 1) @ b) ** 2 / k)


def chance_mean(n: int, k: int) -> float:
    """Exact E[sim_k] for two independent Haar-random k-subspaces of R^n."""
    if k == 0 or k > n:
        return float("nan")
    return k / n


def format_sim_line(side: str, k: int, sim: float, n: int) -> str:
    """One log line; the plot_*_overlap.py parsers depend on this exact format."""
    mean = chance_mean(n, k)
    multiple = sim / mean if mean > 1e-12 else float("nan")
    return (f"    {side:<5} k={k:<4} sim_k={sim:.4f}  "
            f"chance(k/n)={mean:.6f}  "
            f"observed/chance={multiple:.2f}x")


def format_summary_line(side: str, k: int, sims: list[float], n: int) -> str:
    """Mean +/- std of sim_k across layers. Deliberately NOT matched by the plot parsers
    ("mean_sim_k=" instead of "sim_k="), so the summary is never double-counted there."""
    mean = statistics.mean(sims)
    std = statistics.stdev(sims) if len(sims) > 1 else 0.0
    chance = chance_mean(n, k)
    multiple = mean / chance if chance > 1e-12 else float("nan")
    return (f"    {side:<5} k={k:<4} mean_sim_k={mean:.4f}+/-{std:.4f}  "
            f"chance(k/n)={chance:.6f}  "
            f"observed/chance={multiple:.2f}x  (n_tensors={len(sims)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH",
                         help="repeat for each checkpoint; all pairs are compared")
    parser.add_argument("--tensors", default=None,
                         help="comma-separated full tensor names (profiled individually) and/or kinds such as "
                              "'down_proj' (every layer, plus the layer average); default: every kind")
    parser.add_argument("--k", default="4,8,16,32,64,128,256", help="comma-separated top-k values to test")
    parser.add_argument("--compare-base", action="store_true",
                         help=f"also compare each checkpoint's delta with the base weight's own top-k "
                              f"singular subspace (pair label '{BASE_W_NAME}')")
    parser.add_argument("--cross-kind", action="store_true",
                         help="also compare, within each layer and checkpoint, the top-k subspaces of different "
                              "tensor kinds on every side whose dimensions match (e.g. q_proj vs k_proj input space)")
    parser.add_argument("--device", default="cpu",
                         help="where the deltas, SVDs and overlaps are computed, e.g. 'cpu' or 'cuda' / 'cuda:1'; "
                              "the same torch.linalg.svd call dispatches to LAPACK or cuSOLVER")
    parser.add_argument("--svd-dtype", default="float32", choices=tuple(SVD_DTYPES),
                         help="precision of the deltas, SVDs and overlaps; bf16 checkpoints are upcast exactly")
    args = parser.parse_args()
    svd_dtype = SVD_DTYPES[args.svd_dtype]
    device = torch.device(args.device)
    started = time.perf_counter()

    checkpoints = {}
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        checkpoints[name] = Path(path)
    if BASE_W_NAME in checkpoints:
        raise ValueError(f"--checkpoint name '{BASE_W_NAME}' is reserved for --compare-base rows")
    if len(checkpoints) < (1 if args.compare_base else 2):
        raise ValueError("Need >=2 --checkpoint entries to compare pairwise (or >=1 with --compare-base)")

    k_values = [int(k.strip()) for k in args.k.split(",") if k.strip()]
    base_locations = tensor_locations(args.base)
    tensor_names, summary_kinds = resolve_targets(args.tensors, base_locations)
    print(f"profiling {len(tensor_names)} tensors, k={k_values}, device={device}, svd_dtype={args.svd_dtype}, "
          f"analytic chance baseline\n",
          flush=True)

    ckpt_locations = {name: tensor_locations(path) for name, path in checkpoints.items()}
    # (kind, left, right, side, k, n) -> sim_k per tensor; n is in the key so tensors of one
    # kind but different shapes never share a chance baseline.
    per_kind_sims: dict[tuple, list[float]] = defaultdict(list)
    max_k = max(k_values)
    # --cross-kind: kind -> model -> side -> top-max_k basis (cloned, so the full SVD factors are
    # freed), plus kind -> shape; emptied once the layer's cross-kind block is printed.
    layer_bases: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    layer_shapes: dict[str, tuple[int, int]] = {}
    current_layer = None

    def flush_cross_kind(layer: int) -> None:
        kinds = sorted(layer_bases, key=KIND_ORDER.index)
        for kind_a, kind_b in itertools.combinations(kinds, 2):
            shape_a, shape_b = layer_shapes[kind_a], layer_shapes[kind_b]
            sides = [(side, shape_a[dim]) for dim, side in enumerate(("left", "right"))
                     if shape_a[dim] == shape_b[dim]]
            if not sides:
                continue
            cross_kind = f"{kind_a}~{kind_b}"
            print(f"=== model.layers.{layer}.cross.{cross_kind}.weight  "
                  f"(shape ({kind_a}={shape_a[0]}x{shape_a[1]}, {kind_b}={shape_b[0]}x{shape_b[1]})) ===")
            for model in layer_bases[kind_a]:
                left_label, right_label = f"{model}.{kind_a}", f"{model}.{kind_b}"
                print(f"  -- {left_label} vs {right_label} --")
                for side, n_dim in sides:
                    basis_a, basis_b = layer_bases[kind_a][model][side], layer_bases[kind_b][model][side]
                    for k in k_values:
                        if k > min(basis_a.shape[1], basis_b.shape[1]):
                            continue
                        sim = subspace_sim(basis_a[:, :k], basis_b[:, :k])
                        print(format_sim_line(side, k, sim, n_dim))
                        if kind_a in summary_kinds and kind_b in summary_kinds:
                            per_kind_sims[(cross_kind, left_label, right_label, side, k, n_dim)].append(sim)
            print()
        layer_bases.clear()
        layer_shapes.clear()

    for tensor_name in tensor_names:
        layer = int(CANONICAL_RE.match(tensor_name).group(1)) if CANONICAL_RE.match(tensor_name) else None
        if args.cross_kind and layer != current_layer:
            if current_layer is not None:
                flush_cross_kind(current_layer)
            current_layer = layer
        base_tensor = load_tensor(args.base, base_locations, tensor_name).to(device, svd_dtype)
        n_out, n_in = base_tensor.shape
        print(f"=== {tensor_name}  (shape {tuple(base_tensor.shape)}) ===", flush=True)
        deltas = {}
        for name, path in checkpoints.items():
            other = load_tensor(path, ckpt_locations[name], tensor_name).to(device, svd_dtype)
            deltas[name] = other - base_tensor
        bases = {name: top_k_bases(delta, k_values, svd_dtype) for name, delta in deltas.items()}
        pairs = list(itertools.combinations(checkpoints, 2))
        if args.compare_base:
            bases[BASE_W_NAME] = top_k_bases(base_tensor, k_values, svd_dtype)
            pairs += [(name, BASE_W_NAME) for name in checkpoints]

        for left_name, right_name in pairs:
            print(f"  -- {left_name} vs {right_name} --")
            for side, n_dim in (("left", n_out), ("right", n_in)):
                for k in k_values:
                    if k > min(n_out, n_in):
                        continue
                    sim = subspace_sim(bases[left_name][side][k], bases[right_name][side][k])
                    print(format_sim_line(side, k, sim, n_dim))
                    if tensor_kind(tensor_name) in summary_kinds:
                        per_kind_sims[(tensor_kind(tensor_name), left_name, right_name, side, k, n_dim)].append(sim)
        print()
        if args.cross_kind and layer is not None:
            layer_shapes[tensor_kind(tensor_name)] = (n_out, n_in)
            layer_bases[tensor_kind(tensor_name)] = {
                model: {side: basis[max_k].clone() for side, basis in model_bases.items()}
                for model, model_bases in bases.items()
            }
        del bases

    if args.cross_kind and current_layer is not None:
        flush_cross_kind(current_layer)

    print(f"##### per-tensor profiling took {time.perf_counter() - started:.1f}s on {device} #####\n", flush=True)
    if not per_kind_sims:
        return
    print("##### mean +/- std over layers, per tensor kind #####\n")
    kinds = list(dict.fromkeys(key[0] for key in per_kind_sims))
    kinds.sort(key=lambda kind: (KIND_ORDER.index(kind.split("~")[0]) if kind.split("~")[0] in KIND_ORDER
                                 else len(KIND_ORDER), "~" in kind, kind))
    for kind in kinds:
        keys = [key for key in per_kind_sims if key[0] == kind]
        print(f"=== summary: {kind} ===")
        for pair in dict.fromkeys(key[1:3] for key in keys):
            print(f"  -- {pair[0]} vs {pair[1]} --")
            for key in keys:
                if key[1:3] == pair:
                    _, _, _, side, k, n_dim = key
                    print(format_summary_line(side, k, per_kind_sims[key], n_dim))
        print()


if __name__ == "__main__":
    main()
