"""Top-k singular-subspace overlap between two weight deltas, per tensor.

Motivating question: two coordinate masks (oracle vs random, ~20% density,
near-disjoint literal coordinate support -- an oracle-chosen and a uniformly
random 20% subset of the same tensor overlap at roughly chance level, ~20%
of the oracle set, by construction) end up with very similar *functional*
behavior (probability-space cosine ~0.9+, see plot_cosine_heatmap.py
--space prob) despite both having very high weight-space stable_rank
(svd_rank_profile.py: ~110-130, spread across most of the tensor). Does that
functional agreement come with *any* shared weight-space direction, or is it
happening via completely different, non-overlapping weight-space routes?

Metric: normalized Frobenius subspace cosine between the top-k singular
subspaces of two deltas A, B (same convention as
reproduce_effopd_geometry_figures.py's EffOPD comparison):

    sim_k(A, B) = ||Q_k(A)^T Q_k(B)||_F / sqrt(k)   in [0, 1]

1.0 = identical k-dim subspace, 0 = orthogonal subspaces. Reported for both
sides of each (out x in) weight matrix:
  - "left"  (Q = top-k left singular vectors,  U[:, :k]):  which *output*
    directions the update engages.
  - "right" (Q = top-k right singular vectors, V[:, :k]):  which *input*
    feature directions the update engages.

Chance level (two things you should read side by side):
  - closed-form asymptotic for two independent uniformly random k-subspaces
    of R^n: sqrt(k/n) (exact only as k << n; printed for context/sanity).
  - empirical: actually draws `--random-trials` independent Haar-random
    k-subspaces (via QR of a Gaussian matrix) and reports the mean +/- std
    of their pairwise sim_k on the *same* (n, k) as your tensors. This is
    the baseline to compare against, not the asymptotic formula -- use it.

Interpretation:
  - observed sim_k close to the empirical chance level -> no shared
    weight-space subspace; the functional agreement is happening through
    genuinely different routes ("many roads to the same function").
  - observed sim_k well above chance -> there IS a shared "core" direction
    inside the otherwise spread-out (high stable-rank) mask updates, and the
    two masks are finding it despite starting from disjoint coordinate sets.

CPU-only, no model reload -- reuses svd_rank_profile.py's tensor I/O.

Usage:
    python subspace_overlap_profile.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --checkpoint oracle=/mnt/.../m6_coord_mask_1p6pct_oracle_step300 \\
        --checkpoint random=/mnt/.../m6_coord_mask_1p6pct_random_step300 \\
        --k 4,8,16,32,64,128,256 --random-trials 20 --seed 20260907
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from svd_rank_profile import load_tensor, pick_representative_tensors, tensor_locations


def top_k_bases(delta: torch.Tensor, k_values: list[int]) -> dict[str, dict[int, torch.Tensor]]:
    """SVD once, slice out top-k left/right singular vectors for every k."""
    u, _, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    v = vh.transpose(0, 1)
    max_k = min(u.shape[1], v.shape[1])
    return {
        "left": {k: u[:, : min(k, max_k)] for k in k_values},
        "right": {k: v[:, : min(k, max_k)] for k in k_values},
    }


def subspace_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """||A^T B||_F / sqrt(k) for two (n, k) orthonormal-column matrices."""
    k = a.shape[1]
    if k == 0:
        return float("nan")
    return float(torch.linalg.norm(a.transpose(0, 1) @ b) / (k ** 0.5))


def random_orthonormal_basis(n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    gaussian = torch.randn(n, k, generator=generator)
    q, _ = torch.linalg.qr(gaussian)
    return q


def empirical_chance(n: int, k: int, trials: int, generator: torch.Generator) -> tuple[float, float]:
    if k == 0 or k > n:
        return float("nan"), float("nan")
    sims = []
    for _ in range(trials):
        a = random_orthonormal_basis(n, k, generator)
        b = random_orthonormal_basis(n, k, generator)
        sims.append(subspace_sim(a, b))
    values = torch.tensor(sims)
    return float(values.mean()), float(values.std(unbiased=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH",
                         help="repeat for each checkpoint; all pairs are compared")
    parser.add_argument("--tensors", default=None, help="comma-separated tensor names; default: auto-pick a representative spread")
    parser.add_argument("--k", default="4,8,16,32,64,128,256", help="comma-separated top-k values to test")
    parser.add_argument("--random-trials", type=int, default=20,
                         help="independent Haar-random subspace pairs sampled per (n, k) for the empirical chance baseline")
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    checkpoints = {}
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        checkpoints[name] = Path(path)
    if len(checkpoints) < 2:
        raise ValueError("Need >=2 --checkpoint entries to compare pairwise")

    k_values = [int(k.strip()) for k in args.k.split(",") if k.strip()]
    base_locations = tensor_locations(args.base)
    tensor_names = [t.strip() for t in args.tensors.split(",")] if args.tensors else pick_representative_tensors(base_locations)
    print(f"profiling {len(tensor_names)} tensors, k={k_values}, {args.random_trials} random-baseline trials each\n", flush=True)

    ckpt_locations = {name: tensor_locations(path) for name, path in checkpoints.items()}
    generator = torch.Generator().manual_seed(args.seed)
    chance_cache: dict[tuple[int, int], tuple[float, float]] = {}

    for tensor_name in tensor_names:
        base_tensor = load_tensor(args.base, base_locations, tensor_name).float()
        n_out, n_in = base_tensor.shape
        print(f"=== {tensor_name}  (shape {tuple(base_tensor.shape)}) ===")
        deltas = {}
        for name, path in checkpoints.items():
            other = load_tensor(path, ckpt_locations[name], tensor_name).float()
            deltas[name] = other - base_tensor
        bases = {name: top_k_bases(delta, k_values) for name, delta in deltas.items()}

        for left_name, right_name in itertools.combinations(checkpoints, 2):
            print(f"  -- {left_name} vs {right_name} --")
            for side, n_dim in (("left", n_out), ("right", n_in)):
                for k in k_values:
                    if k > min(n_out, n_in, base_tensor.shape[0], base_tensor.shape[1]):
                        continue
                    sim = subspace_sim(bases[left_name][side][k], bases[right_name][side][k])
                    cache_key = (n_dim, k)
                    if cache_key not in chance_cache:
                        chance_cache[cache_key] = empirical_chance(n_dim, k, args.random_trials, generator)
                    chance_mean, chance_std = chance_cache[cache_key]
                    asymptotic = (k / n_dim) ** 0.5
                    multiple = sim / chance_mean if chance_mean > 1e-12 else float("nan")
                    print(f"    {side:<5} k={k:<4} sim_k={sim:.4f}  "
                          f"chance_empirical={chance_mean:.4f}+/-{chance_std:.4f}  "
                          f"chance_asymptotic_sqrt(k/n)={asymptotic:.4f}  "
                          f"observed/chance={multiple:.2f}x")
        print()


if __name__ == "__main__":
    main()
