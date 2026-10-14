"""Export energy-matched random input subspaces V = orth(A^{1/2} Z) as a remove_delta_subspace.py --subspace file.

A Haar-random k-dim subspace of the input space reads almost none of the activation energy (its
directions mostly sit in the low-variance tail of A = E[x x^T]), so "keep a random subspace" fails
for reasons unrelated to which directions a task needs. This control instead draws random
directions distributed like the data: Z ~ N(0, I) of shape (n_in, k_max), M = A^{1/2} Z,
V = QR(M).Q. Column j of M is a random combination of activation directions weighted by
sqrt(eigenvalue), i.e. a direction the gradient (rows in span{x}) could plausibly have used.
Gram-Schmidt is nested, so V[:, :k] is itself such a k-dim subspace for every k <= k_max and one
file serves a whole k sweep (exclude/keep with --exclude-subspace-k).

The --eig file (activation_covariance_overlap.py --save-eig) stores all eigenvalues but only the
top eigenvectors E, so A^{1/2} = E diag(sqrt(lambda_top)) E^T + sqrt(mean tail lambda) (I - E E^T).
Directions are shared by every tensor of a layer's input group (q/k/v; gate/up), as in
export_activation_eig_subspace.py. --center-out-mean removes the activation-mean direction before
the QR, giving a variant without the implicit-bias channel.

For each k in --report-k the captured activation energy tr(V^T A V) / tr(A) is printed (median over
layers per group) next to the activation top-k (the maximum), and next to every --compare basis
(e.g. a build_exclusion_subspace.py ΔW basis), so energy matching can be checked before training.

    python export_activation_random_subspace.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --eig /mnt/.../actcov_eig_math_t07_mean.safetensors --k-max 256 --seed 0 \\
        --compare opd19=/mnt/.../exclusion_subspaces/opd19_top256.safetensors \\
        --output /mnt/.../task_split_export_energy/actrand_math_k256_seed0.safetensors
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from subspace_overlap_profile import GROUP_OF_KIND, all_target_tensors, tensor_kind
from svd_rank_profile import CANONICAL_RE, tensor_locations


def a_power_times(z: torch.Tensor, vecs: torch.Tensor, vals: torch.Tensor, power: float = 0.5) -> torch.Tensor:
    """A^power z with A known through its top eigenpairs (vecs: n x m) and all eigenvalues (vals)."""
    m = vecs.shape[1]
    top = vals[:m].clamp(min=1e-30).pow(power)
    tail = vals[m:].clamp(min=1e-30).mean().pow(power) if vals.numel() > m else torch.zeros((), dtype=vals.dtype)
    coef = vecs.T @ z
    return vecs @ (top[:, None] * coef) + tail * (z - vecs @ coef)


def sqrt_a_times(z: torch.Tensor, vecs: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """A^{1/2} z (alpha = 1 of orth(A^{alpha/2} Z))."""
    return a_power_times(z, vecs, vals, 0.5)


def orth(m: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(m)
    return q * torch.sign(torch.diagonal(r))


def matched_alpha(z: torch.Tensor, vecs: torch.Tensor, vals: torch.Tensor, k: int, target: float,
                  lo: float = 0.0, hi: float = 2.0, iters: int = 30) -> tuple[float, torch.Tensor]:
    """Bisect alpha so that orth(A^{alpha/2} Z)[:, :k] captures `target` activation energy (monotone in alpha)."""
    def basis(alpha: float) -> torch.Tensor:
        return orth(a_power_times(z, vecs, vals, alpha / 2))
    if captured_energy(basis(hi)[:, :k], vecs, vals) <= target:
        return hi, basis(hi)
    if captured_energy(basis(lo)[:, :k], vecs, vals) >= target:
        return lo, basis(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if captured_energy(basis(mid)[:, :k], vecs, vals) < target:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2
    return alpha, basis(alpha)


def captured_energy(v: torch.Tensor, vecs: torch.Tensor, vals: torch.Tensor) -> float:
    """tr(V^T A V) / tr(A) for an orthonormal V, tail beyond the saved eigenvectors at mean tail eigenvalue."""
    m = vecs.shape[1]
    proj = (vecs.T @ v).pow(2).sum(1)  # (m,) energy of V along each saved eigenvector
    head = float((vals[:m].clamp(min=0) * proj).sum())
    tail_lambda = float(vals[m:].clamp(min=0).mean()) if vals.numel() > m else 0.0
    leftover = max(v.shape[1] - float(proj.sum()), 0.0)
    return (head + tail_lambda * leftover) / float(vals.clamp(min=0).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path, help="base HF checkpoint (tensor names and shapes)")
    parser.add_argument("--eig", required=True, type=Path, help="activation_covariance_overlap.py --save-eig file")
    parser.add_argument("--k-max", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--center-out-mean", action="store_true",
                        help="project the activation-mean direction out of A^{1/2} Z before the QR (needs '.mean')")
    parser.add_argument("--report-k", default="16,64,256")
    parser.add_argument("--compare", action="append", default=[], metavar="NAME=PATH",
                        help="other '<name>.V' basis files whose captured activation energy is printed alongside")
    parser.add_argument("--match", default=None, metavar="PATH",
                        help="a '<name>.V' basis file (e.g. a ΔW reference); per layer/group, tune alpha in "
                             "orth(A^{alpha/2} Z) so the random basis captures the same activation energy as the "
                             "reference's top --match-k (mean over the group's tensors). Default: alpha = 1.")
    parser.add_argument("--match-k", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; remove it or pick another --output")
    report_k = [int(k) for k in args.report_k.split(",") if k.strip() and int(k) <= args.k_max]

    locations = tensor_locations(args.base)
    names = all_target_tensors(locations)
    generator = torch.Generator().manual_seed(args.seed)
    compare = {}
    for item in args.compare:
        name, _, path = item.partition("=")
        compare[name] = safe_open(path, framework="pt")

    match = safe_open(str(args.match), framework="pt") if args.match else None
    group_names = defaultdict(list)  # (layer, group) -> tensor names reading that input
    for name in names:
        group_names[(int(CANONICAL_RE.match(name).group(1)), GROUP_OF_KIND[tensor_kind(name)])].append(name)
    alphas = defaultdict(list)  # group -> per layer

    tensors: dict[str, torch.Tensor] = {}
    group_bases: dict[tuple[int, str], torch.Tensor] = {}
    energy = defaultdict(list)  # (group, basis, k) -> per layer
    with safe_open(str(args.eig), framework="pt") as eig:
        keys = set(eig.keys())
        for name in names:
            layer, group = int(CANONICAL_RE.match(name).group(1)), GROUP_OF_KIND[tensor_kind(name)]
            vecs = eig.get_tensor(f"layers.{layer}.{group}.eigvecs").double()
            vals = eig.get_tensor(f"layers.{layer}.{group}.eigvals").double()
            n_in = vecs.shape[0]
            if (layer, group) not in group_bases:
                z = torch.randn(n_in, args.k_max, generator=generator, dtype=torch.float64)
                if args.center_out_mean:
                    key = f"layers.{layer}.{group}.mean"
                    if key not in keys:
                        raise KeyError(f"--center-out-mean needs {key} in {args.eig}")
                    mu = eig.get_tensor(key).double()
                    mu = mu / mu.norm()
                    # remove the mean direction from Z's span before shaping; A^{p} keeps mu ~ eigvec 1
                    z = z - mu[:, None] * (mu @ z)[None, :]
                if match is not None:
                    target = statistics.mean(
                        captured_energy(match.get_tensor(f"{n}.V").double()[:, :args.match_k], vecs, vals)
                        for n in group_names[(layer, group)])
                    alpha, v = matched_alpha(z, vecs, vals, args.match_k, target)
                    alphas[group].append(alpha)
                else:
                    v = orth(sqrt_a_times(z, vecs, vals))
                group_bases[(layer, group)] = v
                for k in report_k:
                    energy[(group, "act-random", k)].append(captured_energy(v[:, :k], vecs, vals))
                    top = vals.clamp(min=0)
                    energy[(group, "act-top (max)", k)].append(float(top[:k].sum() / top.sum()))
            v = group_bases[(layer, group)]
            with safe_open(str(locations[name]), framework="pt") as handle:
                n_out, n_in_w = handle.get_slice(name).get_shape()
            if n_in_w != n_in:
                raise ValueError(f"{name} has n_in={n_in_w}, eigvecs of {group} have {n_in}")
            tensors[f"{name}.V"] = v.float().contiguous()
            tensors[f"{name}.U"] = torch.zeros(n_out, 1)
            for cname, handle in compare.items():
                ckey = f"{name}.V"
                if ckey in handle.keys():
                    cv = handle.get_tensor(ckey).double()
                    for k in report_k:
                        if k <= cv.shape[1]:
                            energy[(f"{group} [{tensor_kind(name)}]", cname, k)].append(
                                captured_energy(cv[:, :k], vecs, vals))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata={
        "k_max": str(args.k_max), "kind": "activation_energy_random", "side": "right", "seed": str(args.seed),
        "center_out_mean": str(args.center_out_mean), "eig": str(args.eig),
        "match": str(args.match or ""), "match_k": str(args.match_k if args.match else "")})
    print(f"exported energy-matched random bases (k_max={args.k_max}) for {len(tensors) // 2} tensors -> {args.output}")
    if alphas:
        print(f"alpha matched to {args.match} at k={args.match_k}: " + "  ".join(
            f"{g}: median {statistics.median(a):.2f} [{min(a):.2f},{max(a):.2f}]" for g, a in sorted(alphas.items())))

    print("\n##### captured activation energy tr(V^T A V)/tr(A): median over layers [min, max] #####")
    fmt = lambda v: f"{statistics.median(v):.3f} [{min(v):.3f},{max(v):.3f}]"  # noqa: E731
    for row in sorted({(g, b) for g, b, _ in energy}):
        print(f"  {row[0]:<22} {row[1]:<16} " + "  ".join(
            f"k={k}:{fmt(energy[(row[0], row[1], k)])}" for k in report_k if energy[(row[0], row[1], k)]))


if __name__ == "__main__":
    main()
