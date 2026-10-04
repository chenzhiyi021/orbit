"""How large is each fine-tune's implicit bias, and do different fine-tunes write the same one?

For every q/k/v/o/gate/up/down weight and every --model, b = (e_1^T mu) * D e_1 (implicit_bias.py)
with D = W_ft - W_base. Per tensor kind (median over layers [min, max]) it reports

  |b|          norm of the bias the fine-tune adds to this layer's output on every token
  |b|/|b_W|    relative to the base weight's own implicit bias (e_1^T mu) * W e_1
  biasShare    |b|^2 / E||D x||^2 -- share of the update's output change that is this constant bias
               (E||D x||^2 = tr(D A D^T), from the saved eigenpairs; the tail beyond the saved
               eigenvectors is approximated by the mean leftover eigenvalue)
  deltaEnergy  ||D e_1||^2 / ||D||^2 -- the same component in weight-energy terms (what --k 1 removes)

and, for every pair of models, the cosine between their bias vectors and their norm ratio.

    python bias_vector_compare.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --eig /mnt/.../actcov_eig_math_t07_mean.safetensors \\
        --model opd300=/mnt/... --model rl300=/mnt/... --model rlp300=/mnt/... --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from implicit_bias import ImplicitBias
from subspace_overlap_profile import GROUP_OF_KIND, KIND_ORDER, all_target_tensors, tensor_kind
from svd_rank_profile import CANONICAL_RE, load_tensor, tensor_locations


def output_energy(delta: torch.Tensor, eigvecs: torch.Tensor, eigvals: torch.Tensor) -> float:
    """E||D x||^2 = tr(D A D^T) = sum_k lambda_k ||D e_k||^2; eigenvectors beyond those saved are
    approximated by the mean of the remaining eigenvalues times D's leftover energy."""
    k = eigvecs.shape[1]
    proj = delta @ eigvecs
    head = float((eigvals[:k] * proj.pow(2).sum(0)).sum())
    leftover = max(float(delta.pow(2).sum() - proj.pow(2).sum()), 0.0)
    tail_lambda = float(eigvals[k:].mean()) if eigvals.numel() > k else 0.0
    return head + tail_lambda * leftover


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--eig", required=True, type=Path, help="--save-eig file that includes '.mean' entries")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ib = ImplicitBias(args.base, args.eig, args.device)
    models = dict(item.split("=", 1) for item in args.model)
    locations = {name: tensor_locations(Path(path)) for name, path in models.items()}
    names = all_target_tensors(ib.base_locations)
    stats = defaultdict(list)  # (kind, model, metric) -> per layer
    pair_stats = defaultdict(list)  # (kind, a, b, metric)
    for name in names:
        kind = tensor_kind(name)
        layer, group = int(CANONICAL_RE.match(name).group(1)), GROUP_OF_KIND[kind]
        w0 = ib.base_weight(name)
        e1, m = ib.direction(name)
        eigvecs = ib.eig.get_tensor(f"layers.{layer}.{group}.eigvecs").to(ib.device).double()
        eigvals = ib.eig.get_tensor(f"layers.{layer}.{group}.eigvals").to(ib.device).double()
        b_w = float((m * (w0 @ e1)).norm())
        biases = {}
        for model, path in models.items():
            delta = load_tensor(Path(path), locations[model], name).to(ib.device).double() - w0
            b = m * (delta @ e1)
            biases[model] = b
            out_e = output_energy(delta, eigvecs, eigvals)
            stats[(kind, model, "norm")].append(float(b.norm()))
            stats[(kind, model, "rel_base")].append(float(b.norm()) / max(b_w, 1e-30))
            stats[(kind, model, "bias_share")].append(float(b.pow(2).sum()) / max(out_e, 1e-30))
            stats[(kind, model, "delta_energy")].append(float((delta @ e1).pow(2).sum() / delta.pow(2).sum()))
        for a, c in itertools.combinations(models, 2):
            ba, bc = biases[a], biases[c]
            pair_stats[(kind, a, c, "cos")].append(float(ba @ bc / (ba.norm() * bc.norm()).clamp(min=1e-30)))
            pair_stats[(kind, a, c, "ratio")].append(float(ba.norm() / bc.norm().clamp(min=1e-30)))
        if kind == "down_proj":
            print(f"  layer {layer} done", flush=True)

    fmt = lambda v: f"{statistics.median(v):8.3f} [{min(v):7.3f},{max(v):8.3f}]"  # noqa: E731
    print("\n##### implicit bias b = (e1^T mu) * dW e1 per tensor kind: median over layers [min, max]")
    for kind in KIND_ORDER:
        print(f"=== {kind}")
        print(f"    {'model':<12} {'|b|':>27} {'|b|/|b_W|':>27} {'biasShare':>27} {'deltaEnergy':>27}")
        for model in models:
            print(f"    {model:<12} " + " ".join(fmt(stats[(kind, model, k)])
                                             for k in ("norm", "rel_base", "bias_share", "delta_energy")))
        for a, c in itertools.combinations(models, 2):
            print(f"    cos(b_{a}, b_{c}) {fmt(pair_stats[(kind, a, c, 'cos')])}   "
                  f"|b_{a}|/|b_{c}| {fmt(pair_stats[(kind, a, c, 'ratio')])}")


if __name__ == "__main__":
    main()
