"""Export one task's top input-activation eigenvectors as a remove_delta_subspace.py --subspace file.

Reads an activation_covariance_overlap.py --save-eig file (per layer and input group, the
eigenvectors of the input second moment E[x x^T], sorted by eigenvalue = energy, descending) and
writes, for every q/k/v/o/gate/up/down weight, its layer/group's top --k-max eigenvectors as
"<name>.V" (n_in, k_max), plus a zero "<name>.U" (n_out, 1) placeholder: input-side bases, so use
them with --side right. Because the columns are energy-ordered, remove_delta_subspace.py --k
4,16,64,... removes each tensor's top-k highest-energy input directions -- one file serves a whole
dose curve in a single call.

    python export_activation_eig_subspace.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --eig /mnt/L202500431/orbit/output/subspace_overlap/actcov_eig_math_t07.safetensors \\
        --k-max 256 --output /mnt/L202500431/orbit/output/task_split_export_energy/top_energy_math_k256.safetensors
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from subspace_overlap_profile import GROUP_OF_KIND, all_target_tensors, tensor_kind
from svd_rank_profile import CANONICAL_RE, tensor_locations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path, help="base HF checkpoint (tensor names and shapes)")
    parser.add_argument("--eig", required=True, type=Path, help="activation_covariance_overlap.py --save-eig file")
    parser.add_argument("--k-max", type=int, default=256)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    locations = tensor_locations(args.base)
    tensors, missing = {}, []
    with safe_open(str(args.eig), framework="pt") as eig:
        keys = set(eig.keys())
        for name in all_target_tensors(locations):
            layer, group = int(CANONICAL_RE.match(name).group(1)), GROUP_OF_KIND[tensor_kind(name)]
            key = f"layers.{layer}.{group}.eigvecs"
            if key not in keys:
                missing.append(name)
                continue
            vectors = eig.get_tensor(key).float()
            if vectors.shape[1] < args.k_max:
                raise ValueError(f"{key} has {vectors.shape[1]} eigenvectors < --k-max {args.k_max}")
            with safe_open(str(locations[name]), framework="pt") as handle:
                n_out, n_in = handle.get_slice(name).get_shape()
            if vectors.shape[0] != n_in:
                raise ValueError(f"{key} is {tuple(vectors.shape)}, but {name} has n_in={n_in}")
            tensors[f"{name}.V"] = vectors[:, :args.k_max].clone()  # q/k/v (gate/up) share a group: no aliasing
            tensors[f"{name}.U"] = torch.zeros(n_out, 1)
    if missing:
        raise ValueError(f"{len(missing)} tensors have no eigenvectors in {args.eig} (e.g. {missing[0]}); "
                         f"was activation_covariance_overlap.py run with all --groups?")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata={"k_max": str(args.k_max), "kind": "activation_top_energy",
                                                  "side": "right", "eig": str(args.eig)})
    print(f"exported top-{args.k_max} activation eigenvectors for {len(tensors) // 2} tensors -> {args.output}")


if __name__ == "__main__":
    main()
