"""Split two tasks' input-activation subspaces into shared / task-specific parts and ask where
each fine-tune delta's input side lives.

Inputs are two activation_covariance_overlap.py --save-eig files (e.g. math and writing prompts),
holding per layer and input group the top eigenvectors E_A, E_B (n x k) of the input second
moment. Per layer/group, principal vectors of the two top-k subspaces come from the SVD
E_A^T E_B = P diag(cos) Q^T:  U_A = E_A P,  U_B = E_B Q,  with U_A[:, i]^T U_B[:, j] = cos_i [i = j].
Each principal pair i is bucketed by cos_i^2:

    shared      cos^2 >= --shared-cos2     basis (U_A[:, i] + U_B[:, i]) / ||.||  (bisectors)
    A-specific  cos^2 <  --specific-cos2   basis U_A[:, i]   (nearly orthogonal to all of E_B)
    B-specific  cos^2 <  --specific-cos2   basis U_B[:, i]
    partial     in between                 counted, not used as a bucket

All bucket bases are orthonormal; A- and B-specific directions are orthogonal to each other except
within a pair (cos^2 < --specific-cos2, so nearly). For every --checkpoint delta D (and W itself,
as a control) and each bucket basis B of dimension d, the script reports

    energy  = ||D B||_F^2 / ||D||_F^2          chance d / n
    top16   = ||V_16(D)^T B||_F^2 / 16         chance d / n   (D's top-16 right singular vectors)

and their enrichment over chance. Prediction under "the task's activations decide where D goes":
math deltas are enriched in math-specific directions, writing deltas in writing-specific ones,
and both in the shared ones.

--export-dir writes each bucket as a remove_delta_subspace.py --subspace file, for ablations:
    <A>_only.safetensors, <B>_only.safetensors, shared.safetensors
    random_like_<A>_only.safetensors, random_like_shared.safetensors   (Haar-random, same dimension
                                                                        per layer and input group)
    top_energy_<A>_like_shared.safetensors        A's top-d eigenvectors, d = that layer/group's
                                                  shared dimension (is it energy, not sharing?)
    random_in_<A>_top<k>_like_shared.safetensors  random d-dim subspace inside span(E_A top-k)
keyed "<hf tensor name>.V" (n_in, width) for every q/k/v/o/gate/up/down weight, the bucket basis of
the tensor's layer and input group zero-padded to one width (zero columns leave the projection
unchanged), and "<name>.U" as a zero (n_out, 1) placeholder: these are input-side bases, so use them
with --side right and --k <the file's k_max>.

Usage:
    python task_subspace_split.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --eig-a math=actcov_eig_math_t07.safetensors --eig-b writing=actcov_eig_writing_t07.safetensors \\
        --checkpoint opd19=/mnt/.../opd19 --checkpoint wri19=/mnt/.../wri19 --k 256 --device cuda
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from subspace_overlap_profile import (
    GROUP_OF_KIND,
    KIND_ORDER,
    SVD_DTYPES,
    all_target_tensors,
    subspace_sim,
    tensor_kind,
    top_k_bases,
)
from build_exclusion_subspace import random_orthonormal
from svd_rank_profile import CANONICAL_RE, load_tensor, tensor_locations

BASE_W_NAME = "W_base"
TOP_DIRECTIONS = 16
SIM_KS = (1, 4, 16, 64, 256)


def parse_named(item: str) -> tuple[str, Path]:
    name, _, path = item.partition("=")
    if not path:
        raise ValueError(f"expected NAME=PATH, got {item!r}")
    return name, Path(path)


def split_subspaces(e_a: torch.Tensor, e_b: torch.Tensor, shared_cos2: float,
                    specific_cos2: float) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Bucket bases {shared, a, b} (n x d each) and the principal cos^2 (descending)."""
    p, cos, qh = torch.linalg.svd(e_a.transpose(0, 1) @ e_b)
    cos = cos.clamp(max=1.0)
    u_a, u_b = e_a @ p, e_b @ qh.transpose(0, 1)
    cos2 = cos.pow(2)
    shared, specific = cos2 >= shared_cos2, cos2 < specific_cos2
    bisect = u_a[:, shared] + u_b[:, shared]
    bisect = bisect / torch.linalg.norm(bisect, dim=0, keepdim=True).clamp(min=1e-12)
    return {"shared": bisect, "a": u_a[:, specific], "b": u_b[:, specific]}, cos2


def export_buckets(args, buckets: dict[tuple[int, str], dict[str, torch.Tensor]],
                   base_locations: dict[str, Path], groups: list[str], names: dict[str, str],
                   energy_controls: dict[tuple[int, str], dict[str, torch.Tensor]]) -> None:
    """Write every bucket, random controls matched to the a-only and shared dimensions, and the
    shared-dimension-matched energy controls (top eigenvectors of A, random inside A's top-k)."""
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(args.seed)
    tensor_names = [name for name in all_target_tensors(base_locations) if GROUP_OF_KIND[tensor_kind(name)] in groups]
    shapes = {}
    for name in tensor_names:
        with safe_open(str(base_locations[name]), framework="pt") as handle:
            shapes[name] = tuple(handle.get_slice(name).get_shape())
    random_bases = {}  # (layer, group, source bucket) -> basis, shared by the tensors of one group
    for (layer, group), layer_buckets in buckets.items():
        for source in ("a", "shared"):
            n, d = layer_buckets[source].shape
            random_bases[(layer, group, source)] = random_orthonormal(n, d, generator) if d else layer_buckets[source][:, :0].cpu()
    outputs = {"shared": "shared", "a": f"{names['a']}_only", "b": f"{names['b']}_only",
               "random_a": f"random_like_{names['a']}_only", "random_shared": "random_like_shared",
               "energy_top": f"top_energy_{names['a']}_like_shared",
               "energy_rand_in": f"random_in_{names['a']}_top{args.k}_like_shared"}
    args.export_dir.mkdir(parents=True, exist_ok=True)
    for bucket, stem in outputs.items():
        per_tensor = {}
        for name in tensor_names:
            layer, group = int(CANONICAL_RE.match(name).group(1)), GROUP_OF_KIND[tensor_kind(name)]
            if bucket.startswith("energy_"):
                per_tensor[name] = energy_controls[(layer, group)][bucket.removeprefix("energy_")]
            elif bucket.startswith("random_"):
                per_tensor[name] = random_bases[(layer, group, bucket.removeprefix("random_"))]
            else:
                per_tensor[name] = buckets[(layer, group)][bucket].float().cpu()
        width = max(1, max(basis.shape[1] for basis in per_tensor.values()))
        tensors = {}
        for name, basis in per_tensor.items():
            padded = torch.zeros(basis.shape[0], width)
            padded[:, :basis.shape[1]] = basis
            tensors[f"{name}.V"] = padded.contiguous()
            tensors[f"{name}.U"] = torch.zeros(shapes[name][0], 1)
        dims = [basis.shape[1] for basis in per_tensor.values()]
        metadata = {"k_max": str(width), "kind": f"task_split_{stem}", "side": "right",
                    "eig_a": args.eig_a, "eig_b": args.eig_b, "k": str(args.k),
                    "shared_cos2": str(args.shared_cos2), "specific_cos2": str(args.specific_cos2),
                    "seed": str(args.seed)}
        path = args.export_dir / f"{stem}.safetensors"
        save_file(tensors, str(path), metadata=metadata)
        print(f"exported {stem}: {len(per_tensor)} tensors, dims {min(dims)}..{max(dims)} "
              f"(mean {statistics.mean(dims):.1f}), k_max={width} -> {path}", flush=True)
    print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--eig-a", required=True, metavar="NAME=PATH", help="first task's --save-eig file")
    parser.add_argument("--eig-b", required=True, metavar="NAME=PATH", help="second task's --save-eig file")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--k", type=int, default=256, help="top-k eigenvectors per task subspace")
    parser.add_argument("--shared-cos2", type=float, default=0.5)
    parser.add_argument("--specific-cos2", type=float, default=0.1)
    parser.add_argument("--groups", default="attn_in,o_in,mlp_in,down_in")
    parser.add_argument("--export-dir", type=Path, default=None,
                         help="also write each bucket (and dimension-matched random controls) as a "
                              "remove_delta_subspace.py --subspace file")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random control bases")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--svd-dtype", default="float32", choices=tuple(SVD_DTYPES))
    args = parser.parse_args()

    started = time.perf_counter()
    device, dtype = torch.device(args.device), SVD_DTYPES[args.svd_dtype]
    (name_a, path_a), (name_b, path_b) = parse_named(args.eig_a), parse_named(args.eig_b)
    bucket_label = {"shared": "shared", "a": f"{name_a}-only", "b": f"{name_b}-only"}
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    checkpoints = dict(parse_named(item) for item in args.checkpoint)
    if BASE_W_NAME in checkpoints:
        raise ValueError(f"--checkpoint name {BASE_W_NAME!r} is reserved")

    handle_a, handle_b = safe_open(str(path_a), framework="pt"), safe_open(str(path_b), framework="pt")

    def eigvecs(handle, layer: int, group: str) -> torch.Tensor:
        vectors = handle.get_tensor(f"layers.{layer}.{group}.eigvecs")
        if args.k > vectors.shape[1]:
            raise ValueError(f"--k {args.k} > the {vectors.shape[1]} eigenvectors saved for layer {layer} {group}")
        return vectors[:, :args.k].to(device, dtype)

    # 1. How much do the two tasks' activation subspaces overlap, and the bucket split itself.
    layers = sorted({int(key.split(".")[1]) for key in handle_a.keys()})
    sim_ks = sorted({k for k in SIM_KS if k <= args.k} | {args.k})
    buckets: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    # Energy controls, dimension-matched to each (layer, group)'s shared bucket: A's top-d
    # eigenvectors ("top"), and a random d-dim subspace inside span(E_a top-k) ("rand_in").
    energy_controls: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    control_generator = torch.Generator().manual_seed(args.seed + 1)
    overlap = defaultdict(list)  # (group, stat) -> per layer
    print(f"##### {name_a} vs {name_b} activation subspaces: top-k overlap and bucket sizes "
          f"(k={args.k}, shared cos^2 >= {args.shared_cos2}, specific cos^2 < {args.specific_cos2}) #####")
    for group in groups:
        for layer in layers:
            e_a, e_b = eigvecs(handle_a, layer, group), eigvecs(handle_b, layer, group)
            for k in sim_ks:
                overlap[(group, f"sim_k{k}")].append(subspace_sim(e_a[:, :k], e_b[:, :k]))
            buckets[(layer, group)], cos2 = split_subspaces(e_a, e_b, args.shared_cos2, args.specific_cos2)
            d_shared = buckets[(layer, group)]["shared"].shape[1]
            e_a_cpu = e_a.float().cpu()
            energy_controls[(layer, group)] = {
                "top": e_a_cpu[:, :d_shared].contiguous(),
                "rand_in": e_a_cpu @ random_orthonormal(e_a.shape[1], d_shared, control_generator) if d_shared
                           else e_a_cpu[:, :0],
            }
            if d_shared:
                shared_cpu = buckets[(layer, group)]["shared"].float().cpu()
                overlap[(group, "sim_shared_top")].append(subspace_sim(shared_cpu, energy_controls[(layer, group)]["top"]))
            overlap[(group, "n")].append(e_a.shape[0])
            overlap[(group, "cos2_1")].append(float(cos2[0]))
            for bucket, basis in buckets[(layer, group)].items():
                overlap[(group, f"dim_{bucket}")].append(basis.shape[1])
            overlap[(group, "dim_partial")].append(int(((cos2 >= args.specific_cos2) & (cos2 < args.shared_cos2)).sum()))
        mean = lambda stat: statistics.mean(overlap[(group, stat)])  # noqa: E731
        sims = "  ".join(f"k={k}:{mean(f'sim_k{k}'):.3f}" for k in sim_ks)
        print(f"  {group:<8} n={int(mean('n')):<5} sim_k(E_{name_a}, E_{name_b}) {sims}  (chance k/n)")
        print(f"  {'':<8} mean dims: shared {mean('dim_shared'):.1f}  {name_a}-only {mean('dim_a'):.1f}  "
              f"{name_b}-only {mean('dim_b'):.1f}  partial {mean('dim_partial'):.1f}  (of {args.k}; "
              f"largest principal cos^2 {mean('cos2_1'):.3f})")
        if overlap[(group, "sim_shared_top")]:
            print(f"  {'':<8} sim(shared, top-d energy of {name_a}) = {mean('sim_shared_top'):.3f}  "
                  f"(1 = the shared bucket IS {name_a}'s top-energy subspace)")
    print(flush=True)

    base_locations = tensor_locations(args.base)
    if args.export_dir is not None:
        export_buckets(args, buckets, base_locations, groups, {"a": name_a, "b": name_b}, energy_controls)

    # 2. Where each delta (and W) puts its input-side energy and its top directions.
    ckpt_locations = {name: tensor_locations(path) for name, path in checkpoints.items()}
    tensor_names = [name for name in all_target_tensors(base_locations) if GROUP_OF_KIND[tensor_kind(name)] in groups]
    stats: dict[tuple, list[float]] = defaultdict(list)  # (kind, model, bucket, metric) -> per layer
    for tensor_name in tensor_names:
        layer, kind = int(CANONICAL_RE.match(tensor_name).group(1)), tensor_kind(tensor_name)
        base_tensor = load_tensor(args.base, base_locations, tensor_name).to(device, dtype)
        n_in = base_tensor.shape[1]
        matrices = {name: load_tensor(path, ckpt_locations[name], tensor_name).to(device, dtype) - base_tensor
                    for name, path in checkpoints.items()}
        matrices[BASE_W_NAME] = base_tensor
        print(f"=== {tensor_name}  (shape {tuple(base_tensor.shape)}) ===", flush=True)
        for model, matrix in matrices.items():
            total = torch.linalg.norm(matrix) ** 2
            top = top_k_bases(matrix, [TOP_DIRECTIONS], dtype)["right"][TOP_DIRECTIONS]
            parts = []
            for bucket, basis in buckets[(layer, GROUP_OF_KIND[kind])].items():
                d = basis.shape[1]
                chance = d / n_in
                energy = float(torch.linalg.norm(matrix @ basis) ** 2 / total) if d else 0.0
                top_mass = float(torch.linalg.norm(top.transpose(0, 1) @ basis) ** 2 / top.shape[1]) if d else 0.0
                for metric, value in (("energy", energy), ("top16", top_mass), ("chance", chance)):
                    stats[(kind, model, bucket, metric)].append(value)
                parts.append(f"{bucket_label[bucket]}(d={d}) energy={energy:.4f} top16={top_mass:.4f} "
                             f"chance={chance:.4f}")
            print(f"    {model:<10} " + "  |  ".join(parts))
        print()

    print(f"##### total {time.perf_counter() - started:.1f}s #####\n")
    print("##### mean over layers: energy / top16 mass per bucket, and enrichment over chance (d/n) #####\n")
    for kind in sorted({key[0] for key in stats}, key=KIND_ORDER.index):
        print(f"=== summary: {kind} (input group {GROUP_OF_KIND[kind]}) ===")
        for model in dict.fromkeys(key[1] for key in stats if key[0] == kind):
            cells = []
            for bucket in ("shared", "a", "b"):
                energy, top16, chance = (statistics.mean(stats[(kind, model, bucket, m)])
                                         for m in ("energy", "top16", "chance"))
                enrich = lambda value: value / chance if chance > 0 else float("nan")  # noqa: E731
                cells.append(f"{bucket_label[bucket]}: energy {energy:.3f} ({enrich(energy):.1f}x) "
                             f"top16 {top16:.3f} ({enrich(top16):.1f}x)")
            print(f"    {model:<10} " + "  |  ".join(cells))
        print()


if __name__ == "__main__":
    main()
