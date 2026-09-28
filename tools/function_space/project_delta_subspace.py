"""Transplant one fine-tune's update through another fine-tune's input- or output-side subspace.

Question: the reasoning update and, e.g., a creative-writing update share a lot of their top
*input-side* (right singular) directions but few *output-side* (left) ones. Do the shared
directions carry the reasoning capability, or is it carried elsewhere? Projecting an update
onto its OWN top-k left or right subspace cannot answer this -- U_k U_k^T dW = dW V_k V_k^T =
the rank-k truncation either way -- so the source update is projected onto the REFERENCE
update's subspaces instead. For every canonical q/k/v/o/gate/up/down weight:

    dS = W_source - W_base           (optionally truncated to its top --source-rank directions)
    U_R, V_R = top-k left / right singular vectors of (W_reference - W_base)

    in_keep     W_base + dS V_R V_R^T         dS restricted to the reference's input directions
    in_remove   W_base + dS (I - V_R V_R^T)   ... with them removed
    out_keep    W_base + U_R U_R^T dS         dS restricted to the reference's output directions
    out_remove  W_base + (I - U_R U_R^T) dS   ... with them removed
    rand_in_keep / rand_out_keep              in_keep / out_keep with V_R / U_R replaced by a
                                              random orthonormal basis of the same size (chance)

Every other tensor (norms, embeddings) comes from the source fine-tune, so all variants differ
only in the projected linear-layer updates. Each variant is written as its own HF directory
<output-root>/<prefix>_<mode>_k<k>/ with projection_report.json holding the fraction of
||dS||_F^2 it keeps (overall and per kind) -- read every eval number against that fraction.

Memory: the source deltas are kept on the host in fp32 (about 4 bytes per linear-layer
parameter) together with the reference's top-k factors; variants are then built and written
one at a time, so only one output copy is resident.

    python project_delta_subspace.py \\
        --base /mnt/L202500431/models/qwen3-1.7b \\
        --source /mnt/L202500431/models/zh_models/m6_st_full_non_thinking_hf_step300 \\
        --reference /mnt/.../writing/hf_iter_0000019 \\
        --source-rank 64 --ref-k 64 --random-control \\
        --output-root /mnt/L202500431/models/subspace_transplant --prefix reason300_via_writing \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from svd_rank_profile import CANONICAL_RE, shard_files, tensor_locations

MODES = ("in_keep", "in_remove", "out_keep", "out_remove")
RANDOM_MODES = ("rand_in_keep", "rand_out_keep")


def kind_of(name: str) -> str:
    return name.split(".")[-2]


def random_basis(n: int, k: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, k, generator=generator))
    return q.to(device)


def project(mode: str, delta: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if mode in ("in_keep", "rand_in_keep"):
        return (delta @ v) @ v.T
    if mode == "in_remove":
        return delta - (delta @ v) @ v.T
    if mode in ("out_keep", "rand_out_keep"):
        return u @ (u.T @ delta)
    if mode == "out_remove":
        return delta - u @ (u.T @ delta)
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="fine-tune whose update is transplanted")
    parser.add_argument("--reference", required=True, type=Path, help="fine-tune whose subspaces are used")
    parser.add_argument("--source-rank", default="64",
                        help="truncate the source update to its top-r directions first ('full' = no truncation)")
    parser.add_argument("--ref-k", default="64", help="comma-separated reference subspace sizes, e.g. '16,64'")
    parser.add_argument("--modes", default=",".join(MODES), help=f"comma-separated subset of {MODES}")
    parser.add_argument("--random-control", action="store_true", help=f"also write {RANDOM_MODES}")
    parser.add_argument("--seed", type=int, default=20260928, help="for the random-control bases")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    source_rank = None if args.source_rank == "full" else int(args.source_rank)
    ks = [int(k) for k in args.ref_k.split(",") if k.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if unknown := set(modes) - set(MODES):
        raise ValueError(f"unknown --modes {sorted(unknown)}; choose from {MODES}")
    if args.random_control:
        modes += list(RANDOM_MODES)
    variants = [(mode, k) for mode in modes for k in ks]
    out_dirs = {v: args.output_root / f"{args.prefix}_{v[0]}_k{v[1]}" for v in variants}
    for out_dir in out_dirs.values():
        if out_dir.exists() and any(out_dir.glob("*.safetensors")):
            raise FileExistsError(f"{out_dir} already has weights; remove it or change --output-root/--prefix")

    base_loc = tensor_locations(args.base)
    ref_loc = tensor_locations(args.reference)
    k_max = max(ks)
    started = time.perf_counter()

    # Pass 1: per target tensor, the (optionally truncated) source delta and the reference's
    # top-k_max singular vectors, all on the host. Non-target tensors come from the source.
    source_shards = shard_files(args.source)
    passthrough: dict[str, dict[str, torch.Tensor]] = {}   # shard name -> non-target tensors
    targets: dict[str, dict] = {}                           # tensor name -> cached pieces
    shard_of: dict[str, str] = {}
    with ExitStack() as stack:
        handles = {}
        for path in sorted({*base_loc.values(), *ref_loc.values()}):
            handles[path] = stack.enter_context(safe_open(path, framework="pt"))
        n_targets = sum(1 for name in tensor_locations(args.source) if CANONICAL_RE.match(name))
        done = 0
        for shard in source_shards:
            passthrough[shard.name] = {}
            with safe_open(shard, framework="pt") as handle:
                for name in handle.keys():
                    shard_of[name] = shard.name
                    tuned = handle.get_tensor(name)
                    if not CANONICAL_RE.match(name):
                        passthrough[shard.name][name] = tuned
                        continue
                    base = handles[base_loc[name]].get_tensor(name)
                    reference = handles[ref_loc[name]].get_tensor(name)
                    delta = tuned.to(device).float() - base.to(device).float()
                    if source_rank is not None and source_rank < min(delta.shape):
                        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                        delta = (u[:, :source_rank] * s[:source_rank]) @ vh[:source_rank]
                    ref_delta = reference.to(device).float() - base.to(device).float()
                    ru, _, rvh = torch.linalg.svd(ref_delta, full_matrices=False)
                    r = min(k_max, ru.shape[1])
                    targets[name] = dict(base=base, dtype=tuned.dtype, delta=delta.cpu(),
                                         u=ru[:, :r].cpu(), v=rvh[:r].T.contiguous().cpu())
                    done += 1
                    print(f"  [{done}/{n_targets}] {name}  shape={tuple(delta.shape)}  "
                          f"({time.perf_counter() - started:.0f}s)", flush=True)

    # Pass 2: build and write one variant at a time.
    shard_names = [shard.name for shard in source_shards]
    for variant_index, (mode, k) in enumerate(variants):
        kept = defaultdict(float)
        total = defaultdict(float)
        per_shard = {s: dict(passthrough[s]) for s in shard_names}
        for tensor_index, (name, t) in enumerate(targets.items()):
            delta = t["delta"].to(device)
            kk = min(k, t["u"].shape[1])
            if mode.startswith("rand_"):
                seed = args.seed + 1000003 * tensor_index + kk
                u = random_basis(delta.shape[0], kk, seed, device)
                v = random_basis(delta.shape[1], kk, seed + 1, device)
            else:
                u, v = t["u"][:, :kk].to(device), t["v"][:, :kk].to(device)
            projected = project(mode, delta, u, v)
            kept[kind_of(name)] += float(projected.pow(2).sum())
            total[kind_of(name)] += float(delta.pow(2).sum())
            per_shard[shard_of[name]][name] = (t["base"].to(device).float() + projected).to(t["dtype"]).cpu()
        out_dir = out_dirs[(mode, k)]
        out_dir.mkdir(parents=True, exist_ok=True)
        for shard_name, tensors in per_shard.items():
            save_file(tensors, str(out_dir / shard_name), metadata={"format": "pt"})
        for item in args.source.iterdir():
            if item.is_file() and item.suffix != ".safetensors":
                shutil.copy2(item, out_dir / item.name)
        overall = sum(kept.values()) / max(sum(total.values()), 1e-30)
        per_kind = {kind: kept[kind] / total[kind] for kind in sorted(total)}
        (out_dir / "projection_report.json").write_text(json.dumps(dict(
            base=str(args.base), source=str(args.source), reference=str(args.reference),
            source_rank=args.source_rank, mode=mode, ref_k=k, kept_energy_fraction_all=overall,
            kept_energy_fraction_per_kind=per_kind), indent=2))
        print(f"[{variant_index + 1}/{len(variants)}] {mode:14s} k={k:<4d} keeps {overall:6.3f} of ||dS||^2  "
              + " ".join(f"{kind}={frac:.2f}" for kind, frac in per_kind.items())
              + f"  -> {out_dir}  ({time.perf_counter() - started:.0f}s)", flush=True)
        del per_shard


if __name__ == "__main__":
    main()
