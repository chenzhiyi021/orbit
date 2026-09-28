"""Per-tensor top-k singular subspaces of a fine-tune's weight deltas, saved for retraining.

For every canonical q/k/v/o/gate/up/down weight this computes

    U S V^T = SVD(W_finetuned - W_base)

and stores the leading --k-max left singular vectors U[:, :k_max] (output space), right
singular vectors V[:, :k_max] (input space) and singular values S[:k_max]. A later training
run can then keep its own update out of the first k <= k_max of these directions, e.g. by
projecting its delta after every optimizer step:

    left  side:  Delta <- (I - U_k U_k^T) Delta
    right side:  Delta <- Delta (I - V_k V_k^T)
    both sides:  Delta <- (I - U_k U_k^T) Delta (I - V_k V_k^T)

Only k_max is fixed here; the excluded k and the side are chosen at training time by
slicing the first k columns, so one file serves k = 1, 16, 64, ... alike.

--random writes a control with the same shapes: independent Haar-random orthonormal
columns per tensor (QR of a Gaussian), so "exclude the fine-tune's top-k" can be compared
against "exclude k random directions". Random files carry no singular values.

Output is a single safetensors file keyed by HF tensor name:
    "<name>.U"  (out, k_max) fp32      "<name>.V"  (in, k_max) fp32
    "<name>.S"  (k_max,)     fp32      (fine-tune subspaces only)
with the run's settings in the safetensors metadata. Mapping these HF-layout (per q/k/v,
per gate/up) bases onto Megatron's fused linear_qkv / linear_fc1 is left to the trainer.

    python build_exclusion_subspace.py \\
        --base /mnt/L202500431/models/qwen3-1.7b \\
        --finetuned /mnt/L202500431/models/zh_models/m6_st_full_non_thinking_hf_step300 \\
        --k-max 256 --device cuda \\
        --output /mnt/L202500431/models/exclusion_subspaces/m6_full_step300_top256.safetensors
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from svd_rank_profile import CANONICAL_RE, shard_files, tensor_locations


def random_orthonormal(n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(n, k, generator=generator, dtype=torch.float64))
    # fix column signs so the result is Haar-distributed, not biased by QR's convention
    return (q * torch.sign(torch.diagonal(r))).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path, help="base HF checkpoint directory")
    parser.add_argument("--finetuned", type=Path, help="fine-tuned HF checkpoint directory (not needed with --random)")
    parser.add_argument("--k-max", type=int, required=True, help="number of leading directions to store per tensor")
    parser.add_argument("--output", required=True, type=Path, help="output .safetensors file")
    parser.add_argument("--random", action="store_true", help="store Haar-random orthonormal bases instead")
    parser.add_argument("--seed", type=int, default=0, help="seed for --random")
    parser.add_argument("--device", default="cpu", help="where the SVDs run, e.g. 'cpu' or 'cuda'")
    args = parser.parse_args()
    if not args.random and args.finetuned is None:
        parser.error("--finetuned is required unless --random is given")
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; remove it or pick another --output")

    device = torch.device(args.device)
    base_locations = tensor_locations(args.base)
    names = sorted(name for name in base_locations if CANONICAL_RE.match(name))
    generator = torch.Generator().manual_seed(args.seed)
    tensors: dict[str, torch.Tensor] = {}
    kept_energy: dict[str, list[float]] = {}
    started = time.perf_counter()

    with ExitStack() as stack:
        base_handles = {path: stack.enter_context(safe_open(path, framework="pt"))
                        for path in sorted(set(base_locations.values()))}
        tuned_handles, tuned_locations = {}, {}
        if not args.random:
            tuned_locations = tensor_locations(args.finetuned)
            tuned_handles = {path: stack.enter_context(safe_open(path, framework="pt"))
                             for path in shard_files(args.finetuned)}

        for i, name in enumerate(names, 1):
            base = base_handles[base_locations[name]].get_tensor(name)
            n_out, n_in = base.shape
            k = min(args.k_max, n_out, n_in)
            if args.random:
                tensors[f"{name}.U"] = random_orthonormal(n_out, k, generator).contiguous()
                tensors[f"{name}.V"] = random_orthonormal(n_in, k, generator).contiguous()
            else:
                tuned = tuned_handles[tuned_locations[name]].get_tensor(name)
                delta = tuned.to(device).float() - base.to(device).float()
                u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                tensors[f"{name}.U"] = u[:, :k].cpu().contiguous()
                tensors[f"{name}.V"] = vh[:k].T.cpu().contiguous()
                tensors[f"{name}.S"] = s[:k].cpu().contiguous()
                energy = s.pow(2)
                kept_energy.setdefault(name.split(".")[-2], []).append(float(energy[:k].sum() / energy.sum()))
                del delta, u, s, vh
            print(f"  [{i}/{len(names)}] {name}  shape=({n_out}, {n_in})  k={k}  "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    metadata = {
        "k_max": str(args.k_max),
        "kind": "random" if args.random else "delta_svd",
        "base": str(args.base),
        "finetuned": "" if args.random else str(args.finetuned),
        "seed": str(args.seed) if args.random else "",
        "layout": "hf",  # per q/k/v and gate/up; not Megatron's fused linear_qkv / linear_fc1
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=metadata)

    # orthonormality spot check on the first tensor
    first = names[0]
    for side in ("U", "V"):
        q = tensors[f"{first}.{side}"]
        err = (q.T @ q - torch.eye(q.shape[1])).abs().max().item()
        print(f"orthonormality |Q^T Q - I|_max for {first}.{side}: {err:.2e}")
    if kept_energy:
        print(f"\nfraction of ||Delta W||_F^2 inside the stored top-{args.k_max}, mean over layers:")
        for kind, fractions in sorted(kept_energy.items()):
            print(f"  {kind:10s} {sum(fractions) / len(fractions):.3f}")
    print(f"\nwrote {len(names)} tensors' bases -> {args.output}  ({time.perf_counter() - started:.0f}s)")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
