"""Remove a fixed per-tensor subspace from a fine-tune's weight deltas after training.

Post-hoc counterpart of --exclude-subspace-* (orbit/backends/megatron_utils/subspace_exclusion.py),
using the same bases file (tools/function_space/build_exclusion_subspace.py) and the same projector:

    left:  P(D) = (I - U_k U_k^T) D        right: P(D) = D (I - V_k V_k^T)        both: both

    W_out = W_base + P(W_finetuned - W_base)      for every canonical q/k/v/o/gate/up/down weight

--mode keep uses the complementary projector, as --exclude-subspace-mode keep does in training
(left: P(D) = U_k U_k^T D, right: P(D) = D V_k V_k^T), to clean a fixed-subspace-trained
checkpoint the same way. In keep mode "inside" below means outside the kept subspace, i.e.
always the part P forbids.

Two uses:
  * clean an exclusion-trained checkpoint: its saved weights are bf16(W_base + P(D)), and at
    small |D| the bf16 rounding (half an ulp of W_base, ~6e-5 at |w| ~ 0.02 -- the same order
    as 20 steps of Adam at lr 5e-6) puts a residual back into the excluded subspace;
  * ablate an unconstrained fine-tune: remove its own top-k directions and see what survives
    (the complement of truncate_delta_rank.py, which keeps them).

--dtype bf16 writes a checkpoint servable as usual but, for the reason above, not exactly
outside the subspace; --dtype fp32 writes every tensor in fp32 and is exact to fp32 precision
(serve it with SGLang --dtype float32, e.g. SERVE_DTYPE=float32 for eval-math-evalchemy.sh,
and compare against baselines served the same way). projection_report.json records, per kind,

    inside_before  ||D - P(D)||^2 / ||D||^2      of the input checkpoint's delta
    kept           ||P(D)||^2 / ||D||^2
    inside_after   ||R - P(R)||^2 / ||R||^2      R = W_out - W_base as actually stored

    python remove_delta_subspace.py \\
        --base /mnt/L202500431/models/qwen3-1.7b \\
        --finetuned /mnt/.../m6_..._excl_reasontop_k64_both_hf_iter19 \\
        --subspace /mnt/L202500431/models/exclusion_subspaces/m6_full_step300_top256.safetensors \\
        --k 64 --side both --dtype bf16,fp32 \\
        --output-root /mnt/L202500431/models/subspace_removed --prefix reason_excl_iter19 --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from svd_rank_profile import CANONICAL_RE, shard_files, tensor_locations
from truncate_delta_rank import copy_side_files

SIDES = ("left", "right", "both")
MODES = ("exclude", "keep")
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def kind_of(name: str) -> str:
    return name.split(".")[-2]


def project(delta: torch.Tensor, u: torch.Tensor, v: torch.Tensor, side: str, mode: str = "exclude") -> torch.Tensor:
    keep = mode == "keep"
    if side in ("left", "both"):
        inside = u @ (u.T @ delta)
        delta = inside if keep else delta - inside
    if side in ("right", "both"):
        inside = (delta @ v) @ v.T
        delta = inside if keep else delta - inside
    return delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path, help="base HF checkpoint directory")
    parser.add_argument("--finetuned", required=True, type=Path, help="fine-tuned HF checkpoint directory")
    parser.add_argument("--subspace", required=True, type=Path, help="bases from build_exclusion_subspace.py")
    parser.add_argument("--k", default="64", help="comma-separated numbers of leading directions to remove")
    parser.add_argument("--side", choices=SIDES, default="both")
    parser.add_argument("--mode", choices=MODES, default="exclude",
                        help="exclude: remove the subspace; keep: keep only the component inside it")
    parser.add_argument("--dtype", default="bf16", help=f"comma-separated output dtypes from {tuple(DTYPES)}")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prefix", default=None, help="output dir prefix (default: the fine-tune's dir name)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ks = [int(k) for k in args.k.split(",") if k.strip()]
    dtypes = [d.strip() for d in args.dtype.split(",") if d.strip()]
    if unknown := set(dtypes) - set(DTYPES):
        raise ValueError(f"unknown --dtype {sorted(unknown)}; choose from {tuple(DTYPES)}")
    prefix = args.prefix or args.finetuned.name
    variants = [(k, d) for k in ks for d in dtypes]
    tag = "rm" if args.mode == "exclude" else "keep"
    out_dirs = {v: args.output_root / f"{prefix}_{tag}{args.side}_k{v[0]}_{v[1]}" for v in variants}
    for out_dir in out_dirs.values():
        if out_dir.exists() and any(out_dir.glob("*.safetensors")):
            raise FileExistsError(f"{out_dir} already has weights; remove it or pick another --output-root/--prefix")
        out_dir.mkdir(parents=True, exist_ok=True)

    base_locations = tensor_locations(args.base)
    # per variant, per kind: ||D||^2, ||D - P(D)||^2, ||P(D)||^2, ||R||^2, ||R - P(R)||^2
    stats = {v: defaultdict(lambda: defaultdict(float)) for v in variants}
    n_targets = sum(1 for name in tensor_locations(args.finetuned) if CANONICAL_RE.match(name))
    done = 0
    started = time.perf_counter()

    with ExitStack() as stack:
        bases = stack.enter_context(safe_open(str(args.subspace), framework="pt"))
        meta = bases.metadata() or {}
        k_max = int(meta.get("k_max", "0") or 0)
        if max(ks) > k_max:
            raise ValueError(f"--k {max(ks)} exceeds the subspace file's k_max {k_max}")
        base_handles = {path: stack.enter_context(safe_open(path, framework="pt"))
                        for path in sorted(set(base_locations.values()))}

        for shard in shard_files(args.finetuned):
            outputs: dict = {v: {} for v in variants}
            with safe_open(shard, framework="pt") as handle:
                for name in handle.keys():
                    tuned = handle.get_tensor(name)
                    if not CANONICAL_RE.match(name):
                        for k, d in variants:
                            outputs[(k, d)][name] = tuned.to(DTYPES[d])
                        continue
                    base32 = base_handles[base_locations[name]].get_tensor(name).to(device).float()
                    delta = tuned.to(device).float() - base32
                    u_all = bases.get_tensor(f"{name}.U").to(device)
                    v_all = bases.get_tensor(f"{name}.V").to(device)
                    for k in ks:
                        u, v = u_all[:, :k], v_all[:, :k]
                        kept = project(delta, u, v, args.side, args.mode)
                        for d in dtypes:
                            stored = (base32 + kept).to(DTYPES[d])
                            residual = stored.float() - base32
                            s = stats[(k, d)][kind_of(name)]
                            s["delta"] += float(delta.pow(2).sum())
                            s["inside_before"] += float((delta - kept).pow(2).sum())
                            s["kept"] += float(kept.pow(2).sum())
                            s["residual"] += float(residual.pow(2).sum())
                            s["inside_after"] += float((residual - project(residual, u, v, args.side, args.mode)).pow(2).sum())
                            outputs[(k, d)][name] = stored.cpu()
                    done += 1
                    print(f"  [{done}/{n_targets}] {name}  ({time.perf_counter() - started:.0f}s)", flush=True)
            for v in variants:
                save_file(outputs[v], str(out_dirs[v] / shard.name), metadata={"format": "pt"})
            del outputs

    for (k, d), out_dir in out_dirs.items():
        copy_side_files(args.finetuned, args.base, out_dir)
        if d == "fp32":
            config_path = out_dir / "config.json"
            config = json.loads(config_path.read_text())
            config["torch_dtype"] = "float32"
            config_path.write_text(json.dumps(config, indent=2))
        per_kind = {}
        for kind, s in sorted(stats[(k, d)].items()):
            per_kind[kind] = dict(inside_before=s["inside_before"] / s["delta"], kept=s["kept"] / s["delta"],
                                  inside_after=s["inside_after"] / max(s["residual"], 1e-30))
        totals = defaultdict(float)
        for s in stats[(k, d)].values():
            for key, value in s.items():
                totals[key] += value
        overall = dict(inside_before=totals["inside_before"] / totals["delta"], kept=totals["kept"] / totals["delta"],
                       inside_after=totals["inside_after"] / max(totals["residual"], 1e-30))
        (out_dir / "projection_report.json").write_text(json.dumps(dict(
            base=str(args.base), finetuned=str(args.finetuned), subspace=str(args.subspace),
            subspace_kind=meta.get("kind"), k=k, side=args.side, mode=args.mode, dtype=d,
            all=overall, per_kind=per_kind), indent=2))
        print(f"k={k:<4d} {d}  inside_before={overall['inside_before']:.3e}  kept={overall['kept']:.3f}  "
              f"inside_after={overall['inside_after']:.3e}  -> {out_dir}")
    print(f"done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
