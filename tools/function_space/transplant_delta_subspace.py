"""Swap or shrink a fine-tune's input-side delta component, post hoc, for function ablations.

Companion of remove_delta_subspace.py (which only deletes a component). With B the per-tensor
input-side basis from a --subspace file ("<name>.V", n_in x k, e.g. task_subspace_split.py
--export-dir output) and P = B B^T, for every q/k/v/o/gate/up/down weight it writes

    transplant   W = W_base + D_body (I - P) + D_donor P      the body keeps everything outside
                                                               span(B) and takes the donor's
                                                               component inside it
    scale-dist   W = W_base + c D_body,  c = 1 - sqrt(f)       moves W_body toward W_base by the SAME
                                                               distance as deleting D_body P would
    scale-norm   W = W_base + c D_body,  c = sqrt(1 - f)       keeps the SAME ||delta|| as deleting
                                                               D_body P would
    alpha<a>     W = W_base + a D_body                         fixed interpolation (WiSE-FT style)

with D = W_finetuned - W_base and f = ||D_body P||^2 / ||D_body||^2 per tensor. The scale variants
are the energy-matched controls for "deleting the component helps": they remove as much as the
projection does, but uniformly in every direction. --kinds restricts all variants to some
tensor kinds (the rest is copied from the body). Non-q/k/v/o/gate/up/down tensors always come from
the body; config/tokenizer files from the base (as remove_delta_subspace.py).

Output dirs: <output-root>/<prefix>_<variant>[_only-<kinds>]_<dtype>, each with a
transplant_report.json of per-kind energies.

    python transplant_delta_subspace.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --body /mnt/.../rl300_hf --donor /mnt/.../opd300_hf \\
        --subspace /mnt/L202500431/orbit/output/task_split_export/shared.safetensors \\
        --variants transplant,scale-dist,scale-norm --dtype bf16 \\
        --output-root /mnt/L202500431/models/task_split_transplant --prefix rl300_shared --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from svd_rank_profile import CANONICAL_RE, shard_files, tensor_locations
from truncate_delta_rank import copy_side_files

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
FIXED_VARIANTS = ("transplant", "scale-dist", "scale-norm")


def kind_of(name: str) -> str:
    return name.split(".")[-2]


def parse_variants(spec: str) -> list[str]:
    variants = [v.strip() for v in spec.split(",") if v.strip()]
    for v in variants:
        if v not in FIXED_VARIANTS and not (v.startswith("alpha") and _is_float(v[5:])):
            raise ValueError(f"unknown variant {v!r}: use {FIXED_VARIANTS} or alpha<float>, e.g. alpha0.5")
    return list(dict.fromkeys(variants))


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def build(variant: str, d_body: torch.Tensor, d_donor: torch.Tensor | None, basis: torch.Tensor) -> torch.Tensor:
    inside_body = (d_body @ basis) @ basis.T
    if variant == "transplant":
        return d_body - inside_body + (d_donor @ basis) @ basis.T
    f = float(inside_body.pow(2).sum() / d_body.pow(2).sum().clamp(min=1e-30))
    if variant == "scale-dist":
        return (1.0 - math.sqrt(f)) * d_body
    if variant == "scale-norm":
        return math.sqrt(max(0.0, 1.0 - f)) * d_body
    return float(variant[5:]) * d_body  # alpha<a>


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--body", required=True, type=Path, help="fine-tune whose delta is kept / scaled")
    parser.add_argument("--donor", type=Path, default=None, help="fine-tune giving the inside component (transplant)")
    parser.add_argument("--subspace", required=True, type=Path,
                        help="input-side bases '<name>.V' (task_subspace_split.py --export-dir / build_exclusion_subspace.py)")
    parser.add_argument("--k", type=int, default=None, help="leading basis columns to use (default: the file's k_max)")
    parser.add_argument("--variants", default="transplant,scale-dist,scale-norm",
                        help=f"comma-separated from {FIXED_VARIANTS} and alpha<float> (e.g. alpha0.5)")
    parser.add_argument("--kinds", default=None, help="comma-separated tensor kinds to modify (default: all)")
    parser.add_argument("--dtype", default="bf16", choices=tuple(DTYPES))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device, out_dtype = torch.device(args.device), DTYPES[args.dtype]
    variants = parse_variants(args.variants)
    if "transplant" in variants and args.donor is None:
        raise ValueError("variant 'transplant' needs --donor")
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()} if args.kinds else None
    kinds_tag = "" if kinds is None else "_only-" + "-".join(sorted(k.removesuffix("_proj") for k in kinds))
    out_dirs = {v: args.output_root / f"{args.prefix}_{v}{kinds_tag}_{args.dtype}" for v in variants}
    for out_dir in out_dirs.values():
        if out_dir.exists() and any(out_dir.glob("*.safetensors")):
            raise FileExistsError(f"{out_dir} already has weights; remove it or pick another --output-root/--prefix")
        out_dir.mkdir(parents=True, exist_ok=True)

    base_locations = tensor_locations(args.base)
    donor_locations = tensor_locations(args.donor) if args.donor is not None else {}
    targets = [n for n in tensor_locations(args.body)
               if CANONICAL_RE.match(n) and (kinds is None or kind_of(n) in kinds)]
    # per variant, per kind: ||D_body||^2, ||D_body P||^2, ||D_donor P||^2, ||D_new||^2, ||D_new - D_body||^2
    stats = {v: defaultdict(lambda: defaultdict(float)) for v in variants}
    done, started = 0, time.perf_counter()

    with ExitStack() as stack:
        bases = stack.enter_context(safe_open(str(args.subspace), framework="pt"))
        meta = bases.metadata() or {}
        k_max = int(meta.get("k_max", "0") or 0)
        k = args.k if args.k is not None else k_max
        if k > k_max:
            raise ValueError(f"--k {k} exceeds the subspace file's k_max {k_max}")
        open_all = lambda locs: {p: stack.enter_context(safe_open(str(p), framework="pt"))  # noqa: E731
                                 for p in sorted(set(locs.values()))}
        base_handles, donor_handles = open_all(base_locations), open_all(donor_locations)

        for shard in shard_files(args.body):
            outputs: dict = {v: {} for v in variants}
            with safe_open(shard, framework="pt") as handle:
                for name in handle.keys():
                    tuned = handle.get_tensor(name)
                    if name not in targets:
                        for v in variants:
                            outputs[v][name] = tuned.to(out_dtype)
                        continue
                    base32 = base_handles[base_locations[name]].get_tensor(name).to(device).float()
                    d_body = tuned.to(device).float() - base32
                    d_donor = None
                    if args.donor is not None:
                        d_donor = donor_handles[donor_locations[name]].get_tensor(name).to(device).float() - base32
                    basis = bases.get_tensor(f"{name}.V").to(device).float()[:, :k]
                    for v in variants:
                        d_new = build(v, d_body, d_donor, basis)
                        stored = (base32 + d_new).to(out_dtype)
                        s = stats[v][kind_of(name)]
                        s["body"] += float(d_body.pow(2).sum())
                        s["body_inside"] += float(((d_body @ basis) @ basis.T).pow(2).sum())
                        if d_donor is not None:
                            s["donor_inside"] += float(((d_donor @ basis) @ basis.T).pow(2).sum())
                        s["new"] += float(d_new.pow(2).sum())
                        s["moved"] += float((d_new - d_body).pow(2).sum())
                        outputs[v][name] = stored.cpu()
                    done += 1
                    print(f"  [{done}/{len(targets)}] {name}  ({time.perf_counter() - started:.0f}s)", flush=True)
            for v in variants:
                save_file(outputs[v], str(out_dirs[v] / shard.name), metadata={"format": "pt"})
            del outputs

    for v, out_dir in out_dirs.items():
        copy_side_files(args.body, args.base, out_dir)
        if args.dtype == "fp32":
            config_path = out_dir / "config.json"
            config = json.loads(config_path.read_text())
            config["torch_dtype"] = "float32"
            config_path.write_text(json.dumps(config, indent=2))

        def ratios(s: dict) -> dict:
            body = max(s["body"], 1e-30)
            return dict(body_inside=s["body_inside"] / body, donor_inside_over_body=s["donor_inside"] / body,
                        new_over_body=s["new"] / body, moved_over_body=s["moved"] / body)

        totals = defaultdict(float)
        for s in stats[v].values():
            for key, value in s.items():
                totals[key] += value
        overall = ratios(totals)
        (out_dir / "transplant_report.json").write_text(json.dumps(dict(
            base=str(args.base), body=str(args.body), donor=None if args.donor is None else str(args.donor),
            subspace=str(args.subspace), subspace_kind=meta.get("kind"), k=k, variant=v, dtype=args.dtype,
            kinds=None if kinds is None else sorted(kinds), all=overall,
            per_kind={kind: ratios(s) for kind, s in sorted(stats[v].items())}), indent=2))
        print(f"{v:<12} body_inside={overall['body_inside']:.3f}  donor_inside/body={overall['donor_inside_over_body']:.3f}  "
              f"||new||^2/||body||^2={overall['new_over_body']:.3f}  ||new-body||^2/||body||^2={overall['moved_over_body']:.3f}"
              f"  -> {out_dir}")
    print(f"done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
