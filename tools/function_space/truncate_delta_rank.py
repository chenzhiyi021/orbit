"""Rank-k truncation of a fine-tune's weight deltas, saved as ready-to-eval HF checkpoints.

For every canonical q/k/v/o/gate/up/down weight,

    W_rank_k = W_base + U_k S_k V_k^T,   U S V^T = SVD(W_finetuned - W_base)

i.e. the fine-tune's update is kept only along its top-k singular directions. Every other
tensor (RMSNorm scales, embeddings) is copied from the fine-tune by default, so the only
difference from the fine-tuned model is the truncation itself; pass --other-params base to
revert those to the base model as well.

Each tensor is decomposed once and every requested k is written from the same factors, so
one run produces one HF directory per k:

    <output-root>/<prefix>_rank<k>/   (config, tokenizer and index copied from the fine-tune)

k larger than a tensor's smaller dimension keeps that tensor's full delta; "full" keeps
every delta (a sanity check -- it should score like the fine-tuned model, up to bf16
rounding). Each output directory also gets rank_truncation_report.json with the fraction of
||Delta W||_F^2 kept, per tensor kind.

    python truncate_delta_rank.py \\
        --base /mnt/L202500431/models/qwen3-1.7b \\
        --finetuned /mnt/L202500431/models/zh_models/m6_st_full_non_thinking_hf_step300 \\
        --k 1,4,16,64,256,full --output-root /mnt/L202500431/models/rank_truncated --device cuda
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

FULL = "full"

# Files that make up the tokenizer / chat template. Taken from the base model rather than the
# fine-tune: the tokenizer never changes during fine-tuning, and a fine-tune exported by a
# different transformers version can carry a tokenizer_config.json this environment cannot load
# (e.g. TRL exports with a list-valued "extra_special_tokens").
TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "chat_template.json"}


def copy_side_files(source_dir: Path, base_dir: Path, out_dir: Path) -> None:
    """Copy the non-weight files an HF directory needs: config / generation settings from the
    fine-tune, tokenizer and chat-template files from the base model."""
    for item in source_dir.iterdir():
        if item.is_file() and item.suffix != ".safetensors" and item.name not in TOKENIZER_FILES:
            shutil.copy2(item, out_dir / item.name)
    for name in TOKENIZER_FILES:
        if (base_dir / name).is_file():
            shutil.copy2(base_dir / name, out_dir / name)


def parse_ks(spec: str) -> list[int | str]:
    ks: list[int | str] = []
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        ks.append(FULL if item == FULL else int(item))
    if any(isinstance(k, int) and k < 0 for k in ks):
        raise ValueError("--k values must be >= 0 or 'full'")
    return list(dict.fromkeys(ks))


def kind_of(name: str) -> str:
    return name.split(".")[-2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path, help="base HF checkpoint directory")
    parser.add_argument("--finetuned", required=True, type=Path, help="fine-tuned HF checkpoint directory")
    parser.add_argument("--k", required=True, help="comma-separated ranks, e.g. '1,4,16,64,256,full'")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prefix", default=None, help="output dir prefix (default: the fine-tune's dir name)")
    parser.add_argument("--other-params", choices=("finetuned", "base"), default="finetuned",
                        help="source for every non-q/k/v/o/gate/up/down tensor (norms, embeddings)")
    parser.add_argument("--device", default="cpu", help="where the SVDs run, e.g. 'cpu' or 'cuda'")
    args = parser.parse_args()

    ks = parse_ks(args.k)
    device = torch.device(args.device)
    prefix = args.prefix or args.finetuned.name
    out_dirs = {k: args.output_root / f"{prefix}_rank{k}" for k in ks}
    for out_dir in out_dirs.values():
        if out_dir.exists() and any(out_dir.glob("*.safetensors")):
            raise FileExistsError(f"{out_dir} already has weights; remove it or pick another --output-root/--prefix")
        out_dir.mkdir(parents=True, exist_ok=True)

    base_locations = tensor_locations(args.base)
    # kept energy per k, per kind: sums of ||Delta_k||^2 and ||Delta||^2
    kept = {k: defaultdict(float) for k in ks}
    total = defaultdict(float)
    other_delta_sq = 0.0
    n_targets = sum(1 for name in tensor_locations(args.finetuned) if CANONICAL_RE.match(name))
    done = 0
    started = time.perf_counter()

    # Base shards opened once for the whole run, not once per tensor.
    base_handles: dict[Path, object] = {}
    with ExitStack() as stack:
        for path in sorted(set(base_locations.values())):
            base_handles[path] = stack.enter_context(safe_open(path, framework="pt"))

        for shard in shard_files(args.finetuned):
            print(f"--- {shard.name}", flush=True)
            # One tensor at a time, everything on `device`: delta, SVD, and every k's reconstruction.
            # Only the final (low-precision) outputs come back to host memory.
            outputs: dict = {k: {} for k in ks}
            with safe_open(shard, framework="pt") as handle:
                for name in handle.keys():
                    tuned = handle.get_tensor(name)
                    base = base_handles[base_locations[name]].get_tensor(name)
                    if not CANONICAL_RE.match(name):
                        other_delta_sq += float((tuned.float() - base.float()).pow(2).sum())
                        kept_tensor = tuned if args.other_params == "finetuned" else base
                        for k in ks:
                            outputs[k][name] = kept_tensor
                        continue
                    base_dev = base.to(device).float()
                    delta = tuned.to(device).float() - base_dev
                    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                    energy = s.pow(2)
                    total[kind_of(name)] += float(energy.sum())
                    for k in ks:
                        kk = s.numel() if k == FULL else min(k, s.numel())
                        kept[k][kind_of(name)] += float(energy[:kk].sum())
                        if kk >= s.numel():
                            approx = delta
                        elif kk == 0:
                            approx = torch.zeros_like(delta)
                        else:
                            approx = (u[:, :kk] * s[:kk]) @ vh[:kk]
                        outputs[k][name] = (base_dev + approx).to(tuned.dtype).cpu()
                    done += 1
                    print(f"  [{done}/{n_targets}] {name}  shape={tuple(delta.shape)}  "
                          f"({time.perf_counter() - started:.0f}s)", flush=True)
                    del base_dev, delta, u, s, vh
            for k in ks:
                save_file(outputs[k], str(out_dirs[k] / shard.name), metadata={"format": "pt"})
            del outputs
            print(f"  wrote {shard.name} for k={ks} ({time.perf_counter() - started:.0f}s elapsed)", flush=True)

    for out_dir in out_dirs.values():
        copy_side_files(args.finetuned, args.base, out_dir)

    all_kinds = sorted(total)
    grand_total = sum(total.values())
    print(f"\nfraction of ||Delta W||_F^2 kept (q/k/v/o/gate/up/down only; "
          f"other tensors' delta energy = {other_delta_sq:.4g}, taken from --other-params={args.other_params})")
    print(f"{'k':>6s} {'all':>7s} " + " ".join(f"{kind:>9s}" for kind in all_kinds))
    for k in ks:
        overall = sum(kept[k].values()) / grand_total if grand_total else float("nan")
        per_kind = {kind: kept[k][kind] / total[kind] for kind in all_kinds}
        print(f"{str(k):>6s} {overall:7.3f} " + " ".join(f"{per_kind[kind]:9.3f}" for kind in all_kinds))
        report = dict(base=str(args.base), finetuned=str(args.finetuned), k=k, other_params=args.other_params,
                      kept_energy_fraction_all=overall, kept_energy_fraction_per_kind=per_kind,
                      other_tensors_delta_sq=other_delta_sq)
        (out_dirs[k] / "rank_truncation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\ndone in {time.perf_counter() - started:.0f}s; outputs:")
    for out_dir in out_dirs.values():
        print(f"  {out_dir}")


if __name__ == "__main__":
    main()
