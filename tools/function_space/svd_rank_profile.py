"""SVD rank profile of (checkpoint - base) weight deltas, per tensor.

Answers a narrower question than plain "changed_from_base density"
(verify_checkpoints_distinct.py): not just how many entries changed, but
*how many independent directions* the change actually uses. A sparse
coordinate mask can be ~98% zero and still be full rank (sparse random
matrices are generically full rank once density is above the row-connectivity
threshold); a LoRA-style update is exactly rank-capped by construction. This
tells the two apart, which "changed_from_base density" alone cannot.

CPU-only, no model reload -- reads tensors straight out of safetensors, one
at a time.

Usage:
    python svd_rank_profile.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --checkpoint oracle=/mnt/.../m6_coord_mask_1p6pct_oracle_step300 \\
        --checkpoint random=/mnt/.../m6_coord_mask_1p6pct_random_step300 \\
        --tensors "model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.13.self_attn.q_proj.weight,model.layers.13.mlp.down_proj.weight,model.layers.27.self_attn.q_proj.weight,model.layers.27.mlp.down_proj.weight"

Omit --tensors to auto-pick a spread of representative target-module tensors
(one q_proj and one down_proj at the first, middle, and last transformer
layer) instead of scanning all ~196 target tensors -- add --all-target-tensors
to scan every canonical q/k/v/o/gate/up/down tensor instead (slower, more
complete).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open


CANONICAL_RE = re.compile(
    r"^model\.layers\.(\d+)\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$"
)


def shard_files(checkpoint_dir: Path) -> list[Path]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return sorted({checkpoint_dir / shard for shard in index["weight_map"].values()})
    single = checkpoint_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"No model.safetensors or model.safetensors.index.json in {checkpoint_dir}")


def tensor_locations(checkpoint_dir: Path) -> dict[str, Path]:
    locations: dict[str, Path] = {}
    for shard in shard_files(checkpoint_dir):
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                locations[name] = shard
    return locations


def load_tensor(checkpoint_dir: Path, locations: dict[str, Path], name: str) -> torch.Tensor:
    with safe_open(locations[name], framework="pt") as handle:
        return handle.get_tensor(name)


def pick_representative_tensors(locations: dict[str, Path]) -> list[str]:
    layers = sorted({int(m.group(1)) for name in locations if (m := CANONICAL_RE.match(name))})
    if not layers:
        raise RuntimeError("No canonical q/k/v/o/gate/up/down tensors found -- pass --tensors explicitly")
    picks = sorted({layers[0], layers[len(layers) // 2], layers[-1]})
    names = []
    for layer in picks:
        for suffix in ("self_attn.q_proj.weight", "mlp.down_proj.weight"):
            name = f"model.layers.{layer}.{suffix}"
            if name in locations:
                names.append(name)
    return names


def rank_profile(delta: torch.Tensor, energy_thresholds=(0.90, 0.95, 0.99)) -> dict:
    matrix = delta.float()
    singular = torch.linalg.svdvals(matrix)
    total_energy = float((singular ** 2).sum())
    cumulative = torch.cumsum(singular ** 2, dim=0) / max(total_energy, 1e-30)
    hard_rank = int((singular > singular.max() * 1e-6).sum()) if singular.numel() else 0
    stable_rank = total_energy / max(float(singular[0] ** 2), 1e-30) if singular.numel() else 0.0
    ranks_at = {}
    for threshold in energy_thresholds:
        idx = torch.searchsorted(cumulative, threshold).item()
        ranks_at[threshold] = min(idx + 1, singular.numel())
    return dict(
        shape=tuple(matrix.shape), min_dim=min(matrix.shape),
        frobenius_norm=float(singular.square().sum() ** 0.5),
        hard_rank=hard_rank, stable_rank=stable_rank,
        **{f"rank_at_{int(t * 100)}pct_energy": v for t, v in ranks_at.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--tensors", default=None, help="comma-separated tensor names; default: auto-pick a representative spread")
    parser.add_argument("--all-target-tensors", action="store_true",
                         help="scan every canonical q/k/v/o/gate/up/down tensor instead of a representative sample (slower)")
    args = parser.parse_args()

    checkpoints = {}
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        checkpoints[name] = Path(path)

    base_locations = tensor_locations(args.base)
    if args.tensors:
        tensor_names = [t.strip() for t in args.tensors.split(",") if t.strip()]
    elif args.all_target_tensors:
        tensor_names = sorted(name for name in base_locations if CANONICAL_RE.match(name))
    else:
        tensor_names = pick_representative_tensors(base_locations)
    print(f"profiling {len(tensor_names)} tensors: {tensor_names}\n", flush=True)

    ckpt_locations = {name: tensor_locations(path) for name, path in checkpoints.items()}

    for tensor_name in tensor_names:
        base_tensor = load_tensor(args.base, base_locations, tensor_name).float()
        print(f"=== {tensor_name}  (shape {tuple(base_tensor.shape)}) ===")
        for name, path in checkpoints.items():
            other = load_tensor(path, ckpt_locations[name], tensor_name).float()
            delta = other - base_tensor
            profile = rank_profile(delta)
            frac_90 = profile["rank_at_90pct_energy"] / profile["min_dim"]
            frac_95 = profile["rank_at_95pct_energy"] / profile["min_dim"]
            frac_99 = profile["rank_at_99pct_energy"] / profile["min_dim"]
            print(f"  {name:<10}  hard_rank={profile['hard_rank']:>5}/{profile['min_dim']}  "
                  f"stable_rank={profile['stable_rank']:>7.2f}  "
                  f"rank@90/95/99%={profile['rank_at_90pct_energy']}({frac_90:.2%})"
                  f"/{profile['rank_at_95pct_energy']}({frac_95:.2%})"
                  f"/{profile['rank_at_99pct_energy']}({frac_99:.2%})  "
                  f"||delta||_F={profile['frobenius_norm']:.4f}")
        print()


if __name__ == "__main__":
    main()
