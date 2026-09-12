"""Verify that N checkpoints are genuinely different models, not an accidental
duplicate (same file copied/symlinked twice, wrong path in a training script,
mask installed but never actually applied, etc).

Does NOT need a GPU or a full model load -- reads tensors straight out of the
safetensors files, one tensor at a time, so it's cheap enough to run before
spending GPU time scoring anything.

Checks, cheapest first (each one that fails is a strong, specific signal --
read the printed reason, don't just look at the final PASS/FAIL):

  1. File hash: are any two checkpoints' safetensors shards byte-identical?
     (instant "these are literally the same file" catch)
  2. Tensor-level identity: even if files differ (different tensor ordering,
     metadata, dtype-on-disk...), are the actual weight values identical?
  3. Per-tensor changed-from-base density: for each checkpoint, what fraction
     of each tensor's elements differ from --base at all -- should match the
     mask's intended density (e.g. ~1.6% for the coordinate-mask runs), not
     0% (nothing trained) or 100% (mask not actually applied / this is a
     full-finetune checkpoint mislabeled as a mask run).
  4. Cross-checkpoint changed-position overlap: for two mask runs meant to
     use *different* coordinate sets, how much do their changed-position
     sets overlap? Should be small and close to what independent random
     subsets of that density would give by chance -- a very high overlap
     (most changed positions the same) means the two masks likely selected
     (nearly) the same coordinates, which would explain a suspiciously high
     functional cosine without it being a real "mask identity doesn't
     matter" finding.
  5. Raw weight-space cosine between the two checkpoints' (checkpoint-base)
     deltas, flattened across every tensor -- a cross-check against the
     CountSketch functional cosine from plot_cosine_heatmap.py. If this is
     also ~0.93+, the high functional agreement is corroborated at the
     weight level too, not just an artifact of the logit-space metric.

Usage:
    python verify_checkpoints_distinct.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --checkpoint oracle=/mnt/L202500431/models/hyy_models/m6_coord_mask_1p6pct_oracle_step300 \\
        --checkpoint random=/mnt/L202500431/models/hyy_models/m6_coord_mask_1p6pct_random_step300

Optionally also cross-check the constraint mask files themselves (before any
training), if you have them:
    python verify_checkpoints_distinct.py --base ... \\
        --checkpoint oracle=... --checkpoint random=... \\
        --mask oracle=.../mask_top20_oracle.safetensors \\
        --mask random=.../mask_random_seed20260819.safetensors
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """{tensor_name: shard_file} covering every tensor in the checkpoint."""
    locations: dict[str, Path] = {}
    for shard in shard_files(checkpoint_dir):
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                locations[name] = shard
    return locations


class LazyCheckpoint:
    """Keeps at most one shard's safe_open handle open per checkpoint;
    reads tensors on demand instead of loading everything into RAM at once."""

    def __init__(self, checkpoint_dir: Path):
        self.dir = checkpoint_dir
        self.locations = tensor_locations(checkpoint_dir)
        self._open_path: Path | None = None
        self._open_handle = None

    def keys(self):
        return self.locations.keys()

    def get(self, name: str) -> torch.Tensor:
        shard = self.locations[name]
        if shard != self._open_path:
            if self._open_handle is not None:
                self._open_handle.__exit__(None, None, None)
            self._open_handle = safe_open(shard, framework="pt")
            self._open_handle.__enter__()
            self._open_path = shard
        return self._open_handle.get_tensor(name)

    def close(self):
        if self._open_handle is not None:
            self._open_handle.__exit__(None, None, None)


def check_file_hashes(checkpoints: dict[str, Path]) -> None:
    print("\n--- 1. file hash ---")
    hashes: dict[str, list[str]] = {}
    for name, ckpt_dir in checkpoints.items():
        for shard in shard_files(ckpt_dir):
            digest = sha256_file(shard)
            hashes.setdefault(digest, []).append(f"{name}:{shard.name}")
    duplicates = {digest: owners for digest, owners in hashes.items() if len(owners) > 1}
    if duplicates:
        print("FAIL: byte-identical shard files across checkpoints:")
        for digest, owners in duplicates.items():
            print(f"  {digest[:12]}...: {owners}")
    else:
        print("PASS: no two checkpoints share a byte-identical shard file")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH",
                         help="repeatable, e.g. --checkpoint oracle=/path --checkpoint random=/path")
    parser.add_argument("--mask", action="append", default=[], metavar="NAME=PATH",
                         help="optional: the boolean constraint safetensors file used to *train* "
                              "each checkpoint (not the trained checkpoint itself), same NAMEs as --checkpoint")
    parser.add_argument("--atol", type=float, default=0.0, help="tolerance for 'changed from base' (bf16 exact-equality by default)")
    args = parser.parse_args()

    checkpoints = {}
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        checkpoints[name] = Path(path)
    if len(checkpoints) < 2:
        raise ValueError("Need at least 2 --checkpoint entries to compare")

    check_file_hashes(checkpoints)

    print("\n--- 2-3. per-tensor changed-from-base ---")
    base = LazyCheckpoint(args.base)
    loaded = {name: LazyCheckpoint(path) for name, path in checkpoints.items()}
    shared_keys = set(base.keys())
    for ck in loaded.values():
        shared_keys &= set(ck.keys())
    shared_keys = sorted(shared_keys)
    print(f"comparing {len(shared_keys)} tensors present in base and all {len(checkpoints)} checkpoints")

    changed_masks: dict[str, dict[str, torch.Tensor]] = {name: {} for name in checkpoints}
    total_params = 0
    changed_counts = dict.fromkeys(checkpoints, 0)
    identical_to_base_counts = dict.fromkeys(checkpoints, 0)
    delta_flat_sq_norm = dict.fromkeys(checkpoints, 0.0)

    for key in shared_keys:
        base_tensor = base.get(key).float()
        total_params += base_tensor.numel()
        for name, ck in loaded.items():
            other = ck.get(key).float()
            if other.shape != base_tensor.shape:
                raise ValueError(f"{name}:{key} shape {other.shape} != base shape {base_tensor.shape}")
            delta = other - base_tensor
            changed = delta.abs() > args.atol
            changed_masks[name][key] = changed
            changed_counts[name] += int(changed.sum())
            delta_flat_sq_norm[name] += float(delta.double().square().sum())
        del base_tensor

    print(f"total tensor elements compared: {total_params:,}")
    for name in checkpoints:
        density = changed_counts[name] / total_params
        print(f"  {name}: changed_from_base={changed_counts[name]:,} ({density:.4%} of params), "
              f"||delta||_2={delta_flat_sq_norm[name]**0.5:.4f}")
    print("(compare this density to the mask's intended density, e.g. ~1.6% for the coordinate-mask runs; "
          "0% means nothing trained, ~100% means the mask wasn't actually applied)")

    print("\n--- 4. tensor-level identity + changed-position overlap between checkpoints ---")
    for a, b in itertools.combinations(checkpoints, 2):
        identical_tensors = 0
        for key in shared_keys:
            ta, tb = loaded[a].get(key), loaded[b].get(key)
            if torch.equal(ta, tb):
                identical_tensors += 1
        if identical_tensors == len(shared_keys):
            print(f"FAIL: {a} and {b} have IDENTICAL weights in every one of {len(shared_keys)} tensors "
                  f"-- these are the same trained model.")
        elif identical_tensors > 0:
            print(f"NOTE: {a} and {b} are identical in {identical_tensors}/{len(shared_keys)} tensors "
                  f"(e.g. frozen embedding/norm layers outside the mask -- fine if those layers are meant "
                  f"to be untouched by both).")
        else:
            print(f"PASS: {a} and {b} differ in every one of {len(shared_keys)} tensors from base.")

        overlap_changed = 0
        union_changed = 0
        for key in shared_keys:
            ma, mb = changed_masks[a][key], changed_masks[b][key]
            overlap_changed += int((ma & mb).sum())
            union_changed += int((ma | mb).sum())
        overlap_frac = overlap_changed / max(union_changed, 1)
        density_a = changed_counts[a] / total_params
        density_b = changed_counts[b] / total_params
        expected_if_independent = density_a * density_b / max(density_a + density_b - density_a * density_b, 1e-30)
        print(f"  {a} vs {b}: changed-position overlap (Jaccard) = {overlap_frac:.4%} "
              f"(expected ~{expected_if_independent:.4%} if the two masks were independent random subsets)")

    print("\n--- 5. raw weight-space cosine of (checkpoint - base) deltas ---")
    for a, b in itertools.combinations(checkpoints, 2):
        dot, norm_a, norm_b = 0.0, 0.0, 0.0
        for key in shared_keys:
            da = (loaded[a].get(key).float() - base.get(key).float()).double()
            db = (loaded[b].get(key).float() - base.get(key).float()).double()
            dot += float((da * db).sum())
            norm_a += float(da.square().sum())
            norm_b += float(db.square().sum())
        cosine = dot / max((norm_a ** 0.5) * (norm_b ** 0.5), 1e-30)
        print(f"  cos(delta_{a}, delta_{b}) = {cosine:.4f}  (cross-check against the CountSketch "
              f"functional cosine -- same ballpark is corroborating, very different is worth a closer look)")

    for ck in loaded.values():
        ck.close()
    base.close()

    if args.mask:
        print("\n--- optional: constraint mask files (pre-training, not the trained checkpoints) ---")
        masks = {}
        for item in args.mask:
            name, _, path = item.partition("=")
            masks[name] = Path(path)
        loaded_masks = {}
        for name, path in masks.items():
            with safe_open(path, framework="pt") as handle:
                loaded_masks[name] = {key: handle.get_tensor(key).bool() for key in handle.keys()}
        for a, b in itertools.combinations(masks, 2):
            keys_a, keys_b = set(loaded_masks[a]), set(loaded_masks[b])
            if keys_a != keys_b:
                print(f"FAIL: mask {a} and {b} cover different tensor sets: "
                      f"only in {a}: {keys_a - keys_b}; only in {b}: {keys_b - keys_a}")
                continue
            overlap = union = count_a = count_b = 0
            for key in sorted(keys_a):
                ma, mb = loaded_masks[a][key], loaded_masks[b][key]
                if ma.shape != mb.shape:
                    raise ValueError(f"mask {key} shape mismatch: {a}={ma.shape} {b}={mb.shape}")
                overlap += int((ma & mb).sum())
                union += int((ma | mb).sum())
                count_a += int(ma.sum())
                count_b += int(mb.sum())
            print(f"  mask {a} selects {count_a:,} coordinates, mask {b} selects {count_b:,}; "
                  f"overlap (Jaccard) = {overlap / max(union, 1):.4%}")
            if overlap == count_a == count_b:
                print(f"  FAIL: masks {a} and {b} select the EXACT SAME coordinates -- not actually "
                      f"independent oracle/random masks.")


if __name__ == "__main__":
    main()
