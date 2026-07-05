#!/usr/bin/env python3
"""Concatenate per-domain MOPD training files into one --prompt-data file.

`--prompt-data` only accepts a single path (see orbit/utils/arguments.py),
so training on N domains at once means physically merging N per-domain
JSONL files (each already in the common `{"prompt","label","metadata"}`
schema produced by mopd_prepare_gsm8k.py / mopd_prepare_nemotron_if.py)
into one file. This script does that merge and writes a sibling
`<name>.README.md` manifest recording exactly what went in, plus a
top-level `orbit_mopd/README.md` describing the directory layout (written
once, never overwritten).

Usage:
    python scripts/mopd_mix_train.py \\
        --domain-file /mnt/L202500431/datasets/orbit_mopd/domains/math_gsm8k_train.jsonl \\
        --domain-file /mnt/L202500431/datasets/orbit_mopd/domains/if_nemotron_train.jsonl \\
        --output-dir  /mnt/L202500431/datasets/orbit_mopd/mixes \\
        --name        train_math-if_v1
"""

from __future__ import annotations

import argparse
import os
import random

from mopd_data_common import ensure_top_level_readme, iter_records, write_jsonl, write_mix_manifest


def _domain_and_rm_type(first_row: dict) -> tuple[str, str]:
    metadata = first_row.get("metadata") or {}
    return metadata.get("domains", "unknown"), metadata.get("rm_type", "unknown")


def build_mix(domain_files: list[str], *, limits: list[int | None], shuffle: bool, seed: int) -> tuple[list[dict], list[dict]]:
    all_records: list[dict] = []
    sources: list[dict] = []

    for path, limit in zip(domain_files, limits, strict=True):
        rows = list(iter_records(path))
        rows_total = len(rows)
        if limit is not None:
            rows = rows[:limit]
        if not rows:
            raise ValueError(f"No rows read from {path}")

        domain, rm_type = _domain_and_rm_type(rows[0])
        sources.append(
            {
                "path": path,
                "domains": domain,
                "rm_type": rm_type,
                "rows_used": len(rows),
                "rows_total": rows_total,
            }
        )
        all_records.extend(rows)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_records)

    return all_records, sources


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--domain-file",
        dest="domain_files",
        action="append",
        required=True,
        help="Path to a per-domain JSONL (from mopd_prepare_*.py). Repeat for each domain to mix in.",
    )
    ap.add_argument(
        "--limit",
        dest="limits",
        action="append",
        type=int,
        default=None,
        help="Optional per-domain row cap, aligned by position with --domain-file (repeat once per file, or omit entirely).",
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write <name>.jsonl + <name>.README.md into (usually orbit_mopd/mixes/).")
    ap.add_argument("--name", required=True, help="Mix name, e.g. train_math-if_v1 (see orbit_mopd/README.md for the naming convention).")
    ap.add_argument("--shuffle", action="store_true", default=False, help="Shuffle row order on disk (runtime --rollout-shuffle already reshuffles per epoch, so this is optional).")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed, recorded in the manifest for reproducibility.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing mix with the same name.")
    args = ap.parse_args()

    if args.limits is not None and len(args.limits) != len(args.domain_files):
        ap.error(f"--limit given {len(args.limits)} times but --domain-file given {len(args.domain_files)} times; give one per file or omit --limit entirely.")
    limits = args.limits if args.limits is not None else [None] * len(args.domain_files)

    output_path = os.path.join(args.output_dir, f"{args.name}.jsonl")
    manifest_path = os.path.join(args.output_dir, f"{args.name}.README.md")
    if not args.force and os.path.exists(output_path):
        raise SystemExit(f"refusing to overwrite {output_path} (use --force, or bump the _vN suffix in --name).")

    records, sources = build_mix(args.domain_files, limits=limits, shuffle=args.shuffle, seed=args.seed)
    count = write_jsonl(records, output_path)
    print(f"wrote {output_path}: {count} rows from {len(sources)} domain file(s)")

    write_mix_manifest(
        manifest_path,
        mix_name=args.name,
        output_path=output_path,
        sources=sources,
        total_rows=count,
        shuffled=args.shuffle,
        seed=args.seed if args.shuffle else None,
    )

    # orbit_mopd/README.md lives one level up from mixes/.
    orbit_mopd_dir = os.path.dirname(os.path.normpath(args.output_dir))
    ensure_top_level_readme(orbit_mopd_dir)


if __name__ == "__main__":
    main()
