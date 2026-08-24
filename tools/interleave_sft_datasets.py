"""Interleave two tools/generate_teacher_sft_data.py outputs by prompt position.

Both inputs are expected to come from the same underlying --dataset (so their
``metadata.index`` values refer to the same prompt sequence). For prompt
position N (1-based, i.e. index+1):
    - N odd  (1st, 3rd, 5th, ...) -> taken from --odd-input
    - N even (2nd, 4th, 6th, ...) -> taken from --even-input

If the file assigned to a given position doesn't have that index (e.g. a run
that's still in progress, or a prompt skipped by --max-prompt-length), the
other file is used instead as a fallback so the merged dataset doesn't just
silently drop rows. Indices missing from both inputs are skipped and counted.

    python tools/interleave_sft_datasets.py \
        --odd-input data/sft/ministral_8b/train.jsonl \
        --even-input data/sft/openreasoning_mixed_100k/train.jsonl \
        --output data/sft/interleaved_ministral_qwen3/train.jsonl

Each output record keeps its original schema and gains one field,
``metadata.interleave_source``, set to the basename of the input directory it
was actually pulled from (useful once the two are merged and you want to
check the mix later, e.g. for computing per-teacher token/length stats).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger("interleave_sft_datasets")


def _load_indexed(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    duplicates = 0
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index = record.get("metadata", {}).get("index")
            if not isinstance(index, int):
                continue
            if index in records:
                duplicates += 1
            records[index] = record
    if duplicates:
        logger.warning("%s: %d duplicate indices, last occurrence wins", path, duplicates)
    logger.info("%s: loaded %d records (indices %d..%d)", path, len(records), min(records), max(records))
    return records


def _source_label(path: Path) -> str:
    return path.parent.name


def build_interleaved(
    odd_records: dict[int, dict],
    even_records: dict[int, dict],
    odd_label: str,
    even_label: str,
) -> tuple[list[dict], dict[str, int]]:
    all_indices = sorted(set(odd_records) | set(even_records))
    stats = {
        "total": 0,
        "from_odd_intended": 0,
        "from_even_intended": 0,
        "fallback_used": 0,
        "missing_both": 0,
        "prompt_hash_mismatch": 0,
    }
    output: list[dict] = []
    for index in all_indices:
        position = index + 1  # 1-based "第几条"
        want_odd = position % 2 == 1
        primary, primary_label, fallback, fallback_label = (
            (odd_records, odd_label, even_records, even_label)
            if want_odd
            else (even_records, even_label, odd_records, odd_label)
        )

        if index in primary:
            record, source_label = primary[index], primary_label
            stats["from_odd_intended" if want_odd else "from_even_intended"] += 1
        elif index in fallback:
            record, source_label = fallback[index], fallback_label
            stats["fallback_used"] += 1
        else:
            stats["missing_both"] += 1
            continue

        if index in odd_records and index in even_records:
            odd_hash = odd_records[index].get("metadata", {}).get("prompt_sha256")
            even_hash = even_records[index].get("metadata", {}).get("prompt_sha256")
            if odd_hash and even_hash and odd_hash != even_hash:
                stats["prompt_hash_mismatch"] += 1
                logger.warning("index %d: prompt_sha256 differs between inputs -- sequences may not be aligned", index)

        record = dict(record)
        record["metadata"] = {**record.get("metadata", {}), "interleave_source": source_label}
        output.append(record)
        stats["total"] += 1

    return output, stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--odd-input",
        type=Path,
        default=Path("data/sft/ministral_8b/train.jsonl"),
        help="Source for odd prompt positions (1st, 3rd, 5th, ...).",
    )
    parser.add_argument(
        "--even-input",
        type=Path,
        default=Path("data/sft/openreasoning_mixed_100k/train.jsonl"),
        help="Source for even prompt positions (2nd, 4th, 6th, ...).",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    odd_records = _load_indexed(args.odd_input)
    even_records = _load_indexed(args.even_input)
    odd_label = _source_label(args.odd_input)
    even_label = _source_label(args.even_input)

    output, stats = build_interleaved(odd_records, even_records, odd_label, even_label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for record in output:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "done: total=%d from_odd(%s)=%d from_even(%s)=%d fallback_used=%d missing_both=%d prompt_hash_mismatch=%d output=%s",
        stats["total"],
        odd_label,
        stats["from_odd_intended"],
        even_label,
        stats["from_even_intended"],
        stats["fallback_used"],
        stats["missing_both"],
        stats["prompt_hash_mismatch"],
        args.output,
    )


if __name__ == "__main__":
    main()
