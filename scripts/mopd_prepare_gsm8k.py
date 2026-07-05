#!/usr/bin/env python3
"""Convert a GSM8K parquet split into the common MOPD per-domain row schema.

Reads the raw `question`/`answer` columns and writes JSONL rows of the form
`{"prompt": ..., "label": ..., "metadata": {"domains": ..., "rm_type": ...}}`
so this file can later be concatenated with other domains' files (e.g. the
Nemotron instruction-following domain from `mopd_prepare_nemotron_if.py`) by
`mopd_mix_train.py`.

Usage:
    python scripts/mopd_prepare_gsm8k.py \\
        --input  /mnt/L202500431/datasets/gsm8k/main/train-00000-of-00001.parquet \\
        --output /mnt/L202500431/datasets/orbit_mopd/domains/math_gsm8k_train.jsonl
"""

from __future__ import annotations

import argparse

from mopd_data_common import iter_records, make_record, write_jsonl


def convert(input_path: str, output_path: str, *, input_key: str, label_key: str, domain: str, rm_type: str, limit: int | None) -> int:
    records = []
    for row in iter_records(input_path):
        prompt = row[input_key]
        label = row.get(label_key)
        metadata = {"domains": domain, "rm_type": rm_type}
        records.append(make_record(prompt, label, metadata))
        if limit is not None and len(records) >= limit:
            break
    return write_jsonl(records, output_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Source GSM8K parquet (question/answer columns).")
    ap.add_argument("--output", required=True, help="Destination JSONL path (usually under orbit_mopd/domains/).")
    ap.add_argument("--input-key", default="question", help="Source column holding the prompt text (default: question).")
    ap.add_argument("--label-key", default="answer", help="Source column holding the label (default: answer).")
    ap.add_argument("--domain", default="math", help="Value to write into metadata.domains (default: math).")
    ap.add_argument("--rm-type", default="math", help="Value to write into metadata.rm_type (default: math).")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows (for smoke tests).")
    args = ap.parse_args()

    count = convert(
        args.input,
        args.output,
        input_key=args.input_key,
        label_key=args.label_key,
        domain=args.domain,
        rm_type=args.rm_type,
        limit=args.limit,
    )
    print(f"wrote {args.output}: {count} rows (domain={args.domain}, rm_type={args.rm_type})")


if __name__ == "__main__":
    main()
