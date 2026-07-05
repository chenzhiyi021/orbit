#!/usr/bin/env python3
"""Convert nvidia/Nemotron-RL-instruction_following into the common MOPD row schema.

Source schema (per https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following):
    {"id": int, "instruction_id_list": [str, ...], "prompt": str, "kwargs": [dict, ...], ...}

Written row:
    {
      "prompt": <prompt text>,
      "label": "",
      "metadata": {
        "domains": "if",
        "rm_type": "ifbench",
        "instruction_id_list": [...],
        "kwargs": [...],
        "prompt_text": <same as prompt, required by compute_ifbench_reward's matching>,
        "record_id": <id>,
      },
    }

This mirrors the fields orbit/rollout/rm_hub/ifbench.py::compute_ifbench_reward
expects in `sample.metadata` (instruction_id_list/kwargs/prompt_text/record_id),
so the *same* rm_type used here for training also works unchanged for eval
(see mopd_prepare_ifbench_eval.py).

Usage:
    python scripts/mopd_prepare_nemotron_if.py \\
        --input  /mnt/L202500431/datasets/Nemotron-RL-instruction_following/instruction_following.jsonl \\
        --output /mnt/L202500431/datasets/orbit_mopd/domains/if_nemotron_train.jsonl
"""

from __future__ import annotations

import argparse

from mopd_data_common import iter_records, make_record, write_jsonl

# Field-name fallbacks in case the released schema uses different key casing/naming.
_ID_KEYS = ("id", "key", "record_id")
_PROMPT_KEYS = ("prompt", "text")
_INSTRUCTION_ID_KEYS = ("instruction_id_list", "instruction_ids")
_KWARGS_KEYS = ("kwargs", "instruction_kwargs")


def _first_present(row: dict, keys: tuple[str, ...], default=None):
    for key in keys:
        if key in row:
            return row[key]
    return default


def convert(input_path: str, output_path: str, *, domain: str, rm_type: str, limit: int | None) -> int:
    records = []
    for row in iter_records(input_path):
        prompt = _first_present(row, _PROMPT_KEYS)
        if prompt is None:
            raise KeyError(f"Could not find a prompt field (tried {_PROMPT_KEYS}) in row: {row}")
        record_id = _first_present(row, _ID_KEYS, default=len(records))
        instruction_id_list = _first_present(row, _INSTRUCTION_ID_KEYS, default=[])
        kwargs = _first_present(row, _KWARGS_KEYS, default=[])

        metadata = {
            "domains": domain,
            "rm_type": rm_type,
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs,
            "prompt_text": prompt,
            "record_id": record_id,
        }
        records.append(make_record(prompt, None, metadata))
        if limit is not None and len(records) >= limit:
            break
    return write_jsonl(records, output_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Source instruction_following.jsonl.")
    ap.add_argument("--output", required=True, help="Destination JSONL path (usually under orbit_mopd/domains/).")
    ap.add_argument("--domain", default="if", help="Value to write into metadata.domains (default: if).")
    ap.add_argument("--rm-type", default="ifbench", help="Value to write into metadata.rm_type (default: ifbench).")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows (for smoke tests).")
    args = ap.parse_args()

    count = convert(args.input, args.output, domain=args.domain, rm_type=args.rm_type, limit=args.limit)
    print(f"wrote {args.output}: {count} rows (domain={args.domain}, rm_type={args.rm_type})")


if __name__ == "__main__":
    main()
