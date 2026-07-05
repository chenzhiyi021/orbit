#!/usr/bin/env python3
"""Convert allenai/IFBench's IFBench_test.jsonl into an Orbit eval JSONL.

IFBench_test.jsonl follows the Google IFEval schema: each row has a flat
`key`/`prompt`/`instruction_id_list`/`kwargs` -- there is no nested
`metadata` column, so it can't be pointed at directly via `--eval-config`
overrides (EvalDatasetConfig only *adds* static overrides, it can't compose
a metadata dict out of other top-level columns). This script does that
composition once, up front.

Written row:
    {
      "prompt": <prompt text>,
      "label": "",
      "metadata": {
        "instruction_id_list": [...],
        "kwargs": [...],
        "prompt_text": <same as prompt>,
        "record_id": <key>,
      },
    }

`rm_type` is intentionally NOT written per-row here -- set it once at the
dataset level in the `--eval-config` yaml (`rm_type: ifbench` for this
dataset's entry) so this file matches how eval datasets are normally
configured (see orbit/utils/eval_config.py::EvalDatasetConfig).

Usage:
    python scripts/mopd_prepare_ifbench_eval.py \\
        --input  /mnt/L202500431/datasets/IFBench_test/IFBench_test.jsonl \\
        --output /mnt/L202500431/datasets/orbit_mopd/domains/if_ifbench_test.jsonl
"""

from __future__ import annotations

import argparse

from mopd_data_common import iter_records, make_record, write_jsonl

_ID_KEYS = ("key", "id", "record_id")
_PROMPT_KEYS = ("prompt", "text")
_INSTRUCTION_ID_KEYS = ("instruction_id_list", "instruction_ids")
_KWARGS_KEYS = ("kwargs", "instruction_kwargs")


def _first_present(row: dict, keys: tuple[str, ...], default=None):
    for key in keys:
        if key in row:
            return row[key]
    return default


def convert(input_path: str, output_path: str, *, limit: int | None) -> int:
    records = []
    for row in iter_records(input_path):
        prompt = _first_present(row, _PROMPT_KEYS)
        if prompt is None:
            raise KeyError(f"Could not find a prompt field (tried {_PROMPT_KEYS}) in row: {row}")
        record_id = _first_present(row, _ID_KEYS, default=len(records))
        instruction_id_list = _first_present(row, _INSTRUCTION_ID_KEYS, default=[])
        kwargs = _first_present(row, _KWARGS_KEYS, default=[])

        metadata = {
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
    ap.add_argument("--input", required=True, help="Source IFBench_test.jsonl.")
    ap.add_argument("--output", required=True, help="Destination JSONL path (usually under orbit_mopd/domains/).")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows (for smoke tests).")
    args = ap.parse_args()

    count = convert(args.input, args.output, limit=args.limit)
    print(f"wrote {args.output}: {count} rows")


if __name__ == "__main__":
    main()
