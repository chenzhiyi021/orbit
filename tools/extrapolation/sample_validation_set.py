#!/usr/bin/env python3
"""Sample Dv (paper: "50 examples randomly sampled from the training set") from
an Orbit OPD training file, in the `{problem, expected_answer}` schema
`examples/on_policy_distillation/eval/run_evalchemy_math_eval.py` expects for
a custom Evalchemy task (matching its AIME24 loader).

Only needed if you want validate_checkpoint.py to score candidates against
the actual training distribution instead of an existing benchmark task (its
default) -- closer to the paper, at the cost of registering a small custom
task with your Evalchemy checkout (see the printed instructions at the end).

Accepts a `.jsonl` or `.parquet` training file. Handles two input row
shapes:
  - flat: a problem-like column + an answer-like column (auto-detected from
    {problem,question,prompt} x {answer,expected_answer,label}, or pass
    --problem-key/--answer-key explicitly).
  - chat: a `messages` column (list of {role, content} dicts) -- the last
    user turn's content is taken as the problem; --answer-key is required
    since chat-formatted rollout data typically carries the reference
    answer in a separate column, not inside `messages`.

Usage:

    python tools/extrapolation/sample_validation_set.py \\
        --train-file /path/to/train_qa.parquet \\
        --num-samples 50 --seed 0 \\
        --output validation_sets/dv_lr_2e-6_seed0.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROBLEM_KEY_CANDIDATES = ("problem", "question", "prompt")
ANSWER_KEY_CANDIDATES = ("answer", "expected_answer", "label", "reference_answer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-file", type=Path, required=True, help=".jsonl or .parquet training file.")
    parser.add_argument("--num-samples", type=int, default=50, help="|Dv| (paper default: 50).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--problem-key", default=None, help="Override auto-detected problem column.")
    parser.add_argument("--answer-key", default=None, help="Override auto-detected answer column.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def detect_key(row: dict, candidates: tuple[str, ...], override: str | None, label: str) -> str:
    if override:
        if override not in row:
            raise KeyError(f"--{label}-key {override!r} not found in row; available keys: {sorted(row)}")
        return override
    for candidate in candidates:
        if candidate in row:
            return candidate
    raise KeyError(
        f"Could not auto-detect a {label} column among {candidates}; available keys: {sorted(row)}. "
        f"Pass --{label}-key explicitly."
    )


def extract_problem(row: dict, problem_key: str | None) -> str:
    if problem_key and problem_key in row:
        return str(row[problem_key])
    if "messages" in row:
        messages = row["messages"]
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message["content"])
        raise ValueError(f"No user-role message found in row: {row}")
    raise KeyError(f"Row has neither a problem-like column nor 'messages': {sorted(row)}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.train_file.expanduser().resolve())
    if not rows:
        raise ValueError(f"{args.train_file} has no rows")

    sample = rows[0]
    problem_key = args.problem_key
    if problem_key is None and "messages" not in sample:
        problem_key = detect_key(sample, PROBLEM_KEY_CANDIDATES, None, "problem")
    answer_key = detect_key(sample, ANSWER_KEY_CANDIDATES, args.answer_key, "answer")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(rows)), k=min(args.num_samples, len(rows)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for order, index in enumerate(indices):
            row = rows[index]
            record = {
                "id": f"dv_{order}",
                "problem": extract_problem(row, problem_key),
                "expected_answer": str(row[answer_key]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(indices)} examples to {args.output}")
    print(
        "\nTo use this with run_evalchemy_math_eval.py (and hence validate_checkpoint.py --task), "
        "add a task entry pointing at it, e.g. in a copy of that file's TASKS dict:\n"
        f'  "dv": ("{args.output}", "problem", "expected_answer"),\n'
        "then pass --task dv (and DATA_NAMES=dv / --task dv downstream)."
    )


if __name__ == "__main__":
    main()
