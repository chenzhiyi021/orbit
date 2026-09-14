#!/usr/bin/env python3
"""Audit (and optionally repair) a `run_evalchemy_math_eval.py`-produced
`completions.jsonl` for duplicate/inconsistent rows, and recompute its
reported accuracy from scratch to catch a corrupted `metrics.json`.

Why this can go wrong: `run_evalchemy_math_eval.py`'s resume logic keys each
row by `f"{task}/{example_id}/{repetition}"` -- NOT by which checkpoint
produced it. Point two different eval runs (e.g. two different extrapolation
candidates) at the same `OUTPUT_DIR`, or interrupt-and-restart the wrapper
against a stale one, and its "already have this key, skip regenerating" logic
will happily accept rows a completely different model produced, or leave
duplicate lines for the same key from separate attempts. Neither is visible
from `metrics.json` alone (it's just a rolled-up scalar) -- only the raw
`completions.jsonl` shows it.

This script:
1. Rebuilds the canonical set of `(task, example_id, repetition)` keys the
   eval run *should* contain (same `load_examples`/`PAPER_REPETITIONS`
   logic `run_evalchemy_math_eval.py` uses -- reimplemented here, not
   imported, specifically so checking a completions file doesn't require
   the aiohttp/lm_eval-bearing environment that live generation needs).
2. Flags: duplicate keys (and whether their `output` text actually agrees --
   agreeing duplicates are just redundant writes; disagreeing ones are the
   contamination smell described above), keys present in the file but not
   in the canonical set ("unexpected", another contamination smell), keys
   the canonical set expects but the file never attempted ("missing", just
   means the sweep didn't finish), and malformed JSON lines.
3. Recomputes acc/pass_acc/pass@k from the (deduplicated, last-write-wins --
   the same rule `run_evalchemy_math_eval.py`'s own resume logic uses)
   rows, and diffs that against `--metrics-json` if you point it at one.
4. With --fix: writes a deduplicated `completions.jsonl` (last occurrence
   per key kept, matching that same resume rule), after backing up the
   original. A subsequent eval-math-evalchemy.sh re-run then only
   regenerates genuinely missing keys instead of being stuck on rows a
   different run wrote.

Usage:

    python tools/extrapolation/check_eval_completions.py \\
        --completions eval_results/m6_lr_2e-6_simple_anchor3_current7_alpha4/math/aime24/completions.jsonl \\
        --task aime24 \\
        --evalchemy-root /mnt/L202500431/third_party/evalchemy \\
        --metrics-json eval_results/m6_lr_2e-6_simple_anchor3_current7_alpha4/math/aime24/metrics.json

Add --fix to have it deduplicate in place (original backed up next to it).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


# Mirrors examples/on_policy_distillation/eval/run_evalchemy_math_eval.py's
# TASKS / PAPER_REPETITIONS exactly (kept in sync by hand, not imported --
# see module docstring for why).
TASKS = {
    "aime24": ("eval/chat_benchmarks/AIME24/data/aime24.json", "problem", "expected_answer"),
    "aime25": ("eval/chat_benchmarks/AIME25/data/aime25.json", "problem", "answer"),
    "amc23": ("eval/chat_benchmarks/AMC23/data/amc23.json", "question", "answer"),
    "math500": ("eval/chat_benchmarks/MATH500/data/math500.jsonl", "problem", "answer"),
}
PAPER_REPETITIONS = {"aime24": 16, "aime25": 16, "amc23": 10, "math500": 10}


def load_examples(task: str, evalchemy_root: Path) -> list[tuple[str, str, str]]:
    relative_path, problem_key, answer_key = TASKS[task]
    path = evalchemy_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing Evalchemy data file: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [
        (str(row.get("id", row.get("unique_id", index))), str(row[problem_key]), str(row[answer_key]))
        for index, row in enumerate(rows)
    ]


def make_grader(grader: str):
    if grader == "evalchemy":
        from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

        def grade(prediction: str, reference: str) -> bool:
            boxed = last_boxed_only_string(prediction)
            try:
                extracted = remove_boxed(boxed) if boxed else ""
            except AssertionError:
                extracted = ""
            return bool(is_equiv(reference, extracted))

        return grade

    from orbit.rollout.rm_hub.math_utils import grade_answer_verl

    return lambda prediction, reference: bool(grade_answer_verl(prediction, reference))


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if k > num_samples:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.prod((num_samples - num_correct - i) / (num_samples - i) for i in range(k))


@dataclass(frozen=True)
class WorkKey:
    task: str
    example_id: str
    repetition: int

    @property
    def key(self) -> str:
        return f"{self.task}/{self.example_id}/{self.repetition}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--completions", type=Path, required=True, help="Path to completions.jsonl.")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--evalchemy-root", type=Path, required=True)
    parser.add_argument("--n-sampling", type=int, default=0, help="0 = the paper repetition count for --task (same default as run_evalchemy_math_eval.py).")
    parser.add_argument("--limit", type=int, default=0, help="First N examples; 0 = all (mirror the original run's --limit/NUM_SAMPLES if it used one).")
    parser.add_argument("--grader", choices=["evalchemy", "orbit"], default="evalchemy")
    parser.add_argument("--pass-k-values", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--metrics-json", type=Path, default=None, help="Optional metrics.json to diff the recomputed acc against.")
    parser.add_argument("--fix", action="store_true", help="Write a deduplicated completions.jsonl in place (backs up the original first).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completions_path = args.completions.expanduser().resolve()
    if not completions_path.is_file():
        raise FileNotFoundError(completions_path)

    examples = load_examples(args.task, args.evalchemy_root.expanduser().resolve())
    if args.limit > 0:
        examples = examples[: args.limit]
    repetitions = args.n_sampling if args.n_sampling > 0 else PAPER_REPETITIONS[args.task]
    canonical_keys: dict[str, WorkKey] = {}
    example_answer: dict[str, str] = {}
    for example_id, _problem, answer in examples:
        example_answer[example_id] = answer
        for repetition in range(repetitions):
            work_key = WorkKey(args.task, example_id, repetition)
            canonical_keys[work_key.key] = work_key

    total_lines = 0
    malformed_lines: list[int] = []
    rows_by_key: dict[str, list[dict]] = defaultdict(list)
    line_order: list[str] = []  # order of first appearance, for a stable dedup pass

    with completions_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines.append(line_no)
                continue
            key = row.get("key")
            if key is None:
                malformed_lines.append(line_no)
                continue
            if key not in rows_by_key:
                line_order.append(key)
            rows_by_key[key].append(row)

    present_keys = set(rows_by_key)
    expected_keys = set(canonical_keys)
    missing_keys = sorted(expected_keys - present_keys)
    unexpected_keys = sorted(present_keys - expected_keys)
    duplicate_keys = sorted(key for key, rows in rows_by_key.items() if len(rows) > 1)
    conflicting_keys = [
        key for key in duplicate_keys if len({row.get("output", "") for row in rows_by_key[key]}) > 1
    ]

    print(f"completions file: {completions_path}")
    print(f"  total lines (non-blank): {total_lines}")
    print(f"  malformed lines: {len(malformed_lines)}{' ' + str(malformed_lines[:10]) if malformed_lines else ''}")
    print(f"  expected keys (task={args.task}, {len(examples)} examples x {repetitions} reps): {len(expected_keys)}")
    print(f"  present unique keys: {len(present_keys)}")
    print(f"  missing keys (never attempted -- sweep just incomplete): {len(missing_keys)}")
    print(f"  unexpected keys (not in this task/config's canonical set -- contamination smell): {len(unexpected_keys)}")
    if unexpected_keys:
        print(f"    e.g.: {unexpected_keys[:10]}")
    print(f"  duplicate keys (>1 line): {len(duplicate_keys)}")
    print(f"  duplicate keys with DIFFERING output across duplicates (contamination smell): {len(conflicting_keys)}")
    if conflicting_keys:
        print(f"    e.g.: {conflicting_keys[:10]}")

    # Recompute metrics from the deduplicated (last-write-wins) rows, exactly
    # matching run_evalchemy_math_eval.py's own resume-time dict semantics.
    grade = make_grader(args.grader)
    per_example: dict[str, list[bool]] = defaultdict(list)
    for example_id, _problem, answer in examples:
        for repetition in range(repetitions):
            key = f"{args.task}/{example_id}/{repetition}"
            rows = rows_by_key.get(key)
            row = rows[-1] if rows else {}
            correct = row.get("status") == "ok" and grade(row.get("output", ""), answer)
            per_example[example_id].append(bool(correct))

    total = sum(len(flags) for flags in per_example.values())
    correct_total = sum(sum(flags) for flags in per_example.values())
    recomputed = {
        "acc": correct_total / total if total else 0.0,
        "pass_acc": sum(any(flags) for flags in per_example.values()) / len(per_example) if per_example else 0.0,
        "pass@k": {
            str(k): sum(pass_at_k(len(flags), sum(flags), k) for flags in per_example.values()) / len(per_example)
            for k in args.pass_k_values
            if per_example
        },
        "num_examples": len(per_example),
        "num_samples": total,
    }
    print("\nrecomputed from completions.jsonl (deduplicated, last-write-wins):")
    print(json.dumps(recomputed, indent=2))

    if args.metrics_json:
        reported = json.loads(args.metrics_json.expanduser().resolve().read_text())
        print(f"\nreported in {args.metrics_json}:")
        print(json.dumps(reported, indent=2))
        if abs(reported.get("acc", float("nan")) - recomputed["acc"]) > 1e-9:
            print("\n  MISMATCH: reported acc != recomputed acc from raw completions -- metrics.json is stale or wrong.")
        else:
            print("\n  OK: reported acc matches recomputed acc.")

    has_problems = bool(malformed_lines or unexpected_keys or conflicting_keys or duplicate_keys)
    if not has_problems:
        print("\nNo duplicates, no unexpected keys, no malformed lines. File looks clean.")
        return

    if not args.fix:
        print("\nRun again with --fix to deduplicate in place (keeps the LAST occurrence per key, "
              "matching the eval script's own resume semantics; original is backed up first).")
        return

    backup_path = completions_path.with_name(
        f"{completions_path.name}.bak-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    backup_path.write_bytes(completions_path.read_bytes())

    lines_written = 0
    keys_dropped_unexpected = 0
    with completions_path.open("w", encoding="utf-8") as handle:
        for key in line_order:
            if key not in expected_keys:
                keys_dropped_unexpected += 1
                continue  # drop rows that don't belong to this task/config at all
            handle.write(json.dumps(rows_by_key[key][-1], ensure_ascii=False) + "\n")
            lines_written += 1

    print(f"\nBacked up original to {backup_path}")
    print(
        f"Wrote deduplicated file: {total_lines} lines -> {lines_written} lines "
        f"({total_lines - lines_written} removed total: {keys_dropped_unexpected} unexpected key(s) "
        f"dropped entirely, the rest were duplicate lines for the {len(duplicate_keys)} keys that had them)."
    )
    print("Re-run the eval command (same OUTPUT_DIR) to fill in any still-missing keys; "
          "the resume logic will now only regenerate what's genuinely absent.")


if __name__ == "__main__":
    main()
