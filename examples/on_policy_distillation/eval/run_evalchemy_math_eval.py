#!/usr/bin/env python3
"""Score a served checkpoint on Evalchemy's public math benchmarks.

This drives an already-running OpenAI-compatible endpoint (sglang; see
`eval-math-evalchemy.sh`) and reads the benchmark rows straight out of an
Evalchemy checkout, so the prompt text and the answer grader are the ones
Evalchemy itself would use. One JSONL row is appended per completion, so an
interrupted run resumes where it stopped.

Writes `<output-dir>/<dataset>/metrics.json` in the shape
`tools/summarize_eval_results.py` already reads (`acc`, `pass_acc`, `pass@k`),
so the existing `tools/eval_checkpoints_once.sh` curve tooling works unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import aiohttp

# Evalchemy's hendrycks_math-derived prompt, shared verbatim by its AIME24,
# AIME25, AMC23 and MATH500 benchmarks (eval/chat_benchmarks/*/eval_instruct.py).
MATH_PROMPT = """Problem: {problem}
Mark your solution with \\boxed
Answer:"""

# Data file + (problem, answer) column names per task, relative to --evalchemy-root.
TASKS = {
    "aime24": ("eval/chat_benchmarks/AIME24/data/aime24.json", "problem", "expected_answer"),
    "aime25": ("eval/chat_benchmarks/AIME25/data/aime25.json", "problem", "answer"),
    "amc23": ("eval/chat_benchmarks/AMC23/data/amc23.json", "question", "answer"),
    "math500": ("eval/chat_benchmarks/MATH500/data/math500.jsonl", "problem", "answer"),
}

# Repetition counts the OPD paper reports for each cell; used when --n-sampling
# is left unset so the defaults reproduce the published protocol.
PAPER_REPETITIONS = {"aime24": 16, "aime25": 16, "amc23": 10, "math500": 10}


@dataclass(frozen=True)
class WorkItem:
    task: str
    example_id: str
    prompt: str
    answer: str
    repetition: int

    @property
    def key(self) -> str:
        return f"{self.task}/{self.example_id}/{self.repetition}"


def seed_for(base_seed: int, repetition: int) -> int:
    """Evalchemy's sampling-seed scheme: every request in repetition n gets base + n.

    Mirrors eval/chat_benchmarks/{AIME24,AIME25,AMC23}/eval_instruct.py, which
    computes `seed = [s + i for s in self.seed]` once per repetition and hands the
    same value to every example in it; eval/task.py then forwards seed[0] to the
    backend's sampling params. base_seed 0 reproduces Evalchemy's own default
    ([0, 1234, 1234, 1234]).

    The seed depends only on the repetition index, so all examples in one
    repetition share a random-number stream. That is deliberate (parity with
    Evalchemy), and it is what makes a rerun reproducible; note that it also
    correlates the per-example outcomes within a repetition, so the spread of the
    per-repetition accuracies is wider than independent-stream sampling would give.
    """
    return base_seed + repetition


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
    """Return `(prediction_text, reference) -> bool`.

    `evalchemy` is the parity choice: the exact `\\boxed` extraction and
    `is_equiv` comparison Evalchemy runs, which needs lm_eval importable.
    `orbit` is the grader Orbit trains against (`--rm-type math`); it accepts
    strictly more answers, so the two are not interchangeable -- pick one and
    keep it fixed across the checkpoints you compare.
    """
    if grader == "evalchemy":
        from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

        def grade(prediction: str, reference: str) -> bool:
            boxed = last_boxed_only_string(prediction)
            try:
                extracted = remove_boxed(boxed) if boxed else ""
            except AssertionError:
                # last_boxed_only_string also returns \fbox{...} and \boxed
                # followed by whitespace/newline before the brace; remove_boxed
                # only accepts \boxed{...} / "\boxed " and asserts otherwise.
                # Score those as wrong instead of aborting the whole run.
                extracted = ""
            return bool(is_equiv(reference, extracted))

        return grade

    from orbit.rollout.rm_hub.math_utils import grade_answer_verl

    return lambda prediction, reference: bool(grade_answer_verl(prediction, reference))


async def generate_one(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    item: WorkItem,
) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": item.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        # Qwen3 chat templates gate the <think> block on this; sglang forwards it
        # to the tokenizer's chat template.
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    if args.seed >= 0:
        payload["seed"] = seed_for(args.seed, item.repetition)
    last_error = ""
    for attempt in range(args.max_attempts):
        try:
            async with session.post(
                f"{args.base_url.rstrip('/')}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=args.timeout_seconds),
            ) as response:
                body = await response.json()
            return {"key": item.key, "status": "ok", "output": body["choices"][0]["message"]["content"]}
        except Exception as error:  # noqa: BLE001 - retried, then recorded on the row
            last_error = f"{type(error).__name__}: {error}"
            await asyncio.sleep(min(2**attempt, 30))
    return {"key": item.key, "status": "error", "output": "", "error": last_error}


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., Codex), 0 when k exceeds the sample count."""
    if k > num_samples:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.prod((num_samples - num_correct - i) / (num_samples - i) for i in range(k))


def write_metrics(task_dir: Path, results: dict[str, list[bool]], pass_k_values: list[int]) -> dict:
    per_example = list(results.values())
    total = sum(len(flags) for flags in per_example)
    correct = sum(sum(flags) for flags in per_example)
    metrics = {
        "acc": correct / total if total else 0.0,
        "pass_acc": sum(any(flags) for flags in per_example) / len(per_example) if per_example else 0.0,
        "pass@k": {
            str(k): sum(pass_at_k(len(flags), sum(flags), k) for flags in per_example) / len(per_example)
            for k in pass_k_values
            if per_example
        },
        "num_examples": len(per_example),
        "num_samples": total,
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


async def run_task(args: argparse.Namespace, task: str, grade) -> dict:
    task_dir = args.output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    completions_path = task_dir / "completions.jsonl"

    examples = load_examples(task, args.evalchemy_root)
    if args.limit > 0:
        examples = examples[: args.limit]
    repetitions = args.n_sampling if args.n_sampling > 0 else PAPER_REPETITIONS[task]
    work = [
        WorkItem(task, example_id, MATH_PROMPT.format(problem=problem), answer, repetition)
        for example_id, problem, answer in examples
        for repetition in range(repetitions)
    ]

    rows = {}
    if completions_path.is_file():
        with completions_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[row["key"]] = row
    pending = [item for item in work if item.key not in rows]
    print(f"[{task}] {len(examples)} examples x {repetitions} reps = {len(work)} samples, {len(pending)} pending", flush=True)

    if pending:
        semaphore = asyncio.Semaphore(args.concurrency)
        write_lock = asyncio.Lock()
        completed = 0

        # trust_env=False: cluster http_proxy settings otherwise intercept the
        # localhost endpoint (orbit's own scoring client does the same).
        async with aiohttp.ClientSession(trust_env=False) as session:

            async def bounded(item: WorkItem) -> None:
                nonlocal completed
                async with semaphore:
                    row = await generate_one(session, args, item)
                async with write_lock:
                    with completions_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows[item.key] = row
                    completed += 1
                    if completed % args.progress_every == 0 or completed == len(pending):
                        print(f"[{task}] {completed}/{len(pending)}", flush=True)

            await asyncio.gather(*(bounded(item) for item in pending))

    results: dict[str, list[bool]] = {}
    for item in work:
        row = rows.get(item.key, {})
        results.setdefault(item.example_id, []).append(
            row.get("status") == "ok" and grade(row["output"], item.answer)
        )
    metrics = write_metrics(task_dir, results, args.pass_k_values)
    print(f"[{task}] acc={metrics['acc']:.4f} pass_acc={metrics['pass_acc']:.4f}", flush=True)
    return metrics


async def main_async(args: argparse.Namespace) -> None:
    grade = make_grader(args.grader)
    summary = {task: await run_task(args, task, grade) for task in args.tasks}
    (args.output_dir / "summary.json").write_text(
        json.dumps({"model": args.model, "grader": args.grader, "tasks": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:18001")
    parser.add_argument("--model", required=True, help="served model name registered with the endpoint")
    parser.add_argument("--evalchemy-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=lambda value: [part.strip() for part in value.split(",") if part.strip()],
        default=["math500", "aime24", "amc23"],
    )
    parser.add_argument(
        "--n-sampling", type=int, default=0, help="repetitions per example; 0 uses the paper count per task"
    )
    parser.add_argument("--limit", type=int, default=0, help="first N examples per task; 0 means all")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "base sampling seed, Evalchemy-style: repetition n of every example is sent "
            "with seed base+n. 0 (default) matches Evalchemy's own default. -1 sends no "
            "seed at all and lets the server's RNG run free (the old behaviour). Pinning "
            "this fixes the sampling draw but not the batching; see --help on the wrapper."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--grader", choices=["evalchemy", "orbit"], default="evalchemy")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--pass-k-values", type=int, nargs="+", default=[1, 8, 16])
    args = parser.parse_args()
    unknown = [task for task in args.tasks if task not in TASKS]
    if unknown:
        parser.error(f"unsupported task(s) {unknown}; choose from {sorted(TASKS)}")
    return args


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
