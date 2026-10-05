#!/usr/bin/env python3
"""Score a served checkpoint on Evalchemy's public math, code and science benchmarks.

This drives an already-running OpenAI-compatible endpoint (sglang; see
`eval-math-evalchemy.sh`) and mirrors an Evalchemy checkout's prompts and
graders, so the scores are the ones Evalchemy itself would produce:
  - aime24/aime25/amc23/math500: rows read from the checkout's data files.
  - gpqa_diamond: Idavidrein/gpqa (gated on HF), options shuffled and the letter
    extracted exactly as eval/chat_benchmarks/GPQADiamond does.
  - lcbv5: mlfoundations-dev/LCBv5-v2; the last fenced code block is executed
    against the private tests by Evalchemy's own `lcb_run` (Linux only: it forks
    a sandboxed child per solution).
One JSONL row is appended per completion, so an interrupted run resumes where it
stopped.

Writes `<output-dir>/<dataset>/metrics.json` in the shape
`tools/summarize_eval_results.py` already reads (`acc`, `pass_acc`, `pass@k`),
so the existing `tools/eval_checkpoints_once.sh` curve tooling works unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import aiohttp

# Evalchemy's hendrycks_math-derived prompt, shared verbatim by its AIME24,
# AIME25, AMC23 and MATH500 benchmarks (eval/chat_benchmarks/*/eval_instruct.py).
MATH_PROMPT = """Problem: {problem}
Mark your solution with \\boxed
Answer:"""

# SimpleRL-Zoo's "simple prompt" (Zeng et al. 2025, Fig. 10), used to zero-RL-train base models with weak
# instruction following (Llama-3.1-8B, Mistral-7B, Qwen-2.5-0.5B/1.5B). Sent as a raw completion, no chat
# template; generation stops before the model starts inventing the next "Question:".
SIMPLERL_PROMPT = """Question:
{problem}
Answer:
Let's think step by step.
"""
SIMPLERL_STOP = ["\nQuestion:", "\n\nQuestion"]

# Data file + (problem, answer) column names per task, relative to --evalchemy-root.
MATH_TASKS = {
    "aime24": ("eval/chat_benchmarks/AIME24/data/aime24.json", "problem", "expected_answer"),
    "aime25": ("eval/chat_benchmarks/AIME25/data/aime25.json", "problem", "answer"),
    "amc23": ("eval/chat_benchmarks/AMC23/data/amc23.json", "question", "answer"),
    "math500": ("eval/chat_benchmarks/MATH500/data/math500.jsonl", "problem", "answer"),
}
TASKS = (*MATH_TASKS, "gpqa_diamond", "lcbv5")

# eval/chat_benchmarks/GPQADiamond/eval_instruct.py, verbatim.
GPQA_PROMPT = """Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response.
Problem: {problem}
Options: {options}
Answer:"""

# eval/chat_benchmarks/LiveCodeBenchv5/eval_instruct.py, verbatim: the instruction
# is glued straight onto the question text, and the last fenced block is graded.
LCB_STDIN_INSTRUCTION = "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
LCB_FUNCTIONAL_INSTRUCTION = "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
LCB_CODE_BLOCK = re.compile(r"```(?:[a-zA-Z]*)\n(.*?)```", re.DOTALL)
LCB_TEST_TIMEOUT = 6

# Repetition counts the OPD paper reports for each cell (its Table 12); used when
# --n-sampling is left unset so the defaults reproduce the published protocol.
# Evalchemy's own GPQADiamond default is 3, not 10.
PAPER_REPETITIONS = {"aime24": 16, "aime25": 16, "amc23": 10, "math500": 10, "gpqa_diamond": 10, "lcbv5": 3}


@dataclass(frozen=True)
class Example:
    example_id: str
    prompt: str
    answer: str = ""
    # Reported as acc_by_group in metrics.json (LiveCodeBench difficulty).
    group: str = ""
    meta: dict = field(default_factory=dict)


# (example, one output per repetition, None for a failed request) -> verdict per repetition.
GradeFn = Callable[[Example, list[str | None]], list[bool]]


@dataclass(frozen=True)
class WorkItem:
    task: str
    example_id: str
    prompt: str
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


def load_math_examples(task: str, evalchemy_root: Path, prompt_style: str = "evalchemy") -> list[Example]:
    relative_path, problem_key, answer_key = MATH_TASKS[task]
    template = SIMPLERL_PROMPT if prompt_style == "simplerl" else MATH_PROMPT
    path = evalchemy_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing Evalchemy data file: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [
        Example(
            str(row.get("id", row.get("unique_id", index))),
            template.format(problem=row[problem_key]),
            str(row[answer_key]),
        )
        for index, row in enumerate(rows)
    ]


def load_evalchemy_module(evalchemy_root: Path, relative_path: str, name: str):
    """Import one Evalchemy helper file by path.

    Going through the `eval.chat_benchmarks` package would run its
    eval_instruct.py, which imports lm_eval; the helpers themselves only need
    the stdlib (and scipy for LiveCodeBench).
    """
    path = evalchemy_root / "eval/chat_benchmarks" / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing Evalchemy file: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_gpqa(args: argparse.Namespace) -> tuple[list[Example], GradeFn, int]:
    from datasets import load_dataset

    utils = load_evalchemy_module(args.evalchemy_root, "GPQADiamond/testing_utils.py", "evalchemy_gpqa_utils")
    examples = []
    for index, row in enumerate(load_dataset(args.gpqa_path, "gpqa_diamond")["train"]):
        answers = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Evalchemy reseeds per question, so every question gets the same permutation.
        random.Random(42).shuffle(answers)
        letters = "ABCD"
        options = ", ".join(f"{letter}) {answer}" for letter, answer in zip(letters, answers))
        examples.append(
            Example(
                str(row.get("Record ID") or index),
                GPQA_PROMPT.format(problem=row["Question"], options=options),
                letters[answers.index(row["Correct Answer"])],
            )
        )

    def grade(example: Example, outputs: list[str | None]) -> list[bool]:
        return [output is not None and utils.get_multiple_choice_answer(output) == example.answer for output in outputs]

    return examples, grade, 1


def prepare_lcb(args: argparse.Namespace) -> tuple[list[Example], GradeFn, int]:
    from datasets import load_dataset

    utils = load_evalchemy_module(
        args.evalchemy_root, "LiveCodeBenchv5/livecodebench_utils.py", "evalchemy_lcb_utils"
    )
    dataset = load_dataset(args.lcb_path, split="test")
    # The private tests are ~2.3GB of compressed blobs; leave them in the Arrow
    # table and decode one problem at a time while grading.
    light = dataset.remove_columns(["private_test_cases"])
    examples = []
    for index, row in enumerate(light):
        is_stdin = utils.has_test_type(row["public_test_cases"], "stdin")
        instruction = LCB_STDIN_INSTRUCTION if is_stdin else LCB_FUNCTIONAL_INSTRUCTION
        examples.append(
            Example(
                str(row["question_id"]),
                instruction + row["question_content"],
                group=str(row["difficulty"]),
                meta={"row": index, "is_stdin": is_stdin},
            )
        )
    dataset_lock = threading.Lock()

    def grade(example: Example, outputs: list[str | None]) -> list[bool]:
        with dataset_lock:
            encoded = dataset[example.meta["row"]]["private_test_cases"]
        problem = {"test": utils.translate_private_test_cases(encoded)}
        verdicts = []
        for output in outputs:
            blocks = LCB_CODE_BLOCK.findall(output) if output is not None else []
            if not blocks:
                verdicts.append(False)
                continue
            try:
                results = utils.lcb_run(
                    problem,
                    utils.post_process_code(blocks[-1]),
                    LCB_TEST_TIMEOUT,
                    not example.meta["is_stdin"],
                )
                verdicts.append(bool(results) and all(result[0] for result in results))
            except Exception:  # noqa: BLE001 - Evalchemy scores evaluation errors as wrong
                verdicts.append(False)
        return verdicts

    return examples, grade, args.grade_workers


def prepare_task(task: str, args: argparse.Namespace) -> tuple[list[Example], GradeFn, int]:
    """Return the task's examples, its per-example grader, and how many examples to grade at once."""
    if task == "gpqa_diamond":
        return prepare_gpqa(args)
    if task == "lcbv5":
        return prepare_lcb(args)
    grade_one = make_grader(args.grader)

    def grade(example: Example, outputs: list[str | None]) -> list[bool]:
        return [output is not None and grade_one(output, example.answer) for output in outputs]

    # Serial: Orbit's math grader may rely on signal-based timeouts (main thread only).
    return load_math_examples(task, args.evalchemy_root, args.prompt_style), grade, 1


def make_grader(grader: str):
    """Return `(prediction_text, reference) -> bool`.

    `evalchemy` is the parity choice: the exact `\\boxed` extraction and
    `is_equiv` comparison Evalchemy runs, which needs lm_eval importable.
    `orbit` is the grader Orbit trains against (`--rm-type math`); it accepts
    strictly more answers, so the two are not interchangeable -- pick one and
    keep it fixed across the checkpoints you compare.
    `simplerl` is `evalchemy` with a fallback for models that were never asked to
    box (SimpleRL-Zoo's simple prompt): the last \\boxed{} if any, else the text
    after the last "answer is", else the last number.
    """
    if grader in ("evalchemy", "simplerl"):
        from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

        def extract(prediction: str) -> str:
            boxed = last_boxed_only_string(prediction)
            try:
                return remove_boxed(boxed) if boxed else ""
            except AssertionError:
                # last_boxed_only_string also returns \fbox{...} and \boxed
                # followed by whitespace/newline before the brace; remove_boxed
                # only accepts \boxed{...} / "\boxed " and asserts otherwise.
                # Score those as wrong instead of aborting the whole run.
                return ""

        def extract_lenient(prediction: str) -> str:
            if last_boxed_only_string(prediction):
                return extract(prediction)
            stated = re.findall(r"answer is[:\s]*\$?([^\n$]+?)\$?\s*(?:\.\s*)?(?:\n|$)", prediction, re.IGNORECASE)
            if stated:
                return stated[-1].strip()
            numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", prediction.replace(",", ""))
            return numbers[-1] if numbers else ""

        pick = extract_lenient if grader == "simplerl" else extract
        return lambda prediction, reference: bool(is_equiv(reference, pick(prediction)))

    from orbit.rollout.rm_hub.math_utils import grade_answer_verl

    return lambda prediction, reference: bool(grade_answer_verl(prediction, reference))


async def generate_one(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    item: WorkItem,
) -> dict:
    raw = args.prompt_style == "simplerl"
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if raw:
        # Base / zero-RL models: a plain completion, no chat template.
        payload.update(prompt=item.prompt, stop=SIMPLERL_STOP)
    else:
        payload.update(
            messages=[{"role": "user", "content": item.prompt}],
            # Qwen3 chat templates gate the <think> block on this; sglang forwards it
            # to the tokenizer's chat template.
            chat_template_kwargs={"enable_thinking": args.enable_thinking},
        )
    if args.seed >= 0:
        payload["seed"] = seed_for(args.seed, item.repetition)
    endpoint = "/v1/completions" if raw else "/v1/chat/completions"
    last_error = ""
    for attempt in range(args.max_attempts):
        try:
            async with session.post(
                f"{args.base_url.rstrip('/')}{endpoint}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=args.timeout_seconds),
            ) as response:
                body = await response.json()
            choice = body["choices"][0]
            return {"key": item.key, "status": "ok", "output": choice["text"] if raw else choice["message"]["content"]}
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


def write_metrics(
    task_dir: Path, results: dict[str, list[bool]], pass_k_values: list[int], groups: dict[str, str]
) -> dict:
    per_example = list(results.values())
    total = sum(len(flags) for flags in per_example)
    correct = sum(sum(flags) for flags in per_example)
    by_group: dict[str, list[bool]] = {}
    for example_id, group in groups.items():
        by_group.setdefault(group, []).extend(results[example_id])
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
    if by_group:
        metrics["acc_by_group"] = {group: sum(flags) / len(flags) for group, flags in sorted(by_group.items())}
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


async def run_task(args: argparse.Namespace, task: str) -> dict:
    task_dir = args.output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    completions_path = task_dir / "completions.jsonl"

    examples, grade, grade_workers = prepare_task(task, args)
    if args.limit > 0:
        examples = examples[: args.limit]
    repetitions = args.n_sampling if args.n_sampling > 0 else PAPER_REPETITIONS[task]
    work = [
        WorkItem(task, example.example_id, example.prompt, repetition)
        for example in examples
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

    def grade_example(example: Example) -> list[bool]:
        outputs = []
        for repetition in range(repetitions):
            row = rows.get(WorkItem(task, example.example_id, example.prompt, repetition).key, {})
            outputs.append(row["output"] if row.get("status") == "ok" else None)
        return grade(example, outputs)

    if grade_workers > 1:
        print(f"[{task}] grading {len(examples)} examples with {grade_workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=grade_workers) as pool:
            verdicts = list(pool.map(grade_example, examples))
    else:
        verdicts = [grade_example(example) for example in examples]
    results = {example.example_id: flags for example, flags in zip(examples, verdicts)}
    groups = {example.example_id: example.group for example in examples if example.group}
    metrics = write_metrics(task_dir, results, args.pass_k_values, groups)
    print(f"[{task}] acc={metrics['acc']:.4f} pass_acc={metrics['pass_acc']:.4f}", flush=True)
    return metrics


async def main_async(args: argparse.Namespace) -> None:
    summary = {task: await run_task(args, task) for task in args.tasks}
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
    parser.add_argument(
        "--grader", choices=["evalchemy", "orbit", "simplerl"], default="evalchemy", help="math tasks only"
    )
    parser.add_argument(
        "--prompt-style",
        choices=["evalchemy", "simplerl"],
        default="evalchemy",
        help="math tasks only: 'simplerl' sends SimpleRL-Zoo's raw 'Question/Answer' completion prompt "
        "(no chat template) for base and zero-RL models",
    )
    parser.add_argument(
        "--gpqa-path", default="Idavidrein/gpqa", help="HF repo id or local snapshot (needs its README.md)"
    )
    parser.add_argument("--lcb-path", default="mlfoundations-dev/LCBv5-v2", help="HF repo id or local snapshot")
    parser.add_argument(
        "--grade-workers", type=int, default=32, help="LiveCodeBench problems executed concurrently"
    )
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
