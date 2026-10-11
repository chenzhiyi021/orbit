"""Build a non-thinking long-CoT SFT set from nvidia/OpenMathReasoning's cot split.

The target recipe is examples/sft/run-qwen3-1_7b-bf16-sft-openmathreasoning-100step.sh: the same
hyperparameters as the unified M1 100-step SFT (--loss-mask-type generation_prompt with
enable_thinking=false), evaluated with eval-math-evalchemy.sh at its defaults (ENABLE_THINKING=0).
So every row has to train the exact sequence that eval samples from:

    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{reasoning}\\n\\n{summary with \\boxed}<eos>

This script prepares those rows from the local parquet shards (data/cot-*.parquet):

  * keep problem_type == "has_answer_extracted" (the only type whose answer was checked) and,
    by default, generation_model == "DeepSeek-R1" (the rest is QwQ-32B);
  * one solution per problem: candidates are shuffled with --seed and the first one per
    problem is kept, so N rows means N distinct problems;
  * the <think>/</think> tags are stripped -- the reasoning and the summary become one plain
    assistant message -- and rows without exactly one tag pair or without a \\boxed in the
    summary are dropped;
  * the user turn is examples/on_policy_distillation/eval/run_evalchemy_math_eval.py's
    MATH_PROMPT, the prompt eval uses;
  * rows whose full tokenized conversation (rendered with Orbit's generation_prompt mask
    generator, the one the launcher uses) exceeds --max-tokens are dropped, never truncated.
    The default 12288 is the M1 source yaml's max_length.

Rows are written in the shuffled order, which is the training order (the launcher does not
pass --rollout-shuffle), as {"messages": [...], "metadata": {...}} JSONL.

Only the candidate pool (--num-examples x --oversample, enough to absorb the length filter)
has its solution text read, one shard at a time.

    python tools/build_openmathreasoning_sft.py \\
        --data-dir /mnt/L202500431/datasets/OpenMathReasoning/data \\
        --tokenizer /mnt/L202500431/models/qwen3-1.7b \\
        --num-examples 25600 --max-tokens 12288 \\
        --output /mnt/L202500431/datasets/openmathreasoning_nothink_25k6/train.jsonl \\
        --check-tokenizer
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Mirrors run_evalchemy_math_eval.py's MATH_PROMPT; keep in sync by hand.
MATH_PROMPT = """Problem: {problem}
Mark your solution with \\boxed
Answer:"""


def strip_think(solution: str) -> str | None:
    """Turn '<think>reasoning</think>summary' into 'reasoning\\n\\nsummary'; None if malformed."""
    if solution.count("<think>") != 1 or solution.count("</think>") != 1:
        return None
    head, rest = solution.split("<think>", 1)
    if head.strip():
        return None
    reasoning, summary = rest.split("</think>", 1)
    reasoning, summary = reasoning.strip(), summary.strip()
    if not reasoning or "\\boxed" not in summary:
        return None
    return f"{reasoning}\n\n{summary}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path, help="directory holding cot-*.parquet")
    parser.add_argument("--tokenizer", required=True, help="HF dir whose tokenizer/chat template training uses")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-examples", type=int, default=25600)
    parser.add_argument("--max-tokens", type=int, default=12288, help="cap on the full tokenized conversation")
    parser.add_argument("--generation-model", default="DeepSeek-R1", help="'' keeps every teacher")
    parser.add_argument("--oversample", type=float, default=1.4, help="candidate pool = num-examples x this")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-tokenizer", action="store_true")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator
    from orbit.utils.processing_utils import load_tokenizer

    tokenizer = load_tokenizer(args.tokenizer, chat_template_path=None, trust_remote_code=True)
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit(f"{args.tokenizer} has no chat template")
    # Same generator and kwargs as the launcher's --loss-mask-type / --apply-chat-template-kwargs.
    masker = MultiTurnLossMaskGenerator(
        tokenizer, tokenizer_type="generation_prompt", chat_template_kwargs={"enable_thinking": False}
    )

    shards = sorted(args.data_dir.glob("cot-*.parquet"))
    if not shards:
        raise SystemExit(f"no cot-*.parquet under {args.data_dir}")

    # Pass 1: filter on the light columns only.
    candidates = []  # (shard index, row index, problem)
    for s, path in enumerate(shards):
        table = pq.read_table(path, columns=["problem", "problem_type", "generation_model"]).to_pydict()
        for r, (problem, ptype, model) in enumerate(
            zip(table["problem"], table["problem_type"], table["generation_model"])
        ):
            if ptype != "has_answer_extracted":
                continue
            if args.generation_model and model != args.generation_model:
                continue
            candidates.append((s, r, problem))
        print(f"  scanned {s + 1}/{len(shards)} shards, {len(candidates)} candidates", flush=True)

    random.Random(args.seed).shuffle(candidates)
    seen, pool = set(), []
    pool_size = math.ceil(args.num_examples * args.oversample)
    for s, r, problem in candidates:
        key = problem.strip()
        if key in seen:
            continue
        seen.add(key)
        pool.append((s, r, problem))
        if len(pool) >= pool_size:
            break
    print(f"{len(candidates)} filtered solutions; pool of {len(pool)} distinct problems", flush=True)
    del candidates, seen

    # Pass 2: read the solution text of the pool only, one shard at a time.
    by_shard = defaultdict(list)
    for pos, (s, r, _) in enumerate(pool):
        by_shard[s].append((pos, r))
    texts = [None] * len(pool)
    for n, (s, items) in enumerate(sorted(by_shard.items())):
        table = pq.read_table(shards[s], columns=["generated_solution", "expected_answer", "problem_source"])
        picked = table.take([r for _, r in items]).to_pydict()
        for i, (pos, _) in enumerate(items):
            texts[pos] = (picked["generated_solution"][i], picked["expected_answer"][i], picked["problem_source"][i])
        print(f"  read solutions from {n + 1}/{len(by_shard)} shards", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = malformed = too_long = 0
    lengths, response_lengths = [], []
    with args.output.open("w", encoding="utf-8") as out:
        for (s, r, problem), (solution, answer, source) in zip(pool, texts):
            if kept >= args.num_examples:
                break
            content = strip_think(solution)
            if content is None:
                malformed += 1
                continue
            messages = [
                {"role": "user", "content": MATH_PROMPT.format(problem=problem)},
                {"role": "assistant", "content": content},
            ]
            token_ids, loss_mask = masker.get_loss_mask(messages)
            if len(token_ids) > args.max_tokens:
                too_long += 1
                continue
            lengths.append(len(token_ids))
            response_lengths.append(sum(loss_mask))
            meta = {"shard": shards[s].name, "row": r, "source": source, "answer": answer,
                    "num_tokens": len(token_ids)}
            out.write(json.dumps({"messages": messages, "metadata": meta}, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 2000 == 0:
                print(f"  kept {kept}/{args.num_examples}", flush=True)

    print(f"wrote {kept} rows -> {args.output}  (skipped: {malformed} malformed think/boxed, "
          f"{too_long} over {args.max_tokens} tokens)")
    if lengths:
        lengths.sort()
        response_lengths.sort()
        print(f"tokens/row: median {lengths[len(lengths) // 2]}  p90 {lengths[int(0.9 * len(lengths))]}  "
              f"max {lengths[-1]}  total {sum(lengths) / 1e6:.1f}M")
        print(f"supervised tokens/row: median {response_lengths[len(response_lengths) // 2]}  "
              f"mean {sum(response_lengths) / len(response_lengths):.0f}  total {sum(response_lengths) / 1e6:.1f}M")
    if kept < args.num_examples:
        print(f"WARNING: only {kept} rows, fewer than --num-examples {args.num_examples}; raise --oversample")

    if args.check_tokenizer and kept:
        with args.output.open(encoding="utf-8") as handle:
            messages = json.loads(handle.readline())["messages"]
        token_ids, loss_mask = masker.get_loss_mask(messages)
        supervised = tokenizer.decode([t for t, m in zip(token_ids, loss_mask) if m])
        masked = tokenizer.decode([t for t, m in zip(token_ids, loss_mask) if not m])
        print("\n--- masked (prompt) text ---\n" + masked)
        print("\n--- supervised text: first 300 / last 300 chars ---\n"
              + supervised[:300] + "\n  [...]\n" + supervised[-300:])
        if not masked.rstrip().endswith("</think>"):
            raise SystemExit("ERROR: the prompt does not end in the empty non-thinking <think></think> block")
        if "<think>" in supervised or "</think>" in supervised:
            raise SystemExit("ERROR: a think tag leaked into the supervised span")
        print("\nOK: non-thinking prompt, no think tags in the supervised span")


if __name__ == "__main__":
    main()
