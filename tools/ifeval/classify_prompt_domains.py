"""Classify the task domain of chat prompts (e.g. RLVR-IFeval) with an LLM judge.

Question this answers: how much of an "instruction-following" training set is really math
(or other reasoning) underneath its formatting constraint? RLVR-IFeval's prompts come from
the Tulu 2 SFT mixture -- which includes FLAN chain-of-thought data such as GSM8K / AQuA --
with one IFEval constraint appended, and the dataset keeps no per-row source label.

Each prompt's user text is sent to the judge (non-thinking, greedy) with an instruction to
ignore formatting constraints and name the underlying task's category. Writes one JSONL row
per prompt ({"index", "category", "raw", "prompt"}) and prints the category counts.

Run where sglang is available (e.g. orbit_env_v2):

    python tools/ifeval/classify_prompt_domains.py \\
        --data /mnt/L202500431/datasets/RLVR-IFeval/data/train-00000-of-00001.parquet \\
        --judge /mnt/L202500431/models/qwen3-4b-instruct-2507 \\
        --output /mnt/L202500431/datasets/RLVR-IFeval/domain_labels.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CATEGORIES = ["math", "code", "logic_reasoning", "science", "knowledge_qa", "writing", "other"]

JUDGE_PROMPT = """You are labeling the task domain of a user request for dataset analysis.
The request may end or begin with a formatting instruction (e.g. word counts, lowercase,
number of paragraphs, JSON output). IGNORE any such formatting instruction and classify
only the underlying task.

Categories:
- math: requires mathematical calculation or solving a math / arithmetic / word problem
- code: writing, explaining or debugging code
- logic_reasoning: non-mathematical reasoning (logic puzzles, NLI, commonsense multi-step inference)
- science: explaining or answering science questions without substantial calculation
- knowledge_qa: factual questions, reading comprehension, classification of text
- writing: essays, stories, emails, rewriting, summarization, creative writing
- other: anything else

Request:
<<<
{prompt}
>>>

Answer with exactly one category name from the list and nothing else."""


def parse_label(raw: str) -> str:
    text = raw.strip().lower()
    for category in CATEGORIES:
        if text.startswith(category):
            return category
    for category in CATEGORIES:
        if category in text:
            return category
    return "unparsed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="parquet or jsonl with a chat 'messages' column")
    parser.add_argument("--judge", required=True, help="HF checkpoint of the judge model")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--messages-key", default="messages")
    parser.add_argument("--limit", type=int, default=0, help="classify only the first N prompts (0 = all)")
    parser.add_argument("--max-prompt-chars", type=int, default=6000, help="truncate very long prompts")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mem-fraction", type=float, default=0.8)
    args = parser.parse_args()

    import pandas as pd
    import sglang as sgl
    from transformers import AutoTokenizer

    frame = pd.read_parquet(args.data) if args.data.suffix == ".parquet" else pd.read_json(args.data, lines=True)
    if args.limit > 0:
        frame = frame.head(args.limit)
    prompts = [
        "\n".join(str(m["content"]) for m in messages if m["role"] == "user")
        for messages in frame[args.messages_key]
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.judge, trust_remote_code=True)
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": JUDGE_PROMPT.format(prompt=p[: args.max_prompt_chars])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p in prompts
    ]

    engine = sgl.Engine(model_path=args.judge, tp_size=args.tp, mem_fraction_static=args.mem_fraction)
    try:
        outputs = engine.generate(rendered, {"temperature": 0.0, "max_new_tokens": 8})
    finally:
        engine.shutdown()

    counts: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for index, (prompt, out) in enumerate(zip(prompts, outputs)):
            label = parse_label(out["text"])
            counts[label] += 1
            f.write(json.dumps({"index": index, "category": label, "raw": out["text"], "prompt": prompt},
                               ensure_ascii=False) + "\n")

    total = sum(counts.values())
    print(f"\n{total} prompts classified -> {args.output}")
    for category, n in counts.most_common():
        print(f"  {category:16s} {n:6d}  ({n / total:.1%})")


if __name__ == "__main__":
    main()
