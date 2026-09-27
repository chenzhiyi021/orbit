"""Turn euclaise/WritingPrompts_curated into single-turn chat prompts for OPD training.

The dataset has one row per (r/WritingPrompts post, highly-voted story) pair:
    prompt          the post title, tag already stripped ("Write a story about a hero gone bad")
    body            a human story (unused: OPD's training signal is the teacher, not references)
    post_score / comment_score   Reddit votes

A post with several curated stories appears on several rows, so prompts are de-duplicated.
Each unique prompt becomes {"messages": [{"role": "user", "content": ...}]} -- one user turn,
no system prompt, the same shape as the openreasoning / RLVR-IFeval prompts -- so it drops
into the existing launchers with --input-key messages. Any leftover "[WP]"-style tag is
stripped; prompts outside the length bounds are dropped; rows are shuffled with a fixed seed.

    python tools/convert_writingprompts_curated_to_opd.py \\
        --input /mnt/L202500431/datasets/WritingPrompts_curated \\
        --output /mnt/L202500431/datasets/WritingPrompts_curated_opd/train.parquet
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

TAG_RE = re.compile(r"^\s*[\[\(]\s*[A-Za-z]{1,6}\s*[\]\)]\s*")
INSTRUCTION = "Write an original piece of creative writing that responds to the following writing prompt."


def load(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted([*path.rglob("*.parquet"), *path.rglob("*.jsonl")])
        if not files:
            raise FileNotFoundError(f"no .parquet/.jsonl under {path}")
        return pd.concat([load(f) for f in files], ignore_index=True)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_json(path, lines=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="WritingPrompts_curated file or directory")
    parser.add_argument("--output", required=True, type=Path, help="output .parquet")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--min-prompt-chars", type=int, default=5)
    parser.add_argument("--max-prompt-chars", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0, help="keep at most N prompts (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load(args.input)
    kept, seen = [], set()
    dropped = {"length": 0, "duplicate": 0}
    for raw in frame[args.prompt_key].fillna(""):
        prompt = re.sub(r"\s+", " ", TAG_RE.sub("", str(raw), count=1)).strip()
        if not args.min_prompt_chars <= len(prompt) <= args.max_prompt_chars:
            dropped["length"] += 1
            continue
        key = re.sub(r"\W+", " ", prompt.lower()).strip()
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append({"messages": [{"role": "user", "content": f"{INSTRUCTION}\n\nPrompt: {prompt}"}],
                     "prompt": prompt})

    out = pd.DataFrame(kept).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.limit > 0:
        out = out.head(args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output)
    print(f"{len(frame)} rows -> {len(out)} unique prompts -> {args.output}")
    print(f"dropped: {dropped}")
    for message in out["messages"].head(3):
        print("\n---\n" + message[0]["content"])


if __name__ == "__main__":
    main()
