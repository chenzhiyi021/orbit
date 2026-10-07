"""Rejection-sample teacher SFT data against the M9 verifier gold answers.

Takes the jsonl written by ``tools/generate_teacher_sft_data.py`` and:

1. drops responses that hit the teacher's generation cap (truncated, no EOS);
2. for rows whose ``metadata.prompt_sha256`` has a gold answer in the M9 verifier
   set (``m9-verifier-38k-aligned``), keeps the row only if
   ``unified_exact_answer_reward`` -- the exact reward M9 GRPO trained with --
   scores it 1. Rows without a gold are kept unfiltered, except in
   ``--require-gold-domains``, where they are dropped so every kept row of those
   domains is verified (code has no gold at all, so it is never listed there);
3. samples a domain-balanced subset of ``--total`` rows (the smallest domain is
   taken whole, the leftover quota is split across the others) and shuffles it.

    python tools/filter_sft_by_verifier.py \
        --sft-jsonl data/sft/openreasoning_mixed_100k/train.jsonl \
        --verifier-data /mnt/L202500431/datasets/m9-verifier-38k-aligned/data/train-00000-of-00001.parquet \
        --tokenizer /mnt/L202500430/orbit/data/hf_ckpts/Qwen3-1.7B \
        --require-gold-domains math science         --total 25600         --output data/sft/openreasoning_mixed_100k/train_rs_bal25600.jsonl

Writes ``<output>.stats.json`` next to the output with per-domain counts.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path


_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _score(task: tuple[int, str, str, str]) -> tuple[int, float | None]:
    from orbit.rollout.rm_hub.unified_exact_answer import unified_exact_answer_reward

    line_no, response, solution, answer_type = task
    (reward,) = unified_exact_answer_reward([response], [solution], [answer_type])
    return line_no, reward


def _balanced_sample(by_domain: dict[str, list[str]], total: int) -> dict[str, int]:
    """Smallest domain first; its unused share of the quota flows to the larger ones."""
    quota, left = {}, total
    domains = sorted(by_domain, key=lambda d: len(by_domain[d]))
    for i, domain in enumerate(domains):
        quota[domain] = min(len(by_domain[domain]), left // (len(domains) - i))
        left -= quota[domain]
    return quota


def main(args: argparse.Namespace) -> None:
    import pandas as pd
    from transformers import AutoTokenizer

    from orbit.rollout.rm_hub.unified_exact_answer import is_math_verify_available

    if not is_math_verify_available():
        raise SystemExit("math_verify is required (pip install math-verify); it is the M9 verifier backend.")

    gold_df = pd.read_parquet(args.verifier_data)
    gold = {h: (s, t) for h, s, t in zip(gold_df["prompt_sha256"], gold_df["solution"], gold_df["answer_type"])}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    lines = [line for line in args.sft_jsonl.open(encoding="utf-8") if line.strip()]
    records = [json.loads(line) for line in lines]
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    responses = [r["messages"][-1]["content"] for r in records]
    lengths = [len(ids) for ids in tokenizer(responses, add_special_tokens=False)["input_ids"]]

    tasks, keep = [], {}
    for i, (record, length) in enumerate(zip(records, lengths)):
        domain = record["metadata"].get("domain")
        stats[domain]["input"] += 1
        if length >= args.max_response_tokens:
            stats[domain]["drop_truncated"] += 1
            continue
        hit = gold.get(record["metadata"].get("prompt_sha256"))
        if hit is None:
            if domain in args.require_gold_domains:
                stats[domain]["drop_no_gold"] += 1
                continue
            stats[domain]["keep_no_gold"] += 1
            keep[i] = True
            continue
        tasks.append((i, responses[i], hit[0], hit[1]))

    with Pool(args.workers) as pool:
        for i, reward in pool.imap_unordered(_score, tasks, chunksize=64):
            domain = records[i]["metadata"].get("domain")
            if reward is None:  # gold unparseable by the verifier: cannot judge, keep
                stats[domain]["keep_gold_unscorable"] += 1
                keep[i] = True
            elif reward >= 1.0:
                stats[domain]["keep_verified"] += 1
                keep[i] = True
            else:
                stats[domain]["drop_wrong"] += 1

    by_domain: dict[str, list[str]] = collections.defaultdict(list)
    for i in sorted(keep):
        by_domain[records[i]["metadata"].get("domain")].append(lines[i] if lines[i].endswith("\n") else lines[i] + "\n")

    rng = random.Random(args.seed)
    available = sum(len(v) for v in by_domain.values())
    total = min(args.total, available)
    quota = _balanced_sample(by_domain, total)
    out = []
    for domain in sorted(by_domain):
        rows = by_domain[domain]
        rng.shuffle(rows)
        out += rows[: quota[domain]]
        stats[domain]["output"] = quota[domain]
    rng.shuffle(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(out), encoding="utf-8")
    summary = {
        "requested_total": args.total,
        "available_after_filter": available,
        "output_total": len(out),
        "per_domain": {d: dict(c) for d, c in sorted(stats.items())},
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    Path(f"{args.output}.stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if len(out) < args.total:
        print(f"WARNING: only {len(out)} rows available, fewer than --total {args.total}; training will wrap epochs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft-jsonl", type=Path, required=True)
    parser.add_argument("--verifier-data", type=Path, required=True, help="M9 verifier parquet with gold answers.")
    parser.add_argument("--tokenizer", required=True, help="Student tokenizer, used to detect truncated responses.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=8180,
        help="Responses at least this long are treated as truncated by the teacher's --max-tokens 8192.",
    )
    parser.add_argument(
        "--require-gold-domains",
        nargs="*",
        default=[],
        help="Domains whose rows are dropped when no gold answer exists (e.g. math science).",
    )
    parser.add_argument("--total", type=int, default=25_600, help="Rows to emit; 100 steps x global batch 256.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
