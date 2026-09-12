"""Build a fixed prefix bank: base-model rollout + normalized position selection.

Not part of the trl-side snapshot as a single script -- the trl `prepare.py`
only *re-derives* an already-built bank from pre-existing rollout artifacts on
that cluster. This script does the actual rollout-generation step, using the
same position-selection rule (``normalized_positions``, "primary" panel: 16
positions evenly spaced over the first quarter of the completion, 32 over the
middle half, 16 over the last quarter, capped at the first 512 completion
tokens) copied verbatim from
``unifying_posttrain/experiments/geometry_kill_tests/run_a1_local_field.py``.
Rows shorter than 64 completion tokens are dropped (same rule: empty position
list below length 64).

Output schema (one HF ``datasets`` row per eligible prompt) matches exactly
what ``scoring.score_bank`` reads: ``input_ids``, ``prompt_ids``,
``selected_positions``, ``selected_index``, ``prompt_sha256``, ``domain``.

Usage (parquet source, e.g. an OPD training-data file):
    python build_prefix_bank.py --model /path/to/base_model \\
        --prompts /path/to/openreasoning_mixed_100k/train.parquet \\
        --prompt-field messages --chat --domain math --num-prompts 128 \\
        --seed 20260907 --output bank_a

Usage (HF dataset_name/config, e.g. gsm8k main split):
    python build_prefix_bank.py --model /path/to/base_model \\
        --prompts gsm8k --prompts-config main --prompts-split test \\
        --prompt-field question --domain math --num-prompts 128 \\
        --seed 20260907 --output bank_b

Run `--dry-run` first (loads the prompt source and prints the first row) to
confirm --prompt-field / --chat before spending GPU time on generation --
I could not verify the column names of `openreasoning_mixed_100k` or `gsm8k`
on your cluster from here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalized_positions_primary(length: int) -> list[int]:
    """Verbatim port of ``normalized_positions(length, "primary")`` from
    geometry_kill_tests/run_a1_local_field.py. 16 + 32 + 16 = 64 positions,
    normalized over the first min(length, 512) completion tokens. Returns []
    (row is dropped) if length < 64."""
    if length < 64:
        return []
    window = min(length, 512)
    bounds = (0, window // 4, 3 * window // 4, window)
    counts = (16, 32, 16)
    result: list[int] = []
    for start, stop, count in zip(bounds[:-1], bounds[1:], counts, strict=True):
        result.extend(round(start + i * (stop - start - 1) / (count - 1)) for i in range(count))
    return result


def sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_prompt_rows(args) -> list[dict]:
    path = Path(args.prompts)
    if path.suffix == ".parquet":
        import pandas as pd
        frame = pd.read_parquet(path)
        rows = frame.to_dict("records")
    elif path.is_dir():
        rows = list(load_from_disk(str(path)))
    else:
        # Treat as an HF hub dataset name (e.g. "gsm8k").
        dataset = load_dataset(args.prompts, args.prompts_config, split=args.prompts_split)
        rows = list(dataset)
    if args.prompt_field not in rows[0]:
        raise KeyError(f"--prompt-field {args.prompt_field!r} not found; available columns: {sorted(rows[0])}")
    return rows


def parse_messages(value):
    """Accept an already-parsed list, valid JSON, or a Python repr() string
    (single-quoted dicts, e.g. pandas/parquet round-tripped a list-of-dict
    column as its str() -- seen on openreasoning_mixed_100k's `messages`
    column)."""
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        import ast
        return ast.literal_eval(value)


def prompt_ids_for(row: dict, args, tokenizer) -> list[int]:
    value = row[args.prompt_field]
    if args.chat:
        messages = parse_messages(value)
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
    text = str(value)
    if args.chat_wrap:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
    return tokenizer(text, add_special_tokens=True)["input_ids"]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="base model to roll out from (this is your Base for scoring, too -- must match config.json's base_model)")
    parser.add_argument("--prompts", required=True, help="parquet path, HF datasets save_to_disk dir, or HF hub dataset name")
    parser.add_argument("--prompts-config", default=None, help="HF hub dataset config, e.g. 'main' for gsm8k")
    parser.add_argument("--prompts-split", default="train")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--chat", action="store_true", help="--prompt-field holds a chat messages list (or JSON string of one)")
    parser.add_argument("--chat-wrap", action="store_true", help="--prompt-field holds a raw string; wrap it as a single user turn via the chat template (use for gsm8k-style plain-text prompts on a chat/instruct base)")
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--domain", default=None, help="fixed domain label written onto every row of this bank; "
                         "if omitted, uses the source dataset's own 'domain' column (required if it has none)")
    parser.add_argument("--filter-domain", default=None, help="keep only source rows whose own 'domain' column "
                         "equals this value (applied before --domain, which still overrides what gets *written*)")
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true", help="load the prompt source, print row 0, and exit -- no model load, no generation")
    parser.add_argument("--print-samples", type=int, default=0,
                         help="decode and print the prompt + generated completion for the first N kept rows "
                              "(and whether they were long enough to keep, for dropped rows too) -- use this for "
                              "a smoke test before spending time/tokens on a full run")
    args = parser.parse_args()

    rows = load_prompt_rows(args)
    print(f"loaded {len(rows)} candidate prompts from {args.prompts}; first row keys: {sorted(rows[0])}", flush=True)
    if args.dry_run:
        print(json.dumps({k: (v if not isinstance(v, (list, dict)) else str(v)[:500]) for k, v in rows[0].items()}, indent=2, default=str))
        return

    if args.filter_domain:
        before = len(rows)
        rows = [r for r in rows if r.get("domain") == args.filter_domain]
        print(f"--filter-domain {args.filter_domain!r}: kept {len(rows)}/{before} candidates", flush=True)
    if args.domain is None and "domain" not in rows[0]:
        raise ValueError("Source has no 'domain' column and --domain was not given; pass --domain to label this bank.")

    random.Random(args.seed).shuffle(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", trust_remote_code=True,
    ).eval().requires_grad_(False).to(args.device)

    kept: list[dict] = []
    printed = 0
    torch.manual_seed(args.seed)
    for start in range(0, len(rows), args.batch_size):
        if len(kept) >= args.num_prompts:
            break
        batch = rows[start : start + args.batch_size]
        prompt_id_lists = [prompt_ids_for(row, args, tokenizer) for row in batch]
        max_len = max(len(p) for p in prompt_id_lists)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=args.device)
        attention = torch.zeros_like(input_ids)
        for i, ids in enumerate(prompt_id_lists):
            # left-pad so generation starts at the same column for every row
            input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention[i, max_len - len(ids):] = 1
        generated = model.generate(
            input_ids=input_ids, attention_mask=attention, max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0, temperature=max(args.temperature, 1e-6), top_p=args.top_p,
            pad_token_id=pad_id,
        )
        for i, (row, prompt_ids) in enumerate(zip(batch, prompt_id_lists, strict=True)):
            completion_ids = generated[i, max_len:].tolist()
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in completion_ids:
                completion_ids = completion_ids[: completion_ids.index(tokenizer.eos_token_id)]
            positions = normalized_positions_primary(len(completion_ids))
            if printed < args.print_samples:
                printed += 1
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
                status = "KEPT" if positions else f"DROPPED (completion={len(completion_ids)} tokens, need >=64)"
                print(f"\n{'=' * 80}\n[{printed}/{args.print_samples}] {status}\n"
                      f"--- prompt (decoded, includes chat-template special tokens) ---\n{prompt_text}\n"
                      f"--- completion ({len(completion_ids)} tokens) ---\n{completion_text}\n{'=' * 80}", flush=True)
            if not positions:
                continue
            selected_index = len(kept)
            kept.append({
                "input_ids": prompt_ids + completion_ids[: max(positions) + 1],
                "prompt_ids": prompt_ids,
                "selected_positions": positions,
                "selected_index": selected_index,
                "prompt_sha256": sha256(prompt_ids),
                "domain": args.domain if args.domain is not None else row["domain"],
                "completion_length": len(completion_ids),
            })
            if len(kept) >= args.num_prompts:
                break
        print(f"kept {len(kept)}/{args.num_prompts} (scanned {min(start + args.batch_size, len(rows))}/{len(rows)} candidates)", flush=True)

    if len(kept) < args.num_prompts:
        raise RuntimeError(
            f"Only found {len(kept)}/{args.num_prompts} prompts with completions >=64 tokens after scanning "
            f"all {len(rows)} candidates -- pass more --num-prompts headroom in --prompts, or lower --num-prompts."
        )
    Dataset.from_list(kept).save_to_disk(args.output)
    manifest = {
        "model": args.model, "prompts": args.prompts, "domain": args.domain, "seed": args.seed,
        "num_prompts": len(kept), "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_p": args.top_p, "content_sha256": sha256([r["prompt_sha256"] for r in kept]),
    }
    Path(args.output, "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
