"""Attach upstream official CoT traces to the sealed B_target prompts, keeping B_target's order.

Output rows are prepare_sft.py input (``source_id, prompt_messages, response``) in exactly the
B_target row order, so the offline SFT provider (no shuffle) trains them in that order.

    B_target row i --canonical_row--> openreasoning_mixed_100k row --upstream key--> trace

* B_target prompt text must equal the mixed row's user message; with ``--tokenizer`` the
  non-thinking chat template of ``prompt_messages`` must reproduce B_target's prompt and
  prompt_ids exactly, so prepare_sft emits the same prompt tokens M6 trained on.
* Upstream keys: math -> OpenMathReasoning cot ``problem``; science -> OpenScienceReasoning-2
  ``input`` (sha256 = mixed source_id); code -> OpenCodeReasoning split_0 ``id``.
* A prompt that occurs k times (B_target keeps duplicate prompts) takes its upstream traces in a
  fixed order (math: ``--math-model`` first, then shard/row order); when the upstream has fewer
  traces than occurrences they are reused cyclically and flagged. Nothing is dropped: a row with
  no upstream trace is an error.
* The response is the raw upstream text (think tags kept); prepare_sft's manifest
  ``think_tag_policy`` decides what to do with them.

    PYTHONNOUSERSITE=1 python -m orbit.research.posttrain.select_cot \\
        --b-target /mnt/.../B_target.jsonl \\
        --mixed /mnt/.../openreasoning_mixed_100k/train.parquet \\
        --omr-dir /mnt/.../OpenMathReasoning/data \\
        --osr-dir /mnt/.../raw/OpenScienceReasoning-2 \\
        --ocr-dir /mnt/.../raw/OpenCodeReasoning/split_0 \\
        --tokenizer /mnt/.../qwen3-1.7b --output /mnt/.../b_target_cot/selected_cot.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import read_jsonl
from .protocol import M6_DATA_SHA256, digest

USER_PREFIX, ASSISTANT_PREFIX = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"
UPSTREAM = {"math": "nvidia/OpenMathReasoning", "science": "nvidia/OpenScienceReasoning-2",
            "code": "nvidia/OpenCodeReasoning"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def user_text(prompt: str) -> str:
    if not prompt.startswith(USER_PREFIX) or prompt.count(ASSISTANT_PREFIX) != 1:
        raise ValueError("B_target prompt is not a single user turn followed by the assistant header")
    return prompt[len(USER_PREFIX):prompt.index(ASSISTANT_PREFIX)]


def scan(paths, columns, keep):
    """Yield (shard, row, values) for rows whose key columns pass ``keep``. Works one row group
    at a time and reads the (large) trace column only for matching rows, so a 15 GB single-file
    upstream never has to fit in memory."""
    import pyarrow.parquet as pq

    key_cols, trace_col = columns[:-1], columns[-1]
    for shard, path in enumerate(paths):
        pf, offset, hits = pq.ParquetFile(path), 0, 0
        for g in range(pf.num_row_groups):
            table = pf.read_row_group(g, columns=key_cols)
            cols = [table.column(c).to_pylist() for c in key_cols]
            hit = [r for r, values in enumerate(zip(*cols)) if keep(*values)]
            if hit:
                # Pick rows chunk by chunk: ``take`` concatenates chunks and overflows the 2 GB
                # offsets of a plain ``string`` column on OpenScienceReasoning-2's big row groups.
                chunks = pf.read_row_group(g, columns=[trace_col]).column(trace_col).chunks
                starts = [0]
                for chunk in chunks:
                    starts.append(starts[-1] + len(chunk))
                traces, c = [], 0
                for r in hit:
                    while r >= starts[c + 1]:
                        c += 1
                    traces.append(chunks[c][r - starts[c]].as_py())
                for r, trace in zip(hit, traces):
                    yield shard, offset + r, [c[r] for c in cols] + [trace]
            offset += table.num_rows
            hits += len(hit)
        print(f"  {Path(path).name}: {hits} matching rows", flush=True)


def collect(args, needed):
    """needed[domain] = set of upstream keys -> traces[(domain, key)] = [(trace, model), ...]."""
    traces = defaultdict(list)
    if needed["math"]:
        paths = sorted(Path(args.omr_dir).glob("cot-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no cot-*.parquet under {args.omr_dir}")
        found = []
        for shard, r, (problem, model, solution) in scan(paths, ["problem", "generation_model", "generated_solution"],
                                                          lambda p, m: p in needed["math"]):
            found.append(((model != args.math_model, shard, r), problem, solution, model))
        for _, problem, solution, model in sorted(found, key=lambda x: x[0]):
            traces[("math", problem)].append((solution, model))
    if needed["science"]:
        paths = sorted(Path(args.osr_dir).rglob("*.parquet"))
        for _, _, (inp, output) in scan(paths, ["input", "output"], lambda i: sha256(i) in needed["science"]):
            traces[("science", sha256(inp))].append((output, "DeepSeek-R1-0528"))
    if needed["code"]:
        paths = sorted(Path(args.ocr_dir).glob("*.parquet"))
        for _, _, (_id, output) in scan(paths, ["id", "output"], lambda i: i in needed["code"]):
            traces[("code", _id)].append((output, "DeepSeek-R1"))
    return traces


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--b-target", required=True, type=Path)
    p.add_argument("--mixed", required=True, type=Path, help="openreasoning_mixed_100k train.parquet")
    p.add_argument("--omr-dir", type=Path, help="OpenMathReasoning data/ (cot-*.parquet)")
    p.add_argument("--osr-dir", type=Path, help="OpenScienceReasoning-2 download")
    p.add_argument("--ocr-dir", type=Path, help="OpenCodeReasoning split_0/")
    p.add_argument("--math-model", default="DeepSeek-R1", help="preferred OMR generation_model")
    p.add_argument("--tokenizer", help="verify the chat template reproduces B_target prompt/prompt_ids")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Selected CoT banks are immutable; choose a new output")

    import pandas as pd

    b_sha = digest(a.b_target)
    rows = list(read_jsonl(a.b_target))
    mixed = pd.read_parquet(a.mixed)
    plan, needed = [], {"math": set(), "science": set(), "code": set()}
    for i, row in enumerate(rows):
        text = user_text(row["prompt"])
        m = mixed.iloc[row["canonical_row"]]
        if m["messages"][0]["content"] != text:
            raise ValueError(f"B_target row {i}: canonical_row {row['canonical_row']} is a different prompt in {a.mixed}")
        domain = m["domain"]
        key = sha256(text) if domain == "science" else (text if domain == "math" else m["source_id"])
        if domain == "science" and key != m["source_id"]:
            raise ValueError(f"B_target row {i}: science prompt sha256 != mixed source_id")
        needed[domain].add(key)
        plan.append((i, row, text, domain, key))
    print(f"{len(rows)} B_target rows: " + ", ".join(f"{d}={len(k)} unique" for d, k in needed.items()), flush=True)
    for domain, flag in [("math", a.omr_dir), ("science", a.osr_dir), ("code", a.ocr_dir)]:
        if needed[domain] and flag is None:
            raise ValueError(f"B_target has {domain} rows; pass its upstream directory")

    if a.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        for i, row, text, _, _ in plan:
            rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            if rendered != row["prompt"] or tok.encode(rendered, add_special_tokens=False) != row["prompt_ids"]:
                raise ValueError(f"B_target row {i}: chat template does not reproduce its prompt/prompt_ids")
        print("tokenizer check: every prompt_messages renders to B_target's prompt and prompt_ids", flush=True)

    traces = collect(a, needed)
    missing = [(d, k) for d, keys in needed.items() for k in keys if not traces[(d, k)]]
    if missing:
        raise ValueError(f"{len(missing)} prompts have no upstream trace, e.g. {missing[:3]}")

    used, out, stats = Counter(), [], Counter()
    for i, row, text, domain, key in plan:
        pool = traces[(domain, key)]
        k = used[(domain, key)]
        used[(domain, key)] += 1
        response, model = pool[k % len(pool)]
        reused = k >= len(pool)
        stats[f"{domain}_rows"] += 1
        stats[f"{domain}_reused_trace_rows"] += reused
        stats[f"model={model}"] += 1
        out.append({"source_id": f"b{i}_c{row['canonical_row']}",
                    "prompt_messages": [{"role": "user", "content": text}], "response": response,
                    "metadata": {"b_target_index": i, "canonical_row": row["canonical_row"], "domain": domain,
                                 "upstream_dataset": UPSTREAM[domain], "trace_index": k % len(pool),
                                 "trace_reused": reused, "trace_model": model,
                                 "prompt_sha256": row["prompt_sha256"]}})

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        for r in out:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    audit = {"b_target": str(a.b_target), "b_target_sha256": b_sha, "b_target_is_sealed_m6": b_sha == M6_DATA_SHA256,
             "mixed_sha256": digest(a.mixed), "rows": len(out), "output_sha256": digest(a.output),
             "tokenizer_checked": bool(a.tokenizer), "math_model_preference": a.math_model,
             "unique_prompts": {d: len(k) for d, k in needed.items()}, "stats": dict(stats),
             "response_chars": {"min": min(len(r["response"]) for r in out),
                                "max": max(len(r["response"]) for r in out),
                                "mean": sum(len(r["response"]) for r in out) / len(out)}}
    a.output.with_suffix(".selection.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
