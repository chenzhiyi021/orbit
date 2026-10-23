"""Tokenize an explicitly selected CoT bank; never select, deduplicate or silently filter rows."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from .data import read_jsonl
from .protocol import digest


def tokenize_row(row, tokenizer, policy, think_tag_policy):
    messages = row["prompt_messages"]
    response = row["response"]
    if not isinstance(response, str) or not response:
        raise ValueError("SFT source needs a nonempty complete assistant response")
    if think_tag_policy in ("strip_literal_tags_keep_content", "strip_literal_tags_keep_content_trim_outer_whitespace"):
        response = response.replace("<think>", "").replace("</think>", "")
        if think_tag_policy.endswith("_trim_outer_whitespace"):
            # "<think>\n..." leaves a leading newline that BPE-merges with the prompt's
            # trailing "\n\n" (the empty think block); trimming only outer whitespace keeps
            # every content character and the prompt tokens intact.
            response = response.strip()
    elif think_tag_policy != "preserve":
        raise ValueError("Explicit think-tag policy required")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > policy["max_prompt_tokens"]:
        raise ValueError("Prompt exceeds the approved cap; do not silently drop/trim it")
    # Only complete source answers get the natural assistant terminator.
    truncated = bool(row.get("truncated", False))
    ending = "" if truncated or response.endswith(tokenizer.eos_token) else tokenizer.eos_token
    full = tokenizer.encode(prompt + response + ending, add_special_tokens=False)
    if full[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Template/BPE boundary changed prompt tokens (source_id {row.get('source_id')}); "
                         f"response starts with {response[:20]!r}; inspect the approved formatting")
    target = full[len(prompt_ids) :]
    cap = min(policy["max_response_tokens"], policy["max_total_tokens"] - len(prompt_ids))
    if len(target) > cap:
        if policy["truncation_policy"] != "explicit_right_truncation":
            raise ValueError("CoT exceeds length policy; no automatic filtering or truncation")
        target = target[:cap]
        truncated = True
    if not target:
        raise ValueError("No supervised response tokens")
    return {"source_id": str(row["source_id"]), "prompt_ids": prompt_ids, "response_ids": target, "ended_with_eos": target[-1] == tokenizer.eos_token_id, "truncated": truncated, "response_mask": [1] * len(target)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if a.output.exists():
        raise ValueError("Prepared datasets are immutable; choose a new output")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(manifest["tokenizer"]["path"], local_files_only=True)
    rows = []
    for row in read_jsonl(a.input):
        rows.append(tokenize_row(row, tok, manifest["processing"], manifest["source"]["think_tag_policy"]))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    audit = {
        "source_sha256": digest(a.input),
        "output_sha256": digest(a.output),
        "input_rows": len(rows),
        "output_rows": len(rows),
        "dropped_rows": 0,
        "truncated_rows": sum(r["truncated"] for r in rows),
        "complete_answer_coverage": sum(not r["truncated"] for r in rows) / len(rows) if rows else 0,
        "processing": manifest["processing"],
        "think_tag_policy": manifest["source"]["think_tag_policy"],
    }
    a.output.with_suffix(".processing.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
