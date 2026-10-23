"""Held-out Dev NLL of SFT adapters: the LR-calibration metric (never a benchmark score).

Teacher-forced NLL (T=1, raw logits) of the Dev bank's official-CoT responses, i.e. the SFT loss on
questions no run trained on. Reported as the global token mean (the training reducer) and per
domain when ``--meta`` (select_cot output for the Dev bank) is given. Logits are computed in
sequence chunks so 40k-token CoTs fit on one GPU.

    PYTHONNOUSERSITE=1 python -m orbit.research.posttrain.dev_nll \\
        --base /mnt/.../qwen3-1.7b --tokens /mnt/.../calibration/dev_sft_tokens.jsonl \\
        --meta /mnt/.../calibration/dev_selected_cot.jsonl \\
        --adapter base=none --adapter lr2e-4=/mnt/.../runs/lr2e-4/iter_0000099/adapter \\
        --output /mnt/.../calibration/dev_nll.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from .data import read_jsonl


@torch.no_grad()
def nll(model, rows, chunk):
    """sum of -log p(response token) and token count per row (positions predict the next token)."""
    core, head = model.model, model.lm_head
    out = []
    for r in rows:
        ids = torch.tensor([r["prompt_ids"] + r["response_ids"]], device="cuda")
        hidden = core(input_ids=ids).last_hidden_state[0]
        start, end = len(r["prompt_ids"]) - 1, ids.shape[1] - 1
        targets = ids[0, start + 1:end + 1]
        mask = torch.tensor(r.get("response_mask", [1] * len(r["response_ids"])), device="cuda", dtype=torch.bool)
        total = 0.0
        for s in range(start, end, chunk):
            e = min(s + chunk, end)
            logp = torch.log_softmax(head(hidden[s:e]).float(), -1)
            tok = logp.gather(-1, targets[s - start:e - start, None]).squeeze(-1)
            total -= float(tok[mask[s - start:e - start]].sum())
        out.append((total, int(mask.sum())))
        del hidden
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--tokens", required=True, type=Path, help="prepare_sft output for the Dev bank")
    p.add_argument("--meta", type=Path, help="select_cot output for the Dev bank (per-domain breakdown)")
    p.add_argument("--adapter", action="append", required=True, metavar="NAME=PATH|none")
    p.add_argument("--chunk", type=int, default=4096, help="positions per lm_head chunk")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    rows = list(read_jsonl(a.tokens))
    domain = {}
    if a.meta:
        domain = {r["source_id"]: r["metadata"]["domain"] for r in read_jsonl(a.meta)}
    results = {}
    for item in a.adapter:
        name, _, path = item.partition("=")
        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda().eval()
        if path != "none":
            model = PeftModel.from_pretrained(model, path).merge_and_unload().eval()
        per_row = nll(model, rows, a.chunk)
        tot, n = sum(x for x, _ in per_row), sum(c for _, c in per_row)
        by = defaultdict(lambda: [0.0, 0])
        for r, (x, c) in zip(rows, per_row):
            d = domain.get(r["source_id"], "all")
            by[d][0] += x
            by[d][1] += c
        results[name] = {"adapter": path, "token_mean_nll": tot / n, "token_ppl": math.exp(tot / n), "tokens": n,
                         "sequence_mean_nll": sum(x / c for x, c in per_row) / len(per_row),
                         "by_domain_token_mean_nll": {d: v[0] / v[1] for d, v in sorted(by.items())}}
        print(f"{name:<16} token-mean NLL {tot / n:.4f}  " +
              "  ".join(f"{d}={v[0] / v[1]:.4f}" for d, v in sorted(by.items())), flush=True)
        del model
        torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"tokens_file": str(a.tokens), "rows": len(rows), "results": results}, indent=2) + "\n")
    best = min((k for k in results if results[k]["adapter"] != "none"), key=lambda k: results[k]["token_mean_nll"], default=None)
    print(f"lowest Dev NLL: {best}")


if __name__ == "__main__":
    main()
