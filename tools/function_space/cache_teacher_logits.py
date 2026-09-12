"""Cache teacher logits over a fixed prefix bank, for teacher-RKL/FKL scalars.

Ported from ``unifying_posttrain/experiments/m5_m6_param_native_trajectory/cache_teacher_logits.py``.
Only change: the hardcoded ``/data/people/yhuang/...`` default teacher path was
removed -- ``--model`` is now required. Everything else (batching, position
selection, fp16-on-disk memmap) is unchanged.

Usage:
    python cache_teacher_logits.py --bank /path/to/bank_a --model Qwen/Qwen3-4B-Instruct-2507 \\
        --output cache/teacher/bank_a.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--label", default="teacher")
    args = parser.parse_args()
    dataset = load_from_disk(str(args.bank))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(args.device)
    model.eval().requires_grad_(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    array = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(len(dataset), 64, model.config.vocab_size),
    )
    for start in range(0, len(dataset), args.batch_size):
        records = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
        max_length = max(len(record["input_ids"]) for record in records)
        input_ids = torch.zeros(len(records), max_length, dtype=torch.long, device=args.device)
        attention = torch.zeros_like(input_ids)
        for batch_index, record in enumerate(records):
            ids = torch.tensor(record["input_ids"], dtype=torch.long, device=args.device)
            input_ids[batch_index, : ids.numel()] = ids
            attention[batch_index, : ids.numel()] = 1
        hidden = model.model(input_ids=input_ids, attention_mask=attention, use_cache=False).last_hidden_state
        for batch_index, record in enumerate(records):
            prompt_length = len(record["prompt_ids"])
            positions = torch.tensor(
                [prompt_length + value - 1 for value in record["selected_positions"]],
                dtype=torch.long,
                device=args.device,
            )
            logits = model.lm_head(hidden[batch_index].index_select(0, positions))
            array[start + batch_index] = logits.float().cpu().numpy().astype(np.float16)
        array.flush()
        print(f"{args.label} cache {min(start + args.batch_size, len(dataset))}/{len(dataset)}", flush=True)
    del array
    temporary.rename(args.output)
    metadata = {
        "model": str(args.model),
        "label": args.label,
        "bank": str(args.bank),
        "shape": [len(dataset), 64, model.config.vocab_size],
        "dtype": "float16 logits before temperature scaling",
        "diagnostic_temperature": 0.7,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
