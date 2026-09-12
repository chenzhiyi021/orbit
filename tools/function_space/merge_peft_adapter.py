"""Merge an HF-PEFT adapter (LoRA or OFT) onto its base model.

Orbit's Megatron PEFT training exports a standard HF-PEFT-format adapter
sidecar under ``iter_XXXXXXX/adapter/`` (``adapter_config.json`` +
``adapter_model.safetensors``, already gathered across tensor-parallel
ranks -- the ``adapter_megatron_tp*_pp*.pt`` files next to it are the raw
per-rank shards used internally by Megatron and are not needed here).

The main ``iter_XXXXXXX/`` checkpoint next to ``adapter/`` is the frozen
base and is *not* what you want to feed to ``convert_torch_dist_to_hf.py``
for a PEFT run -- it does not reflect the adapter's training at all.

This script loads the base model, applies the adapter via the ``peft``
library, calls ``merge_and_unload()`` to bake the LoRA/OFT delta into plain
dense weights, and saves a checkpoint ``score_checkpoints.py`` can load
directly with ``AutoModelForCausalLM.from_pretrained`` -- no different from
a FullFT checkpoint at that point.

Requires the ``peft`` package (``pip install peft`` if missing).

Usage:
    python merge_peft_adapter.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --adapter /mnt/L202500431/models/zh_models/m6_st_lora_non_thinking/iter_0000299/adapter \\
        --output /mnt/L202500431/models/zh_models/m6_st_lora_non_thinking_hf_step300
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

# peft's LoRA dispatch probes is_torchao_available() to decide whether to use
# a torchao-quantized LoRA layer variant. On some installs that probe itself
# raises (rather than returning False) if an incompatible torchao version is
# present, even though we never touch torchao -- we're merging into plain
# bf16 tensors. Patch it to a hard False so the probe can't blow up; this
# only affects this process, not the installed torchao package.
try:
    import peft.tuners.lora.torchao as _lora_torchao
    _lora_torchao.is_torchao_available = lambda: False
except ImportError:
    pass

# Same asset list convert_torch_dist_to_hf.py copies from --origin-hf-dir:
# tokenizer + misc config files that aren't part of the merged weight files
# themselves, so plain transformers/vLLM/SGLang tooling can load the result.
TOKENIZER_ASSETS = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "generation_config.json",
    "LICENSE", "README.md",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="base model dir (what the adapter was trained on top of)")
    parser.add_argument("--adapter", required=True, type=Path,
                         help="the .../iter_XXXXXXX/adapter directory (containing adapter_config.json)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="overwrite --output if it already exists")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise ValueError(f"{args.output} already exists; pass --force to overwrite")
    config_path = args.adapter / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No adapter_config.json in {args.adapter} -- did you point at the "
                                 f"iter_XXXXXXX/adapter subdirectory, not iter_XXXXXXX itself?")

    print(f"loading base {args.base}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    print(f"applying adapter {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(base, str(args.adapter))
    print("merging (merge_and_unload)", flush=True)
    merged = model.merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    print(f"merged weights saved to {args.output}", flush=True)

    copied = []
    for name in TOKENIZER_ASSETS:
        source = Path(args.base) / name
        if source.exists():
            shutil.copy(source, args.output / name)
            copied.append(name)
    print(f"copied from base: {copied}")
    print("Now sanity-check this isn't secretly identical to base -- verify_checkpoints_distinct.py needs "
          ">=2 --checkpoint entries, so compare it against any other already-verified checkpoint, e.g.:\n"
          f"  python verify_checkpoints_distinct.py --base {args.base} --checkpoint merged={args.output} "
          f"--checkpoint fullft_orbit=/mnt/L202500431/models/zh_models/m6_st_full_non_thinking_hf_step300\n"
          "and check 'merged' shows changed_from_base > 0% in the printed density line.")


if __name__ == "__main__":
    main()
