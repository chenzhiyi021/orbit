"""Generate IFEval responses with an HF checkpoint, for the official google-research grader.

Reads the 541 prompts from IFEval's input_data.jsonl, renders each as a single user turn
with the model's own chat template (non-thinking by default, matching the OPD runs),
generates with SGLang's offline engine, and writes {"prompt", "response"} lines. The
prompt is copied verbatim from input_data.jsonl because the grader looks responses up by
exact prompt text.

Run in an environment with sglang + transformers + a GPU (e.g. orbit_env_v2):

    python tools/ifeval/generate_ifeval_responses.py \\
        --model /mnt/L202500431/models/qwen3-1.7b \\
        --output /mnt/L202500431/google-research/ifeval_runs/qwen3-1.7b/responses.jsonl

then grade in the grader's environment (e.g. ifeval_env), from the google-research dir:

    python -m instruction_following_eval.evaluation_main \\
        --input_data=./instruction_following_eval/data/input_data.jsonl \\
        --input_response_data=./ifeval_runs/qwen3-1.7b/responses.jsonl \\
        --output_dir=./ifeval_runs/qwen3-1.7b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_INPUT = "/mnt/L202500431/google-research/instruction_following_eval/data/input_data.jsonl"


def strip_thinking(text: str) -> str:
    """Drop a reasoning block if the model emitted one (thinking mode); keep only the answer."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.lstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF checkpoint directory")
    parser.add_argument("--output", required=True, type=Path, help="responses .jsonl to write")
    parser.add_argument("--input-data", default=DEFAULT_INPUT, type=Path, help="IFEval input_data.jsonl")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="render with enable_thinking=True (default: non-thinking, as in the OPD runs)")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (the usual IFEval setting)")
    parser.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    parser.add_argument("--mem-fraction", type=float, default=0.8)
    args = parser.parse_args()

    import sglang as sgl
    from transformers import AutoTokenizer

    prompts = [json.loads(line)["prompt"] for line in args.input_data.open() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        for prompt in prompts
    ]
    print(f"{len(prompts)} prompts; rendered example:\n{rendered[0]!r}\n", flush=True)

    engine = sgl.Engine(model_path=args.model, tp_size=args.tp, mem_fraction_static=args.mem_fraction)
    try:
        outputs = engine.generate(rendered, {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens})
    finally:
        engine.shutdown()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    truncated = 0
    with args.output.open("w") as f:
        for prompt, out in zip(prompts, outputs):
            finish = (out.get("meta_info") or {}).get("finish_reason") or {}
            truncated += isinstance(finish, dict) and finish.get("type") == "length"
            f.write(json.dumps({"prompt": prompt, "response": strip_thinking(out["text"])}, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} responses -> {args.output} ({truncated} hit max_new_tokens={args.max_new_tokens})")


if __name__ == "__main__":
    main()
