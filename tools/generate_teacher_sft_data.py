"""Generate rollouts from a teacher model for SFT training.

Loads the teacher in-process with SGLang's offline batch engine (``sglang.Engine``)
-- no HTTP server, no port, no client-side retries. Chat templates (including
Qwen3's thinking/non-thinking toggle) are rendered locally via the tokenizer.

    python tools/generate_teacher_sft_data.py \
    --dataset /mnt/L202500431/datasets/openreasoning_mixed_100k \
    --split train \
    --teacher-model-path /mnt/L202500431/models/qwen3-4b-instruct-2507 \
    --teacher-tp-size 1 --teacher-dp-size 8 \
    --output data/sft/openreasoning_mixed_100k/train.jsonl \
    --concurrency 256

Re-running with the same ``--output`` skips prompts already written (matched by
``prompt_sha256``), so an interrupted run can be resumed by rerunning the same command.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Sequence
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger("generate_teacher_sft_data")

# SGLang's own "disable top-k" sentinel. Some reference recipes built on vLLM use
# 0 for the same intent -- the two engines don't share that convention.
_TOP_K_DISABLED = -1


def _parse_extra_kwargs(spec: str) -> dict:
    """Parse ``key=value,key2=value2`` into kwargs for ``sglang.Engine(**kwargs)``.

    Values go through ``ast.literal_eval`` so numbers/bools/None arrive as their
    Python type (e.g. ``mem_fraction_static=0.85``); anything that doesn't parse
    as a literal is kept as a plain string.
    """
    kwargs: dict = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, raw_value = item.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        kwargs[key] = value
    return kwargs


def _build_engine_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {
        "model_path": args.teacher_model_path,
        "tp_size": args.teacher_tp_size,
        "dp_size": args.teacher_dp_size,
        "dtype": args.teacher_dtype,
    }
    if args.teacher_mem_fraction is not None:
        kwargs["mem_fraction_static"] = args.teacher_mem_fraction
    if args.teacher_context_length is not None:
        kwargs["context_length"] = args.teacher_context_length
    if args.teacher_extra_args:
        kwargs.update(_parse_extra_kwargs(args.teacher_extra_args))
    return kwargs


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_user_prompt(row: dict) -> str | None:
    """Pull the (last) user message content out of a row's ``messages`` field."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = _clean_text(message.get("content"))
            if content:
                return content
    return None


def _prompt_sha256(row: dict, prompt: str) -> str:
    existing = row.get("prompt_sha256")
    if isinstance(existing, str) and existing:
        return existing
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def render_prompt(tokenizer, prompt: str, *, system_prompt: str | None, enable_thinking: bool | None) -> str:
    """Render a user prompt through the tokenizer's chat template.

    Local equivalent of what an OpenAI-compatible server does server-side with
    ``messages``/``chat_template_kwargs`` -- ``sglang.Engine.generate()`` takes
    already-rendered text, not a ``messages`` list.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)


def _load_rows(dataset: str, split: str, *, streaming: bool, cache_dir: str | None) -> Iterable[dict]:
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": streaming}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(dataset, **kwargs)


def _load_existing_hashes(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    hashes: set[str] = set()
    with output_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt_sha256 = record.get("metadata", {}).get("prompt_sha256")
            if prompt_sha256:
                hashes.add(prompt_sha256)
    return hashes


def _reuse_records(reuse_from: Path, output_path: Path, done_hashes: set[str]) -> int:
    """Copy records from an earlier run's output into ``output_path``, skipping ones already there.

    Mirrors the reference bank-builder's ``--reuse-bank``: rows already generated in a prior
    (possibly differently-scoped) run are reused verbatim instead of re-calling the teacher.
    Mutates ``done_hashes`` in place so the caller's generation queue also skips them.
    """
    if not reuse_from.exists():
        raise FileNotFoundError(f"--reuse-from path does not exist: {reuse_from}")
    reused = 0
    with reuse_from.open("r", encoding="utf-8") as fin, output_path.open("a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt_sha256 = record.get("metadata", {}).get("prompt_sha256")
            if not prompt_sha256 or prompt_sha256 in done_hashes:
                continue
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            done_hashes.add(prompt_sha256)
            reused += 1
    return reused


def build_record(row: dict, prompt: str, prompt_sha256: str, assistant_content: str, dataset: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "dataset": dataset,
            "domain": row.get("domain"),
            "source_dataset": row.get("source_dataset"),
            "source_config": row.get("source_config"),
            "source_split": row.get("source_split"),
            "prompt_sha256": prompt_sha256,
            "teacher_generated": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="YangyiH/openreasoning_mixed_100k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--domain", nargs="+", default=None, help="Only keep rows whose `domain` is in this list.")
    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument(
        "--teacher-model-path",
        required=True,
        help="Local HF checkpoint loaded in-process via sglang.Engine; this script starts and stops it.",
    )
    parser.add_argument("--teacher-tp-size", type=int, default=1)
    parser.add_argument("--teacher-dp-size", type=int, default=1, help="SGLang data-parallel replica count.")
    parser.add_argument("--teacher-dtype", default="bfloat16")
    parser.add_argument(
        "--teacher-mem-fraction", type=float, default=None, help="sglang mem_fraction_static; omitted uses sglang's own default."
    )
    parser.add_argument(
        "--teacher-context-length", type=int, default=None, help="sglang context_length cap; omitted uses the model's own max."
    )
    parser.add_argument(
        "--teacher-extra-args",
        default=None,
        help="Extra sglang.Engine kwargs as 'key=value,key2=value2' (Python literals), "
        "e.g. \"attention_backend='fa3'\".",
    )
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--top-k",
        type=int,
        default=_TOP_K_DISABLED,
        help=f"SGLang top-k; {_TOP_K_DISABLED} disables it (vLLM's equivalent is 0, a different convention).",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Qwen3 thinking/non-thinking toggle applied via the tokenizer's chat template. "
        "Omitted (template default) unless set; use --no-enable-thinking for non-thinking mode.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=256,
        help="Prompts per sglang.Engine.generate() batch call; also the resume/write checkpoint granularity.",
    )

    parser.add_argument(
        "--reuse-from",
        type=Path,
        default=None,
        help="An earlier run's output jsonl (or any file in this script's schema); records whose "
        "prompt_sha256 aren't already in --output are copied over instead of re-calling the teacher.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=4096,
        help="Skip prompts longer than this many tokens under --tokenizer.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer name/path for chat-template rendering and --max-prompt-length filtering. "
        "Defaults to --teacher-model-path.",
    )

    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.tokenizer is None:
        args.tokenizer = args.teacher_model_path
    return args


def _generate_dataset(args: argparse.Namespace, llm, tokenizer) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_hashes = _load_existing_hashes(args.output)
    if done_hashes:
        logger.info("resuming: %d prompts already present in %s", len(done_hashes), args.output)

    if args.reuse_from is not None:
        reused = _reuse_records(args.reuse_from, args.output, done_hashes)
        logger.info("reused %d prompts from %s", reused, args.reuse_from)

    rows = _load_rows(args.dataset, args.split, streaming=args.streaming, cache_dir=args.cache_dir)

    domain_filter = set(args.domain) if args.domain else None
    pending: list[tuple[dict, str, str]] = []
    skipped_too_long = 0
    for row in rows:
        if domain_filter is not None and row.get("domain") not in domain_filter:
            continue
        prompt = _extract_user_prompt(row)
        if not prompt:
            continue
        prompt_sha256 = _prompt_sha256(row, prompt)
        if prompt_sha256 in done_hashes:
            continue
        if args.max_prompt_length is not None and len(tokenizer(prompt).input_ids) > args.max_prompt_length:
            skipped_too_long += 1
            continue
        pending.append((row, prompt, prompt_sha256))
        if args.max_rows is not None and len(pending) >= args.max_rows:
            break

    if skipped_too_long:
        logger.info("skipped %d prompts longer than --max-prompt-length %d", skipped_too_long, args.max_prompt_length)
    logger.info("queued %d prompts for teacher generation", len(pending))

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_tokens,
        "sampling_seed": args.seed,
        "n": 1,
    }

    written = 0
    failed = 0
    with args.output.open("a", encoding="utf-8") as fout:
        for start in range(0, len(pending), args.concurrency):
            chunk = pending[start : start + args.concurrency]
            prompt_texts = [
                render_prompt(tokenizer, prompt, system_prompt=args.system_prompt, enable_thinking=args.enable_thinking)
                for _, prompt, _ in chunk
            ]
            try:
                outputs = llm.generate(prompt_texts, sampling_params=sampling_params)
            except Exception:
                logger.exception("giving up on a batch of %d prompts (offset %d)", len(chunk), start)
                failed += len(chunk)
                continue

            for (row, prompt, prompt_sha256), output in zip(chunk, outputs):
                text = output.get("text") if isinstance(output, dict) else getattr(output, "text", None)
                assistant_content = _clean_text(text)
                if not assistant_content:
                    failed += 1
                    continue
                record = build_record(row, prompt, prompt_sha256, assistant_content, dataset=args.dataset)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1

            logger.info(
                "progress %d/%d (written=%d, failed=%d)", min(start + args.concurrency, len(pending)), len(pending), written, failed
            )

    logger.info("done: written=%d failed=%d output=%s", written, failed, args.output)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    import sglang as sgl

    engine_kwargs = _build_engine_kwargs(args)
    logger.info(
        "launching teacher engine: model=%s tp_size=%s dp_size=%s dtype=%s",
        args.teacher_model_path,
        args.teacher_tp_size,
        args.teacher_dp_size,
        args.teacher_dtype,
    )
    llm = sgl.Engine(**engine_kwargs)
    try:
        _generate_dataset(args, llm, tokenizer)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
