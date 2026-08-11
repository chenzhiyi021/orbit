"""Generate rollouts from a teacher model for SFT training.

Example:

    python tools\teacher_rollout_sft.py \\
        --dataset YangyiH/openreasoning_mixed_100k \\
        --split train \\
        --teacher-base-url http://localhost:30000/v1 \\
        --teacher-model my-teacher-model \\
        --output data/sft/openreasoning_mixed_100k/train.jsonl \\
        --concurrency 32

Re-running with the same ``--output`` skips prompts already written (matched by
``prompt_sha256``), so an interrupted run can be resumed by rerunning the same command.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger("generate_teacher_rollout_for_sft")


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


def call_teacher(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint and return the reply text."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < max_retries:
                backoff = min(2**attempt, 30)
                logger.warning("teacher call failed (attempt %d/%d): %s; retrying in %ds", attempt + 1, max_retries + 1, exc, backoff)
                time.sleep(backoff)
    raise RuntimeError(f"teacher call failed after {max_retries + 1} attempts") from last_error


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

    parser.add_argument("--teacher-base-url", required=True, help="e.g. http://localhost:30000/v1")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument(
        "--teacher-api-key",
        default=os.environ.get("TEACHER_API_KEY", "EMPTY"),
        help="Defaults to $TEACHER_API_KEY, or 'EMPTY' for local vLLM/SGLang servers.",
    )
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=16)

    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_hashes = _load_existing_hashes(args.output)
    if done_hashes:
        logger.info("resuming: %d prompts already present in %s", len(done_hashes), args.output)

    rows = _load_rows(args.dataset, args.split, streaming=args.streaming, cache_dir=args.cache_dir)

    domain_filter = set(args.domain) if args.domain else None
    pending: list[tuple[dict, str, str]] = []
    for row in rows:
        if domain_filter is not None and row.get("domain") not in domain_filter:
            continue
        prompt = _extract_user_prompt(row)
        if not prompt:
            continue
        prompt_sha256 = _prompt_sha256(row, prompt)
        if prompt_sha256 in done_hashes:
            continue
        pending.append((row, prompt, prompt_sha256))
        if args.max_rows is not None and len(pending) >= args.max_rows:
            break

    logger.info("queued %d prompts for teacher generation", len(pending))

    write_lock = threading.Lock()
    written = 0
    failed = 0

    def _generate(item: tuple[dict, str, str]) -> dict | None:
        row, prompt, prompt_sha256 = item
        try:
            assistant_content = call_teacher(
                args.teacher_base_url,
                args.teacher_api_key,
                args.teacher_model,
                prompt,
                system_prompt=args.system_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
        except Exception:
            logger.exception("giving up on prompt_sha256=%s", prompt_sha256)
            return None
        assistant_content = _clean_text(assistant_content)
        if not assistant_content:
            return None
        return build_record(row, prompt, prompt_sha256, assistant_content, dataset=args.dataset)

    with args.output.open("a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_generate, item) for item in pending]
        for i, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            if record is None:
                failed += 1
            else:
                with write_lock:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                written += 1
            if i % args.log_every == 0:
                logger.info("progress %d/%d (written=%d, failed=%d)", i, len(pending), written, failed)

    logger.info("done: written=%d failed=%d output=%s", written, failed, args.output)


if __name__ == "__main__":
    main()
