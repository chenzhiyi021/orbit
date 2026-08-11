"""Generate rollouts from a teacher model for SFT training.

Point it at an already-running OpenAI-compatible server:

    python tools/teacher_rollout_sft.py \
    --dataset /mnt/L202500431/datasets/openreasoning_mixed_100k \
    --split train \
    --teacher-base-url http://localhost:30000/v1 \
    --teacher-model my-teacher-model \
    --output data/sft/openreasoning_mixed_100k/train.jsonl \
    --concurrency 32

... or let this script launch the server itself (SGLang or vLLM) from a local HF
checkpoint and tear it down when generation finishes:

    python tools/teacher_rollout_sft.py \
    --dataset /mnt/L202500431/datasets/openreasoning_mixed_100k \
    --split train \
    --teacher-model-path /path/to/hf/teacher-checkpoint \
    --teacher-model my-teacher-model \
    --teacher-tp-size 4 \
    --output data/sft/openreasoning_mixed_100k/train.jsonl \
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
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger("generate_teacher_rollout_for_sft")

# Launch module for each supported inference engine, keyed by --teacher-engine.
_ENGINE_LAUNCH_MODULES = {
    "sglang": "sglang.launch_server",
    "vllm": "vllm.entrypoints.openai.api_server",
}


def _pick_free_port(host: str) -> int:
    """Best-effort free port (small race window between pick and launch)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _build_teacher_launch_cmd(
    engine: str,
    model_path: str,
    served_model_name: str,
    host: str,
    port: int,
    tp_size: int,
    extra_args: str | None,
) -> list[str]:
    module = _ENGINE_LAUNCH_MODULES.get(engine)
    if module is None:
        raise ValueError(f"unknown --teacher-engine {engine!r}; choose from {sorted(_ENGINE_LAUNCH_MODULES)}")
    cmd = [sys.executable, "-m", module, "--host", host, "--port", str(port)]
    if engine == "sglang":
        cmd += ["--model-path", model_path, "--served-model-name", served_model_name, "--tp-size", str(tp_size)]
    else:  # vllm
        cmd += [
            "--model",
            model_path,
            "--served-model-name",
            served_model_name,
            "--tensor-parallel-size",
            str(tp_size),
        ]
    if extra_args:
        cmd += shlex.split(extra_args)
    return cmd


def _tail_file(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<no log available>"
    return "\n".join(lines[-n:])


def _launch_teacher_server(cmd: list[str], log_path: Path) -> subprocess.Popen:
    logger.info("launching teacher server: %s", " ".join(cmd))
    logger.info("teacher server log: %s", log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    kwargs: dict = {}
    if os.name == "posix":
        # Own process group so we can also reap the engine's worker subprocesses (e.g. TP ranks).
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, **kwargs)


def _wait_for_teacher_ready(base_url: str, proc: subprocess.Popen, timeout: float, log_path: Path) -> None:
    deadline = time.time() + timeout
    url = f"{base_url.rstrip('/')}/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"teacher server process exited early (code={proc.returncode}); log tail:\n{_tail_file(log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5):
                logger.info("teacher server ready at %s", base_url)
                return
        except urllib.error.URLError:
            time.sleep(2)
    raise RuntimeError(f"teacher server not ready after {timeout}s; see {log_path}")


def _shutdown_teacher_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    logger.info("shutting down teacher server (pid=%d)", proc.pid)
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


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

    teacher_source = parser.add_mutually_exclusive_group(required=True)
    teacher_source.add_argument(
        "--teacher-base-url", default=None, help="An already-running OpenAI-compatible server, e.g. http://localhost:30000/v1"
    )
    teacher_source.add_argument(
        "--teacher-model-path",
        default=None,
        help="Local HF checkpoint to launch as the teacher server; this script starts and stops it.",
    )
    parser.add_argument(
        "--teacher-model", required=True, help="Model name used in API requests (and --served-model-name when launching)."
    )
    parser.add_argument(
        "--teacher-api-key",
        default=os.environ.get("TEACHER_API_KEY", "EMPTY"),
        help="Defaults to $TEACHER_API_KEY, or 'EMPTY' for local vLLM/SGLang servers.",
    )
    parser.add_argument("--teacher-engine", choices=sorted(_ENGINE_LAUNCH_MODULES), default="sglang")
    parser.add_argument("--teacher-host", default="127.0.0.1")
    parser.add_argument("--teacher-port", type=int, default=None, help="Defaults to an auto-picked free port.")
    parser.add_argument("--teacher-tp-size", type=int, default=1)
    parser.add_argument(
        "--teacher-extra-args",
        default=None,
        help="Extra raw args appended to the launch command, e.g. '--mem-fraction-static 0.85 --dtype bfloat16'.",
    )
    parser.add_argument("--teacher-startup-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--teacher-log-file", type=Path, default=None, help="Defaults to <output>.teacher_server.log"
    )
    parser.add_argument(
        "--keep-teacher-server", action="store_true", help="Leave the launched teacher server running after this script exits."
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


def _generate_dataset(args: argparse.Namespace) -> None:
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


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    teacher_proc = None
    if args.teacher_model_path:
        port = args.teacher_port if args.teacher_port is not None else _pick_free_port(args.teacher_host)
        log_path = args.teacher_log_file or Path(f"{args.output}.teacher_server.log")
        cmd = _build_teacher_launch_cmd(
            args.teacher_engine,
            args.teacher_model_path,
            args.teacher_model,
            args.teacher_host,
            port,
            args.teacher_tp_size,
            args.teacher_extra_args,
        )
        teacher_proc = _launch_teacher_server(cmd, log_path)
        args.teacher_base_url = f"http://{args.teacher_host}:{port}/v1"
        try:
            _wait_for_teacher_ready(args.teacher_base_url, teacher_proc, args.teacher_startup_timeout, log_path)
        except Exception:
            _shutdown_teacher_server(teacher_proc)
            raise

    try:
        _generate_dataset(args)
    finally:
        if teacher_proc is not None and not args.keep_teacher_server:
            _shutdown_teacher_server(teacher_proc)


if __name__ == "__main__":
    main()
