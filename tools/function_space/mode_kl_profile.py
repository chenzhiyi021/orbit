"""Does a non-thinking fine-tune behave like its base model in *thinking* mode?

Tests the "unlock" reading of non-thinking OPD (Qwen3 non-thinking 68 -> 82 on MATH500 within
20 steps): the fine-tune may be switching on the base model's own thinking behaviour rather than
learning the teacher's. Every model scores the SAME completion text y (sampled from the
reference model, read from an eval run's completions.jsonl) under its own chat context:

    think    template(enable_thinking=True)  + "<think>\\n" + y     y read as the thinking trace
    nothink  template(enable_thinking=False) + y                   after the empty <think></think> block
    plain    template() + y                                        models without a thinking switch
                                                                   (e.g. Qwen3-*-Instruct-2507 teachers)

and, at every position of y (all models at --temperature), we record against the reference P:

    kl    KL(P || Q) = sum_v P(v) (log P(v) - log Q(v))      expectation under the model that wrote y
    nll   -log Q(y_t)                                        how natural y is to Q
    top1  argmax P == argmax Q

"Unlock" predicts KL(ft_nothink || base_think) << KL(ft_nothink || base_nothink), and also below
KL(ft_nothink || teacher); the opposite ordering means the fine-tune moved toward the teacher's
style instead. Paired differences between comparison models come with sample-level bootstrap CIs.

Completions sampled in thinking mode ("<think>...</think>answer") are cut to their thinking trace;
use them with a base_think reference to run the question in reverse.

Outputs, under --output-dir:
    positions.parquet     one row per (sample, position, model)
    summary.csv           per (model, position bin): mean kl / nll / top1 (token-pooled) + 95% CI
    contrasts.csv         per (model a, model b, bin): mean(kl_a - kl_b), mean(nll_a - nll_b) + 95% CI

    python mode_kl_profile.py \\
        --completions /mnt/.../eval_results/<no_excl_run>/iter_0000019/math/math500/completions.jsonl \\
        --task math500 --evalchemy-root /mnt/L202500431/third_party/evalchemy \\
        --model ft_nothink=/mnt/.../no_excl_iter19_hf:nothink \\
        --model base_think=/mnt/L202500431/models/qwen3-1.7b:think \\
        --model base_nothink=/mnt/L202500431/models/qwen3-1.7b:nothink \\
        --model teacher=/mnt/L202500431/models/qwen3-4b-instruct-2507:plain \\
        --reference ft_nothink --num-samples 200 --output-dir /mnt/.../mode_kl/no_excl_iter19_math500
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from score_checkpoints import effective_model_config, load_scoring_model

MODES = ("think", "nothink", "plain")
POSITION_BINS = [(0, 64), (64, 256), (256, 1024), (1024, 1 << 30)]

# Mirrors examples/on_policy_distillation/eval/run_evalchemy_math_eval.py (MATH_PROMPT, TASKS,
# load_examples), kept in sync by hand so this runs without the aiohttp/lm_eval eval environment.
MATH_PROMPT = """Problem: {problem}
Mark your solution with \\boxed
Answer:"""
TASKS = {
    "aime24": ("eval/chat_benchmarks/AIME24/data/aime24.json", "problem"),
    "aime25": ("eval/chat_benchmarks/AIME25/data/aime25.json", "problem"),
    "amc23": ("eval/chat_benchmarks/AMC23/data/amc23.json", "question"),
    "math500": ("eval/chat_benchmarks/MATH500/data/math500.jsonl", "problem"),
}


def load_problems(task: str, evalchemy_root: Path) -> dict[str, str]:
    relative_path, problem_key = TASKS[task]
    with (evalchemy_root / relative_path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row.get("id", row.get("unique_id", index))): str(row[problem_key]) for index, row in enumerate(rows)}


def thinking_trace(text: str) -> str:
    """A thinking-mode completion "<think>\\n...</think>answer" -> its trace; anything else unchanged."""
    if text.lstrip().startswith("<think>"):
        text = text.lstrip()[len("<think>"):].lstrip("\n")
        return text.split("</think>", 1)[0]
    return text


def context_prefix(tokenizer, prompt: str, mode: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if mode == "plain":
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=(mode == "think"))
    if mode == "nothink":
        if not prefix.rstrip().endswith("</think>"):
            raise ValueError("chat template ignores enable_thinking=False (no empty think block); use mode 'plain'")
        return prefix
    return prefix if prefix.endswith("<think>\n") else prefix + "<think>\n"


def parse_model(spec: str) -> tuple[str, str, str]:
    label, rest = spec.split("=", 1)
    path, mode = rest.rsplit(":", 1)
    if mode not in MODES:
        raise ValueError(f"--model {spec}: mode must be one of {MODES}")
    return label, path, mode


@torch.inference_mode()
def log_probs_on(model, ids: torch.Tensor, start: int, temperature: float, chunk: int) -> list[torch.Tensor]:
    """Temperature-scaled log-softmax predicting ids[start:], in position chunks (fp32 on the GPU)."""
    hidden = model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    return [torch.log_softmax(model.lm_head(hidden[p - 1: min(p + chunk, len(ids)) - 1]).float() / temperature, -1)
            for p in range(start, len(ids), chunk)]


def bootstrap_ci(values: np.ndarray, sample: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """95% CI of a token-pooled mean, resampling whole samples."""
    n = weights.shape[1]
    sums = np.bincount(sample, weights=values, minlength=n)
    counts = np.bincount(sample, minlength=n).astype(np.float64)
    estimates = (weights @ sums) / np.maximum(weights @ counts, 1e-12)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def aggregate(frame: pd.DataFrame, reference: str, bootstrap: int, seed: int, out: Path) -> None:
    others = [m for m in frame["model"].unique() if m != reference]
    num_samples = int(frame["sample"].max()) + 1
    weights = np.random.default_rng(seed).multinomial(
        num_samples, np.full(num_samples, 1 / num_samples), size=bootstrap).astype(np.float64)
    bins = {"all": (0, 1 << 30), **{f"{lo}-{hi if hi < 1 << 30 else 'end'}": (lo, hi) for lo, hi in POSITION_BINS}}
    wide = {m: frame[frame["model"] == m].sort_values(["sample", "position"]).reset_index(drop=True)
            for m in frame["model"].unique()}
    position = wide[reference]["position"].to_numpy()
    sample = wide[reference]["sample"].to_numpy()

    rows, contrasts = [], []
    for name, (lo, hi) in bins.items():
        mask = (position >= lo) & (position < hi)
        if not mask.any():
            continue
        for m in [reference, *others]:
            row = dict(model=m, bin=name, n_tokens=int(mask.sum()))
            for metric in ("kl", "nll", "top1"):
                values = wide[m][metric].to_numpy(np.float64)[mask]
                row[metric] = float(values.mean())
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bootstrap_ci(values, sample[mask], weights)
            rows.append(row)
        for i, a in enumerate(others):
            for b in others[i + 1:]:
                row = dict(model_a=a, model_b=b, bin=name, n_tokens=int(mask.sum()))
                for metric in ("kl", "nll"):
                    diff = (wide[a][metric].to_numpy(np.float64) - wide[b][metric].to_numpy(np.float64))[mask]
                    row[f"{metric}_diff"] = float(diff.mean())
                    row[f"{metric}_diff_ci_low"], row[f"{metric}_diff_ci_high"] = bootstrap_ci(diff, sample[mask], weights)
                contrasts.append(row)
    summary, contrast = pd.DataFrame(rows), pd.DataFrame(contrasts)
    summary.to_csv(out / "summary.csv", index=False)
    contrast.to_csv(out / "contrasts.csv", index=False)
    for metric in ("kl", "nll"):
        print(f"\n{metric} vs reference {reference} (mean over tokens), by position bin:")
        print(summary.pivot(index="model", columns="bin", values=metric)[list(dict.fromkeys(summary["bin"]))]
              .to_string(float_format=lambda v: f"{v:.4f}"))
    print("\npaired kl differences (a - b), 95% CI:")
    for _, r in contrast[contrast["bin"] == "all"].iterrows():
        print(f"  {r.model_a} - {r.model_b}: {r.kl_diff:+.4f}  [{r.kl_diff_ci_low:+.4f}, {r.kl_diff_ci_high:+.4f}]")
    print(f"\nwrote {out / 'summary.csv'} and {out / 'contrasts.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", type=Path, required=True, help="completions.jsonl from run_evalchemy_math_eval.py")
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--evalchemy-root", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="label=hf_dir:mode, mode in think|nothink|plain")
    parser.add_argument("--reference", default=None, help="label of the model that wrote the completions (default: first --model)")
    parser.add_argument("--num-samples", type=int, default=200, help="random subset of ok completions; 0 = all")
    parser.add_argument("--max-tokens", type=int, default=4096, help="score at most this many tokens of each completion")
    parser.add_argument("--temperature", type=float, default=0.7, help="softmax temperature for every model (eval sampled at 0.7)")
    parser.add_argument("--chunk", type=int, default=1024, help="positions per lm_head chunk")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate-only", action="store_true", help="rebuild summary/contrasts from positions.parquet")
    args = parser.parse_args()

    specs = [parse_model(s) for s in args.model]
    labels = [label for label, _, _ in specs]
    paths = {label: path for label, path, _ in specs}
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate --model labels: {labels}")
    reference = args.reference or labels[0]
    if reference not in labels:
        raise ValueError(f"--reference {reference} is not a --model label")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    positions_path = args.output_dir / "positions.parquet"
    if args.aggregate_only:
        aggregate(pd.read_parquet(positions_path), reference, args.bootstrap, args.seed, args.output_dir)
        return

    problems = load_problems(args.task, args.evalchemy_root)
    rows = {}
    with args.completions.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["key"]] = row  # last write wins, as in the eval runner's resume logic
    keys = sorted(k for k, r in rows.items() if r.get("status") == "ok" and r["output"].strip())
    if args.num_samples and len(keys) > args.num_samples:
        keys = sorted(np.random.default_rng(args.seed).choice(keys, args.num_samples, replace=False).tolist())
    print(f"{len(keys)} completions from {args.completions}", flush=True)

    # One copy per checkpoint directory, shared by its modes; every model resident at once so each
    # position's full distributions can be compared without storing vocab-sized tensors.
    tokenizers, models = {}, {}
    for _, path, _ in specs:
        if path not in models:
            _, overrides = effective_model_config(path)
            tokenizers[path] = AutoTokenizer.from_pretrained(path)
            models[path] = load_scoring_model(path, overrides)
    vocab = {models[p].config.vocab_size for p in models}
    if len(vocab) != 1:
        raise ValueError(f"models disagree on vocab size {vocab}; KL needs a shared vocabulary")

    frames = []
    started = time.perf_counter()
    for sample_index, key in enumerate(keys):
        task, example_id, repetition = key.split("/")
        prompt = MATH_PROMPT.format(problem=problems[example_id])
        text = thinking_trace(rows[key]["output"])
        inputs, y_ids = {}, None
        for label, path, mode in specs:
            tokenizer = tokenizers[path]
            prefix = tokenizer(context_prefix(tokenizer, prompt, mode), add_special_tokens=False)["input_ids"]
            y = tokenizer(text, add_special_tokens=False)["input_ids"][: args.max_tokens]
            if y_ids is None:
                y_ids = y
            elif y != y_ids:
                raise ValueError(f"{label}: tokenizes the completion differently from {specs[0][0]}")
            inputs[label] = (torch.tensor(prefix + y, device="cuda"), len(prefix))
        ids, start = inputs[reference]
        ref_chunks = log_probs_on(models[paths[reference]], ids, start, args.temperature, args.chunk)
        target = torch.tensor(y_ids, device="cuda")
        for label, path, _ in specs:
            ids, start = inputs[label]
            chunks = ref_chunks if label == reference else log_probs_on(models[path], ids, start, args.temperature, args.chunk)
            kl, nll, top1, offset = [], [], [], 0
            for p_log, q_log in zip(ref_chunks, chunks):
                span = target[offset: offset + len(q_log)]
                kl.append((p_log.exp() * (p_log - q_log)).sum(-1))
                nll.append(-q_log.gather(-1, span[:, None])[:, 0])
                top1.append((p_log.argmax(-1) == q_log.argmax(-1)).float())
                offset += len(q_log)
            frames.append(pd.DataFrame({
                "sample": sample_index, "key": key, "model": label, "position": np.arange(len(y_ids)),
                "kl": torch.cat(kl).cpu().numpy(), "nll": torch.cat(nll).cpu().numpy(),
                "top1": torch.cat(top1).cpu().numpy(),
            }))
            del chunks
        del ref_chunks
        if (sample_index + 1) % 10 == 0 or sample_index + 1 == len(keys):
            print(f"  {sample_index + 1}/{len(keys)} ({time.perf_counter() - started:.0f}s)", flush=True)

    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(positions_path, index=False)
    (args.output_dir / "config.json").write_text(json.dumps(dict(
        completions=str(args.completions), task=args.task, models=args.model, reference=reference,
        num_samples=len(keys), max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed), indent=2))
    aggregate(frame, reference, args.bootstrap, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
