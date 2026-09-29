"""Teacher RKL/FKL restricted to high-entropy positions of a fixed prefix bank.

``score_checkpoints.py`` only keeps per-prompt *means* of teacher RKL/FKL
(``mean_teacher_{rkl,fkl}_t0p7``), which are dominated by the many
low-entropy style/format positions. This script re-runs the forward pass on
the same bank and teacher cache, stores per-position statistics, and then
reports KL separately on high- vs low-entropy positions.

Positions are bucketed by *fixed* references so that every run is compared on
the exact same token set:
  - H_T: teacher entropy (from the teacher cache)
  - H_B: Base-model entropy (from scoring Base on the same bank)
Both at ``config["teacher_temperature"]`` (0.7 by default), the same
temperature the KL is computed at. Thresholds are per-bank quantiles
(``--quantile``, default 0.8 = top 20% is "high").

Buckets reported per run:
  all, teacher_high, teacher_low,
  T_low_B_low   (easy / format tokens)
  T_low_B_high  (knowledge gap: teacher certain, base unsure)
  T_high_B_low  (base overconfident where teacher sees options)
  T_high_B_high (genuine forks)

Math is kept identical to ``scoring.score_bank`` (same position indexing,
same temperature-scaled log_softmax for RKL/FKL), so the ``all`` bucket
reproduces ``mean_teacher_{rkl,fkl}_t0p7`` averaged over the bank.

Outputs, under ``<output_root>/<bank>/entropy_kl/``:
  ``<run>.positions.parquet``  one row per (prompt, selected position)
  ``summary_q<quantile>.csv``   per (run, bucket): means, deltas vs Base,
                                 prompt-level paired bootstrap 95% CI

Usage:
    python teacher_kl_by_entropy.py --config config.json --bank bank_a
    python teacher_kl_by_entropy.py --config config.json --bank bank_a --aggregate-only --quantile 0.9
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk

from score_checkpoints import AUDITED_CONFIG_FIELDS, effective_model_config, load_config, load_scoring_model


METRICS = ("fkl", "rkl", "top1_agree", "h_student")


@torch.inference_mode()
def score_positions(model, dataset, teacher_logits: np.ndarray, *, temperature: float, batch_size: int,
                    device: str = "cuda") -> pd.DataFrame:
    vocab = model.config.vocab_size
    if teacher_logits.shape[-1] != vocab:
        raise ValueError(f"Teacher cache vocab {teacher_logits.shape[-1]} != student vocab {vocab}")
    frames = []
    for start in range(0, len(dataset), batch_size):
        records = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        max_length = max(len(record["input_ids"]) for record in records)
        input_ids = torch.zeros(len(records), max_length, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for index, record in enumerate(records):
            ids = torch.tensor(record["input_ids"], dtype=torch.long, device=device)
            input_ids[index, : ids.numel()] = ids
            attention[index, : ids.numel()] = 1
        hidden = model.model(input_ids=input_ids, attention_mask=attention, use_cache=False).last_hidden_state
        for batch_index, record in enumerate(records):
            prompt_length = len(record["prompt_ids"])
            completion_positions = np.asarray(record["selected_positions"], dtype=np.int64)
            hidden_positions = torch.from_numpy(prompt_length + completion_positions - 1).to(device)
            logits = model.lm_head(hidden[batch_index].index_select(0, hidden_positions)).float()
            teacher = torch.from_numpy(
                np.asarray(teacher_logits[start + batch_index][: len(completion_positions)], dtype=np.float32)
            ).to(device)
            student_log_probs = torch.log_softmax(logits / temperature, dim=-1)
            teacher_log_probs = torch.log_softmax(teacher / temperature, dim=-1)
            student_probs = student_log_probs.exp()
            teacher_probs = teacher_log_probs.exp()
            frames.append(pd.DataFrame({
                "prompt_index": start + batch_index,
                "selected_index": record["selected_index"],
                "prompt_sha256": record["prompt_sha256"],
                "position_rank": np.arange(len(completion_positions)),
                "completion_position": completion_positions,
                "h_teacher": (-(teacher_probs * teacher_log_probs).sum(-1)).cpu().numpy(),
                "h_student": (-(student_probs * student_log_probs).sum(-1)).cpu().numpy(),
                "rkl": (student_probs * (student_log_probs - teacher_log_probs)).sum(-1).cpu().numpy(),
                "fkl": (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1).cpu().numpy(),
                "top1_agree": (logits.argmax(-1) == teacher.argmax(-1)).float().cpu().numpy(),
            }))
        del hidden, input_ids, attention
        print(f"positions {min(start + batch_size, len(dataset))}/{len(dataset)}", flush=True)
    return pd.concat(frames, ignore_index=True)


def score_run(label: str, checkpoint, *, dataset, teacher_logits, out: Path, base_config, config: dict,
              overwrite: bool) -> None:
    target = out / f"{label}.positions.parquet"
    if target.exists() and not overwrite:
        print(f"skip complete {label}", flush=True)
        return
    started = time.monotonic()
    print(f"START {label} {checkpoint}", flush=True)
    model_config, overrides = effective_model_config(checkpoint)
    mismatched = {k: (getattr(base_config, k, None), getattr(model_config, k, None)) for k in AUDITED_CONFIG_FIELDS
                  if getattr(base_config, k, None) != getattr(model_config, k, None)}
    if mismatched:
        raise AssertionError(f"{label}: config diverges from base_model on {mismatched}")
    model = load_scoring_model(checkpoint, overrides)
    frame = score_positions(model, dataset, teacher_logits, temperature=config["teacher_temperature"],
                            batch_size=config["batch_size"])
    frame.insert(0, "run", label)
    frame.to_parquet(target, index=False)
    print(f"DONE {label} seconds={time.monotonic() - started:.1f}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def bucket_masks(base: pd.DataFrame, quantile: float) -> tuple[dict[str, np.ndarray], dict]:
    h_teacher = base["h_teacher"].to_numpy()
    h_base = base["h_student"].to_numpy()
    teacher_cut = float(np.quantile(h_teacher, quantile))
    base_cut = float(np.quantile(h_base, quantile))
    t_high = h_teacher >= teacher_cut
    b_high = h_base >= base_cut
    masks = {
        "all": np.ones_like(t_high),
        "teacher_high": t_high,
        "teacher_low": ~t_high,
        "T_low_B_low": ~t_high & ~b_high,
        "T_low_B_high": ~t_high & b_high,
        "T_high_B_low": t_high & ~b_high,
        "T_high_B_high": t_high & b_high,
    }
    return masks, {"teacher_entropy_cut": teacher_cut, "base_entropy_cut": base_cut}


def bootstrap_ratio(values: np.ndarray, mask: np.ndarray, prompt: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """95% CI of a token-pooled masked mean, resampling prompts (paired across runs)."""
    num_prompts = weights.shape[1]
    sums = np.bincount(prompt, weights=values * mask, minlength=num_prompts)
    counts = np.bincount(prompt, weights=mask.astype(np.float64), minlength=num_prompts)
    estimates = (weights @ sums) / np.maximum(weights @ counts, 1e-12)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def aggregate(out: Path, run_order: list[str], quantile: float, bootstrap: int, seed: int) -> pd.DataFrame:
    base = pd.read_parquet(out / "Base.positions.parquet")
    masks, cuts = bucket_masks(base, quantile)
    prompt = base["prompt_index"].to_numpy()
    num_prompts = int(prompt.max()) + 1
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(num_prompts, np.full(num_prompts, 1 / num_prompts), size=bootstrap).astype(np.float64)

    rows = []
    for label in run_order:
        path = out / f"{label}.positions.parquet"
        if not path.exists():
            print(f"missing {path.name}, skipped", flush=True)
            continue
        frame = pd.read_parquet(path)
        same_positions = (len(frame) == len(base)
                          and (frame["prompt_sha256"].to_numpy() == base["prompt_sha256"].to_numpy()).all()
                          and (frame["completion_position"].to_numpy() == base["completion_position"].to_numpy()).all())
        if not same_positions:
            raise ValueError(f"{label}: positions do not line up with Base -- scored on a different bank?")
        for bucket, mask in masks.items():
            row = dict(run=label, bucket=bucket, n_tokens=int(mask.sum()),
                       h_teacher=float(base["h_teacher"].to_numpy()[mask].mean()))
            for metric in METRICS:
                values = frame[metric].to_numpy(dtype=np.float64)
                delta = values - base[metric].to_numpy(dtype=np.float64)
                base_value = float(base[metric].to_numpy()[mask].mean())
                row[metric] = float(values[mask].mean())
                row[f"{metric}_delta"] = float(delta[mask].mean())
                row[f"{metric}_rel"] = row[f"{metric}_delta"] / base_value if base_value else np.nan
                if metric in ("fkl", "rkl") and label != "Base":
                    row[f"{metric}_delta_ci_low"], row[f"{metric}_delta_ci_high"] = bootstrap_ratio(delta, mask, prompt, weights)
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / f"summary_q{quantile:g}.csv", index=False)
    (out / f"summary_q{quantile:g}.json").write_text(json.dumps(dict(quantile=quantile, **cuts), indent=2) + "\n")
    print(f"\nentropy cuts (q={quantile:g}): {cuts}")
    for metric in ("fkl", "rkl"):
        print(f"\n{metric.upper()} relative change vs Base, by bucket:")
        print(summary.pivot(index="run", columns="bucket", values=f"{metric}_rel").reindex(
            [r for r in run_order if r in set(summary["run"])])[list(masks)].to_string(float_format=lambda v: f"{v:+.1%}"))
    print(f"\nwrote {out / f'summary_q{quantile:g}.csv'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", required=True, help="key into config['banks']; needs teacher_cache set")
    parser.add_argument("--runs", default=None, help="comma-separated keys of config['checkpoints']; default: all")
    parser.add_argument("--quantile", type=float, default=0.8, help="entropy >= this per-bank quantile counts as high")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--aggregate-only", action="store_true", help="skip GPU scoring, only rebuild the summary")
    parser.add_argument("--overwrite", action="store_true", help="rescore runs whose .positions.parquet exists")
    args = parser.parse_args()

    config = load_config(args.config)
    bank_cfg = config["banks"][args.bank]
    out = Path(config["output_root"]) / args.bank / "entropy_kl"
    out.mkdir(parents=True, exist_ok=True)

    checkpoints = config["checkpoints"]
    if args.runs:
        wanted = [r.strip() for r in args.runs.split(",") if r.strip()]
        unknown = [r for r in wanted if r not in checkpoints]
        if unknown:
            raise KeyError(f"--runs not in config['checkpoints']: {unknown}")
        checkpoints = {r: checkpoints[r] for r in wanted}
    labels = {name: f"{name}_step{config.get('checkpoint_steps', {}).get(name, 300):03d}" for name in checkpoints}

    if not args.aggregate_only:
        if not bank_cfg.get("teacher_cache"):
            raise ValueError(f"config['banks']['{args.bank}']['teacher_cache'] is not set; run cache_teacher_logits.py first")
        dataset = load_from_disk(bank_cfg["path"])
        teacher_logits = np.load(bank_cfg["teacher_cache"], mmap_mode="r")
        if teacher_logits.shape[0] != len(dataset):
            raise ValueError(f"Teacher cache has {teacher_logits.shape[0]} prompts, bank has {len(dataset)}")
        base_config, _ = effective_model_config(config["base_model"])
        common = dict(dataset=dataset, teacher_logits=teacher_logits, out=out, base_config=base_config,
                      config=config, overwrite=args.overwrite)
        score_run("Base", config["base_model"], **common)
        for name, checkpoint in checkpoints.items():
            score_run(labels[name], checkpoint, **common)

    aggregate(out, ["Base", *labels.values()], args.quantile, args.bootstrap, config.get("bootstrap_seed", 20260826))


if __name__ == "__main__":
    main()
