"""Core forward-pass + CountSketch scoring engine.

Numerics are copied verbatim from the TRL-side
``unifying_posttrain/experiments/m5_m6_param_native_trajectory/score_function_space.py``
(``score_bank``, ``sketch_maps``, ``save_score``). Do not change the math here
without re-running the CountSketch validation gate (see ``relations.sketch_gate``)
against the historical caches -- any drift changes what "cosine of
centered-logit deltas" means and breaks comparability with the trl-side
numbers.

This module has zero trl- or orbit-framework dependencies: it only needs
``torch``, ``numpy``, ``pandas`` and a plain ``transformers`` model. It works
identically on a checkpoint trained with either stack, as long as the
checkpoint loads via ``AutoModelForCausalLM.from_pretrained``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SKETCH_SEEDS = (3407, 3408, 3409)


def sketch_maps(vocab: int, dimension: int, device: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build the (bucket, sign) CountSketch projection for each seed.

    Deterministic given (vocab, dimension, seed): re-running this on a
    different machine reproduces the identical projection, which is what
    makes cross-checkpoint / cross-codebase cosine comparisons valid.
    """
    maps = []
    for seed in SKETCH_SEEDS:
        generator = np.random.default_rng(seed)
        bucket = torch.from_numpy(generator.integers(0, dimension, size=vocab, dtype=np.int64)).to(device)
        sign = torch.from_numpy(generator.choice(np.array([-1.0, 1.0], dtype=np.float32), size=vocab)).to(device)
        maps.append((bucket, sign))
    return maps


@torch.inference_mode()
def score_bank(
    model: torch.nn.Module,
    dataset,
    device: str,
    sketch_dimension: int,
    batch_size: int,
    exact_positions: int,
    teacher_logits: np.ndarray | None = None,
    temperature: float = 0.7,
    space: str = "logit",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Score a fixed prefix bank.

    ``dataset`` rows must carry ``input_ids``, ``prompt_ids``,
    ``selected_positions`` (0-indexed positions within the completion whose
    *next-token* logits are scored), ``selected_index``, ``prompt_sha256``,
    ``domain`` -- see ``README.md`` for the exact schema, matching the
    HF ``datasets`` bank_a / bank_b layout used on the trl side.

    ``space`` controls what gets sketched/stored as "exact":
      - ``"logit"`` (default, unchanged from the original scorer): the
        vocabulary-centered logits, ``z - mean(z)``. Near-zero-probability
        tail tokens get equal weight to head tokens here.
      - ``"prob"``: the post-softmax probabilities ``softmax(z)`` directly
        (no centering needed -- any two probability vectors already have the
        same mean, 1/V, so centering is a no-op and is skipped). Tail tokens
        are automatically down-weighted since their probability is near 0.
        Use this to check whether a centered-logit-space finding (e.g. a
        cosine-similarity clustering) survives in probability space, or was
        partly an artifact of the uniform per-token weighting logits give
        the vocabulary tail. See README.md.

    Returns ``(sketches, exact, scalars)``:
      - ``sketches``: (prompts, 3 seeds, positions, sketch_dimension) float32,
        CountSketch of the vocabulary-centered logits or of the
        probabilities, per ``space``.
      - ``exact``: (min(exact_positions, total_positions), vocab) float32,
        uncompressed centered-logit or probability vectors (per ``space``)
        for the first ``exact_positions`` rows, used only to validate the
        sketch (see ``relations.sketch_gate``).
      - ``scalars``: per-position NLL / entropy / teacher RKL & FKL (if a
        teacher logits array is supplied) -- unaffected by ``space``, always
        computed from the raw logits.
    """
    if space not in ("logit", "prob"):
        raise ValueError(f"space must be 'logit' or 'prob', got {space!r}")
    vocab = model.config.vocab_size
    maps = sketch_maps(vocab, sketch_dimension, device)
    all_sketches: list[np.ndarray] = []
    all_exact: list[np.ndarray] = []
    scalar_rows: list[dict] = []
    for start in range(0, len(dataset), batch_size):
        records = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        lengths = [len(record["input_ids"]) for record in records]
        max_length = max(lengths)
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
            selected_hidden = hidden[batch_index].index_select(0, hidden_positions)
            logits = model.lm_head(selected_hidden).float()
            centered = logits - logits.mean(dim=-1, keepdim=True)
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            # What actually gets sketched/stored as "exact" -- see the `space`
            # docstring above. Everything else (nll, entropy, teacher RKL/FKL,
            # mean_logit_l2) is computed the same way regardless of `space`.
            sketch_source = centered if space == "logit" else probs
            per_seed = []
            for bucket, sign in maps:
                output = torch.zeros(logits.shape[0], sketch_dimension, dtype=torch.float32, device=device)
                output.scatter_add_(1, bucket.expand(logits.shape[0], -1), sketch_source * sign)
                per_seed.append(output)
            all_sketches.append(torch.stack(per_seed).cpu().numpy().astype(np.float32))
            if len(all_exact) * 64 < exact_positions:
                remaining = max(0, exact_positions - len(all_exact) * 64)
                all_exact.append(sketch_source[:remaining].cpu().numpy().astype(np.float32))
            next_positions = torch.from_numpy(prompt_length + completion_positions).to(device)
            next_tokens = input_ids[batch_index].index_select(0, next_positions)
            nll = -log_probs.gather(1, next_tokens[:, None]).squeeze(1)
            entropy = -(probs * log_probs).sum(-1)
            teacher_rkl = teacher_fkl = math.nan
            if teacher_logits is not None:
                teacher = torch.from_numpy(np.asarray(teacher_logits[start + batch_index], dtype=np.float32)).to(device)
                student_log_probs = torch.log_softmax(logits / temperature, dim=-1)
                teacher_log_probs = torch.log_softmax(teacher / temperature, dim=-1)
                student_probs = student_log_probs.exp()
                teacher_probs = teacher_log_probs.exp()
                teacher_rkl = float((student_probs * (student_log_probs - teacher_log_probs)).sum(-1).mean())
                teacher_fkl = float((teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1).mean())
            scalar_rows.append(
                {
                    "selected_index": record["selected_index"],
                    "prompt_sha256": record["prompt_sha256"],
                    "domain": record["domain"],
                    "num_positions": len(completion_positions),
                    "mean_nll": float(nll.mean()),
                    "mean_entropy": float(entropy.mean()),
                    "mean_logit_l2": float(torch.linalg.vector_norm(centered, dim=-1).mean()),
                    "mean_teacher_rkl_t0p7": teacher_rkl,
                    "mean_teacher_fkl_t0p7": teacher_fkl,
                }
            )
        del hidden, input_ids, attention
    sketches = np.stack(all_sketches, axis=0)  # prompt, seed, position, d
    exact = np.concatenate(all_exact, axis=0)[:exact_positions] if all_exact else np.empty((0, vocab), np.float32)
    return sketches, exact, pd.DataFrame(scalar_rows)


def save_score(output: Path, sketches: np.ndarray, exact: np.ndarray, scalars: pd.DataFrame, metadata: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, sketches=sketches.astype(np.float16), exact=exact.astype(np.float16))
    scalars.to_parquet(output.with_suffix(".scalars.parquet"), index=False)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
