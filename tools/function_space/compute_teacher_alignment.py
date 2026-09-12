"""cos(Phi_method, Phi_teacher): how much of the *teacher's* functional update
each scored checkpoint's own update points along, instead of comparing every
method only against each other (plot_cosine_heatmap.py) or only against the
raw teacher-vs-student divergence (score_checkpoints.py's teacher_rkl/fkl).

    Phi_theta(h)   = transform(z_theta(h))   - transform(z_base(h))
    Phi_teacher(h) = transform(z_teacher(h)) - transform(z_base(h))

where ``transform`` is centered-logits or softmax-probabilities depending on
--space, matching scoring.score_bank exactly. Reports, per checkpoint:

  - cosine(Phi_method, Phi_teacher)  -- same cosine definition as
    plot_cosine_heatmap.py's matrix, just against the teacher direction
    instead of another checkpoint.
  - beta = <Phi_method, Phi_teacher> / ||Phi_teacher||^2 -- progress along
    the teacher's own displacement (1.0 = moved exactly as far as the
    teacher did, along the teacher's direction; <1 undershoot, >1 overshoot).
  - r = ||Phi_method - beta*Phi_teacher|| / ||Phi_teacher|| -- the
    orthogonal ("wasted", not-toward-teacher) component, relative to the
    teacher's own displacement size.

Does NOT require a GPU or reloading any model: reuses the already-scored
Base.npz / <run>_step*.npz sketches from score_checkpoints.py and the
cached teacher logits from cache_teacher_logits.py, and re-sketches the
teacher logits with the exact same CountSketch maps (same seeds/dimension,
same vocab as the student) so everything lives in the same sketch space and
cosines/betas are directly comparable to plot_cosine_heatmap.py's numbers.

Usage:
    python compute_teacher_alignment.py --config config.json --bank bank_a --space prob \\
        --runs "M6-FullFT-trl,M6-CoordMaskOracle-trl,M6-CoordMaskRandom-trl,M6-FullFT-orbit"

Requires, for this bank (+ this --space):
  - score_checkpoints.py already run (Base.npz + each requested run's .npz)
  - cache_teacher_logits.py already run, with banks.<bank>.teacher_cache set
    in config.json (this is the same cache score_checkpoints.py itself uses
    for the mean_teacher_rkl_t0p7 / mean_teacher_fkl_t0p7 scalars).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig

from plot_cosine_heatmap import discover_runs, load
from relations import cosine_distance, vector_norm, bootstrap_cosine_distance
from scoring import sketch_maps


def bank_dir_for(config: dict, bank: str, space: str) -> Path:
    out = Path(config["output_root"]) / bank
    return out if space == "logit" else out / space


def transform(logits: torch.Tensor, space: str) -> torch.Tensor:
    """Same per-position transform as scoring.score_bank: centered logits, or
    post-softmax probabilities (no centering needed -- see scoring.py)."""
    if space == "logit":
        return logits - logits.mean(dim=-1, keepdim=True)
    if space == "prob":
        return torch.softmax(logits, dim=-1)
    raise ValueError(f"space must be 'logit' or 'prob', got {space!r}")


def sketch_teacher(teacher_path: Path, *, vocab: int, dimension: int, space: str,
                    device: str, chunk_prompts: int = 8) -> np.ndarray:
    """Re-sketch the cached (N, 64, vocab) teacher logits with the exact same
    CountSketch maps score_bank used, in prompt chunks to bound memory."""
    teacher = np.load(teacher_path, mmap_mode="r")
    if teacher.shape[-1] != vocab:
        raise ValueError(
            f"Teacher logits vocab ({teacher.shape[-1]}) != student/base vocab ({vocab}) -- "
            f"teacher and student don't share a tokenizer/vocab, their logit indices aren't "
            f"comparable, and sketching them with the same CountSketch maps would silently "
            f"produce garbage. Stop and check {teacher_path} / the teacher/base model configs."
        )
    maps = sketch_maps(vocab, dimension, device)
    num_prompts, positions, _ = teacher.shape
    sketches = np.empty((num_prompts, len(maps), positions, dimension), dtype=np.float32)
    for start in range(0, num_prompts, chunk_prompts):
        stop = min(start + chunk_prompts, num_prompts)
        chunk = torch.from_numpy(np.asarray(teacher[start:stop], dtype=np.float32)).to(device)
        flat = transform(chunk.reshape(-1, vocab), space)  # (chunk*positions, vocab)
        for seed_index, (bucket, sign) in enumerate(maps):
            output = torch.zeros(flat.shape[0], dimension, dtype=torch.float32, device=device)
            output.scatter_add_(1, bucket.expand(flat.shape[0], -1), flat * sign)
            sketches[start:stop, seed_index] = output.reshape(stop - start, positions, dimension).cpu().numpy()
        print(f"teacher sketch {stop}/{num_prompts}", flush=True)
    return sketches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", required=True, help="key into config['banks']")
    parser.add_argument("--space", choices=("logit", "prob"), default=None,
                         help="must match what you used for score_checkpoints.py on this bank; "
                              "default: config['space'] (else 'logit')")
    parser.add_argument("--runs", default=None, help="comma-separated run names; default: auto-discover")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None, help="default: <bank_dir>/teacher_alignment.csv")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    space = args.space or config.get("space", "logit")
    bank_dir = bank_dir_for(config, args.bank, space)
    assert (bank_dir / "COMPLETE.json").exists(), (
        f"Run score_checkpoints.py --bank {args.bank}"
        f"{'' if space == 'logit' else f' --space {space}'} first (looked in {bank_dir})"
    )
    bank_cfg = config["banks"][args.bank]
    teacher_cache = bank_cfg.get("teacher_cache")
    if not teacher_cache:
        raise ValueError(
            f"config['banks']['{args.bank}']['teacher_cache'] is not set -- run cache_teacher_logits.py "
            f"for this bank first (score_checkpoints.py needs the same cache for its own teacher_rkl/fkl "
            f"scalars, so if those are populated, this should already exist)."
        )

    with np.load(bank_dir / "Base.npz") as archive:
        base_sketch = archive["sketches"].astype(np.float32)
    num_prompts = base_sketch.shape[0]
    dimension = base_sketch.shape[-1]

    vocab = AutoConfig.from_pretrained(config["base_model"]).vocab_size
    teacher_sketch = sketch_teacher(Path(teacher_cache), vocab=vocab, dimension=dimension,
                                     space=space, device=args.device)
    if teacher_sketch.shape[0] != num_prompts:
        raise ValueError(f"Teacher cache has {teacher_sketch.shape[0]} prompts, Base.npz has {num_prompts} -- "
                          f"mismatched bank / stale cache?")
    teacher_delta = teacher_sketch - base_sketch

    run_paths = discover_runs(bank_dir, None)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()] if args.runs else sorted(run_paths)
    missing = [r for r in runs if r not in run_paths]
    if missing:
        raise KeyError(f"Requested runs not found in {bank_dir}: {missing}. Available: {sorted(run_paths)}")

    rng = np.random.default_rng(config.get("bootstrap_seed", 20260826))
    weights = rng.multinomial(num_prompts, np.full(num_prompts, 1 / num_prompts), size=args.bootstrap).astype(np.float64)

    rows = []
    teacher_norm = vector_norm(teacher_delta)
    for run in runs:
        method_delta = load(run_paths[run])
        cos_d, _ = cosine_distance(method_delta, teacher_delta)
        cosine = 1 - cos_d
        boot = 1 - bootstrap_cosine_distance(method_delta, teacher_delta, weights)
        dot = float(np.einsum("psij,psij->", method_delta.astype(np.float64), teacher_delta.astype(np.float64)))
        beta = dot / max(teacher_norm ** 2, 1e-30)
        residual_norm = vector_norm(method_delta - beta * teacher_delta)
        r = residual_norm / max(teacher_norm, 1e-30)
        rows.append(dict(
            bank=args.bank, space=space, run=run, cosine_to_teacher=cosine,
            cosine_ci_low=float(np.quantile(boot, .025)), cosine_ci_high=float(np.quantile(boot, .975)),
            beta_progress_along_teacher=beta, residual_r_wasted=r,
        ))
        print(f"{run}: cos={cosine:.4f} [{rows[-1]['cosine_ci_low']:.4f}, {rows[-1]['cosine_ci_high']:.4f}]  "
              f"beta={beta:.4f}  r={r:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    output = args.output or (bank_dir / "teacher_alignment.csv")
    frame.to_csv(output, index=False)
    print(f"\nwrote {output}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
