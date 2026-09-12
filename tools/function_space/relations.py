"""Cosine / bootstrap / CountSketch-validation utilities.

Merged, math-unchanged, from the trl-side
``unifying_posttrain/experiments/m5_m6_param_native_trajectory/aggregate_function_space.py``
and ``analyze_function_relations.py``. Only the argparse ``main()`` entry
points (which wrote into that experiment's own hardcoded ``ARTIFACT`` tree)
were dropped; every function below is unchanged.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# From aggregate_function_space.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    run: str
    step: int
    path: Path


def seed_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # prompt, seed, position, bucket -> seed-wise global dot
    return np.einsum("psij,psij->s", a, b, dtype=np.float64)


def seed_norm(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(seed_dot(a, a), 0.0))


def seed_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return seed_dot(a, b) / np.maximum(seed_norm(a) * seed_norm(b), 1e-30)


def sketch_gate(all_records: list[Record]) -> tuple[pd.DataFrame, dict]:
    """Validate CountSketch cosine against exact (uncompressed) cosine.

    Only covers the ``exact_positions`` validation slice saved alongside each
    sketch (16 positions of the first prompt by default), not the whole bank.
    """
    validation: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    for record in all_records:
        with np.load(record.path) as archive:
            exact = archive["exact"].astype(np.float32)
            n = exact.shape[0]
            sketch = archive["sketches"][0, :, :n].astype(np.float32)
        validation[record.path] = (sketch, exact)
    pair_rows = []
    exact_distances = []
    sketch_distances = [[], [], []]
    for left, right in itertools.combinations(all_records, 2):
        sk_a, exact_a = validation[left.path]
        sk_b, exact_b = validation[right.path]
        n = min(exact_a.shape[0], exact_b.shape[0])
        exact_delta = exact_a[:n] - exact_b[:n]
        exact_distance = float(np.linalg.norm(exact_delta.astype(np.float64)))
        exact_cosine = float(
            np.vdot(exact_a[:n].astype(np.float64), exact_b[:n].astype(np.float64))
            / max(np.linalg.norm(exact_a[:n].astype(np.float64)) * np.linalg.norm(exact_b[:n].astype(np.float64)), 1e-30)
        )
        sketch_cosines = seed_cos(sk_a[:, :n][None, ...], sk_b[:, :n][None, ...])
        sketch_d = seed_norm((sk_a[:, :n] - sk_b[:, :n])[None, ...])
        ensemble_dot = float(np.vdot(sk_a[:, :n].astype(np.float64), sk_b[:, :n].astype(np.float64)))
        ensemble_cosine = ensemble_dot / max(
            float(np.linalg.norm(sk_a[:, :n].astype(np.float64)))
            * float(np.linalg.norm(sk_b[:, :n].astype(np.float64))),
            1e-30,
        )
        mean_seed_cosine = float(sketch_cosines.mean())
        exact_distances.append(exact_distance)
        for seed in range(3):
            sketch_distances[seed].append(float(sketch_d[seed]))
            pair_rows.append(
                {
                    "left": f"{left.run}@{left.step}",
                    "right": f"{right.run}@{right.step}",
                    "seed_index": seed,
                    "exact_cosine": exact_cosine,
                    "sketch_cosine": float(sketch_cosines[seed]),
                    "absolute_cosine_error": abs(float(sketch_cosines[seed]) - exact_cosine),
                    "mean_seed_cosine": mean_seed_cosine,
                    "mean_seed_absolute_cosine_error": abs(mean_seed_cosine - exact_cosine),
                    "ensemble_cosine": ensemble_cosine,
                    "ensemble_absolute_cosine_error": abs(ensemble_cosine - exact_cosine),
                    "exact_distance": exact_distance,
                    "sketch_distance": float(sketch_d[seed]),
                }
            )
    frame = pd.DataFrame(pair_rows)
    rank_correlations = [float(spearmanr(exact_distances, values).statistic) for values in sketch_distances] if len(exact_distances) >= 3 else [math.nan] * 3
    independent = []
    for a, b in itertools.combinations(range(3), 2):
        independent.append(float(spearmanr(sketch_distances[a], sketch_distances[b]).statistic) if len(exact_distances) >= 3 else math.nan)
    gate = {
        "num_checkpoint_pairs": len(exact_distances),
        "max_individual_seed_absolute_cosine_error": float(frame.absolute_cosine_error.max()) if len(frame) else math.nan,
        "max_mean_seed_absolute_cosine_error": float(frame.mean_seed_absolute_cosine_error.max()) if len(frame) else math.nan,
        "max_ensemble_absolute_cosine_error": float(frame.ensemble_absolute_cosine_error.max()) if len(frame) else math.nan,
        "exact_distance_spearman_by_seed": rank_correlations,
        "independent_sketch_distance_spearman": independent,
        "cosine_error_gate_lt_0_02": bool(
            len(frame)
            and frame.mean_seed_absolute_cosine_error.max() < .02
            and frame.ensemble_absolute_cosine_error.max() < .02
        ),
        "distance_order_gate_gt_0_98": bool(rank_correlations and min(rank_correlations) > .98),
        "independent_sketch_gate_gt_0_98": bool(independent and min(independent) > .98),
    }
    return frame, gate


# ---------------------------------------------------------------------------
# From analyze_function_relations.py
# ---------------------------------------------------------------------------


def cosine_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """1 - mean seed-wise cosine, and the symmetric normalized L2, over the
    whole (prompt, position) stack."""
    dots = np.einsum("psij,psij->s", a, b, dtype=np.float64)
    na = np.sqrt(np.einsum("psij,psij->s", a, a, dtype=np.float64))
    nb = np.sqrt(np.einsum("psij,psij->s", b, b, dtype=np.float64))
    denominator = na * nb
    cosine = dots / np.maximum(denominator, 1e-30)
    cosine[(na <= 1e-20) & (nb <= 1e-20)] = 1.0
    difference = np.sqrt(np.einsum("psij,psij->s", a - b, a - b, dtype=np.float64))
    normalized_l2 = difference / np.maximum(.5 * (na + nb), 1e-30)
    return float(1.0 - cosine.mean()), float(normalized_l2.mean())


def vector_norm(a: np.ndarray) -> float:
    return float(np.sqrt(np.einsum("psij,psij->", a, a, dtype=np.float64)))


def bootstrap_cosine_distance(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Prompt-cluster bootstrap of the three-sketch mean cosine distance.

    ``weights`` is (num_bootstrap_samples, num_prompts) of multinomial
    resample counts, e.g.
    ``rng.multinomial(num_prompts, [1/num_prompts]*num_prompts, size=1000)``.
    """
    dot = np.einsum("psij,psij->ps", a, b, dtype=np.float64)
    norm_a_sq = np.einsum("psij,psij->ps", a, a, dtype=np.float64)
    norm_b_sq = np.einsum("psij,psij->ps", b, b, dtype=np.float64)
    sampled_dot = weights @ dot
    sampled_a = weights @ norm_a_sq
    sampled_b = weights @ norm_b_sq
    cosine = sampled_dot / np.maximum(np.sqrt(sampled_a * sampled_b), 1e-30)
    cosine[(sampled_a <= 1e-30) & (sampled_b <= 1e-30)] = 1.0
    return 1.0 - cosine.mean(axis=1)
