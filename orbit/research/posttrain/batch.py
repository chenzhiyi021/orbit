"""Canonical observation schema; adapters never overwrite a logprob's meaning."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ObservedSample:
    prompt_id: int
    group_id: int
    token_ids: list[int]
    response_mask: list[int]
    response_length: int
    truncated: bool
    natural_termination: bool
    policy_versions: list[str]
    generation_temperature: float | None
    loss_temperature: float
    behavior_logprobs: list[float] | None
    teacher_logprobs: list[float] | None
    raw_reward: float | None


def record_samples(args, rollout_id, samples, *, elapsed_seconds=None):
    rows = []
    p = args.posttrain_protocol
    for sample in samples:
        length = sample.response_length
        mask = sample.loss_mask if sample.loss_mask is not None else [1] * length
        if length <= 0 or len(mask) != length or sum(mask) <= 0 or getattr(sample, "remove_sample", False):
            raise ValueError("Main research runs require nonempty valid response masks; no silent sample removal")
        if any(x not in (0, 1) for x in mask):
            raise ValueError("Invalid response mask")
        status = getattr(getattr(sample, "status", None), "name", None)
        truncated = status == "TRUNCATED" or sample.metadata.get("truncated", False)
        row = ObservedSample(
            sample.index,
            sample.group_index,
            list(sample.tokens),
            list(mask),
            length,
            truncated,
            not truncated,
            list(map(str, sample.weight_versions or [])),
            None if p["kind"] == "sft" else p["sampling"]["temperature"],
            p["loss_temperature"],
            getattr(sample, "rollout_log_probs", None),
            getattr(sample, "teacher_log_probs", None),
            float(sample.reward) if isinstance(sample.reward, (int, float)) else None,
        )
        rows.append(row)
    if getattr(args, "posttrain_observe", True):
        target = Path(args.posttrain_run_dir) / "rollouts"
        target.mkdir(parents=True, exist_ok=True)
        with (target / f"{rollout_id:06d}.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(asdict(row), allow_nan=False) + "\n")
        summary = {"rollout_id": rollout_id, "responses": len(rows), "tokens": sum(x.response_length for x in rows), "truncated": sum(x.truncated for x in rows), "zero_mask_samples": 0, "wall_seconds": elapsed_seconds, "teacher_score_seconds_sum": sum(s.metadata.get("teacher_score_seconds", 0.0) for s in samples)}
        (target / f"{rollout_id:06d}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return rows
