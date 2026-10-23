"""Protocol resolution without importing CUDA, Megatron, Ray or model weights."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

PROFILES = Path(__file__).resolve().parents[3] / "examples/posttrain_subspace/profiles"
DEPENDENCIES = {
    "megatron-bridge": "9d4c97c6a6ccfa1557ac0f0e8d0e0db52529cc2f",
    "megatron-core": "980211ae6308dd541ec24bfe5af664ef31215256",
    "sglang": "da8625376271adcfdb129a2e1ca632005ff9b9a4",
}
M6_DATA_SHA256 = "bfebe57eb0b3bccafcfa7efed981ed946f6af6192db363c661f19c46f78a1e0d"


def load_profile(name: str) -> dict:
    if name not in {"m6_mixed_r2_reference", "rl_math_grpo_r4", "sft_omr_cot", "sft_teacher_matched", "sft_btarget_official_cot"}:
        raise ValueError(f"Unknown profile: {name}")
    return json.loads((PROFILES / f"{name}.json").read_text())


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def applied_lr(profile: dict, completed_updates: int) -> float:
    if not 0 <= completed_updates <= profile["steps"]:
        raise ValueError("LR index must be a completed-update count within the declared horizon")
    peak = profile["optimizer"]["lr"]
    if peak is None:
        raise ValueError("SFT LR must be explicitly selected on Dev before launch")
    warmup = profile["optimizer"]["warmup_steps"]
    floor = profile["optimizer"]["min_lr"]
    if completed_updates < warmup:
        return peak * completed_updates / warmup
    style = profile["optimizer"]["schedule"]
    if style == "constant":
        return peak
    if style != "cosine":
        raise ValueError(f"Unresolved/unsupported schedule: {style}")
    # Preserve Megatron's actual floating-point operation order, not merely an
    # algebraically equal cosine expression: this matches the archived LR trace.
    decay_ratio = float(completed_updates - warmup) / float(profile["steps"] - warmup)
    coefficient = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
    return floor + coefficient * (peak - floor)


def validate(profile: dict, *, require_resolved: bool = True) -> None:
    p = profile
    failures = []
    for key in ["async_rollout", "tis", "score_centering", "delta_reward", "normalize_advantages"]:
        if p[key] is not False:
            failures.append(f"{key} must be false")
    if p["updates_per_rollout"] != 1:
        failures.append("Only one optimizer update per rollout is certified")
    if p["batch"]["prompts"] is not None and p["batch"]["prompts"] * p["batch"]["samples_per_prompt"] != p["batch"]["global_sequences"]:
        failures.append("P * G must equal the optimizer batch")
    if p["loss_reduction"] != "global_token_with_zero_mask_floor":
        failures.append("Global-token reducer including the zero-mask denominator floor is required")
    if not p["checkpoint"]["optimizer"] or not p["checkpoint"]["rng"] or not p["check_nonfinite"]:
        failures.append("Optimizer/RNG checkpointing and non-finite checks are mandatory")
    if p["loss_temperature"] != (0.7 if p["kind"] == "m6" else 1.0):
        failures.append("Loss temperature differs from the named protocol")
    if p["adapter"]["initialization"] not in {None, "name_seeded_kaiming", "xavier_normal"}:
        failures.append("Unsupported factor initialization; do not silently substitute Xavier")
    if p["kind"] == "m6":
        expected = load_profile("m6_mixed_r2_reference")
        for key in ["steps", "batch", "parallel", "optimizer", "adapter", "sampling", "reward_scale", "teacher_temperature", "training_seed", "data_seed", "init_seed"]:
            if p[key] != expected[key]:
                failures.append(f"M6 reference cannot override {key}; define a separately qualified profile")
    if p["kind"] == "rl":
        reference = load_profile("rl_math_grpo_r4")
        for key in ["steps", "batch", "parallel", "optimizer", "adapter", "sampling", "reward_scale"]:
            if p[key] != reference[key]:
                failures.append(f"RL reference cannot override {key}; define a separate protocol")
        if p["group_std"] != "sample" or p["zero_variance_groups"] != "keep":
            failures.append("RL requires sample std (G-1) and retained zero-variance groups")
        if (p["parallel"]["tp"], p["parallel"]["dp"]) != (2, 2):
            failures.append("RL reference is TP2/DP2; another layout requires separate qualification")
        if p["adapter"]["initialization"] != "xavier_normal":
            failures.append("RL reference preserves Xavier normal initialization")
    if p["adapter"]["initialization"] == "name_seeded_kaiming" and p["parallel"]["tp"] != 1:
        failures.append("Name-seeded initialization currently requires TP1 to preserve global factor coordinates")
    if require_resolved:
        micro = p["batch"]["micro_sequences"]
        tokens = p["batch"]["max_tokens_per_gpu"]
        if micro is None and (type(tokens) is not int or tokens <= 0):
            failures.append("Choose a positive microbatch size or a positive dynamic token budget")
        if micro is not None and (type(micro) is not int or micro <= 0):
            failures.append("micro_sequences must be a positive integer")

        def scan(value, path="profile"):
            if value is None:
                failures.append(f"{path} is unresolved")
            elif isinstance(value, dict):
                for key, child in value.items():
                    if key not in {"micro_sequences", "max_tokens_per_gpu", "teacher_temperature", "group_std"}:
                        scan(child, f"{path}.{key}")

        scan(p)
    if failures:
        raise ValueError("Protocol preflight failed:\n- " + "\n- ".join(failures))


def resolve_sft(name: str, decision: dict | None) -> dict:
    p = load_profile(name)
    if p["kind"] != "sft":
        if decision:
            raise ValueError("Only unresolved SFT scientific fields accept a decision file")
        return p
    if decision:
        allowed = {"optimizer", "adapter", "parallel", "batch", "training_seed", "data_seed", "init_seed"}
        if set(decision) - allowed:
            raise ValueError("SFT decision contains unsupported fields")
        for key, value in decision.items():
            if isinstance(value, dict):
                unknown = set(value) - set(p[key])
                if unknown:
                    raise ValueError(f"Unknown SFT fields: {unknown}")
                p[key].update(copy.deepcopy(value))
            else:
                p[key] = value
    if p["optimizer"]["lr"] is not None and p["optimizer"]["min_lr"] == "0.1 * selected_lr":
        p["optimizer"]["min_lr"] = p["optimizer"]["lr"] * 0.1
    return p
