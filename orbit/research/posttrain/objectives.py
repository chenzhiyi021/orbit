"""Production mathematical paths with the audited loss values, not only gradients."""

from __future__ import annotations
import torch


def _safe_clamp_log_ratio(log_ratio):
    return torch.nan_to_num(log_ratio.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)


def policy_tokens(new_logp, old_logp, advantages, *, force_one=False, clip_low=0.2, clip_high=0.2):
    if advantages.requires_grad or old_logp.requires_grad:
        raise ValueError("Cached advantages and learner-old logprobs must be detached")
    if any(not torch.isfinite(v).all() for v in (new_logp, old_logp, advantages)):
        raise ValueError("Non-finite policy inputs; research profiles do not sanitize and continue")
    log_ratio = new_logp - (new_logp.detach() if force_one else old_logp)
    ratio = _safe_clamp_log_ratio(log_ratio).exp()
    unclipped = -ratio * advantages
    clipped = -ratio.clamp(1 - clip_low, 1 + clip_high) * advantages
    return torch.maximum(unclipped, clipped), (clipped > unclipped).float(), ratio


def m6_advantages(student, teacher, scale=0.1):
    if student.requires_grad or teacher.requires_grad:
        raise ValueError("M6 teacher and cached learner must come from no_grad scoring")
    if student.shape != teacher.shape or not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("Invalid sampled logprob alignment/non-finite values")
    return (teacher - student) * scale


def token_terms(values, masks):
    if len(values) != len(masks):
        raise ValueError("Value/mask batch mismatch")
    numerator = sum((v * m.to(v)).sum() for v, m in zip(values, masks, strict=True))
    denominator = sum(m.sum().clamp_min(1) for m in masks)
    return numerator, denominator


def global_token_mean(values, masks):
    numerator, denominator = token_terms(values, masks)
    return numerator / denominator


def raw_sft_nll(logits, targets):
    # SFT loss temperature is exactly one; do not inherit rollout T=.7.
    return -logits.float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def group_advantages(rewards, group_indices, epsilon=1e-6):
    # The actual zh helper already present on main: use one production definition.
    from orbit.utils.reward_normalization import normalize_grouped_rewards

    return normalize_grouped_rewards(rewards, group_indices, normalize_by_std=True, epsilon=epsilon)
