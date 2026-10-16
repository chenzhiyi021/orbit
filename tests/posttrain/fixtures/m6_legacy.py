from __future__ import annotations
import torch
import hashlib
_LOG_RATIO_EXP_CLAMP = 20.0
PEAKS = {2: 3.2e-4, 4: 2.4e-4, 8: 1.6e-4, 16: 1.08e-4}


def _safe_clamp_log_ratio(log_ratio: torch.Tensor) -> torch.Tensor:
    log_ratio = torch.nan_to_num(
        log_ratio.float(),
        nan=0.0,
        posinf=_LOG_RATIO_EXP_CLAMP,
        neginf=-_LOG_RATIO_EXP_CLAMP,
    )
    return torch.clamp(log_ratio, min=-_LOG_RATIO_EXP_CLAMP, max=_LOG_RATIO_EXP_CLAMP)

def _safe_exp_neg_ppo_kl(ppo_kl: torch.Tensor) -> torch.Tensor:
    return _safe_clamp_log_ratio(-ppo_kl).exp()

@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = _safe_exp_neg_ppo_kl(ppo_kl)
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac

def opd_mopd_advantages(
    rollout_data: dict,
    student_log_probs: list[torch.Tensor],
    response_lengths: list[int],
) -> list[torch.Tensor]:
    """Pure on-policy distillation (MOPD) advantage: teacher_logp - student_logp.

    Args:
        rollout_data: Rollout batch dict; must contain `teacher_log_probs`
            (list[torch.Tensor], one per sample, aligned to `response_lengths`).
        student_log_probs: Current policy log-probs per sample.
        response_lengths: Response length per sample.

    Returns:
        list[torch.Tensor]: `teacher_i[-L_i:] - student_i` per sample.
    """
    precomputed_reverse_kls = rollout_data.get("opd_reverse_kl")
    if precomputed_reverse_kls is not None:
        # Top-k OPD: rollout-side scoring already computed the per-token
        # weighted reverse KL; the MOPD advantage is simply its negation.
        if len(precomputed_reverse_kls) != len(student_log_probs):
            raise ValueError(
                f"OPD length mismatch: opd_reverse_kl={len(precomputed_reverse_kls)} "
                f"student={len(student_log_probs)}"
            )
        device = student_log_probs[0].device
        out = []
        for i, reverse_kl in enumerate(precomputed_reverse_kls):
            if not torch.is_tensor(reverse_kl):
                reverse_kl = torch.tensor(reverse_kl, dtype=torch.float32)
            reverse_kl = reverse_kl.to(device=device)
            if reverse_kl.shape != student_log_probs[i].shape:
                raise ValueError(
                    f"OPD shape mismatch at {i}: opd_reverse_kl={tuple(reverse_kl.shape)} "
                    f"student={tuple(student_log_probs[i].shape)}"
                )
            out.append(-reverse_kl)
        return out

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(
            "advantage_estimator='on_policy_distillation' needs teacher_log_probs. "
            "Enable a teacher producer with --opd-type {megatron,sglang}."
        )
    if len(teacher_log_probs) != len(student_log_probs):
        raise ValueError(f"OPD length mismatch: teacher={len(teacher_log_probs)} student={len(student_log_probs)}")
    device = student_log_probs[0].device
    out = []
    for i, (t, s, response_length) in enumerate(
        zip(teacher_log_probs, student_log_probs, response_lengths, strict=False)
    ):
        t = t.to(device=device)[-response_length:]
        if t.shape != s.shape:
            raise ValueError(f"OPD shape mismatch at {i}: teacher={tuple(t.shape)} student={tuple(s.shape)}")
        out.append(t - s)
    return out

def initial_factors(shape,spec,bank,name,device):
    if spec['basis']=='Kaiming-A':
        generator=torch.Generator(device=device).manual_seed(seed('native:'+name))
        m,n=shape
        a=torch.empty(spec['rank'],n,device=device,dtype=torch.float32).uniform_(-n**-.5,n**-.5,generator=generator)
        b=torch.zeros(m,spec['rank'],device=device,dtype=torch.float32)
        return dict(A=a,B=b,train_A=True,train_B=True)
    return factors(shape,bank.lora(name,spec['side']).to(device),spec['side'])

def seed(name):
    return int(hashlib.sha256(('seed22:'+name).encode()).hexdigest()[:8],16)

def lr_at(completed_steps,peak_lr):
    assert 0<=completed_steps<=100 and peak_lr in PEAKS.values()
    if completed_steps<10:return peak_lr*completed_steps/10
    floor=peak_lr*.1
    return floor+(peak_lr-floor)*.5*(1+math.cos(math.pi*(completed_steps-10)/90))
