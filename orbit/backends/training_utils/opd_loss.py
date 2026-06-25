import torch

from .loss import get_log_probs_and_entropy

def opd_loss_function(
    args,
    batch,
    logits,
    sum_of_sample_mean,
):
    """
    Per-token reverse-KL on-policy distillation loss.

    Implements the OPD gradient from Thinking Machines / Qwen3 technical
    report (sampled-token variant): treat (teacher_log_prob - student_log_prob)
    as a per-token advantage-like coefficient on the policy-gradient term
    log pi_theta(y_t). This is mathematically the gradient of the reverse-KL
    D_KL(pi_theta || pi_T) at each token position, NOT a PPO-clipped loss --
    OPD has no trust region / old-policy concept, so no clipping is applied.

    Args:
        args: must have loss_masks/total_lengths/response_lengths in batch,
            plus args.qkv_format for get_log_probs_and_entropy.
        batch: must contain "teacher_log_probs" (list of 1D tensors, one per
            sample, position 0 already masked to 0 by train_student()),
            "unconcat_tokens", "total_lengths", "response_lengths",
            "loss_masks".
        logits: student model logits, shape [1, T, V].
        sum_of_sample_mean: reduction function from loss_function() dispatcher.

    Returns:
        (loss, reported_loss) matching the interface of policy_loss_function
        / value_loss_function.

    NOTE: this has not been numerically validated end-to-end -- it has been
    written to match the calling interface (confirmed via policy_loss_function)
    and the data shapes confirmed via TeacherManager.score() testing, but the
    actual gradient values/training dynamics have not been checked against a
    reference implementation.
    """
    total_lengths = batch["total_lengths"]
    response_lengths = batch["response_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        entropy_no_grad=True,
        max_seq_lens=max_seq_lens,
    )
    student_log_probs = log_probs_and_entropy["log_probs"]  # list[Tensor], one per sample

    teacher_log_probs = batch["teacher_log_probs"]  # list[Tensor], one per sample, pos 0 already 0

    # Per-token reverse-KL advantage: detached, since this is a coefficient
    # multiplying the policy-gradient term, not something we backprop through
    # directly (mirrors how ppo_kl / advantages are detached coefficients in
    # policy_loss_function).
    kl_advantage = [
        (t_lp - s_lp).detach()
        for t_lp, s_lp in zip(teacher_log_probs, student_log_probs, strict=False)
    ]

    student_log_probs_cat = torch.cat(student_log_probs, dim=0)
    kl_advantage_cat = torch.cat(kl_advantage, dim=0)

    # Gradient of reverse-KL: -(teacher_lp - student_lp) * grad(student_lp),
    # i.e. minimize -kl_advantage * log_prob so that student moves toward
    # teacher's distribution at tokens where teacher assigns higher
    # probability than student currently does.
    per_token_loss = -kl_advantage_cat * student_log_probs_cat

    loss = sum_of_sample_mean(per_token_loss)

    # Make sure gradient backprops even on empty batches (mirrors
    # policy_loss_function's same guard).
    if student_log_probs_cat.numel() == 0:
        loss = loss + 0 * logits.sum()

    # Reported metrics: actual per-token reverse-KL magnitude (positive,
    # detached) is the natural "distance to teacher" diagnostic -- report
    # the mean over real (non-padded) tokens.
    reverse_kl = sum_of_sample_mean(kl_advantage_cat.abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "opd_reverse_kl": reverse_kl.clone().detach(),
    }

    return loss, reported_loss