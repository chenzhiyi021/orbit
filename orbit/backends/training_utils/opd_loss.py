import torch

from orbit.backends.training_utils.loss import get_log_probs_and_entropy

def opd_loss_function(args, batch, logits, sum_of_sample_mean):
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
    student_log_probs = log_probs_and_entropy["log_probs"]

    # teacher_log_probs arrives via Ray serialization (TeacherManager.score()
    # -> ray.put/ray.get), which lands tensors on CPU. student_log_probs is
    # the direct output of model forward, on GPU. Move teacher_log_probs to
    # match before any arithmetic between them.
    device = student_log_probs[0].device
    teacher_log_probs = [
        t.to(device) for t in batch["teacher_log_probs"]
    ]

    kl_advantage = [
        (t_lp - s_lp).detach()
        for t_lp, s_lp in zip(teacher_log_probs, student_log_probs, strict=True)
    ]

    student_log_probs_cat = torch.cat(student_log_probs, dim=0)
    kl_advantage_cat = torch.cat(kl_advantage, dim=0)

    per_token_loss = -kl_advantage_cat * student_log_probs_cat
    loss = sum_of_sample_mean(per_token_loss)

    if student_log_probs_cat.numel() == 0:
        loss = loss + 0 * logits.sum()

    reverse_kl = sum_of_sample_mean(
        student_log_probs_cat.exp().detach() * (-kl_advantage_cat)
    )

    reported_loss = {
        "loss": loss.clone().detach(),
        "opd_reverse_kl": reverse_kl.clone().detach(),
    }

    return loss, reported_loss