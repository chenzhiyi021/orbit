"""Fresh M6 sampling and strict verifier dispatch; no async-policy collection."""

from __future__ import annotations
import json
import time
from pathlib import Path


async def teacher_reward(args, sample, **kwargs):
    if kwargs.get("evaluation"):
        raise ValueError("Formal evaluation must use a verifier, never teacher scoring as reward")
    from orbit.utils.http_utils import post

    payload = {"input_ids": sample.tokens, "sampling_params": {"temperature": 0, "max_new_tokens": 0}, "return_logprob": True, "logprob_start_len": 0}
    started = time.monotonic()
    output = await post(args.posttrain_teacher_url.rstrip("/") + "/generate", payload)
    sample.metadata["teacher_score_seconds"] = time.monotonic() - started
    audit = json.loads(Path(args.posttrain_teacher_audit).read_text())
    if audit["temperature"] != 0.7 or audit["temp_scaled_logprobs"] is not True:
        raise ValueError("Teacher prefill T=.7 has not been observed in the actual server")
    if Path(audit["model_path"]).resolve() != Path(args.posttrain_teacher_model).resolve():
        raise ValueError("Teacher process model identity mismatch")
    values = [item[0] for item in output["meta_info"]["input_token_logprobs"]][-sample.response_length :]
    if len(values) != sample.response_length or any(v is None for v in values):
        raise ValueError("Teacher returned an incomplete response-token score span")
    sample.teacher_log_probs = values
    return 0.0


def teacher_post_process(args, samples):
    import torch

    for sample in samples:
        sample.teacher_log_probs = torch.tensor(sample.teacher_log_probs, dtype=torch.float32)
    return [0.0] * len(samples), [0.0] * len(samples)


async def generate_m6_sample(args, sample, sampling_params, evaluation=False):
    from orbit.rollout.sglang_rollout import generate, GenerateState

    if evaluation:
        raise ValueError("M6 reference provider is training-only")
    from .data import immutable_rows

    rows = immutable_rows(args.prompt_data)
    row = rows[sample.index]
    if GenerateState(args).tokenizer.encode(sample.prompt, add_special_tokens=False) != row["prompt_ids"]:
        raise ValueError("M6 sealed prompt tokenization changed")
    sampling = dict(sampling_params, sampling_seed=args.posttrain_protocol["data_seed"] + row["canonical_row"])
    return await generate(args, sample, sampling, evaluation=False)


def m6_rollout(args, rollout_id, data_source, evaluation=False):
    from orbit.rollout.sglang_rollout import generate_rollout

    started = time.monotonic()
    output = generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    samples = [sample for group in output.samples for sample in group]
    if len(samples) != 256 or sorted(s.index for s in samples) != list(range(rollout_id * 256, (rollout_id + 1) * 256)):
        raise ValueError("M6 reference requires the sealed prompt cursor, no retries/reordering of the selection")
    for s in samples:
        if not s.weight_versions or set(map(str, s.weight_versions)) != {str(rollout_id + 1)}:
            raise ValueError("M6 rollout uses a stale or mixed policy version")
        if not 0 < s.response_length <= 8192 or len(s.tokens) > 12288:
            raise ValueError("M6 length contract violated")
        if s.loss_mask is not None and s.loss_mask != [1] * s.response_length:
            raise ValueError("M6 reference retains every generated token, including capped responses")
    from .batch import record_samples

    record_samples(args, rollout_id, samples, elapsed_seconds=time.monotonic() - started)
    return output


async def verified_math_reward(args, sample, **kwargs):
    # The authoritative labels were checked over every row at launch. Also reject
    # malformed rows at runtime instead of silently rewarding them zero.
    if sample.label is None or str(sample.label).strip() == "":
        raise ValueError("Missing ground truth during RL rollout")
    from orbit.rollout.rm_hub.math_utils import grade_answer_verl

    return float(grade_answer_verl(sample.response, sample.label))


def rl_rollout(args, rollout_id, data_source, evaluation=False):
    from orbit.rollout.sglang_rollout import generate_rollout
    from .batch import record_samples

    started = time.monotonic()
    output = generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    samples = [sample for group in output.samples for sample in group]
    if len(samples) != 256:
        raise ValueError("RL reference requires exactly P64 x G4 responses")
    groups = {}
    for sample in samples:
        groups[sample.group_index] = groups.get(sample.group_index, 0) + 1
    if len(groups) != 64 or set(groups.values()) != {4}:
        raise ValueError("RL prompt groups changed before normalization")
    record_samples(args, rollout_id, samples, elapsed_seconds=time.monotonic() - started)
    return output


async def generate_rl_sample(args, sample, sampling_params, evaluation=False):
    from orbit.rollout.sglang_rollout import generate, GenerateState

    prompt_ids = GenerateState(args).tokenizer.encode(sample.prompt, add_special_tokens=False)
    available = args.rollout_max_context_len - len(prompt_ids)
    if available <= 0:
        raise ValueError("Declared context leaves no response tokens; fix the data manifest")
    sampling = dict(sampling_params, max_new_tokens=min(sampling_params["max_new_tokens"], available))
    return await generate(args, sample, sampling, evaluation=evaluation)
