"""Identity-checked offline data. No implicit filtering, truncation or packing."""

from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from pathlib import Path

from .protocol import M6_DATA_SHA256, digest


def read_jsonl(path):
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                raise ValueError(f"Blank data row {index}; silently dropping rows is forbidden")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Row {index} must be an object")
            yield row


@lru_cache(maxsize=4)
def immutable_rows(path):
    return list(read_jsonl(path))


def _required(obj, keys, context):
    missing = [key for key in keys if obj.get(key) in (None, "", [], {})]
    if missing:
        raise ValueError(f"{context} missing: {', '.join(missing)}")


def verify_manifest(path, profile):
    path = Path(path).resolve()
    m = json.loads(path.read_text())
    _required(m, ["schema_version", "state", "identity", "data", "source", "tokenizer", "processing", "row_count"], "data manifest")
    if m["state"] != "ready" or m["identity"] != profile["data_identity"]:
        raise ValueError("Data identity is unresolved or does not match the selected profile")
    processing = m["processing"]
    _required(processing, ["prompt_format", "assistant_format", "eos_policy", "truncation_policy", "packing_policy", "max_prompt_tokens", "max_response_tokens", "max_total_tokens"], "processing")
    if processing["packing_policy"] != "isolated_sequences":
        raise ValueError("Cross-example packing is not certified; use isolated sequences")
    if processing["truncation_policy"] not in {"prevalidated_no_truncation", "explicit_right_truncation"}:
        raise ValueError("Declare the exact truncation policy; no automatic length filtering")
    for field in ["max_prompt_tokens", "max_response_tokens", "max_total_tokens"]:
        if type(processing[field]) is not int or processing[field] <= 0:
            raise ValueError(f"Invalid {field}")
    _required(m["source"], ["dataset", "revision", "split", "provenance_sha256"], "source identity")
    _required(m["tokenizer"], ["path", "files", "chat_template_sha256"], "tokenizer identity")
    for item in [m["data"], *m["tokenizer"]["files"]]:
        _required(item, ["path", "sha256"], "hashed file")
        p = Path(item["path"])
        if not p.is_absolute():
            p = path.parent / p
        if not p.is_file() or digest(p) != item["sha256"]:
            raise ValueError(f"Data/tokenizer hash mismatch: {p}")
        item["path"] = str(p.resolve())
    tokenizer_configs = [item["path"] for item in m["tokenizer"]["files"] if Path(item["path"]).name == "tokenizer_config.json"]
    if len(tokenizer_configs) != 1:
        raise ValueError("Hashed tokenizer_config.json is required to identify the actual chat template")
    template = json.loads(Path(tokenizer_configs[0]).read_text()).get("chat_template")
    if template is None:
        raise ValueError("Explicit tokenizer chat_template required; no implicit fallback")
    template_hash = hashlib.sha256(json.dumps(template, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if template_hash != m["tokenizer"]["chat_template_sha256"]:
        raise ValueError("Chat-template hash mismatch (hash the canonical JSON chat_template value)")
    if profile["kind"] == "m6" and m["data"]["sha256"] != M6_DATA_SHA256:
        raise ValueError("M6 reference requires the audited B_target bytes and order")
    if profile["kind"] == "sft":
        _required(m["source"], ["trace_author", "trace_field", "think_tag_policy"], "SFT CoT identity")
        if profile["name"] == "sft_omr_cot" and m["source"]["dataset"] != "nvidia/OpenMathReasoning":
            raise ValueError("Official-CoT profile requires nvidia/OpenMathReasoning provenance")
        if profile["name"] == "sft_teacher_matched":
            _required(m["source"], ["teacher_model_sha256", "teacher_sampling"], "teacher-matched identity")
    if profile["kind"] != "m6":
        _required(m, ["splits"], "Train/Dev/Eval split manifest")
        _required(m["splits"], ["train_question_ids_sha256", "dev_question_ids_sha256", "eval_question_ids_sha256", "question_disjoint_audit_sha256"], "question-disjoint splits")
    label_index = None
    if profile["kind"] == "rl":
        _required(m, ["ground_truth"], "RL manifest")
        g = m["ground_truth"]
        _required(g, ["path", "sha256", "source_revision"], "authoritative ground truth")
        gp = Path(g["path"])
        if not gp.is_absolute():
            gp = path.parent / gp
        if digest(gp) != g["sha256"]:
            raise ValueError("Ground-truth source hash mismatch")
        label_index = {}
        for row in read_jsonl(gp):
            _required(row, ["question", "expected_answer"], "authoritative label row")
            if row["question"] in label_index and label_index[row["question"]] != str(row["expected_answer"]):
                raise ValueError("Conflicting authoritative labels")
            label_index[row["question"]] = str(row["expected_answer"])
    count = 0
    for row in read_jsonl(m["data"]["path"]):
        if profile["kind"] == "rl":
            _required(row, ["question", "answer", "source_question"], f"RL row {count}")
            expected = label_index.get(row["source_question"])
            if expected is None or str(row["answer"]) != expected:
                raise ValueError(f"Missing/incorrect ground truth at row {count}; never turn it into reward zero")
        elif profile["kind"] == "sft":
            _required(row, ["prompt_ids", "response_ids", "source_id", "ended_with_eos"], f"SFT row {count}")
            # False is a legitimate EOS marker for an explicitly truncated response.
            validate_token_row(row, processing)
        else:
            _required(row, ["prompt", "prompt_ids", "canonical_row"], f"M6 row {count}")
        count += 1
    if count != m["row_count"]:
        raise ValueError(f"row_count mismatch: actual {count}, manifest {m['row_count']}")
    if profile["kind"] in {"m6", "sft"} and count < profile["steps"] * profile["batch"]["prompts"]:
        raise ValueError("Offline/reference bank cannot cover the complete budget without repeating rows")
    m["manifest_sha256"] = digest(path)
    return m


def validate_token_row(row, policy):
    prompt, response = row["prompt_ids"], row["response_ids"]
    if not prompt or not response or any(type(t) is not int or t < 0 for t in prompt + response):
        raise ValueError("SFT requires nonempty, pretokenized prompt and response spans")
    if len(prompt) > policy["max_prompt_tokens"] or len(response) > policy["max_response_tokens"] or len(prompt) + len(response) > policy["max_total_tokens"]:
        raise ValueError("SFT row exceeds declared lengths; prepare/approve data before training")
    if not row["ended_with_eos"] and not (row.get("truncated") is True and policy["truncation_policy"] == "explicit_right_truncation"):
        raise ValueError("Missing EOS without an explicitly declared truncation")
    mask = row.get("response_mask", [1] * len(response))
    if len(mask) != len(response) or any(m not in (0, 1) for m in mask):
        raise ValueError("Invalid response mask")
    return prompt + response, mask


def supervised_pairs(row):
    """Positions predict targets at the NEXT position, including first response/EOS."""
    prompt, response = row["prompt_ids"], row["response_ids"]
    return list(range(len(prompt) - 1, len(prompt) + len(response) - 1)), response


def offline_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        raise ValueError("Offline SFT provider is training-only; use the formal evaluator")
    from orbit.utils.types import Sample
    from orbit.rollout.base_types import RolloutFnTrainOutput

    # Index from rollout_id makes replay/resume independent of mutable loader RNG.
    data = immutable_rows(args.posttrain_data_manifest["data"]["path"])
    first = rollout_id * args.rollout_batch_size
    if data_source.sample_offset != first:
        raise ValueError("Offline SFT cursor mismatch")
    rows = data[first : first + args.rollout_batch_size]
    if len(rows) != args.rollout_batch_size:
        raise ValueError("SFT data exhausted; no implicit repeat or newly filtered subset")
    samples = []
    for offset, row in enumerate(rows):
        tokens, mask = validate_token_row(row, args.posttrain_data_manifest["processing"])
        sample = Sample(index=first + offset, group_index=first + offset)
        sample.prompt = ""
        sample.tokens = tokens
        sample.response_length = len(row["response_ids"])
        sample.loss_mask = mask
        sample.reward = 0.0
        sample.metadata = {"source_id": row["source_id"], "truncated": row.get("truncated", False)}
        samples.append([sample])
    data_source.sample_offset = first + len(rows)
    from .batch import record_samples

    record_samples(args, rollout_id, [g[0] for g in samples])
    return RolloutFnTrainOutput(samples=samples)


class OfflineDataSource:
    """Dataset/cursor interface for pretokenized SFT, with no re-tokenization/filtering."""

    def __init__(self, args):
        self.args = args
        self.dataset = immutable_rows(args.posttrain_data_manifest["data"]["path"])
        self.sample_offset = 0

    def get_samples(self, num_samples):
        raise RuntimeError("offline_rollout consumes the immutable token bank directly")

    def add_samples(self, samples):
        raise RuntimeError("Offline bank is immutable")

    def save(self, rollout_id):
        import torch

        path = Path(self.args.save) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sample_offset": self.sample_offset, "data_sha256": self.args.posttrain_data_manifest["data"]["sha256"]}, path)

    def load(self, rollout_id=None):
        if not self.args.posttrain_resume:
            return
        import torch

        value = torch.load(self.args.posttrain_data_cursor, weights_only=False)
        if value["data_sha256"] != self.args.posttrain_data_manifest["data"]["sha256"]:
            raise ValueError("SFT resume data changed")
        expected = (rollout_id + 1) * self.args.rollout_batch_size
        if value["sample_offset"] != expected:
            raise ValueError("SFT resume cursor does not match optimizer transaction")
        self.sample_offset = expected
