"""Compile a named protocol to Orbit arguments; preflight precedes Ray/GPU allocation."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .data import verify_manifest
from .protocol import resolve_sft, validate, applied_lr, digest, fingerprint

ROOT = Path(__file__).resolve().parents[3]


def compile_arguments(profile, manifest, model, output, *, teacher=None, teacher_url=None, teacher_audit=None, resume=None, cursor=None, freeze="none"):
    p = profile
    b = p["batch"]
    opt = p["optimizer"]
    adapter = p["adapter"]
    par = p["parallel"]
    lengths = manifest["processing"]

    def length(key):
        value = p["sampling"][key]
        return lengths[key] if value == "from_data_manifest" else value

    cfg = dict(
        posttrain_source_root=str(ROOT),
        posttrain_protocol=p,
        posttrain_data_manifest=manifest,
        posttrain_run_dir=str(output),
        posttrain_observe=True,
        posttrain_resume=resume,
        posttrain_data_cursor=cursor,
        posttrain_freeze_factor=freeze,
        posttrain_teacher_model=teacher,
        posttrain_teacher_url=teacher_url,
        posttrain_teacher_audit=teacher_audit,
        actor_num_nodes=1,
        actor_num_gpus_per_node=par["tp"] * par["dp"],
        hf_checkpoint=str(model),
        load=str(model),
        save=str(output / "checkpoints"),
        save_interval=20,
        no_save_optim=False,
        no_save_rng=False,
        megatron_to_hf_mode="bridge",
        prompt_data=manifest["data"]["path"],
        input_key="prompt" if p["kind"] == "m6" else "question",
        rollout_global_dataset=True,
        rollout_shuffle=p["shuffle"],
        rollout_seed=p["data_seed"],
        num_rollout=p["steps"],
        rollout_batch_size=b["prompts"],
        n_samples_per_prompt=b["samples_per_prompt"],
        global_batch_size=b["global_sequences"],
        micro_batch_size=b["micro_sequences"] or 1,
        use_dynamic_batch_size=b["micro_sequences"] is None,
        max_tokens_per_gpu=b["max_tokens_per_gpu"],
        rollout_temperature=p["sampling"]["temperature"],
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_prompt_len=length("max_prompt_tokens"),
        rollout_max_response_len=length("max_response_tokens"),
        rollout_max_context_len=length("max_total_tokens"),
        seq_length=length("max_total_tokens"),
        max_position_embeddings=length("max_total_tokens"),
        optimizer="adam",
        lr=opt["lr"],
        min_lr=opt["min_lr"],
        lr_decay_style=opt["schedule"],
        lr_warmup_iters=opt["warmup_steps"],
        lr_warmup_init=0.0,
        lr_decay_iters=p["steps"],
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        clip_grad=1.0,
        peft_method=adapter["method"],
        lora_type="lora",
        lora_rank=adapter["rank"],
        lora_alpha=adapter["alpha"],
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        tensor_model_parallel_size=par["tp"],
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=par["tp"] > 1,
        qkv_format="bshd" if p["kind"] == "m6" else "thd",
        calculate_per_token_loss=True,
        loss_type="sft_loss" if p["kind"] == "sft" else "policy_loss",
        compute_advantages_and_returns=p["kind"] != "sft",
        advantage_estimator="on_policy_distillation" if p["kind"] == "m6" else "grpo",
        normalize_advantages=False,
        rewards_normalization=True,
        grpo_std_normalization=p["kind"] == "rl",
        use_tis=False,
        get_mismatch_metrics=False,
        use_opsm=False,
        use_rollout_logprobs=False,
        keep_old_actor=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        kl_coef=0.0,
        kl_loss_coef=0.0,
        use_kl_loss=False,
        entropy_coef=0.0,
        use_critic=False,
        eval_interval=None,
        seed=p["training_seed"],
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_backend="flash",
        attention_softmax_in_fp32=True,
        accumulate_allreduce_grads_in_fp32=True,
        gradient_accumulation_fusion=p["kind"] == "m6",
        check_for_nan_in_loss_and_grad=True,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        # Untruncated long-CoT SFT puts ~45k tokens x 152k vocab logits on one GPU: compute the
        # response log-probs in sequence chunks, each recomputed in backward (ppo_utils), instead of
        # keeping full-size fp32 intermediates (same math, memory only).
        log_probs_chunk_size=4096 if p["kind"] == "sft" else -1,
        recompute_loss_function=False,
        keep_bf16_logits=p["kind"] == "sft",
        offload_train=False,
        offload_train_async=False,
        offload_rollout=p["kind"] != "sft",
        num_gpus_per_node=par["tp"] * par["dp"],
        rollout_num_gpus=par["tp"] * par["dp"],
        rollout_num_gpus_per_engine=1,
        colocate=True,
        te_rng_tracker=True,
        debug_train_only=p["kind"] == "sft",
        use_wandb=False,
        router_disable_circuit_breaker=True,
        sglang_mem_fraction_static=0.20 if p["kind"] == "m6" else 0.4,
        sglang_server_concurrency=8 if p["kind"] == "m6" else 64,
        sglang_max_running_requests=16 if p["kind"] == "m6" else 512,
        sglang_attention_backend="fa3",
        sglang_sampling_backend="flashinfer",
        sglang_lora_backend="triton",
    )
    if p["kind"] == "m6":
        if not teacher or not teacher_url or not teacher_audit:
            raise ValueError("M6 needs the fixed teacher model, URL and fresh prefill audit path")
        cfg.update(
            rollout_function_path="orbit.research.posttrain.rollout.m6_rollout", custom_generate_function_path="orbit.research.posttrain.rollout.generate_m6_sample", custom_rm_path="orbit.research.posttrain.rollout.teacher_reward", custom_reward_post_process_path="orbit.research.posttrain.rollout.teacher_post_process"
        )
    elif p["kind"] == "rl":
        cfg.update(
            rollout_function_path="orbit.research.posttrain.rollout.rl_rollout",
            custom_generate_function_path="orbit.research.posttrain.rollout.generate_rl_sample",
            label_key="answer",
            apply_chat_template=True,
            apply_chat_template_kwargs={"enable_thinking": False},
            rm_type="math",
            custom_rm_path="orbit.research.posttrain.rollout.verified_math_reward",
        )
    else:
        cfg.update(rollout_function_path="orbit.research.posttrain.data.offline_rollout", input_key="source_id", apply_chat_template=False, debug_train_only=True, data_source_path="orbit.research.posttrain.data.OfflineDataSource")
    if resume and not cursor:
        raise ValueError("A full resume requires the matching global_dataset_state_dict file")
    return cfg


def source_files():
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / "orbit").rglob("*.py"))}


def source_digest():
    return fingerprint(source_files())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile")
    ap.add_argument("--data-manifest", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--sft-decision", type=Path)
    ap.add_argument("--environment-roots", type=Path)
    ap.add_argument("--teacher")
    ap.add_argument("--teacher-url")
    ap.add_argument("--teacher-audit")
    ap.add_argument("--resume")
    ap.add_argument("--data-cursor")
    ap.add_argument("--freeze-factor", choices=["none", "right", "left"], default="none")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--qualification-updates", type=int, choices=[1, 2, 3])
    args = ap.parse_args()
    decision = json.loads(args.sft_decision.read_text()) if args.sft_decision else None
    p = resolve_sft(args.profile, decision)
    validate(p)
    manifest = verify_manifest(args.data_manifest, p)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; run artifacts are immutable")
    if args.resume:
        required = Path(args.resume) / "posttrain_state_rank0.pt"
        if not required.is_file():
            raise ValueError("Resume must contain complete posttrain state, not only an HF adapter")
        complete_file = Path(args.resume) / "POSTTRAIN_COMPLETE.json"
        if not complete_file.is_file():
            raise ValueError("Resume checkpoint has not completed all ranks and its data cursor")
        complete = json.loads(complete_file.read_text())
        if complete["protocol_hash"] != fingerprint(p):
            raise ValueError("Resume protocol differs from the checkpoint protocol")
        for filename, expected in complete["files"].items():
            if digest(filename) != expected:
                raise ValueError(f"Incomplete or changed checkpoint file: {filename}")
        if Path(args.data_cursor or "").resolve() != Path(complete["data_cursor"]).resolve():
            raise ValueError("Use the data cursor belonging to this exact complete checkpoint")
    cfg = compile_arguments(p, manifest, args.model.resolve(), output, teacher=args.teacher, teacher_url=args.teacher_url, teacher_audit=args.teacher_audit, resume=args.resume, cursor=args.data_cursor, freeze=args.freeze_factor)
    cfg["posttrain_qualification_updates"] = args.qualification_updates
    cfg["posttrain_execution_ready"] = False
    cfg["posttrain_source_hash"] = source_digest()
    environment = None
    if args.execute and args.environment_roots:
        from .environment import inspect_environment

        # The M6 source-anchor and GPU-acceptance gates are disabled: the anchor commits are not
        # reachable and no acceptance tooling exists yet. The environment is still recorded
        # (environment.lock, "verified" says whether it matches the anchor).
        environment = inspect_environment(json.loads(args.environment_roots.read_text()), check=False)
        if not environment["verified"]:
            print("NOTE: environment differs from the M6 anchor (recorded, not enforced):\n  " + "\n  ".join(environment["errors"]), file=sys.stderr)
    cfg["posttrain_execution_ready"] = bool(args.execute)
    model_files = sorted(path for path in args.model.resolve().iterdir() if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"})
    if not model_files or not (args.model / "config.json").is_file():
        raise ValueError("The base model must be a complete local HuggingFace checkpoint")
    model_config = json.loads((args.model / "config.json").read_text())
    expected_shape = {"hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8}
    if any(model_config.get(key) != value for key, value in expected_shape.items()):
        raise ValueError("Profile model geometry must be Qwen3-1.7B")
    for item in manifest["tokenizer"]["files"]:
        actual = args.model / Path(item["path"]).name
        if not actual.is_file() or digest(actual) != item["sha256"]:
            raise ValueError("Manifest tokenizer differs from the tokenizer the learner will actually load")
    model_hashes = {path.name: digest(path) for path in model_files}
    teacher_hashes = None
    if args.teacher:
        if digest(Path(args.teacher) / "tokenizer.json") != digest(args.model / "tokenizer.json"):
            raise ValueError("M6 teacher and learner must share the audited token-id vocabulary")
        teacher_hashes = {path.name: digest(path) for path in sorted(Path(args.teacher).iterdir()) if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_identity.json").write_text(json.dumps({"base": model_hashes, "teacher": teacher_hashes}, indent=2) + "\n")
    config_file = output / "orbit_args.json"
    config_file.write_text(json.dumps(cfg, indent=2) + "\n")
    (output / "requested_protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "expected_applied_lr.json").write_text(json.dumps([applied_lr(p, i) for i in range(p["steps"])]) + "\n")
    if environment:
        (output / "environment.lock").write_text(json.dumps(environment, indent=2) + "\n")
    source_hashes = source_files()
    (output / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    command = ["bash", str(ROOT / "examples/posttrain_subspace/run.sh"), str(config_file)]
    (output / "launch_argv.json").write_text(json.dumps({"outer": sys.argv, "driver": command}, indent=2) + "\n")
    print(shlex.join(command))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
