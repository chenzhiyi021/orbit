#!/usr/bin/env bash
# Qwen3-1.7B LoRA SFT, hyperparameters translated from the M1 300-step "hard
# SFT" TRL SFTConfig-style YAML (accelerate + deepspeed_zero3, not this
# codebase). Two source variants were given and are IDENTICAL on every
# training hyperparameter below -- they only differ in which teacher rollout
# bank populates `dataset_name` / `output_dir` / `run_name`:
#
#   accelerate_config: examples/accelerate_configs/deepspeed_zero3.yaml
#   model_name_or_path: Qwen/Qwen3-1.7B
#   dataset_name: (either a "prompt-aligned" bank matching the M4--M6 seed-42
#                  schedule, or the plain unified_m0_m5 bank)
#   seed: 42 / data_seed: 42
#   max_steps: 300
#   per_device_train_batch_size: 1 / gradient_accumulation_steps: 32
#   learning_rate: 5.0e-6 / optim: adamw_torch
#   lr_scheduler_type: constant / warmup_steps: 0
#   max_grad_norm: 1.0 / weight_decay: 0.0
#   max_length: 12288
#   dtype: bfloat16 / bf16: true / tf32: true
#   attn_implementation: flash_attention_2
#   gradient_checkpointing: true (use_reentrant: false)
#   eval_strategy: "no" / logging_steps: 1
#   save_strategy: steps / save_steps: 20 / save_total_limit: 15
#   dataloader_num_workers: 4
#
# NOTE: the source yaml has no PEFT/LoRA section at all -- it's a full
# fine-tune. PEFT_ARGS below (rank 32 / alpha 64 / dropout 0.0 / all-linear)
# is a requested addition on top of the translated recipe, not something
# translated from the source; see
# examples/sft/run-qwen3-1_7b-bf16-sft-unified-m1-300step.sh for the
# full-finetune version this was derived from.
#
# Unlike examples/sft/run-qwen3-1_7b-bf16-sft-unified-m1.sh (the earlier
# 100-step cosine_with_min_lr recipe), this launcher does NOT read from a
# prebuilt HF `datasets.save_to_disk()` bank. It reads TRAIN_JSONL straight
# from tools/generate_teacher_sft_data.py's own --output: that tool already
# writes {"messages": [...], "metadata": {...}} rows, one per line, which is
# exactly Orbit's --input-key messages schema -- no bank/HF-dataset
# conversion step in between.
#
# Orbit is Megatron-based (no accelerate/deepspeed, no HF Trainer), so this is
# a translation, not a literal port. Mapping notes:
#   - max_steps 300                 -> NUM_ROLLOUT=300 (exact).
#   - learning_rate / lr_scheduler_type: constant / warmup_steps: 0
#                                    -> --lr-decay-style constant, no
#                                       --lr-warmup-fraction / --min-lr flags
#                                       (constant needs neither; warmup_steps
#                                       0 has nothing to translate).
#   - max_grad_norm / weight_decay  -> --clip-grad / --weight-decay (exact).
#   - optim: adamw_torch            -> --optimizer adam (Megatron's built-in
#                                       AdamW; no literal "adamw_torch" knob).
#   - bf16 / attn_implementation /
#     gradient_checkpointing        -> PRECISION_PROFILE=bf16,
#                                       --attention-backend flash,
#                                       --recompute-granularity full (already
#                                       Orbit's default SFT posture).
#   - assistant_only_loss           -> already true by construction: Orbit's
#                                       SFT rollout (orbit.rollout.sft_rollout)
#                                       always masks everything except
#                                       assistant target tokens, regardless of
#                                       this flag's value in the source yaml.
#   - shuffle_dataset: false        -> --rollout-shuffle is NOT set below, so
#                                       Orbit consumes TRAIN_JSONL in file
#                                       order. NOTE: this is where the two
#                                       source variants' "prompt-aligned"
#                                       distinction actually lives -- see the
#                                       CAVEATS section below.
#   - per_device_train_batch_size 1 / gradient_accumulation_steps 32
#                                    -> NOT a 1:1 mapping, and NOT scaled by
#                                       GPU count the way TRL's per-device
#                                       batch is. Orbit's --global-batch-size /
#                                       --rollout-batch-size are already the
#                                       TOTAL sample count for one optimizer
#                                       step across every GPU combined --
#                                       orbit.ray.rollout._split_train_data_by_dp
#                                       partitions that fixed total across the
#                                       DP ranks, it does not multiply it by DP
#                                       world size. So GLOBAL_BATCH_SIZE=32
#                                       below already matches the yaml's
#                                       effective batch (32) as-is, REGARDLESS
#                                       of GPUS_PER_NODE -- changing
#                                       GPUS_PER_NODE only changes how many of
#                                       those 32 samples land on each GPU per
#                                       step (throughput/memory), not the
#                                       total processed. To reproduce a TRL
#                                       run that itself used 8 real GPUs (true
#                                       effective batch 32*8=256, hence 76,800
#                                       examples over 300 steps), set
#                                       ROLLOUT_BATCH_SIZE=GLOBAL_BATCH_SIZE=256
#                                       explicitly -- do not try to get there
#                                       via GPUS_PER_NODE.
#   - max_length: 12288             -> no per-example truncation flag found in
#                                       this codebase's examples; approximated
#                                       via --max-tokens-per-gpu (a packing
#                                       budget, not a truncation cap). Bumped
#                                       above 12288 to fit one full-length
#                                       example; tune for your GPU memory.
#   - save_steps: 20                -> --save-interval 20 (exact).
#   - save_total_limit: 15          -> no equivalent flag found in this
#                                       codebase's examples; not translated.
#   - dataset_kwargs.skip_prepare_dataset, completion_only_loss,
#     include_num_input_tokens_seen, online_trace*, online_rollout_*
#                                    -> TRL SFTTrainer-specific mechanics
#                                       (online drift monitoring, tracing).
#                                       No equivalent in this codebase; not
#                                       translated. If you want periodic
#                                       generation-based monitoring during SFT,
#                                       the closest tool here is EVAL_ARGS
#                                       (needs labeled eval data + --rm-type;
#                                       see examples/high_precision's OPD
#                                       launchers for the pattern) -- left
#                                       empty (disabled) below to match
#                                       eval_strategy: "no".
#   - data_seed, dataloader_num_workers, logging_steps, tf32
#                                    -> no equivalent flags found in this
#                                       codebase's examples; not translated.
#                                       --seed below covers `seed` only.
#
# CAVEATS (things that do NOT line up):
#   - Row order / "prompt-aligned" schedule: tools/generate_teacher_sft_data.py
#     assigns `index` by position in its own --dataset iteration (optionally
#     resumed/reused via --reuse-from), which is independent of whatever
#     produced the M4--M6 seed-42 rollout schedule referenced in the source
#     yaml's comment. Since shuffle is off, TRAIN_JSONL's line order IS the
#     training order here -- reproducing the "prompt-aligned" variant requires
#     you to construct TRAIN_JSONL's row order yourself (e.g. by pointing
#     --dataset at the same prompt sequence/order M4--M6 used); this launcher
#     has no way to verify or enforce that alignment.
#   - 76,800 total examples vs GLOBAL_BATCH_SIZE=32 * NUM_ROLLOUT=300 = 9,600:
#     the gap is NOT closed by GPUS_PER_NODE/DP world size (see mapping note
#     above -- Orbit's batch args are already global, not per-GPU). To
#     process 76,800 examples over 300 steps, raise ROLLOUT_BATCH_SIZE and
#     GLOBAL_BATCH_SIZE to 256 instead.
#   - save_total_limit (checkpoint retention) has no Orbit equivalent; disk
#     usage across 15 checkpoints @ save-interval 20 is on you to manage.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-unified_m1_sft_hard_non_thinking_300step_lora}
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to the Qwen3-1.7B Hugging Face checkpoint}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the Qwen3-1.7B Megatron torch_dist checkpoint}"
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/unified-m1-sft-hard-non-thinking-300step-lora}"
# Point this at the --output path of a tools/generate_teacher_sft_data.py run
# (or a concatenation of several -- Orbit reads it in file order, unshuffled).
TRAIN_JSONL="${TRAIN_JSONL:?set TRAIN_JSONL to a tools/generate_teacher_sft_data.py output jsonl}"

# === Resources ===
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-1.7B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
# max_steps: 300 -> NUM_ROLLOUT (exact).
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
# per_device_train_batch_size(1) * gradient_accumulation_steps(32); scale by
# your DP world size to match the source's true effective batch size -- see
# header note.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL:-20}"
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key messages
    --rollout-function-path orbit.rollout.sft_rollout.generate_rollout
    --loss-mask-type "${LOSS_MASK_TYPE:-qwen}"
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt 1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
)
# shuffle_dataset: false in the source yaml -> no --rollout-shuffle here.

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-5e-6}"
    --lr-decay-style constant
    --weight-decay "${WEIGHT_DECAY:-0.0}"
    --clip-grad "${MAX_GRAD_NORM:-1.0}"
    --adam-beta1 0.9
    --adam-beta2 0.999
)

RL_ARGS=()

LOSS_ARGS=(
    --training-mode sft
    --loss-type sft_loss
    --disable-compute-advantages-and-returns
    --calculate-per-token-loss
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-1}"
    --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
    --sequence-parallel
)

EVAL_ARGS=()
SGLANG_ARGS=()

MISC_ARGS=(
    --seed "${SEED:-42}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-offload-train
    --no-offload-train-async
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --te-rng-tracker
    --no-check-for-nan-in-loss-and-grad
)

if ! is_true "${SFT_GRADIENT_ACCUMULATION_FUSION:-0}"; then
    MISC_ARGS+=( --no-gradient-accumulation-fusion )
fi

DEBUG_ARGS=()

# LoRA (source yaml is a full finetune; this PEFT config was requested
# separately -- see the NOTE near the top of this file).
PEFT_ARGS=(
    --peft-method lora
    --peft-variant standard
    --lora-rank 32
    --lora-alpha 64
    --lora-dropout 0.0
    --target-modules all-linear
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
