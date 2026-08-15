#!/usr/bin/env bash
# M9-P (unique-prompt-aligned): verifier-only GRPO, G=4, 256 unique prompts per
# optimizer step -> 1024 trajectories/step (4x the rollout/token budget of
# M9-T). Full fine-tune. Orbit re-implementation of the TRL/accelerate
# "unified-300step-constant-m9-grpo-g4-unique-prompt-aligned-non-thinking"
# recipe -- see chat history for the parameter-alignment notes; several knobs
# (dataset format, ZeRO3 vs Megatron parallelism, vLLM-side IS correction
# mode) do not have exact Orbit equivalents and were approximated.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=run_qwen3_17b_m9_verifier_grpo_g4_unique_prompt_full
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to the Qwen3-1.7B Hugging Face checkpoint}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the Qwen3-1.7B Megatron torch_dist checkpoint}"
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/exp_ckpt/${LAUNCHER_NAME#run_}_$(date +%Y%m%d_%H%M%S)}"
# NOTE: source config points at a preprocessed HF `datasets` dir
# (../data/m9_verifier_100k_aligned); Orbit's rollout loader wants a flat
# jsonl/parquet with --input-key/--label-key instead, so this needs a one-time
# export/convert step before launch (see alignment notes, "dataset format").
: "${TRAIN_JSONL:?set TRAIN_JSONL to the exported m9_verifier jsonl/parquet path}"

# === Local checkpoint staging (Lustre -> NVMe) ===
LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage}
STAGE_HF_CKPT_TO=${STAGE_HF_CKPT_TO-${LOCAL_STAGE_ROOT}/Qwen3-1.7B}
STAGE_MEGATRON_CKPT_TO=${STAGE_MEGATRON_CKPT_TO-${LOCAL_STAGE_ROOT}/Megatron-Bridge/checkpoints/Qwen3-1.7B}

# === Resources ===
# Single-node colocated layout: 4 GPUs total, matching the source's
# vllm_tensor_parallel_size: 4 (rollout engine spans all 4 GPUs).
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-1.7B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
# 256 unique prompts x G=4 = 1024 trajectories, all consumed in one optimizer
# step per rollout round (matches TRL's num_iterations: 1 / generation_batch_size: 1024).
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"          # == max_steps: 300
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"   # unique prompts / rollout
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}" # G
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"    # trajectories / optimizer step

COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --ref-load "${MEGATRON_LOAD}"
    --ckpt-format torch_dist
    --save "${SAVE_DIR}"
    --save-interval 20
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key "${INPUT_KEY:-question}"
    --label-key "${LABEL_KEY:-answer}"
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --rollout-shuffle
    # Closest built-in rule-based verifier to unified_exact_answer_reward;
    # swap --custom-rm-path in if the exact grader must match bit-for-bit.
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 8192
    --rollout-temperature 0.7
    --rollout-top-p 1.0
    --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 5e-6
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --clip-grad 1.0
)

RL_ARGS=(
    --advantage-estimator grpo
    # beta: 0.0 in the source -> no KL term at all (omit --use-kl-loss).
    --eps-clip 0.2
    --eps-clip-high 0.2
    # Best-effort analogue of vllm_importance_sampling_correction +
    # vllm_importance_sampling_clip_max: 3.0. Orbit's TIS is a sequence-level
    # rollout/train mismatch correction, not the same "sequence_mask" mode
    # TRL exposes -- see alignment notes.
    --use-tis
    --tis-clip 3.0
)

LOSS_ARGS=(
    --calculate-per-token-loss
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

EVAL_ARGS=( --eval-interval 20 --skip-eval-before-train )

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 4
    --sglang-mem-fraction-static 0.3
    --rollout-num-gpus 0
    --sglang-max-running-requests 1024
    --router-disable-circuit-breaker
    --sglang-router-policy round_robin
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-offload-train
    --no-offload-train-async
    --offload-rollout
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --te-rng-tracker
    --no-check-for-nan-in-loss-and-grad
)

DEBUG_ARGS=(
    --log-passrate
    --log-reward-category acc
)

PEFT_ARGS=(
    --peft-method none
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
