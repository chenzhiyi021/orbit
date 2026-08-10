#!/usr/bin/env bash
# Qwen3-1.7B (student, full fine-tune) <- Qwen3-4B-Instruct-2507 (frozen teacher),
# sampled-token on-policy distillation (pure MOPD, reward-free): adv_t =
# teacher_logp_t - student_logp_t. Reverse-KL by construction -- this codebase's
# sampled-token path has no forward-KL variant (--opd-kl-type only applies to the
# top-k loss, opd_topk_loss; plain sampled-token scoring always computes reverse KL
# in the trainer, per orbit/utils/arguments.py). Self-contained launcher.
#
# The pasted reference (Sphere-AI-Lab/orbit-develop@feat/full-vocab-opd's
# run-qwen3-1_7b-bf16-openreasoning-opd-full-vocab-lora-fkl.sh) is a full-vocab-only
# recipe, so it has no RL_ARGS/LOSS_ARGS counterpart for sampled-token distillation;
# training schedule, rollout, optimizer, eval, perf, resource, and misc
# hyperparameters below are still aligned with it 1:1. RL_ARGS/LOSS_ARGS instead
# follow this repo's own examples/on_policy_distillation/run-qwen3-4B-opd-sglang.sh
# sampled-token pattern (--entropy-coef 0.0, --eps-clip 0.2/0.2, no --rm-type math --
# custom-rm-path is the only reward source in that pattern).
#
# Full fine-tuning instead of LoRA per explicit request (--peft-method none): with a
# frozen 4B teacher time-sharing the same 2 GPUs, losing LoRA's memory savings makes
# this configuration memory-tight. Watch for OOM and be ready to lower
# PERF_ARGS/SGLANG_ARGS/--opd-teacher-mem-fraction below, or add GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=run_qwen3_17b_bf16_openreasoning_opd_sampled_token_fullft
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
# Student: mv Qwen3-1.7B -> qwen3-1.7b already done on this machine.
#   export HF_CKPT=/mnt/L202500431/models/qwen3-1.7b
#   export MEGATRON_LOAD=/mnt/L202500431/models/megatron_ckpt/qwen3-1.7b   # NOT YET CONVERTED -- see note below
# Teacher (frozen, served via SGLang, never converted to Megatron):
#   export OPD_TEACHER_CKPT=/mnt/L202500431/models/qwen3-4b-instruct-2507
: "${HF_CKPT:?set HF_CKPT to the Qwen3-1.7B Hugging Face checkpoint}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the Qwen3-1.7B Megatron torch_dist checkpoint}"
: "${OPD_TEACHER_CKPT:?set OPD_TEACHER_CKPT to the frozen Qwen3-4B Hugging Face checkpoint}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to the OpenReasoning training data (.jsonl or .parquet, question/answer schema)}"
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Qwen3-1.7B_4B_openreasoning_sampled_token_opd_fullft}"
AIME24_PATH="${AIME24_PATH:-${ORBIT_ROOT}/data/aime24/test.parquet}"
AIME25_PATH="${AIME25_PATH:-${ORBIT_ROOT}/data/aime25/test.parquet}"
HMMT25_PATH="${HMMT25_PATH:-${ORBIT_ROOT}/data/hmmt25/test.parquet}"

# === Resources ===
# Same two-GPU colocated topology as the full-vocab launchers in this family.
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-1.7B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL:-10}"
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key question
    --label-key answer
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --rollout-shuffle
    --custom-rm-path orbit.rollout.opd_sglang.reward_func
    --custom-reward-post-process-path orbit.rollout.opd_sglang.post_process
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 4096
    --rollout-temperature 0.7
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 5e-6
    --lr-decay-style cosine
    --min-lr 5e-7
    --lr-warmup-fraction 0.1
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

# Pure MOPD, sampled-token, reward-free: adv_t = teacher_logp_t - student_logp_t.
# No --teacher-score-mode/--loss-type opd_jsd_loss here (that's the full-vocab
# path); teacher-forcing scoring still goes through the managed sglang teacher.
RL_ARGS=(
    --advantage-estimator on_policy_distillation
    --opd-type sglang
    --teacher-hf-checkpoint "${OPD_TEACHER_CKPT}"
    --opd-serve-teacher
    --opd-teacher-num-gpus "${OPD_TEACHER_NUM_GPUS:-2}"
    --opd-teacher-mem-fraction "${OPD_TEACHER_MEM_FRACTION:-0.3}"
    --opd-teacher-max-running-requests "${OPD_TEACHER_MAX_RUNNING_REQUESTS:-8}"
    --opd-teacher-max-prefill-tokens "${OPD_TEACHER_MAX_PREFILL_TOKENS:-4096}"
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.2
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
    --max-tokens-per-gpu 8192
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data aime24 "${AIME24_PATH}" aime25 "${AIME25_PATH}" hmmt25 "${HMMT25_PATH}"
    --n-samples-per-eval-prompt 16
    --eval-max-response-len 8192
    --eval-top-k -1
    --eval-top-p 0.95
    --eval-temperature 1.0
    --eval-pass-k-values 1 8 16
)

SGLANG_ARGS=(
    --num-gpus-per-node "${GPUS_PER_NODE}"
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static 0.25
    --sglang-server-concurrency 4
    --sglang-max-running-requests 512
    --router-disable-circuit-breaker
    --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND:-fa3}"
    --sglang-sampling-backend "${SGLANG_SAMPLING_BACKEND:-flashinfer}"
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

DEBUG_ARGS=( --log-passrate )

# Full fine-tuning per explicit request.
PEFT_ARGS=(
    --peft-method none
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
