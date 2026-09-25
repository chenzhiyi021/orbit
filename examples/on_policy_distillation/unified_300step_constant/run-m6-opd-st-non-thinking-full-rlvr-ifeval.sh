#!/usr/bin/env bash
# M6 on RLVR-IFeval -- the run-m6-opd-st-non-thinking-full.sh recipe with only the
# task (and step count) swapped: uncentered sampled-token RKL estimator on refreshed
# student rollouts, full fine-tuning, prompts from allenai/RLVR-IFeval instead of
# openreasoning_mixed_100k.
#
# Purpose: a task-control run for the weight-space analysis in tools/function_space/.
# Comparing its delta against the openreasoning M6 deltas separates update directions
# that are specific to the reasoning task from ones any fine-tune of this model on this
# chat template would produce. Everything except the prompts and the step count is
# kept identical to run-m6-opd-st-non-thinking-full.sh so that comparison is fair:
# global batch 256, constant LR 5e-6 with no warmup or decay, weight decay 0,
# grad-norm clip 1.0, seed 42, one generation per prompt at temperature 0.7 /
# top-p 1.0, 4096-token completions, non-thinking chat template, same teacher.
#
# Data: allenai/RLVR-IFeval (15k rows, ODC-BY). Tulu 2 SFT prompts, each with exactly
# one IFEval constraint appended to the single user message, e.g. "... Your entire
# response should be in English, and in all lowercase letters." The `messages` column
# is already a one-element user turn with no system prompt, matching the openreasoning
# prompts, so it goes straight into --input-key messages. OPD needs no labels: the
# training signal is the teacher's log-prob of each sampled token, so the dataset's
# ground_truth / constraint columns are unused here (they are what you would grade
# against when evaluating).
#
# 300 steps x 256 prompts = 76,800 prompts over 15,000 rows, i.e. ~5.1 epochs: each
# prompt is sampled ~5 times (reshuffled every epoch by orbit/rollout/data_source.py).
# The openreasoning run sees ~100k distinct prompts over the same 300 steps, so this
# control repeats its prompts where that run does not -- keep that in mind when
# comparing the two deltas.
#
# Evaluation is not wired in: the openreasoning recipe's eval is math-only. Score the
# checkpoints on IFEval (lm-eval task `ifeval`) or IFBench separately.
#
#   HF_CKPT=/path/to/hf/Qwen3-1.7B \
#   MEGATRON_LOAD=/path/to/megatron/Qwen3-1.7B \
#   OPD_TEACHER_CKPT=/path/to/hf/Qwen3-4B-Instruct-2507 \
#   TRAIN_JSONL=/path/to/RLVR-IFeval/data/train-00000-of-00001.parquet \
#       bash examples/on_policy_distillation/unified_300step_constant/run-m6-opd-st-non-thinking-full-rlvr-ifeval.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=m6_opd_st_full_non_thinking_rlvr_ifeval_300step
WANDB_PROJECT=${WANDB_PROJECT:-orbit-adapt}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
# Cluster defaults (the /mnt/L202500431 layout, same as tools/extrapolation/live_training);
# override any of them in the environment.
HF_CKPT="${HF_CKPT:-/mnt/L202500431/models/qwen3-1.7b}"
MEGATRON_LOAD="${MEGATRON_LOAD:-/mnt/L202500431/models/megatron_ckpt/qwen3-1.7b}"
OPD_TEACHER_CKPT="${OPD_TEACHER_CKPT:-/mnt/L202500431/models/qwen3-4b-instruct-2507}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/L202500431/datasets/RLVR-IFeval/data/train-00000-of-00001.parquet}"

# Fail here with a readable message rather than deep inside Megatron/SGLang.
for _p in "${HF_CKPT}" "${MEGATRON_LOAD}" "${OPD_TEACHER_CKPT}" "${TRAIN_JSONL}"; do
    if [ ! -e "${_p}" ]; then
        echo "[$(basename "${BASH_SOURCE[0]}")] ERROR: path does not exist: ${_p}" >&2
        exit 1
    fi
done
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/${LAUNCHER_NAME}}"

# No in-training eval (see header).
DISABLE_EVAL=1

# === Resources ===
# --colocate: actor training, student rollout, and the managed teacher time-share the
# same GPUs. Global batch 256 is a global quantity, so the GPU count only changes
# throughput, not the optimization trajectory.
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${GPUS_PER_NODE}}"
OPD_TEACHER_NUM_GPUS="${OPD_TEACHER_NUM_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-1.7B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
# rollout_batch_size x n_samples_per_prompt equals global_batch_size, so one rollout is
# exactly one optimizer step and NUM_ROLLOUT is the step count.
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

# Checkpoint every 30 steps -> 10 checkpoints (iter 29, 59, ..., 299; 0-indexed).
CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL:-30}"
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key "${INPUT_KEY:-messages}"
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --rollout-shuffle
    # Kept from the openreasoning recipe for parity; the reward actually used is the
    # teacher score from --custom-rm-path below.
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 4096
    --rollout-temperature 0.7
    --rollout-top-p 1.0
    --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    # Required transport for a served teacher: reward_func POSTs the sampled sequence
    # for scoring, post_process puts the result on the sample.
    --custom-rm-path orbit.rollout.opd_sglang.reward_func
    --custom-reward-post-process-path orbit.rollout.opd_sglang.post_process
)

# Constant 5e-6, no warmup, no decay.
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-5e-6}"
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --clip-grad 1.0
)

# The frozen teacher is served by this job: orbit launches it as an extra sglang model
# entry with update_weights=false and the scoring-correctness server flags baked in
# (orbit/ray/rollout.py::_teacher_server_overrides).
RL_ARGS=(
    --opd-type sglang
    --teacher-hf-checkpoint "${OPD_TEACHER_CKPT}"
    --opd-serve-teacher
    --opd-teacher-num-gpus "${OPD_TEACHER_NUM_GPUS}"
    --opd-teacher-mem-fraction "${OPD_TEACHER_MEM_FRACTION:-0.35}"
    --opd-teacher-max-running-requests "${OPD_TEACHER_MAX_RUNNING_REQUESTS:-64}"
    --opd-teacher-max-prefill-tokens "${OPD_TEACHER_MAX_PREFILL_TOKENS:-4096}"
    --advantage-estimator on_policy_distillation
    --teacher-score-mode sampled_token
    --force-on-policy-ratio
    --kl-loss-coef 0.0
    --kl-loss-type k1
    --kl-coef 0.0
    --entropy-coef 0.0
)

LOSS_ARGS=(
    --loss-type policy_loss
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
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

# Emptied by validate_eval_args because DISABLE_EVAL=1 above.
EVAL_ARGS=()

SGLANG_ARGS=(
    --num-gpus-per-node "${GPUS_PER_NODE}"
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.25}"
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-32}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-512}"
    --router-disable-circuit-breaker
    # fa3 is rejected on B200/SM100 by the pinned SGLang; use triton there.
    --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND:-fa3}"
    --sglang-sampling-backend "${SGLANG_SAMPLING_BACKEND:-flashinfer}"
)

MISC_ARGS=(
    --seed 42
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

PEFT_ARGS=(
    --peft-method none
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
