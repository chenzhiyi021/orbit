#!/usr/bin/env bash
# M6 on euclaise/WritingPrompts_curated -- the 20-step run-m6-opd-st-non-thinking-full.sh
# recipe with only the prompts swapped: uncentered sampled-token RKL estimator on refreshed
# student rollouts, full fine-tuning, creative-writing prompts instead of
# openreasoning_mixed_100k.
#
# Purpose: a clean non-reasoning task control for the weight-space / rank-truncation
# analysis in tools/function_space/. Unlike RLVR-IFeval (whose Tulu 2 prompts carry ~20%
# chain-of-thought and some math), creative writing involves no math and no step-by-step
# reasoning, while its responses are long -- closer in length to math solutions than
# instruction-following answers are. Kept identical to the openreasoning 20-step run:
# 20 steps, global batch 256, constant LR 5e-6 with no warmup or decay, weight decay 0,
# grad-norm clip 1.0, seed 42, one generation per prompt at temperature 0.7 / top-p 1.0,
# 4096-token completions, non-thinking chat template, same teacher, checkpoint every 2
# steps (iter 1, 3, ..., 19; 0-indexed), so iter_0000019 lines up with the openreasoning
# run's iter_0000019.
#
# Two resource-only settings are taken from the RLVR-IFeval launcher, where long sequences
# OOMed the teacher's full-vocab scoring prefill: --rollout-max-prompt-len 1024 and
# --opd-teacher-mem-fraction 0.2. Neither changes the optimization; the converted prompts
# are one short line each, so the prompt filter drops nothing.
#
# Data: build it first with tools/convert_writingprompts_curated_to_opd.py, which turns
# each unique curated prompt into a single user turn ("Write an original piece of creative
# writing that responds to the following writing prompt. Prompt: <prompt>") -- the same
# one-turn, no-system shape as the openreasoning prompts. 20 x 256 = 5,120 prompts, well
# under the number of unique curated prompts, so none repeats.
#
#   TRAIN_JSONL=/path/to/WritingPrompts_curated_opd/train.parquet \
#       bash examples/on_policy_distillation/unified_300step_constant/run-m6-opd-st-non-thinking-full-writingprompts-curated.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=m6_opd_st_full_non_thinking_writingprompts_curated_20step
WANDB_PROJECT=${WANDB_PROJECT:-orbit-adapt}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
# Cluster defaults (the /mnt/L202500431 layout); override any of them in the environment.
HF_CKPT="${HF_CKPT:-/mnt/L202500431/models/qwen3-1.7b}"
MEGATRON_LOAD="${MEGATRON_LOAD:-/mnt/L202500431/models/megatron_ckpt/qwen3-1.7b}"
OPD_TEACHER_CKPT="${OPD_TEACHER_CKPT:-/mnt/L202500431/models/qwen3-4b-instruct-2507}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/L202500431/datasets/WritingPrompts_curated_opd/train.parquet}"

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
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${GPUS_PER_NODE}}"
OPD_TEACHER_NUM_GPUS="${OPD_TEACHER_NUM_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-1.7B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
# rollout_batch_size x n_samples_per_prompt equals global_batch_size, so one rollout is
# exactly one optimizer step and NUM_ROLLOUT is the step count.
NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

# Checkpoint every 2 steps -> iter 1, 3, ..., 19 (0-indexed), as in the openreasoning run.
CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL:-2}"
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
    # Resource guard from the RLVR-IFeval launcher (teacher prefill OOM on long sequences).
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN:-1024}"
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
    # 0.2 as in the RLVR-IFeval launcher: long stories + 4096-token responses make the
    # teacher's fp32 full-vocab scoring prefill large, and it lives outside the static pool.
    --opd-teacher-mem-fraction "${OPD_TEACHER_MEM_FRACTION:-0.2}"
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
