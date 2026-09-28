#!/usr/bin/env bash
# M6 full fine-tuning on openreasoning with the update kept OUT of a fixed per-tensor
# subspace -- the necessity test for the directions a previous fine-tune used.
#
# Earlier rank truncation showed the top-64 singular directions of the M6 full-FT delta are
# *sufficient* to recover its MATH500 gain. This run asks whether they are *necessary*:
# retrain the same recipe while every q/k/v/o/gate/up/down update is forbidden from the
# first EXCLUDE_K directions of that fine-tune's delta (or of a random basis, as control).
# Mechanism: orbit/backends/megatron_utils/subspace_exclusion.py (the network runs with
# W_base + P(W - W_base); local grads are projected before the DP reduction; params are
# rewritten after each step, so rollouts and checkpoints use the constrained weights).
#
# Recipe = run-m6-opd-st-non-thinking-full.sh (same data, teacher, LR 5e-6 constant, batch
# 256, seed 42, temperature 0.7, 4096-token completions, non-thinking template), except:
#   - tensor-model-parallel size 1 (the exclusion needs whole matrices on every rank), so
#     --sequence-parallel is dropped; with 2 GPUs this is DP=2 instead of TP=2;
#   - 300 steps / checkpoint every 30 by default, matching the 300-step M6 full-FT run the
#     default subspace was built from (NUM_ROLLOUT / SAVE_INTERVAL override).
#
# Build the bases first (tools/function_space/build_exclusion_subspace.py), e.g.
#   top-256 of the 300-step M6 full-FT delta -> exclusion_subspaces/m6_full_step300_top256.safetensors
#   random control                           -> exclusion_subspaces/random_top256_seed0.safetensors
#
#   EXCLUDE_K=64 EXCLUDE_SIDE=both bash .../run-m6-opd-st-non-thinking-full-exclude-subspace.sh
#   EXCLUDE_SUBSPACE_PATH=/.../random_top256_seed0.safetensors EXCLUDE_TAG=random EXCLUDE_K=64 bash ...
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Exclusion settings ===
EXCLUDE_SUBSPACE_PATH="${EXCLUDE_SUBSPACE_PATH:-/mnt/L202500431/models/exclusion_subspaces/m6_full_step300_top256.safetensors}"
EXCLUDE_K="${EXCLUDE_K:-64}"
EXCLUDE_SIDE="${EXCLUDE_SIDE:-both}"          # left (output space) | right (input space) | both
EXCLUDE_TAG="${EXCLUDE_TAG:-reasontop}"       # names the run: reasontop | random | ...

# === Recipe identity ===
LAUNCHER_NAME="m6_opd_st_full_non_thinking_excl_${EXCLUDE_TAG}_k${EXCLUDE_K}_${EXCLUDE_SIDE}"
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
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/L202500431/datasets/openreasoning_mixed_100k/train.parquet}"

# Fail here with a readable message rather than deep inside Megatron/SGLang.
for _p in "${HF_CKPT}" "${MEGATRON_LOAD}" "${OPD_TEACHER_CKPT}" "${TRAIN_JSONL}" "${EXCLUDE_SUBSPACE_PATH}"; do
    if [ ! -e "${_p}" ]; then
        echo "[$(basename "${BASH_SOURCE[0]}")] ERROR: path does not exist: ${_p}" >&2
        exit 1
    fi
done
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/${LAUNCHER_NAME}}"

# No in-training eval; score checkpoints with eval-math-evalchemy.sh afterwards.
DISABLE_EVAL=1

# === Resources ===
# TP=1: with 2 GPUs this is data-parallel 2 (the distributed optimizer shards the fp32
# optimizer state across the two ranks; each rank still holds whole bf16 matrices).
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${GPUS_PER_NODE}}"
OPD_TEACHER_NUM_GPUS="${OPD_TEACHER_NUM_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-1.7B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

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
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 4096
    --rollout-temperature 0.7
    --rollout-top-p 1.0
    --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
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

RL_ARGS=(
    --opd-type sglang
    --teacher-hf-checkpoint "${OPD_TEACHER_CKPT}"
    --opd-serve-teacher
    --opd-teacher-num-gpus "${OPD_TEACHER_NUM_GPUS}"
    # 0.2, not 0.35: at TP=1 the resident train state (full-model train offload is
    # unsupported) leaves ~35 GB per GPU for the student engine + teacher; 0.35 OOMed rollout 1.
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

# TP=1 (required by the exclusion); no --sequence-parallel, which only applies with TP>1.
PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

# Emptied by validate_eval_args because DISABLE_EVAL=1 above.
EVAL_ARGS=()

SGLANG_ARGS=(
    --num-gpus-per-node "${GPUS_PER_NODE}"
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    # 0.18, not 0.25: see the teacher mem-fraction note (TP=1 keeps more train state resident).
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.18}"
    # Student engine only (the managed teacher forces chunked_prefill_size=-1 itself). With the
    # small KV pool, retracted requests are re-prefilled in large extend batches whose fp32
    # full-vocab logits (~0.6 MB/token) OOMed GPU 0 at rollout 4 (1.5 GiB alloc, 0.8 GiB free);
    # 1024-token chunks cap that transient at ~0.6 GB.
    --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE:-1024}"
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
    # --- subspace exclusion (orbit/backends/megatron_utils/subspace_exclusion.py) ---
    --exclude-subspace-path "${EXCLUDE_SUBSPACE_PATH}"
    --exclude-subspace-k "${EXCLUDE_K}"
    --exclude-subspace-side "${EXCLUDE_SIDE}"
)

DEBUG_ARGS=( --log-passrate )

PEFT_ARGS=(
    --peft-method none
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
