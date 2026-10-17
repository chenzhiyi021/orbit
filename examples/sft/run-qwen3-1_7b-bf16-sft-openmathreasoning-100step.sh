#!/usr/bin/env bash
# Qwen3-1.7B full-parameter non-thinking SFT on nvidia/OpenMathReasoning's DeepSeek-R1 cot traces,
# with every training hyperparameter copied from run-qwen3-1_7b-bf16-sft-unified-m1-100step.sh
# (see that launcher for the TRL -> Orbit mapping): 100 steps x global batch 256 = 25,600 rows,
# constant lr 5e-6, no warmup, clip 1.0, wd 0, seed 42, no shuffle, save every 20 steps.
#
# Data: tools/build_openmathreasoning_sft.py. The R1 traces have their <think>/</think> tags
# stripped, so reasoning + summary is one plain assistant message trained after the empty
# non-thinking <think></think> block (--loss-mask-type generation_prompt, enable_thinking=false).
# The user turn is evalchemy's MATH_PROMPT and every row is <= 12288 tokens (the M1 source yaml's
# max_length), so eval-math-evalchemy.sh at its defaults (ENABLE_THINKING=0, 32768-token budget)
# scores these checkpoints exactly like the earlier M1 / OPD ones.
#
# Responses are ~5-6k tokens on average, several times the M1 teacher bank's, so each step
# processes far more tokens than the M1 run at the same step count.
#
#   HF_CKPT=/.../qwen3-1.7b MEGATRON_LOAD=/.../megatron_ckpt/qwen3-1.7b \
#   bash examples/sft/run-qwen3-1_7b-bf16-sft-openmathreasoning-100step.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-openmathreasoning_r1_sft_non_thinking_100step}
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to the Qwen3-1.7B Hugging Face checkpoint}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the Qwen3-1.7B Megatron torch_dist checkpoint}"
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/${LAUNCHER_NAME}}"
# tools/build_openmathreasoning_sft.py --output; read in file order (already shuffled by its --seed).
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/L202500431/datasets/openmathreasoning_nothink_25k6/train.jsonl}"

# === Resources ===
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-1.7B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
# max_steps: 100 -> NUM_ROLLOUT (exact).
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
# Global batch 256, aligned with the unified M4--M6 OPD launchers
# (examples/on_policy_distillation/unified_300step_constant/) and with the
# source TRL run's true effective batch (per_device 1 x grad_accum 32 x 8 GPUs).
# Orbit's batch args are already global, so this holds for any GPUS_PER_NODE.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"

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
    # Same prompt string as the unified OPD launchers' student rollout:
    # user turn + generation prompt rendered with enable_thinking=false, i.e.
    # ending in an empty <think></think> block; loss on teacher content + EOS.
    --loss-mask-type "${LOSS_MASK_TYPE:-generation_prompt}"
    --apply-chat-template-kwargs '{"enable_thinking": false}'
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

# Optional training-time subspace exclusion (orbit/backends/megatron_utils/subspace_exclusion.py),
# same knobs as run-m6-opd-st-non-thinking-full-exclude-subspace.sh; off unless
# EXCLUDE_SUBSPACE_PATH is set. Needs TENSOR_MODEL_PARALLEL_SIZE=1 (the default here).
if [[ -n "${EXCLUDE_SUBSPACE_PATH:-}" ]]; then
    MISC_ARGS+=(
        --exclude-subspace-path "${EXCLUDE_SUBSPACE_PATH}"
        --exclude-subspace-k "${EXCLUDE_K:?set EXCLUDE_K with EXCLUDE_SUBSPACE_PATH}"
        --exclude-subspace-side "${EXCLUDE_SIDE:-right}"
        --exclude-subspace-mode "${EXCLUDE_MODE:-exclude}"
    )
fi

if ! is_true "${SFT_GRADIENT_ACCUMULATION_FUSION:-0}"; then
    MISC_ARGS+=( --no-gradient-accumulation-fusion )
fi

DEBUG_ARGS=()

# Full fine-tuning (source yaml has no PEFT/LoRA config) unless LORA_RANK is set:
# then LoRA on all linears, scale alpha/rank (LORA_ALPHA default 32, as in "LoRA Without
# Regret"; pair with a ~10x LR, e.g. LR=5e-5). Not combinable with EXCLUDE_SUBSPACE_PATH.
if [[ -n "${LORA_RANK:-}" ]]; then
    PEFT_ARGS=(
        --peft-method lora
        --peft-variant standard
        --lora-rank "${LORA_RANK}"
        --lora-alpha "${LORA_ALPHA:-32}"
        --lora-dropout 0.0
        --target-modules all-linear
    )
else
    PEFT_ARGS=(
        --peft-method none
    )
fi

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
