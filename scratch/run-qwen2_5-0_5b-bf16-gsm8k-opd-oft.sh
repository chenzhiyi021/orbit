#!/usr/bin/env bash
# Qwen2.5-0.5B-Instruct BF16 On-Policy Distillation (OPD) on the math dataset.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=run_qwen25_05b_bf16_gsm8k_megatron_opd_oft
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train_opd.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
HF_CKPT="/mnt/L202500431/models/qwen2.5-0.5b-instruct"
# : "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
MEGATRON_LOAD="/mnt/L202500431/models/megatron_ckpt/qwen2.5-0.5b-instruct"
# : "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
SAVE_DIR="${ORBIT_ROOT}/orbit_ckpts/Qwen2.5-0.5B-Instruct_gsm8k_opd_oft_$(date +%Y%m%d_%H%M%S)"
TRAIN_JSONL="/mnt/L202500431/datasets/gsm8k/main/train-00000-of-00001.parquet"
# : "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
TEST_JSONL="/mnt/L202500431/datasets/gsm8k/main/test-00000-of-00001.parquet"
# TEST_JSONL=${TEST_JSONL:-}

# Teacher checkpoint -- 0.5B RLVR-trained (same family/tokenizer as student).
OPD_TEACHER_CKPT="/mnt/L202500431/orbit/orbit_ckpts/Qwen2.5-0.5B-Instruct_math_full_rlvr_20260725_230112/iter_0000875/merged"
# : "${OPD_TEACHER_CKPT:?set OPD_TEACHER_CKPT to a Hugging Face checkpoint path}"

# === Resources ===
GPUS_PER_NODE=2
RAY_NUM_CPUS=128

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen2.5-0.5B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
count_rows() {
    case "$1" in
        *.parquet) python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$1').metadata.num_rows)" ;;
        *) wc -l < "$1" ;;
    esac
}
TRAIN_ROWS=${TRAIN_ROWS:-$(count_rows "${TRAIN_JSONL}")}
NUM_ROLLOUT=${NUM_ROLLOUT:-$(( (TRAIN_ROWS * TOTAL_EPOCHS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))}

# === ARGS arrays ===
# 2-GPU colocate: actor+rollout share GPU 0, teacher gets its own dedicated
# bundle (GPU 1). driver.sh passes --actor-num-gpus-per-node ${GPUS_PER_NODE}
# (=2 here), so we override to 1 to avoid consuming both GPUs for the actor.
COLOCATE_ARGS=(
    --colocate
    --actor-num-gpus-per-node 1
)

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval 200
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key question
    --label-key answer
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 1024
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 3e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

# === OPD args ===
OPD_ARGS=(
    --opd-teacher-model-path "${OPD_TEACHER_CKPT}"
    --opd-teacher-num-gpus 1
    --opd-teacher-tp-size 1
    --advantage-estimator on_policy_distillation
    --loss-type policy_loss
    --opd_teacher_mem_fraction_static 0.7
    --opd-loss-type sampled_token
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
    --tensor-model-parallel-size 1
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
    --eval-interval 10
    --eval-prompt-data math "${TEST_JSONL}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 1024
    --eval-top-k 1
    --eval-pass-k-values 1 2 4 8 16
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.60
    --rollout-num-gpus 1
    --sglang-max-running-requests 1024
    --router-disable-circuit-breaker
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
)

PEFT_ARGS=(
    --peft-method oft
    --peft-variant standard
    --oft-type canonical_oft
    --oft-block-size 128
    --oft-eps 6e-5
    --target-modules all-linear
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"