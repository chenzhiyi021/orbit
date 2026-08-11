#!/usr/bin/env bash
# Qwen3-1.7B (student, full fine-tune) <- Qwen3-4B-Instruct-2507 (frozen teacher),
# full-vocabulary on-policy distillation, pure reverse KL(student||teacher)
# (--opd-jsd-beta 1.0). Self-contained launcher.
#
# Hyperparameters are aligned with Sphere-AI-Lab/orbit-develop@feat/full-vocab-opd's
# examples/high_precision/run-qwen3-1_7b-bf16-openreasoning-opd-full-vocab-lora-fkl.sh
# wherever a counterpart exists in this codebase. Deltas from that source recipe:
#   - Full fine-tuning instead of LoRA (--peft-method none) -- requested explicitly;
#     with a frozen 4B teacher time-sharing the same 2 GPUs, losing LoRA's memory
#     savings makes this configuration memory-tight. Watch for OOM and be ready to
#     lower PERF_ARGS/SGLANG_ARGS/--opd-teacher-mem-fraction below, or add GPUs.
#   - Teacher serving/entrypoint: the source uses ORBIT_ENTRYPOINT=train_opd.py plus
#     a multi-server --sglang-config YAML with a named "teacher" group and
#     --teacher-model-name teacher / --teacher-num-gpus. None of that exists in this
#     codebase -- train_opd.py is not a file here, and --teacher-model-name /
#     --sglang-config-with-named-teacher-group / --teacher-num-gpus are not
#     recognized flags. This launcher instead uses the wiring this codebase actually
#     validates for full-vocab OPD (orbit/utils/arguments.py::_validate_opd_args):
#     --opd-type sglang, --opd-serve-teacher (managed in-job teacher serving), and
#     --opd-teacher-num-gpus/--opd-teacher-mem-fraction/--opd-teacher-max-running-requests/
#     --opd-teacher-max-prefill-tokens carrying over the source's numeric values
#     (num_gpus=2, mem_fraction_static=0.3, max_running_requests=8,
#     max_prefill_tokens=4096) 1:1 from its sglang_config.yaml "teacher" group.
#   - --advantage-estimator grpo (+ --disable-compute-advantages-and-returns) instead
#     of the source's --advantage-estimator on_policy_distillation: this codebase's
#     _validate_opd_args explicitly rejects on_policy_distillation/--use-opd together
#     with --teacher-score-mode full_vocab, so grpo (inert here; advantages/returns
#     are disabled) is the only accepted estimator for the full-vocab loss.
#   - Dropped the source's dead TOTAL_EPOCHS/count_rows()/TRAIN_ROWS-based
#     NUM_ROLLOUT formula (TOTAL_EPOCHS is commented out there, so that branch never
#     actually executes -- NUM_ROLLOUT always resolves from its own default first).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=run_qwen3_17b_bf16_openreasoning_opd_full_vocab_fullft_rkl
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
# Train data: YangyiH/openreasoning_mixed_100k's raw train.parquet -- columns are
# `messages` (one user-turn dict) + source metadata, NO answer/label column at all
# (confirmed against a real row: 'domain': 'science', no ground truth). ROLLOUT_ARGS
# below is wired for this messages-only, no-label schema (--input-key messages, no
# --label-key).
#   export TRAIN_JSONL=/mnt/L202500431/datasets/openreasoning_mixed_100k/train.parquet
: "${HF_CKPT:?set HF_CKPT to the Qwen3-1.7B Hugging Face checkpoint}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the Qwen3-1.7B Megatron torch_dist checkpoint}"
: "${OPD_TEACHER_CKPT:?set OPD_TEACHER_CKPT to the frozen Qwen3-4B Hugging Face checkpoint}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to the OpenReasoning training data (.jsonl or .parquet, messages-only schema)}"
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Qwen3-1.7B_4B_openreasoning_full_vocab_opd_fullft_rkl}"
AIME24_PATH="${AIME24_PATH:-${ORBIT_ROOT}/data/aime24/test.parquet}"
AIME25_PATH="${AIME25_PATH:-${ORBIT_ROOT}/data/aime25/test.parquet}"
HMMT25_PATH="${HMMT25_PATH:-${ORBIT_ROOT}/data/hmmt25/test.parquet}"
# These AIME/HMMT eval files don't exist in this environment yet, and --input-key/
# --label-key are global args shared with training's messages-only, no-label
# schema above -- even if the files existed, eval loading would break unless they
# also carry a `messages` column with no separate answer column. Eval is off by
# default until you prepare matching eval data (see tools/convert_math_eval_to_orbit.py)
# or wire per-dataset keys via --eval-config; set DISABLE_EVAL=0 to re-enable once ready.
DISABLE_EVAL="${DISABLE_EVAL:-1}"

# === Resources ===
# Matches the source's two-GPU colocated topology: actor training, student rollout
# serving, and the managed teacher time-share these GPUs via the offload/onload
# dance. Full fine-tuning removes LoRA's memory savings, so this is tight for a
# 1.7B student + 4B frozen teacher on 2xH100 -- see caveat in the header comment.
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
    --input-key messages
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --rollout-shuffle
    # No --label-key: the training parquet has no ground-truth column. --rm-type
    # math is still required (not optional) even though there's nothing to grade
    # against -- reward_func (orbit/rollout/opd_sglang.py) unconditionally calls
    # default_async_rm for full_vocab-mode training samples to populate a
    # diagnostic reward metric, and default_async_rm raises NotImplementedError if
    # args.rm_type is unset. With sample.label=None, grade_answer_verl short-
    # circuits on `if not ground_truth: return False`, so this reward is always 0
    # -- harmless, but not a real accuracy signal (the actual training signal is
    # the full-vocab JSD loss, not this reward).
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 4096
    --rollout-temperature 0.7
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --custom-rm-path orbit.rollout.opd_sglang.reward_func
    --custom-reward-post-process-path orbit.rollout.opd_sglang.post_process
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

# Full-vocab OPD, pure reverse KL(student||teacher): opd-jsd-beta=1.0. See header
# comment for why --opd-type sglang / --opd-serve-teacher / --advantage-estimator
# grpo replace the source's train_opd.py + --teacher-model-name +
# --advantage-estimator on_policy_distillation wiring.
RL_ARGS=(
    --advantage-estimator grpo
    --opd-type sglang
    --teacher-score-mode full_vocab
    --teacher-hf-checkpoint "${OPD_TEACHER_CKPT}"
    --opd-serve-teacher
    --opd-teacher-num-gpus "${OPD_TEACHER_NUM_GPUS:-2}"
    --opd-teacher-mem-fraction "${OPD_TEACHER_MEM_FRACTION:-0.3}"
    --opd-teacher-max-running-requests "${OPD_TEACHER_MAX_RUNNING_REQUESTS:-8}"
    --opd-teacher-max-prefill-tokens "${OPD_TEACHER_MAX_PREFILL_TOKENS:-4096}"
    --opd-defer-full-vocab-scoring
    --disable-compute-advantages-and-returns
)

LOSS_ARGS=(
    --loss-type opd_jsd_loss
    --opd-jsd-beta 1.0
    --calculate-per-token-loss
    --use-kl-loss
    --kl-loss-type low_var_kl
    --kl-loss-coef 0.0
    --opd-log-topk-overlap
    --opd-topk-overlap-ks 8 16 32 64
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

# Full fine-tuning per explicit request (source recipe uses LoRA rank 64).
PEFT_ARGS=(
    --peft-method none
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
