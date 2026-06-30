#!/usr/bin/env bash
# Qwen2.5-0.5B-Instruct BF16 Multi-Teacher On-Policy Distillation (MOPD) on GSM8K.
#
# Teachers:
#   general (fallback) : Qwen2.5-1.5B-Instruct  (--opd-teacher-model-path)
#   math               : Qwen2.5-3B-Instruct     (via --mopd-teacher-configs, domains=["math"])
#
# All "math" rollouts route to the 3B teacher; anything else falls back to 1.5B.
# In single-GPU colocate mode both teachers share the same GPU bundle.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=run_qwen25_05b_bf16_gsm8k_megatron_mopd_oft
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train_opd.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
HF_CKPT="/mnt/L202500431/models/qwen2.5-0.5b-instruct"
MEGATRON_LOAD="/mnt/L202500431/models/megatron_ckpt/qwen2.5-0.5b-instruct"
SAVE_DIR="${ORBIT_ROOT}/orbit_ckpts/Qwen2.5-0.5B-Instruct_gsm8k_mopd"
TRAIN_JSONL="/mnt/L202500431/datasets/gsm8k/main/train-00000-of-00001.parquet"
TEST_JSONL="/mnt/L202500431/datasets/gsm8k/main/test-00000-of-00001.parquet"

# === MOPD teacher checkpoints ===
# General teacher (fallback for any domain not in MOPD_TEACHER_CONFIGS).
OPD_GENERAL_TEACHER_CKPT="/mnt/L202500431/models/qwen2.5-1.5b-instruct"

# Specialised math teacher -- routes all "math" domain samples.
# Encoded as JSON so it can be passed as a single --mopd-teacher-configs arg.
MOPD_MATH_TEACHER_CKPT="/mnt/L202500431/models/qwen2.5-3b-instruct"
MOPD_TEACHER_CONFIGS=$(python3 -c "
import json
print(json.dumps([
    {
        'name':    'math',
        'path':    '${MOPD_MATH_TEACHER_CKPT}',
        'domains': ['math'],
        'num_gpus': 1,
    }
]))
")

# === Pre-flight: verify MOPD config before touching Ray ===
echo ""
echo "══════════════════════════════════════════════════════"
echo "  MOPD config verification"
echo "══════════════════════════════════════════════════════"
python3 - <<PYEOF
import json, os, sys

general_path = "${OPD_GENERAL_TEACHER_CKPT}"
configs      = json.loads(r"""${MOPD_TEACHER_CONFIGS}""")

# ── print routing table ──────────────────────────────────
print(f"  general (fallback) : {general_path}")
for t in configs:
    print(f"  {t['name']:<18}: {t['path']}")
    print(f"    domains : {t['domains']}")
    print(f"    num_gpus: {t['num_gpus']}")

# ── assert paths exist ───────────────────────────────────
errors = []
for path in [general_path] + [t["path"] for t in configs]:
    if not os.path.isdir(path):
        errors.append(f"  MISSING checkpoint: {path}")
if errors:
    print("\n[MOPD verify] FAILED – missing checkpoints:")
    for e in errors:
        print(e)
    sys.exit(1)

# ── assert domain uniqueness ─────────────────────────────
seen = {}
for t in configs:
    for d in t["domains"]:
        assert d not in seen, f"Domain '{d}' mapped to both '{seen[d]}' and '{t['name']}'"
        seen[d] = t["name"]

# ── print expected routing table ─────────────────────────
print("\n  Expected domain → teacher routing:")
for domain, teacher in seen.items():
    print(f"    '{domain}' → '{teacher}'")
print(f"    <any other> → 'general' (fallback)")
print(f"\n  rm-type for this run: 'math'  →  teacher 'math' (Qwen2.5-3B)")
print("\n[MOPD verify] OK")
PYEOF
echo "══════════════════════════════════════════════════════"
echo ""

# === Resources ===
GPUS_PER_NODE=1
RAY_NUM_CPUS=128

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen2.5-0.5B.sh"

# === Training schedule ===
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
TRAIN_ROWS=${TRAIN_ROWS:-$(wc -l < "${TRAIN_JSONL}")}
NUM_ROLLOUT=${NUM_ROLLOUT:-$(( (TRAIN_ROWS * TOTAL_EPOCHS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))}

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

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

# === MOPD args ===
# --opd-teacher-model-path   : general (fallback) teacher
# --mopd-teacher-configs     : JSON list of specialised teachers + their domain bindings
OPD_ARGS=(
    --opd-teacher-model-path "${OPD_GENERAL_TEACHER_CKPT}"
    --opd-teacher-num-gpus 1
    --opd-teacher-tp-size 1
    --opd_teacher_mem_fraction_static 0.9
    --mopd-teacher-configs "${MOPD_TEACHER_CONFIGS}"
    --loss-type custom_loss
    --custom-loss-function-path "orbit.backends.training_utils.opd_loss.opd_loss_function"
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
    --sglang-mem-fraction-static 0.3
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
    --no-offload-rollout
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
