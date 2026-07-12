#!/usr/bin/env bash
# Qwen2.5-0.5B-Instruct BF16 Multi-Teacher On-Policy Distillation (MOPD) --
# math (GSM8K) + instruction-following (Nemotron-RL-instruction_following)
# routing validation, single-GPU --debug-colocate.
#
# Purpose: confirm per-sample domain routing actually works across two
# *genuinely different* domains sharing one rollout batch -- unlike
# run-qwen-2_5-0_5b-bf16-gsm8k-mopd.sh, which is single-domain under the
# hood and only works because of the --rm-type global fallback.
#
# Both teachers are the SAME 1.5B-instruct checkpoint on purpose: this run
# is about validating the routing plumbing (orbit/ray/teacher.py::MopdRouter,
# the `domains` propagation added to orbit/ray/rollout.py, and the two eval
# datasets in scratch/eval_math_if_mopd.yaml logging separate accuracies),
# not about distillation quality.
#
# Data prerequisites (build with scripts/mopd_prepare_*.py + mopd_mix_train.py):
#   /mnt/L202500431/datasets/orbit_mopd/mixes/train_math-if_v1.jsonl
#   /mnt/L202500431/datasets/orbit_mopd/domains/if_ifbench_test.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=run_qwen25_05b_bf16_mopd_math_if_debug
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train_opd.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
HF_CKPT="/mnt/L202500431/models/qwen2.5-0.5b-instruct"
MEGATRON_LOAD="/mnt/L202500431/models/megatron_ckpt/qwen2.5-0.5b-instruct"
SAVE_DIR="${ORBIT_ROOT}/orbit_ckpts/Qwen2.5-0.5B-Instruct_mopd_math_if_debug"

# Mixed training set (math + if domains), built by scripts/mopd_mix_train.py.
TRAIN_JSONL="/mnt/L202500431/datasets/orbit_mopd/mixes/train_math-if_v1.jsonl"

# Eval datasets stay separate on purpose (math + ifbench each log their own
# eval/<name> accuracy plus a cross-dataset eval/avg -- see
# orbit/ray/rollout.py::_log_eval_rollout_data).
EVAL_CONFIG="${ORBIT_ROOT}/scratch/eval_math_if_mopd.yaml"

# === MOPD teacher checkpoints ===
# Both teachers are the SAME 1.5B-instruct checkpoint on purpose (see header).
OPD_GENERAL_TEACHER_CKPT="/mnt/L202500431/models/qwen2.5-1.5b-instruct"   # fallback -> effectively serves the "math" domain
MOPD_IF_TEACHER_CKPT="/mnt/L202500431/models/qwen2.5-1.5b-instruct"      # domains=["if"]

MOPD_TEACHER_CONFIGS=$(python3 -c "
import json
print(json.dumps([
    {
        'name':                'ifollow',
        'path':                '${MOPD_IF_TEACHER_CKPT}',
        'domains':             ['if'],
        'num_gpus':            1,
        'mem_fraction_static': 0.4,   # starts after general; sees less free GPU, use higher fraction
    }
]))
")

# === Pre-flight: verify MOPD config + data before touching Ray ===
echo ""
echo "══════════════════════════════════════════════════════"
echo "  MOPD config verification"
echo "══════════════════════════════════════════════════════"
python3 - <<PYEOF
import json, os, sys

general_path = "${OPD_GENERAL_TEACHER_CKPT}"
configs      = json.loads(r"""${MOPD_TEACHER_CONFIGS}""")

# ── print routing table ──────────────────────────────────
print(f"  general (fallback, serves 'math' by default) : {general_path}")
for t in configs:
    print(f"  {t['name']:<18}: {t['path']}")
    print(f"    domains : {t['domains']}")
    print(f"    num_gpus: {t['num_gpus']}")

# ── assert checkpoints + data files exist ────────────────
errors = []
for path in [general_path] + [t["path"] for t in configs]:
    if not os.path.isdir(path):
        errors.append(f"  MISSING checkpoint: {path}")
for path in ["${TRAIN_JSONL}", "${EVAL_CONFIG}"]:
    if not os.path.isfile(path):
        errors.append(f"  MISSING data/config file: {path}")
if errors:
    print("\n[MOPD verify] FAILED:")
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
print("\n  Expected domain -> teacher routing:")
for domain, teacher in seen.items():
    print(f"    '{domain}' -> '{teacher}'")
print(f"    <any other, e.g. 'math'> -> 'general' (fallback)")
print("\n[MOPD verify] OK")
PYEOF
echo "══════════════════════════════════════════════════════"
echo ""

# === Resources ===
# Single-GPU debug-colocate: actor + rollout + BOTH teachers share one
# bundle. This is tighter than run-qwen-2_5-0_5b-bf16-gsm8k-opd.sh (one more
# SGLang server on the same GPU) -- the mem_fraction_static values below and
# --sglang-mem-fraction-static are conservative starting points, not tuned;
# if you OOM, shrink those first before adding a second GPU.
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
COLOCATE_ARGS=( --debug-colocate )

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
    --input-key prompt
    --label-key label
    --rollout-shuffle
    # Global fallback only -- every row in TRAIN_JSONL already carries its
    # own metadata.rm_type ("math" or "ifbench"), set by
    # scripts/mopd_prepare_gsm8k.py / mopd_prepare_nemotron_if.py, and
    # orbit/rollout/rm_hub/__init__.py::async_rm prefers the per-row value
    # over this one.
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 2048
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
# --opd-teacher-model-path : general (fallback) teacher
# --mopd-teacher-configs   : JSON list of specialised teachers + domain bindings
OPD_ARGS=(
    --opd-teacher-model-path "${OPD_GENERAL_TEACHER_CKPT}"
    --opd-teacher-num-gpus 1
    --opd-teacher-tp-size 1
    --opd_teacher_mem_fraction_static 0.2
    --mopd-teacher-configs "${MOPD_TEACHER_CONFIGS}"
    --advantage-estimator on_policy_distillation
    --loss-type policy_loss
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

# math and ifbench are configured as two separate named datasets in
# EVAL_CONFIG, so orbit/ray/rollout.py::_log_eval_rollout_data logs them as
# eval/math and eval/ifbench separately (plus eval/avg across both) --
# nothing else to do here to get separate accuracies.
EVAL_ARGS=(
    --eval-interval 10
    --eval-config "${EVAL_CONFIG}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 2048
    --eval-top-k 1
    --eval-pass-k-values 1 2 4 8 16
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.2
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
    --peft-method lora
    --peft-variant standard
    --lora-rank 32
    --lora-alpha 64
    --lora-dropout 0.0
    --target-modules all-linear
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
