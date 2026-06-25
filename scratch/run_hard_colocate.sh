#!/usr/bin/env bash
# Two-engine colocate stress test launcher.
# See run_hard_colocate.py docstring for what this isolates.
set -euo pipefail

VENV_PATH="${VENV_PATH:-../venvs/orbit_env}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    echo "Activating virtual environment: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
else
    echo "Warning: Virtual environment not found at ${VENV_PATH}"
    echo "Set VENV_PATH to override or ensure the environment is already activated."
fi

export ORBIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
echo "ORBIT_ROOT: ${ORBIT_ROOT}"

source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen2.5-0.5B.sh"

HF_CKPT="/mnt/L202500431/models/qwen2.5-0.5b-instruct"

RAY_NUM_GPUS=1
RAY_NUM_CPUS=4

# NOTE: these are required by parse_args()'s full Megatron arg validation
# even though this script never actually runs a rollout or a real training
# step -- parse_args() does not have a "minimal mode" that skips them.
ROLLOUT_BATCH_SIZE=1
NUM_ROLLOUT=1

OPD_TEACHER_MODEL_PATH="${HF_CKPT}"
OPD_TEACHER_TP_SIZE=1

FULL_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --num-rollout "${NUM_ROLLOUT}"

    --opd-teacher-model-path "${OPD_TEACHER_MODEL_PATH}"
    --opd-teacher-tp-size "${OPD_TEACHER_TP_SIZE}"

    "${MODEL_ARGS[@]}"
)

echo "Starting two-engine colocate stress test with arguments:"
printf "  %s\n" "${FULL_ARGS[@]}"

python "${ORBIT_ROOT}/scratch/run_hard_colocate.py" "${FULL_ARGS[@]}"