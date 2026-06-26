#!/usr/bin/env bash
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

source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen2.5-1.5B.sh" 


HF_CKPT="/mnt/L202500431/models/qwen2.5-1.5b-instruct"


RAY_NUM_GPUS=1
RAY_NUM_CPUS=4

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


echo "Starting TeacherManager with arguments:"
printf "  %s\n" "${FULL_ARGS[@]}"

python "${ORBIT_ROOT}/scratch/run_teacher_manager.py" "${FULL_ARGS[@]}"