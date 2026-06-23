#!/usr/bin/env bash

export ORBIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
echo "ORBIT_ROOT=${ORBIT_ROOT}"

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen2.5-0.5B.sh"   # provides MODEL_ARGS=(...)
echo "MODEL_ARGS=${MODEL_ARGS[*]}"

python "${ORBIT_ROOT}/scratch/run_teacher_manager.py" \
    --num-rollout 1 \
    "${MODEL_ARGS[@]}" 