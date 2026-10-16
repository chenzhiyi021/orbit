#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?prepared orbit_args.json required}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SGLANG_RETURN_ORIGINAL_LOGPROB=0
export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=0
export ORBIT_RAY_LIFECYCLE=private ORBIT_LOG_FILTER=0
mapfile -t FIELDS < <(python3 - "$CONFIG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
if c.get('posttrain_execution_ready') is not True:
 raise SystemExit('Plan only: run launch --execute with environment/GPU qualification instead')
for key in ['posttrain_run_dir','hf_checkpoint','load','save','actor_num_gpus_per_node']:
 print(c[key])
PY
)
if [[ "${#FIELDS[@]}" != 5 ]]; then
    exit 2
fi
RUN_DIR="${FIELDS[0]}"
HF_CKPT="${FIELDS[1]}"
MEGATRON_LOAD="${FIELDS[2]}"
SAVE_DIR="${FIELDS[3]}"
GPUS_PER_NODE="${FIELDS[4]}"
ORBIT_ROOT="$ROOT"
ORBIT_ENTRYPOINT="$ROOT/train.py"
LAUNCHER_NAME=posttrain_subspace
PRECISION_PROFILE=bf16
RUN_LOG="$RUN_DIR/train.log"
DISABLE_EVAL=1
RAY_NUM_CPUS=$((4 * GPUS_PER_NODE))
export RAY_TEMP_DIR="$RUN_DIR/ray"
source "$ROOT/orbit_plugins/model_args/qwen3-1.7B.sh"
COLOCATE_ARGS=(); CKPT_ARGS=(--hf-checkpoint "$HF_CKPT" --load "$MEGATRON_LOAD")
ROLLOUT_ARGS=(); OPTIMIZER_ARGS=(); RL_ARGS=(); LOSS_ARGS=(); WANDB_ARGS=()
PERF_ARGS=(); EVAL_ARGS=(); SGLANG_ARGS=(); PEFT_ARGS=(); DEBUG_ARGS=()
MISC_ARGS=(--custom-config-path "$CONFIG")
# Runtime parsed arguments, after hooks, are written to effective_protocol.json.
source "$ROOT/scripts/lib/launcher.sh"
