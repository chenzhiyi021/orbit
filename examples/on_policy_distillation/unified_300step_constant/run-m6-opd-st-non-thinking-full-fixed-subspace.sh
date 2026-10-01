#!/usr/bin/env bash
# M6 full fine-tuning with the update confined to a FIXED per-tensor subspace -- the
# fixed-right / fixed-left (mirror) experiments on the read/write asymmetry of Zhu et al. 2024
# ("Asymmetry in Low-Rank Adapters of Foundation Models").
#
# Same recipe and mechanism as run-m6-opd-st-non-thinking-full-exclude-subspace.sh, run with
# EXCLUDE_MODE=keep: every q/k/v/o/gate/up/down update is restricted to the first FIXED_K
# directions of a basis file (orbit/backends/megatron_utils/subspace_exclusion.py), exactly,
# by reparametrization W_eff = W_base + P(W - W_base):
#
#   FIXED_SIDE=right  Delta = Delta V V^T      fixed input (read) space, output learned  ~ frozen-A LoRA
#   FIXED_SIDE=left   Delta = U U^T Delta      fixed output (write) space, input learned ~ frozen-B LoRA (mirror)
#   FIXED_SIDE=both   Delta = U U^T Delta V V^T  both fixed, only a k x k core learned
#
# With the Haar-random basis (default) these are the random-subspace runs; point
# FIXED_SUBSPACE_PATH at a fine-tune's delta-SVD file for the "optimal subspace" counterpart
# (Final-r vs Haar-r). Defaults: k=16, 200 steps, checkpoint every 20 (as the TRL haar_v16 run).
#
# Build a random basis first (k_max >= FIXED_K; one file serves every k <= k_max):
#   python tools/function_space/build_exclusion_subspace.py --random --seed 0 --k-max 256 \
#       --base /mnt/L202500431/models/qwen3-1.7b \
#       --output /mnt/L202500431/models/exclusion_subspaces/random_top256_seed0.safetensors
#
#   FIXED_SIDE=left  bash .../run-m6-opd-st-non-thinking-full-fixed-subspace.sh
#   FIXED_SIDE=right bash .../run-m6-opd-st-non-thinking-full-fixed-subspace.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export EXCLUDE_MODE=keep
export EXCLUDE_SUBSPACE_PATH="${FIXED_SUBSPACE_PATH:-/mnt/L202500431/models/exclusion_subspaces/random_top256_seed0.safetensors}"
export EXCLUDE_K="${FIXED_K:-16}"
export EXCLUDE_SIDE="${FIXED_SIDE:-left}"     # left (fixed output space) | right (fixed input space) | both
export EXCLUDE_TAG="${FIXED_TAG:-haar}"       # names the run: haar | reasontop | ...
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"

exec bash "${SCRIPT_DIR}/run-m6-opd-st-non-thinking-full-exclude-subspace.sh"
