#!/usr/bin/env bash
# Evaluate Orbit OPD checkpoints on Evalchemy's public math benchmarks.
#
# Serves a checkpoint with SGLang (Orbit's venv ships SGLang, not vLLM) as an
# OpenAI-compatible endpoint, runs run_evalchemy_math_eval.py against it, then
# tears the server down before moving on. The runner reads its benchmark rows out
# of an Evalchemy checkout, so prompts and grading match what Evalchemy would run.
#
# Pick ONE input mode:
#   SAVE_DIR      An orbit_ckpts/<run> directory. Up to MAX_CHECKPOINTS of the
#                 checkpoints under it are evaluated serially in iteration order,
#                 one SGLang server at a time (started, evaluated, killed, GPU
#                 memory released, next). Checkpoints whose metrics already exist
#                 are skipped, so an interrupted sweep resumes.
#   ITER_DIR      A single iter_NNNNNNN/adapter or iter_NNNNNNN torch_dist directory.
#   MODEL_PATH    A single dense HF model directory, served as-is.
#
# SGLang serves dense weights, so each checkpoint is first turned into one:
#   - PEFT runs (iter_*/adapter): OFT is baked with tools/bake_oft_to_hf.py, LoRA is
#     merged with peft. The base model comes from adapter_config.json unless
#     BASE_MODEL is set.
#   - Full-finetune runs (iter_*/.metadata): converted with
#     tools/convert_torch_dist_to_hf.py. BASE_MODEL is REQUIRED here -- a torch_dist
#     checkpoint carries no record of its HF origin.
# Converted copies land under BAKE_ROOT and, unless KEEP_BAKED=1, each one this run
# creates is removed after its checkpoint is scored -- a 1.7B bf16 copy is ~3.4GB,
# which adds up fast across a sweep. A copy reused from an earlier run is never
# deleted.
#
# Required env var:
#   EVALCHEMY_ROOT  Path to an Evalchemy checkout (commit 6ed67415 is what the
#                   OPD reproduction suite pins).
#
# Common overrides (defaults shown):
#   OUTPUT_DIR=${EVAL_RESULTS_ROOT}/<run>/<iter>/math   (single-checkpoint mode only)
#   EVAL_RESULTS_ROOT=${ORBIT_ROOT}/eval_results
#   DATA_NAMES=math500,aime24,amc23     from {aime24,aime25,amc23,math500}
#   N_SAMPLING=0                        0 = the paper repetition count per task
#   TEMPERATURE=0.7  TOP_P=0.95         Evalchemy's distributed-vLLM sampling
#   SEED=0                              Evalchemy's scheme: repetition n of every
#                                       example is sent with sampling seed SEED+n,
#                                       fixed across runs, so a rerun re-issues the
#                                       same draws. 0 matches Evalchemy's default;
#                                       -1 sends no seed (server RNG runs free).
#   DETERMINISTIC=0                     1 = serve with SGLang's batch-invariant
#                                       kernels (--enable-deterministic-inference).
#                                       Needed for bit-identical reruns: without it
#                                       a fixed SEED still drifts, because a request's
#                                       logits depend on which other requests share
#                                       its batch. Costs throughput.
#   ATTENTION_BACKEND=triton            only used when DETERMINISTIC=1
#   MAX_TOKENS_PER_CALL=32768
#   MAX_MODEL_LEN=40960
#   NUM_GPUS=4                          total GPUs for the eval server
#   EVAL_TP_SIZE=2                      tensor-parallel size; replicas = NUM_GPUS/TP
#   GPU_MEMORY_UTILIZATION=0.85
#   NUM_SAMPLES=0                       0 = all examples; >0 truncates (smoke runs)
#   MAX_CHECKPOINTS=6                   cap on checkpoints per sweep; 0 = no cap
#   CHECKPOINT_SELECT=first             which ones when capped: first | last | spread
#                                       (spread = evenly spaced, both ends included)
#   ENABLE_THINKING=0
#   GRADER=evalchemy                    or 'orbit' for Orbit's --rm-type math grader
#   CONCURRENCY=64  PORT=18001
#   KEEP_BAKED=0  BAKE_ROOT  BASE_MODEL
#   PYTHON_BIN           serving + adapter baking / checkpoint conversion
#   RUNNER_PYTHON_BIN    the grading runner only (defaults to PYTHON_BIN)
#
# Sweep every checkpoint of a run:
#   EVALCHEMY_ROOT=/path/to/evalchemy \
#   SAVE_DIR=orbit_ckpts/Qwen3-1.7B_4B_Instruct2507_openreasoning100k_full_vocab_opd_oft_fkl \
#       bash examples/on_policy_distillation/eval/eval-math-evalchemy.sh
#
# Single checkpoint:
#   EVALCHEMY_ROOT=/path/to/evalchemy \
#   ITER_DIR=/.../orbit_ckpts/<run>/iter_0000100/adapter \
#       bash examples/on_policy_distillation/eval/eval-math-evalchemy.sh
#
# The single-checkpoint env contract also matches the PEFT-Arena wrapper's, so this
# works as a drop-in there if you prefer that loop's logging:
#   EVAL_WRAPPER=examples/on_policy_distillation/eval/eval-math-evalchemy.sh \
#   EVALCHEMY_ROOT=... SAVE_DIR=... bash tools/eval_checkpoints_once.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# SGLang's deep_gemm import requires CUDA_HOME; the training launchers get it
# from tool_env.sh, so reuse the same setup here.
# shellcheck source=../../../scripts/lib/tool_env.sh
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"

: "${EVALCHEMY_ROOT:?EVALCHEMY_ROOT (path to an evalchemy checkout) is required}"
if [ ! -d "${EVALCHEMY_ROOT}/eval/chat_benchmarks" ]; then
    echo "[eval-math-evalchemy] ERROR: EVALCHEMY_ROOT='${EVALCHEMY_ROOT}' has no eval/chat_benchmarks/" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${ORBIT_WORKSPACE_ROOT:-${HOME}/.cache/orbit/workspace}/orbit-workspace/.venv/bin/python}"
# The grading runner may need a different interpreter than the serving/conversion
# steps: GRADER=evalchemy imports lm_eval, and Evalchemy pins a lm-eval[vllm] fork
# that is not worth forcing into Orbit's venv. Point this at a small side venv
# holding just aiohttp + lm-eval if you do not want lm_eval in the Orbit env.
RUNNER_PYTHON_BIN="${RUNNER_PYTHON_BIN:-${PYTHON_BIN}}"

EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-${ORBIT_ROOT}/eval_results}"
BAKE_ROOT="${BAKE_ROOT:-${EVAL_RESULTS_ROOT}/_baked}"
DATA_NAMES="${DATA_NAMES:-math500,aime24,amc23}"
N_SAMPLING="${N_SAMPLING:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-0}"
DETERMINISTIC="${DETERMINISTIC:-0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
MAX_TOKENS_PER_CALL="${MAX_TOKENS_PER_CALL:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
# Total GPUs for the eval server, split into NUM_GPUS/EVAL_TP_SIZE replicas.
# Default: 4 GPUs as TP2 x DP2 -- two tensor-parallel replicas. Eval is
# throughput-bound (thousands of completions), so the DP factor is what buys wall
# clock; the TP factor is what buys headroom per replica.
NUM_GPUS="${NUM_GPUS:-2}"
EVAL_TP_SIZE="${EVAL_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

if [ $((NUM_GPUS % EVAL_TP_SIZE)) -ne 0 ]; then
    echo "[eval-math-evalchemy] ERROR: NUM_GPUS (${NUM_GPUS}) must be divisible by EVAL_TP_SIZE (${EVAL_TP_SIZE})" >&2
    exit 1
fi
EVAL_DP_SIZE=$((NUM_GPUS / EVAL_TP_SIZE))
NUM_SAMPLES="${NUM_SAMPLES:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
GRADER="${GRADER:-evalchemy}"
CONCURRENCY="${CONCURRENCY:-64}"
PORT="${PORT:-18001}"
KEEP_BAKED="${KEEP_BAKED:-0}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-6}"
CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-first}"

if ! [[ "${MAX_CHECKPOINTS}" =~ ^[0-9]+$ ]]; then
    echo "[eval-math-evalchemy] ERROR: MAX_CHECKPOINTS must be a non-negative integer, got '${MAX_CHECKPOINTS}'" >&2
    exit 1
fi

SERVED_MODEL="orbit-opd-eval-${PORT}"
BASE_URL="http://127.0.0.1:${PORT}"

if [ "${MAX_TOKENS_PER_CALL}" -ge "${MAX_MODEL_LEN}" ]; then
    echo "[eval-math-evalchemy] ERROR: MAX_TOKENS_PER_CALL (${MAX_TOKENS_PER_CALL}) >= MAX_MODEL_LEN (${MAX_MODEL_LEN}); leave room for the prompt." >&2
    exit 1
fi

# --- Server lifecycle -------------------------------------------------------
server_pid=""

stop_server() {
    [ -n "${server_pid}" ] || return 0
    if kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "${server_pid}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    # SGLang's TP workers are children holding the GPU allocation. While the
    # health endpoint still answers, one of them is alive and the next
    # checkpoint's server would OOM trying to allocate beside it.
    for _ in $(seq 1 60); do
        curl --noproxy '*' -sf "${BASE_URL}/health" >/dev/null 2>&1 || break
        sleep 2
    done
    server_pid=""
}

trap stop_server EXIT
trap 'stop_server; exit 130' INT TERM

start_server() {
    local model_path="$1" server_log="$2"
    local -a serve_args=(
        --model-path "${model_path}"
        --host 127.0.0.1
        --port "${PORT}"
        --served-model-name "${SERVED_MODEL}"
        --dtype bfloat16
        --context-length "${MAX_MODEL_LEN}"
        --tp-size "${EVAL_TP_SIZE}"
        --mem-fraction-static "${GPU_MEMORY_UTILIZATION}"
    )
    # Batch-invariant kernels: a request's logits stop depending on the shape of
    # the batch it happened to land in, which is what makes a fixed SEED actually
    # reproduce. Pays for it in throughput, so it stays opt-in.
    if [ "${DETERMINISTIC}" = "1" ]; then
        serve_args+=(
            --enable-deterministic-inference
            --attention-backend "${ATTENTION_BACKEND}"
        )
    fi
    # Only pass --dp-size when it does something: a single replica is the default
    # and the flag is not worth depending on for it.
    if [ "${EVAL_DP_SIZE}" -gt 1 ]; then
        serve_args+=( --dp-size "${EVAL_DP_SIZE}" )
    fi
    ${PYTHON_BIN} -m sglang.launch_server "${serve_args[@]}" >"${server_log}" 2>&1 &
    server_pid=$!

    local attempt
    for attempt in $(seq 1 120); do
        if curl --noproxy '*' -sf "${BASE_URL}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "[eval-math-evalchemy] ERROR: SGLang exited during startup; see ${server_log}" >&2
            server_pid=""
            return 1
        fi
        if [ "${attempt}" -eq 120 ]; then
            echo "[eval-math-evalchemy] ERROR: timed out waiting for ${BASE_URL}/health; see ${server_log}" >&2
            return 1
        fi
        sleep 5
    done
}

# --- Checkpoint resolution --------------------------------------------------
# Echoes "<created>|<dense HF dir>" for an iter_*/adapter, baking/merging the
# adapter when needed; <created> is 1 when this call produced the bake. It is
# returned rather than set as a variable because the caller reads this through a
# command substitution, whose subshell would discard any assignment.
resolve_adapter() {
    local iter_dir="$1" run_name="$2" iter_name="$3"
    local base_model peft_type dense_dir

    base_model="${BASE_MODEL:-$(${PYTHON_BIN} - <<EOF
import json, sys
cfg = json.load(open("${iter_dir}/adapter_config.json", encoding="utf-8"))
base = cfg.get("base_model_name_or_path") or ""
if not base:
    sys.exit("adapter_config.json missing base_model_name_or_path")
print(base)
EOF
)}"
    if [ ! -d "${base_model}" ]; then
        echo "[eval-math-evalchemy] ERROR: base model '${base_model}' does not exist on disk" >&2
        return 1
    fi

    dense_dir="${BAKE_ROOT}/${run_name}/${iter_name}"
    if [ -f "${dense_dir}/config.json" ]; then
        echo "[eval-math-evalchemy] reusing baked model at ${dense_dir}" >&2
        printf '0|%s' "${dense_dir}"
        return 0
    fi

    peft_type="$(${PYTHON_BIN} -c "import json; print(json.load(open('${iter_dir}/adapter_config.json', encoding='utf-8')).get('peft_type', ''))" | tr '[:upper:]' '[:lower:]')"
    if [ "${peft_type}" = "lora" ]; then
        echo "[eval-math-evalchemy] merging LoRA adapter into ${dense_dir}" >&2
        ${PYTHON_BIN} - >&2 <<EOF
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("${base_model}", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, "${iter_dir}").merge_and_unload()
model.save_pretrained("${dense_dir}")
AutoTokenizer.from_pretrained("${base_model}").save_pretrained("${dense_dir}")
EOF
    else
        echo "[eval-math-evalchemy] baking OFT adapter into ${dense_dir}" >&2
        ${PYTHON_BIN} "${ORBIT_ROOT}/tools/bake_oft_to_hf.py" \
            --base "${base_model}" --adapter "${iter_dir}" --output "${dense_dir}" >&2
    fi
    printf '1|%s' "${dense_dir}"
}

# Same contract as resolve_adapter, for a full-finetune torch_dist iter_* directory.
# BASE_MODEL is required here: unlike an adapter, a torch_dist checkpoint carries no
# record of the HF model it came from, and the converter needs it for the tokenizer,
# config, and parameter naming.
resolve_torch_dist() {
    local iter_dir="$1" run_name="$2" iter_name="$3"
    local dense_dir="${BAKE_ROOT}/${run_name}/${iter_name}"

    if [ -z "${BASE_MODEL:-}" ]; then
        echo "[eval-math-evalchemy] ERROR: full-finetune checkpoints need BASE_MODEL set to the" >&2
        echo "  original HF checkpoint (the launcher's HF_CKPT) so torch_dist can be converted." >&2
        return 1
    fi
    if [ -f "${dense_dir}/config.json" ]; then
        echo "[eval-math-evalchemy] reusing converted model at ${dense_dir}" >&2
        printf '0|%s' "${dense_dir}"
        return 0
    fi

    echo "[eval-math-evalchemy] converting torch_dist checkpoint into ${dense_dir}" >&2
    ${PYTHON_BIN} "${ORBIT_ROOT}/tools/convert_torch_dist_to_hf.py" \
        --input-dir "${iter_dir}" \
        --output-dir "${dense_dir}" \
        --origin-hf-dir "${BASE_MODEL}" \
        --force >&2
    printf '1|%s' "${dense_dir}"
}

metrics_complete() {
    local output_dir="$1" dataset
    local -a _datasets
    IFS=',' read -r -a _datasets <<< "${DATA_NAMES}"
    for dataset in "${_datasets[@]}"; do
        [ -f "${output_dir}/${dataset}/metrics.json" ] || return 1
    done
    return 0
}

run_eval() {
    local model_path="$1" output_dir="$2"
    local runner_args=(
        --base-url "${BASE_URL}"
        --model "${SERVED_MODEL}"
        --evalchemy-root "${EVALCHEMY_ROOT}"
        --output-dir "${output_dir}"
        --tasks "${DATA_NAMES}"
        --n-sampling "${N_SAMPLING}"
        --limit "${NUM_SAMPLES}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --seed "${SEED}"
        --max-tokens "${MAX_TOKENS_PER_CALL}"
        --grader "${GRADER}"
        --concurrency "${CONCURRENCY}"
    )
    if [ "${ENABLE_THINKING}" = "1" ]; then
        runner_args+=( --enable-thinking )
    fi
    PYTHONPATH="${ORBIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        ${RUNNER_PYTHON_BIN} "${SCRIPT_DIR}/run_evalchemy_math_eval.py" "${runner_args[@]}"
}

# Serve one dense model, score it, tear the server down. Never leaves a server
# running, so the caller can immediately start the next one.
serve_and_eval() {
    local model_path="$1" output_dir="$2"
    mkdir -p "${output_dir}"
    local status=0
    if start_server "${model_path}" "${output_dir}/sglang.log"; then
        echo "[eval-math-evalchemy] server up on ${BASE_URL}"
        run_eval "${model_path}" "${output_dir}" || status=$?
    else
        status=1
    fi
    stop_server
    return "${status}"
}

# --- Build the work list ----------------------------------------------------
# Each entry is "<kind>|<path>|<run_name>|<iter_name>"; kind is adapter (PEFT run),
# distcp (full-finetune torch_dist), or dense (already-converted HF directory).
declare -a targets=()
if [ -n "${SAVE_DIR:-}" ]; then
    SAVE_DIR="$(cd -- "${SAVE_DIR}" && pwd)"
    RUN_NAME="${RUN_NAME:-$(basename "${SAVE_DIR}")}"
    for adapter_dir in "${SAVE_DIR}"/iter_*/adapter; do
        [ -f "${adapter_dir}/adapter_config.json" ] || continue
        [ -f "${adapter_dir}/adapter_model.safetensors" ] || continue
        targets+=("adapter|${adapter_dir}|${RUN_NAME}|$(basename "$(dirname "${adapter_dir}")")")
    done
    if [ "${#targets[@]}" -eq 0 ]; then
        # No adapters: a full-finetune run, whose iter_* dirs are torch_dist shards.
        for iter_dir in "${SAVE_DIR}"/iter_*; do
            [ -f "${iter_dir}/.metadata" ] || continue
            targets+=("distcp|${iter_dir}|${RUN_NAME}|$(basename "${iter_dir}")")
        done
    fi
    if [ "${#targets[@]}" -eq 0 ]; then
        echo "[eval-math-evalchemy] ERROR: no iter_*/adapter and no iter_*/.metadata under ${SAVE_DIR}" >&2
        exit 1
    fi

    # Cap the sweep. Applied at selection time, not to the number of evaluations
    # actually run, so a given run always resolves to the same set whether or not
    # some of it is already scored. iter_* names are zero-padded, so the glob is
    # already in step order.
    if [ "${MAX_CHECKPOINTS}" -gt 0 ] && [ "${#targets[@]}" -gt "${MAX_CHECKPOINTS}" ]; then
        total="${#targets[@]}"
        declare -a picked=()
        case "${CHECKPOINT_SELECT}" in
            last)   for ((i = total - MAX_CHECKPOINTS; i < total; i++)); do picked+=("${targets[i]}"); done ;;
            first)  for ((i = 0; i < MAX_CHECKPOINTS; i++)); do picked+=("${targets[i]}"); done ;;
            spread)
                # Evenly spaced, both endpoints included: the first and final
                # checkpoints are always in the set.
                if [ "${MAX_CHECKPOINTS}" -eq 1 ]; then
                    picked+=("${targets[total - 1]}")
                else
                    for ((i = 0; i < MAX_CHECKPOINTS; i++)); do
                        idx=$(( (i * (total - 1) + (MAX_CHECKPOINTS - 1) / 2) / (MAX_CHECKPOINTS - 1) ))
                        picked+=("${targets[idx]}")
                    done
                fi
                ;;
            *)
                echo "[eval-math-evalchemy] ERROR: CHECKPOINT_SELECT must be last, first or spread, got '${CHECKPOINT_SELECT}'" >&2
                exit 1
                ;;
        esac
        echo "[eval-math-evalchemy] ${total} checkpoints found, capped to ${MAX_CHECKPOINTS} (CHECKPOINT_SELECT=${CHECKPOINT_SELECT});"
        echo "[eval-math-evalchemy]   skipping: $(
            for t in "${targets[@]}"; do
                keep=0
                for k in "${picked[@]}"; do [ "${t}" = "${k}" ] && keep=1 && break; done
                [ "${keep}" -eq 0 ] && printf '%s ' "$(echo "${t}" | cut -d'|' -f4)"
            done
        )"
        echo "[eval-math-evalchemy]   set MAX_CHECKPOINTS=0 to evaluate all of them"
        targets=("${picked[@]}")
    fi
elif [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATH="$(cd -- "${MODEL_PATH}" && pwd)"
    targets+=("dense|${MODEL_PATH}|${RUN_NAME:-$(basename "${MODEL_PATH}")}|${ITER_NAME:-iter_0000000}")
else
    : "${ITER_DIR:?set SAVE_DIR (sweep), ITER_DIR (one checkpoint), or MODEL_PATH (one dense HF dir)}"
    ITER_DIR="$(cd -- "${ITER_DIR}" && pwd)"
    _run="${RUN_NAME:-$(basename "$(dirname "$(dirname "${ITER_DIR}")")")}"
    if [ -f "${ITER_DIR}/adapter_config.json" ]; then
        targets+=("adapter|${ITER_DIR}|${_run}|$(basename "$(dirname "${ITER_DIR}")")")
    elif [ -f "${ITER_DIR}/.metadata" ]; then
        targets+=("distcp|${ITER_DIR}|${RUN_NAME:-$(basename "$(dirname "${ITER_DIR}")")}|$(basename "${ITER_DIR}")")
    else
        echo "[eval-math-evalchemy] ERROR: ${ITER_DIR} is neither an adapter dir" >&2
        echo "  (adapter_config.json) nor a torch_dist checkpoint (.metadata)." >&2
        exit 1
    fi
fi

echo "[eval-math-evalchemy] ${#targets[@]} checkpoint(s), tasks=${DATA_NAMES}, grader=${GRADER}"

# --- Serial sweep -----------------------------------------------------------
done_count=0
failed_count=0
skipped_count=0

for target in "${targets[@]}"; do
    IFS='|' read -r kind ckpt_path run_name iter_name <<< "${target}"
    bake_created=0
    dense_dir=""
    [ "${kind}" = "dense" ] && dense_dir="${ckpt_path}"

    done_count=$((done_count + 1))
    # Single-checkpoint callers may pin OUTPUT_DIR; a sweep always derives it.
    if [ "${#targets[@]}" -eq 1 ] && [ -n "${OUTPUT_DIR:-}" ]; then
        output_dir="${OUTPUT_DIR}"
    else
        output_dir="${EVAL_RESULTS_ROOT}/${run_name}/${iter_name}/math"
    fi

    if metrics_complete "${output_dir}"; then
        echo "[eval-math-evalchemy] ($done_count/${#targets[@]}) $(date +%H:%M:%S) ${iter_name}: metrics already complete, skipping"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    echo "[eval-math-evalchemy] ($done_count/${#targets[@]}) $(date +%H:%M:%S) ${iter_name} -> ${output_dir}"
    if [ "${kind}" != "dense" ]; then
        if [ "${kind}" = "adapter" ]; then
            resolved="$(resolve_adapter "${ckpt_path}" "${run_name}" "${iter_name}")" || resolved=""
        else
            resolved="$(resolve_torch_dist "${ckpt_path}" "${run_name}" "${iter_name}")" || resolved=""
        fi
        if [ -z "${resolved}" ]; then
            echo "[eval-math-evalchemy] FAIL ${iter_name}: could not resolve a dense model" >&2
            failed_count=$((failed_count + 1))
            continue
        fi
        bake_created="${resolved%%|*}"
        dense_dir="${resolved#*|}"
    fi

    if serve_and_eval "${dense_dir}" "${output_dir}"; then
        echo "[eval-math-evalchemy] done ${iter_name}"
    else
        echo "[eval-math-evalchemy] FAIL ${iter_name}; see ${output_dir}/sglang.log" >&2
        failed_count=$((failed_count + 1))
    fi

    # Only ever removes a bake this invocation produced, under BAKE_ROOT.
    if [ "${bake_created}" = "1" ] && [ "${KEEP_BAKED}" != "1" ]; then
        echo "[eval-math-evalchemy] removing baked model ${dense_dir} (KEEP_BAKED=1 to keep)"
        rm -rf "${dense_dir}"
    fi
done

# --- Curve summary ----------------------------------------------------------
# Cosmetic, and the scores are already on disk by now -- never fail the sweep over it.
if [ -n "${SAVE_DIR:-}" ]; then
    if summary_path="$(${PYTHON_BIN} "${ORBIT_ROOT}/tools/summarize_eval_results.py" \
            --run-dir "${EVAL_RESULTS_ROOT}/${RUN_NAME}")"; then
        echo "[eval-math-evalchemy] summary: ${summary_path}"
    else
        echo "[eval-math-evalchemy] WARN: summarize_eval_results.py failed; per-checkpoint metrics.json are still written" >&2
    fi
fi

echo "[eval-math-evalchemy] ${done_count} checkpoint(s): $((done_count - failed_count - skipped_count)) evaluated, ${skipped_count} skipped, ${failed_count} failed"
[ "${failed_count}" -eq 0 ]
