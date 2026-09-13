#!/usr/bin/env python3
"""EffOPD: validation-gated directional extrapolation (paper Section 4.1).

Reproduces the released EffOPD procedure from "Learning to Foresee:
Unveiling the Unlocking Efficiency of On-Policy Distillation" (Cai et al.),
Section 4.1, "Accelerating OPD via Directional Extrapolation":

  Let W_t be the model after the t-th OPD update. EffOPD triggers a search at
  exponentially spaced checkpoints t=2^n (n=0,1,2,...; the first search is at
  t=1). For n=0 the local direction is W_1 - W_0 (W_0 = the pretrained/base
  model). For n>=1:

      delta_n = W_(2^n) - W_(2^(n-1))                                   (Eq.1)

  Five candidates are generated with increasing magnitude, k=1..5:

      W~_{n,k} = W_(2^n) + 2k * delta_n                                 (Eq.2)

  (the released implementation's alpha set is {2,4,6,8,10}; see
  `trl/scripts/build_effopd_extrapolated_checkpoint.py`). Only parameters
  matching `checkpoint_io.EFFOPD_KEYWORDS` (attention + MLP blocks) are
  modified -- everything else stays at W_(2^n).

  A lightweight validation set Dv (paper: 50 examples sampled from the
  training set) scores each candidate in order. W_acc starts at W_(2^n) with
  score v_acc = V(W_(2^n)). Each candidate that scores >= v_acc is accepted
  (W_acc, v_acc update to it) and the search continues to k+1; the first
  candidate that fails to improve on v_acc stops the search immediately
  (Eq.3-4). If k=1 already fails, EffOPD degenerates to vanilla OPD (W_acc
  stays W_(2^n), unmodified).

Post-hoc framing, and why: this tool runs against a *fixed, already-fully-
trained* checkpoint sequence (orbit_ckpts/<run>/iter_*), not a live training
loop it can restart from an extrapolated point. So delta_n is always taken
between the two REAL trained checkpoints W_(2^(n-1)) and W_(2^n) at every n
-- never between a previous n's *accepted/extrapolated* candidate and a real
checkpoint. This matches the paper's own W_t definition ("parameters after
the t-th OPD update", i.e. the actual trained trajectory) and mirrors the
oracle-style convention already used on the trl side for this same question
(`trl/scripts/analyze_m6_effopd_extrapolation_oracle.py`). It answers "how
good would EffOPD's candidate have been at checkpoint 2^n," not "what would
live training have produced had EffOPD been wired into the trainer" -- the
latter needs the extrapolated weights fed back into the optimizer loop
itself, out of scope for a post-hoc analysis tool.

t indexes checkpoints by their ordinal position in the run's saved sequence
(1st saved checkpoint = t=1, 2nd = t=2, ...), not by raw Megatron iteration
number -- on the lr=2e-6 run (--save-interval 2, so iter_0000001,
iter_0000003, ..., iter_0000019 are the 10 saved checkpoints), this makes
t=1..10 available, so n=0,1,2,3 (t=1,2,4,8) all run; n=4 (t=16) is skipped,
not enough saved checkpoints.

Validation is pluggable via --validator-cmd (default: this package's
validate_checkpoint.py, which serves a candidate through Orbit's existing
eval-math-evalchemy.sh MODEL_PATH path and reads back its accuracy -- see
that file's docstring for the required EVALCHEMY_ROOT / cluster setup). Any
command that accepts a checkpoint directory as its last argument and prints
a final JSON line `{"acc": <float>}` (or a bare float) to stdout works.

Example (--run-dir and --base-model default to the lr=2e-6 run and this
cluster's base Qwen3-1.7B; run from the orbit repo root in the orbit_env_v2
environment, with EVALCHEMY_ROOT etc. set for whatever --validator-cmd you
point at):

    python tools/extrapolation/effopd_extrapolate.py \\
        --validator-cmd "python tools/extrapolation/validate_checkpoint.py \\
            --evalchemy-root /path/to/evalchemy --task aime24 --num-samples 50" \\
        --output-root extrapolation_results/effopd/lr_2e-6
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from checkpoint_io import (
    CheckpointTensorStore,
    build_candidate,
    convert_iter_to_hf,
    resolve_run_iterations,
    write_hf_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("orbit_ckpts/m6_st_full_non_thinking_lr_2e-6"),
        help="orbit_ckpts/<run> directory holding iter_NNNNNNN DCP checkpoints.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/mnt/L202500431/models/qwen3-1.7b"),
        help="HF checkpoint dir the run was trained from (the launcher's HF_CKPT); this is W_0. "
        "Default: this cluster's base Qwen3-1.7B under ../models/qwen3-1.7b.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 8.0, 10.0],
        help="Candidate magnitudes 2k for k=1..5, tried in order (paper default: 2 4 6 8 10).",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="Cap the largest exponent n searched (default: as many as the saved checkpoint count allows).",
    )
    parser.add_argument(
        "--validator-cmd",
        required=True,
        help="Shell command that scores one HF checkpoint. The candidate directory is appended as the "
        "final argument; the command must print a final stdout line that is either a JSON object with "
        "an 'acc' key or a bare float. See validate_checkpoint.py for the shipped default implementation.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Where converted (Megatron->HF) real checkpoints are cached (default: <run-dir>/../_hf_cache/<run-name>).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where candidate + accepted checkpoints are written (default: extrapolation_results/effopd/<run-name>).",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Interpreter for the torch_dist->HF conversion subprocess.")
    return parser.parse_args()


def parse_validator_score(stdout: str) -> float:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Validator produced no output")
    last = lines[-1]
    try:
        parsed = json.loads(last)
        if isinstance(parsed, dict):
            if "acc" not in parsed:
                raise ValueError(f"Validator JSON output missing 'acc' key: {last!r}")
            return float(parsed["acc"])
        return float(parsed)
    except json.JSONDecodeError:
        return float(last)


def validate(validator_cmd: str, checkpoint_dir: Path) -> float:
    command = shlex.split(validator_cmd) + [str(checkpoint_dir)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_validator_score(result.stdout)


def convert_real_checkpoint(iteration: int | None, run_dir: Path, base_model: Path, hf_cache_dir: Path, python_bin: str) -> Path:
    if iteration is None:
        return base_model
    checkpoint_dir = run_dir / f"iter_{iteration:07d}"
    output_dir = hf_cache_dir / checkpoint_dir.name
    print(f"[effopd] converting real checkpoint iter_{iteration:07d} -> {output_dir}", flush=True)
    return convert_iter_to_hf(checkpoint_dir, base_model, output_dir, python_bin=python_bin)


def run_search_for_n(
    n: int,
    anchor_iter: int | None,
    current_iter: int,
    run_dir: Path,
    base_model: Path,
    hf_cache_dir: Path,
    output_root: Path,
    alphas: list[float],
    validator_cmd: str,
    python_bin: str,
) -> dict:
    anchor_hf = convert_real_checkpoint(anchor_iter, run_dir, base_model, hf_cache_dir, python_bin)
    current_hf = convert_real_checkpoint(current_iter, run_dir, base_model, hf_cache_dir, python_bin)

    n_dir = output_root / f"n{n}_t{2**n}_iter{current_iter}"
    print(f"[effopd] n={n} (t={2**n}, iter_{current_iter:07d}): scoring baseline W_2^n", flush=True)
    v_acc = validate(validator_cmd, current_hf)
    accepted_dir = current_hf
    accepted_alpha = 0.0
    accepted_score = v_acc
    log = [{"k": 0, "alpha": 0.0, "candidate": "W_2^n (unmodified)", "score": v_acc, "accepted": True}]

    with ExitStack() as stack:
        current_store = CheckpointTensorStore(current_hf, stack)
        anchor_store = CheckpointTensorStore(anchor_hf, stack)
        for k, alpha in enumerate(alphas, start=1):
            candidate_dir = n_dir / f"k{k}_alpha{alpha:g}"
            print(f"[effopd] n={n} k={k} alpha={alpha:g}: building + scoring candidate", flush=True)
            tensors, stats = build_candidate(current_store, anchor_store, alpha, keyword_filter=True)
            metadata = {
                "created_at": datetime.now(UTC).isoformat(),
                "method": "EffOPD directional extrapolation (paper Eq. 1-4)",
                "n": n,
                "t": 2**n,
                "k": k,
                "anchor": "base" if anchor_iter is None else anchor_iter,
                "current_iter": current_iter,
                **stats,
            }
            write_hf_checkpoint(tensors, current_hf, candidate_dir, metadata)
            score = validate(validator_cmd, candidate_dir)
            improved = score >= v_acc
            log.append({"k": k, "alpha": alpha, "candidate": str(candidate_dir), "score": score, "accepted": improved})
            print(f"[effopd] n={n} k={k} alpha={alpha:g}: score={score:.4f} (v_acc={v_acc:.4f}) -> {'accept' if improved else 'STOP'}", flush=True)
            if improved:
                v_acc = score
                accepted_dir = candidate_dir
                accepted_alpha = alpha
                accepted_score = score
            else:
                break

    result = {
        "n": n,
        "t": 2**n,
        "anchor": "base" if anchor_iter is None else anchor_iter,
        "anchor_hf_dir": str(anchor_hf),
        "current_iter": current_iter,
        "current_hf_dir": str(current_hf),
        "accepted_alpha": accepted_alpha,
        "accepted_dir": str(accepted_dir),
        "accepted_score": accepted_score,
        "degenerated_to_vanilla_opd": accepted_alpha == 0.0,
        "search_log": log,
    }
    manifest_path = n_dir / "search_manifest.json"
    n_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"[effopd] n={n}: accepted alpha={accepted_alpha:g} score={accepted_score:.4f} -> {manifest_path}")
    return result


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    base_model = args.base_model.expanduser().resolve()
    run_name = run_dir.name

    available = resolve_run_iterations(run_dir)
    n_max_available = int(math.floor(math.log2(len(available))))
    n_max = n_max_available if args.max_n is None else min(args.max_n, n_max_available)

    hf_cache_dir = args.hf_cache_dir or (run_dir.parent / "_hf_cache" / run_name)
    output_root = args.output_root or (Path("extrapolation_results") / "effopd" / run_name)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[effopd] {len(available)} saved checkpoints under {run_dir} "
        f"-> n=0..{n_max} (t={[2**n for n in range(n_max + 1)]})",
        flush=True,
    )

    results = []
    for n in range(n_max + 1):
        t_current = 2**n
        current_iter = available[t_current - 1]
        anchor_iter = None if n == 0 else available[2 ** (n - 1) - 1]
        results.append(
            run_search_for_n(
                n=n,
                anchor_iter=anchor_iter,
                current_iter=current_iter,
                run_dir=run_dir,
                base_model=base_model,
                hf_cache_dir=hf_cache_dir,
                output_root=output_root,
                alphas=args.alphas,
                validator_cmd=args.validator_cmd,
                python_bin=args.python_bin,
            )
        )

    manifest = {
        "run_dir": str(run_dir),
        "base_model": str(base_model),
        "saved_iterations": available,
        "n_max": n_max,
        "alphas": args.alphas,
        "results": results,
    }
    manifest_path = output_root / "effopd_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[effopd] full manifest: {manifest_path}")


if __name__ == "__main__":
    main()
