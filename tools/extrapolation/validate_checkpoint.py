#!/usr/bin/env python3
"""Default V_Dv(.) validator for effopd_extrapolate.py's --validator-cmd.

Scores one HF checkpoint directory by driving Orbit's existing
`examples/on_policy_distillation/eval/eval-math-evalchemy.sh` in its
MODEL_PATH (single dense checkpoint) mode: it serves the checkpoint with
SGLang, runs a handful of examples from an Evalchemy task through it, and
reports accuracy. Requires the same cluster setup that wrapper already
needs -- an EVALCHEMY_ROOT checkout and a GPU to serve on; this is meant to
run in the orbit_env_v2 cluster environment, not this authoring session.

The paper's Dv is "50 examples randomly sampled from the training set", used
only to check whether a candidate's update direction is still net-positive,
not for precise supervision (the paper explicitly notes Dv's difficulty
barely matters, Fig. 7b). This wrapper defaults to 50 examples from an
existing eval task (aime24) as a stand-in Dv, which needs no extra data
preparation. To validate against actual sampled training examples instead
(closer to the paper), pass --task with a benchmark you've regenerated from
your training jsonl in Evalchemy's `{problem, expected_answer}` schema
(`sample_validation_set.py` in this directory builds that file), or point
--validator-cmd at your own script -- effopd_extrapolate.py only requires
that the final stdout line be `{"acc": <float>}` or a bare float.

N_SAMPLING defaults to 1 (one completion per example, no pass@k repetition)
to keep each candidate's validation pass cheap -- the whole point of Dv is
that it is far smaller than a full rollout batch (paper Section 4.1).

Usage (as effopd_extrapolate.py's --validator-cmd; it appends the checkpoint
dir as the final positional argument):

    python tools/extrapolation/validate_checkpoint.py \\
        --evalchemy-root /path/to/evalchemy --task aime24 --num-samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="HF checkpoint directory to score.")
    parser.add_argument("--evalchemy-root", type=Path, required=True)
    parser.add_argument("--task", default="aime24", choices=["aime24", "aime25", "amc23", "math500"])
    parser.add_argument("--num-samples", type=int, default=50, help="|Dv| (paper default: 50).")
    parser.add_argument("--n-sampling", type=int, default=1, help="Completions per example; 1 keeps Dv cheap.")
    parser.add_argument("--num-gpus", type=int, default=2, help="Matches eval-math-evalchemy.sh's own default and this cluster's 2-GPU box.")
    parser.add_argument("--eval-tp-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--grader", default="evalchemy", choices=["evalchemy", "orbit"])
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument(
        "--orbit-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Orbit repo root (default: inferred from this file's location).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where eval-math-evalchemy.sh writes metrics.json (default: a temp dir under the checkpoint).",
    )
    parser.add_argument("--python-bin", default=None, help="PYTHON_BIN forwarded to eval-math-evalchemy.sh.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    wrapper = args.orbit_root / "examples" / "on_policy_distillation" / "eval" / "eval-math-evalchemy.sh"
    if not wrapper.is_file():
        raise FileNotFoundError(f"eval-math-evalchemy.sh not found at {wrapper}")

    output_dir = args.output_dir
    cleanup_output = False
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="effopd_validate_"))
        cleanup_output = True

    env = os.environ.copy()
    env.update(
        {
            "MODEL_PATH": str(checkpoint),
            "OUTPUT_DIR": str(output_dir),
            "EVALCHEMY_ROOT": str(args.evalchemy_root.expanduser().resolve()),
            "DATA_NAMES": args.task,
            "NUM_SAMPLES": str(args.num_samples),
            "N_SAMPLING": str(args.n_sampling),
            "TEMPERATURE": str(args.temperature),
            "NUM_GPUS": str(args.num_gpus),
            "EVAL_TP_SIZE": str(args.eval_tp_size),
            "GRADER": args.grader,
            "PORT": str(args.port),
        }
    )
    if args.python_bin:
        env["PYTHON_BIN"] = args.python_bin

    print(f"[validate] scoring {checkpoint} on {args.task} (n={args.num_samples}) via {wrapper}", file=sys.stderr, flush=True)
    subprocess.run(["bash", str(wrapper)], env=env, check=True)

    metrics_path = output_dir / args.task / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    print(f"[validate] {metrics_path}: {metrics}", file=sys.stderr)

    if cleanup_output:
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)

    print(json.dumps({"acc": metrics["acc"]}))


if __name__ == "__main__":
    main()
