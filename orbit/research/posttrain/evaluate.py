"""Endpoint-only evaluation dispatch with an explicit evaluator/data manifest."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
from .protocol import digest


def validate_evaluation(manifest, completed_updates, profile):
    for field in ["manifest_version", "seed", "samples_per_prompt", "temperature", "top_p", "top_k", "max_response_tokens", "template_sha256", "datasets", "evaluator", "output_schema"]:
        if field not in manifest or manifest[field] is None:
            raise ValueError(f"Evaluation manifest missing {field}")
    if completed_updates != profile["formal_eval_step"]:
        raise ValueError("Formal evaluation is restricted to the declared endpoint, not checkpoint selection")
    required = {"prompt_id", "generation", "judge_output", "score", "response_tokens", "truncated"}
    if not required.issubset(manifest["output_schema"]):
        raise ValueError(f"Per-prompt outputs must include {sorted(required)}")
    for data in manifest["datasets"]:
        if digest(data["path"]) != data["sha256"]:
            raise ValueError("Evaluation dataset hash mismatch")
    e = manifest["evaluator"]
    if digest(e["path"]) != e["sha256"]:
        raise ValueError("Evaluator source hash mismatch")
    if not e.get("argv") or not all(isinstance(v, str) for v in e["argv"]):
        raise ValueError("Evaluator needs an explicit argv array, not a shell string")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--completed-updates", required=True, type=int)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    profile = json.loads((a.run / "requested_protocol.json").read_text())
    manifest = json.loads(a.manifest.read_text())
    validate_evaluation(manifest, a.completed_updates, profile)
    if not a.checkpoint.is_dir():
        raise ValueError("Checkpoint directory missing")
    if a.output.exists():
        raise ValueError("Use a new evaluation output directory")
    a.output.mkdir(parents=True)
    argv = [v.replace("{checkpoint}", str(a.checkpoint.resolve())).replace("{output}", str(a.output.resolve())) for v in manifest["evaluator"]["argv"]]
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.time()
    result = subprocess.run(argv, check=False)
    (a.output / "resources.json").write_text(json.dumps({"wall_seconds": time.time() - started, "exit_code": result.returncode, "argv": argv, "completed_optimizer_transactions": a.completed_updates, "teacher_cost_included": False, "gpu_hours": "must be reconciled with scheduler allocation"}, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)
    rows_path = a.output / "prompt_results.jsonl"
    if not rows_path.is_file():
        raise ValueError("Evaluator did not write prompt_results.jsonl; completion not certified")
    for line in rows_path.read_text().splitlines():
        row = json.loads(line)
        if set(manifest["output_schema"]) - set(row):
            raise ValueError("Incomplete per-prompt evaluation record")
    (a.output / "COMPLETE.json").write_text(json.dumps({"prompt_results_sha256": digest(rows_path)}) + "\n")


if __name__ == "__main__":
    main()
