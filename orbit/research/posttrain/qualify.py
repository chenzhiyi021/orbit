"""Record CPU regression results and compare independently produced fixed-batch runs."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]


def cpu_report(output):
    import torch

    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests/posttrain"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema_version": 1,
        "cpu": {
            "status": "passed" if result.wasSuccessful() else "failed",
            "tests": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "scope": ["exact-action enumeration", "legacy mathematical function parity", "actual core response slicing/reducer", "group normalization", "CPU optimizer moments/resume/freeze", "endpoint dense-delta reconstruction"],
        },
        "gpu_migration": {name: "not_run" for name in ["m6_mixed_r2_reference", "rl_math_grpo_r4", "sft_omr_cot", "sft_teacher_matched", "sft_btarget_official_cot"]},
        "formal_training": "not_run",
        "formal_evaluation": "not_run",
        "limitations": ["CPU optimizer fixtures are not a Megatron distributed-optimizer or fused-kernel certification", "M6 real-batch GPU migration and RL TP2/DP2 short runs remain mandatory", "SFT/RL data identities must be resolved before running"],
    }
    import importlib

    observations = getattr(importlib.import_module("test_protocol"), "OBSERVATIONS", {})
    report["cpu"]["observations"] = observations
    from .launch import source_digest

    report["tested_source_hash"] = source_digest()
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return result.wasSuccessful()


def compare_fixed_batch(reference, candidate, tolerances):
    """Compare raw arrays from two *independent* execution paths.

    The archive must include the same recorded real batch hash and explicit policy,
    mask, temperature and dependency identity. This tool cannot manufacture a pass
    by comparing a run against itself or by silently dropping missing tensors.
    """
    import torch

    if Path(reference).resolve() == Path(candidate).resolve():
        raise ValueError("Need two independent execution paths")
    old = torch.load(reference, map_location="cpu", weights_only=False)
    new = torch.load(candidate, map_location="cpu", weights_only=False)
    required = {"batch_sha256", "policy_sha256", "temperatures", "masks", "raw_advantage", "scaled_advantage", "scalar_loss", "gradients", "adam_m", "adam_v", "adam_step", "applied_lr", "effective_delta", "source_hashes"}
    for label, run in [("reference", old), ("candidate", new)]:
        if required - set(run):
            raise ValueError(f"{label} missing {sorted(required - set(run))}")
    for key in ["batch_sha256", "policy_sha256", "temperatures", "adam_step", "applied_lr"]:
        if old[key] != new[key]:
            raise ValueError(f"Fixed-batch identity mismatch: {key}")
    if not torch.equal(old["masks"], new["masks"]):
        raise ValueError("Mask mismatch")
    result = {}
    for key in ["raw_advantage", "scaled_advantage", "scalar_loss", "gradients", "adam_m", "adam_v", "effective_delta"]:
        if key not in tolerances:
            raise ValueError(f"Predeclare a tolerance for {key}")
        a, b = torch.as_tensor(old[key]).double().flatten(), torch.as_tensor(new[key]).double().flatten()
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Invalid/unaligned arrays for {key}")
        tolerance = tolerances[key]
        delta = (a - b).abs()
        limit = tolerance["atol"] + tolerance["rtol"] * a.abs()
        result[key] = {"passed": bool((delta <= limit).all()), "max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "rms": float(delta.square().mean().sqrt()), "p50_abs": float(delta.quantile(0.5)), "p95_abs": float(delta.quantile(0.95)), "p99_abs": float(delta.quantile(0.99)), "tolerance": tolerance}
    return {
        "status": "passed" if all(v["passed"] for v in result.values()) else "failed",
        "scope": "fixed recorded batch/one optimizer step only",
        "comparisons": result,
        "reference_sha256": hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(candidate).read_bytes()).hexdigest(),
        "batch_sha256": old["batch_sha256"],
        "sources": {"reference": old["source_hashes"], "candidate": new["source_hashes"]},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    cpu = s.add_parser("cpu")
    cpu.add_argument("--output", required=True)
    compare = s.add_parser("compare")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--tolerances", required=True)
    compare.add_argument("--output", required=True)
    a = p.parse_args()
    if a.command == "cpu":
        raise SystemExit(0 if cpu_report(a.output) else 1)
    result = compare_fixed_batch(a.reference, a.candidate, json.loads(Path(a.tolerances).read_text()))
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
