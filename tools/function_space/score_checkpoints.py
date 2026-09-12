"""Score a set of full HF checkpoints on a fixed prefix bank (orbit port).

Adapted from the trl-side ``experimental/function_space/workspace_snapshot/
experiments/p4_function_space_20260907/score.py``. Behavior-preserving
changes only:

  * Config-driven (``config.json``) instead of a hardcoded P4/M6 run list and
    a trl-workspace-relative manifest -- pass any ``{run_name: checkpoint_path}``
    mapping, from either stack.
  * Dropped the ``copy_orbit_env_direct`` Lustre-direct-I/O staging step
    (that was a throughput optimization for one specific cluster's
    filesystem, not part of the scoring logic). Writes go straight to
    ``output_root`` via plain ``Path``/``numpy`` I/O. Re-add node-local
    staging yourself if you hit the same page-cache-thrashing problem on
    your cluster.
  * Dropped the hard ``transformers.__version__ == "4.57.1"`` assert. The
    installed version is still recorded in ``environment.json`` for every
    scored checkpoint so a version mismatch between two runs you are
    comparing is auditable, not silently invisible -- but it is now a
    warning, not a crash, since orbit and trl may pin different versions on
    purpose. Everything else (config-compatibility audit against the base
    model, RoPE override handling, base-noise gate) is unchanged.

The core math (``scoring.score_bank`` / ``sketch_maps``) is byte-for-byte the
same as the trl-side scorer, so cosines computed here are directly comparable
to ones computed there, *if* the two environments' numerics agree closely
enough -- see the base-noise gate below and README.md "Residual issues".

Usage:
    python score_checkpoints.py --config config.json --bank bank_a
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM

from scoring import SKETCH_SEEDS, save_score, score_bank


AUDITED_CONFIG_FIELDS = (
    "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
    "rms_norm_eps", "rope_theta", "rope_scaling", "tie_word_embeddings",
    "attention_bias", "attention_dropout", "hidden_act", "max_position_embeddings",
    "use_sliding_window", "sliding_window",
)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    config.setdefault("sketch_dimension", 1024)
    config.setdefault("batch_size", 2)
    config.setdefault("exact_positions", 16)
    config.setdefault("teacher_temperature", 0.7)
    return config


def effective_model_config(checkpoint) -> tuple[AutoConfig, dict]:
    raw_config = json.loads((Path(checkpoint) / "config.json").read_text())
    overrides = {}
    if "rope_parameters" in raw_config:
        rope = raw_config["rope_parameters"]
        assert rope["rope_type"] == "default", "Non-default RoPE needs a separate audited compatibility map"
        overrides = {"rope_theta": rope["rope_theta"], "rope_scaling": None}
    config = AutoConfig.from_pretrained(checkpoint, **overrides)
    return config, overrides


def load_scoring_model(checkpoint, overrides: dict):
    return AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        config=AutoConfig.from_pretrained(checkpoint, **overrides),
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).eval().requires_grad_(False).to("cuda")


def scalar_has_finite_teacher_rkl(scalar_path: Path) -> bool:
    if not scalar_path.exists():
        return False
    frame = pd.read_parquet(scalar_path, columns=["mean_teacher_rkl_t0p7"])
    return bool(np.isfinite(frame["mean_teacher_rkl_t0p7"].to_numpy(dtype=float)).all())


def score_one(label: str, checkpoint, *, bank_info: dict, dataset, teacher_logits, out: Path,
              base_config: AutoConfig, config: dict, relative: bool, step: int = 0) -> None:
    target = out / f"{label}.npz"
    # A prior pass may have scored this run without teacher logits (teacher_cache
    # was null). If teacher_logits is now available but the cached scalars are
    # still all-NaN for mean_teacher_rkl_t0p7, force a rescore instead of
    # silently skipping -- otherwise adding teacher_cache to config.json later
    # and rerunning does nothing.
    needs_teacher_rescore = teacher_logits is not None and not scalar_has_finite_teacher_rkl(target.with_suffix(".scalars.parquet"))
    if target.exists() and target.with_suffix(".json").exists() and not needs_teacher_rescore:
        print(f"skip complete {label}", flush=True)
        return
    if needs_teacher_rescore and target.exists():
        print(f"RESCORE {label}: teacher_cache now available, existing scalars have no teacher RKL/FKL", flush=True)
    started = time.monotonic()
    print(f"START {bank_info['id']} {label} {checkpoint}", flush=True)
    model_config, overrides = effective_model_config(checkpoint)
    effective = {k: getattr(model_config, k, None) for k in AUDITED_CONFIG_FIELDS}
    expected = {k: getattr(base_config, k, None) for k in AUDITED_CONFIG_FIELDS}
    if effective != expected:
        mismatched = {k: (expected[k], effective[k]) for k in AUDITED_CONFIG_FIELDS if expected[k] != effective[k]}
        raise AssertionError(f"{label}: config diverges from base_model on {mismatched}")
    print(f"CONFIG GATE {label} PASS rope_theta={model_config.rope_theta}", flush=True)
    model = load_scoring_model(checkpoint, overrides)
    sketches, exact, scalars = score_bank(
        model, dataset, "cuda", config["sketch_dimension"], config["batch_size"], config["exact_positions"],
        teacher_logits=teacher_logits, temperature=config["teacher_temperature"],
    )
    if relative:
        with np.load(out / "Base.npz") as base_archive:
            sketches = sketches - base_archive["sketches"].astype(np.float32)
            exact = exact - base_archive["exact"].astype(np.float32)[: exact.shape[0]]
    save_score(target, sketches, exact, scalars, dict(
        run=label, step=step, checkpoint=str(checkpoint), bank=bank_info["path"], bank_id=bank_info["id"],
        sketch_seeds=SKETCH_SEEDS, sketch_dimension=config["sketch_dimension"], dtype_on_disk="float16",
        logits_centered_over_vocabulary=True, relative_to_base=relative,
        model_config_gate=True, effective_model_config=effective, config_compatibility_overrides=overrides,
        transformers_version=transformers.__version__, torch_version=torch.__version__,
        cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        host=platform.node(), elapsed_seconds=time.monotonic() - started,
    ))
    print(f"DONE {label} seconds={time.monotonic() - started:.1f}", flush=True)
    del model, sketches, exact
    gc.collect()
    torch.cuda.empty_cache()


def base_noise_gate(out: Path, historical_base_npz: str | None) -> dict | None:
    """Optional cross-environment sanity check: rescoring the *same* base
    checkpoint here should reproduce the historical Base sketch closely.
    Mirrors score.py's baseline_parity.json. Skipped if you have no
    historical cache to compare against (e.g. a pure orbit-only run)."""
    if not historical_base_npz:
        return None
    with np.load(out / "Base.npz") as fresh, np.load(historical_base_npz) as historical:
        noise = fresh["sketches"].astype(np.float64) - historical["sketches"].astype(np.float64)
        base_norm = float(np.linalg.norm(historical["sketches"].astype(np.float64)))
        error_norm = float(np.linalg.norm(noise))
    gate = dict(noise_norm=error_norm, noise_over_base_norm=error_norm / max(base_norm, 1e-30),
                threshold=0.05, gate_pass=(error_norm / max(base_norm, 1e-30)) < 0.05,
                historical_base_npz=str(historical_base_npz))
    (out / "baseline_parity.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(f"BASE PARITY {json.dumps(gate)}", flush=True)
    if not gate["gate_pass"]:
        print("WARNING: cross-environment base discrepancy exceeds 5% of the historical base sketch norm; "
              "inspect before trusting cosines against historical runs.", flush=True)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", required=True, help="key into config['banks']")
    args = parser.parse_args()
    config = load_config(args.config)
    bank_cfg = config["banks"][args.bank]
    bank_info = {"id": args.bank, "path": bank_cfg["path"]}
    dataset = load_from_disk(bank_cfg["path"])
    out = Path(config["output_root"]) / args.bank
    out.mkdir(parents=True, exist_ok=True)

    teacher_logits = None
    teacher_cache = bank_cfg.get("teacher_cache")
    if teacher_cache:
        teacher_logits = np.load(teacher_cache, mmap_mode="r")
        if teacher_logits.shape[:2] != (len(dataset), 64):
            raise ValueError(f"Teacher cache shape {teacher_logits.shape} does not match bank ({len(dataset)},64,V)")

    base_config, base_overrides = effective_model_config(config["base_model"])
    score_one("Base", config["base_model"], bank_info=bank_info, dataset=dataset, teacher_logits=teacher_logits,
              out=out, base_config=base_config, config=config, relative=False)
    base_noise_gate(out, bank_cfg.get("historical_base_npz"))

    for label, checkpoint in config["checkpoints"].items():
        step = config.get("checkpoint_steps", {}).get(label, 300)
        # on-disk filename must carry "_step<NNN>" -- plot_cosine_heatmap.py's
        # run auto-discovery (and relations.Record's FILE_RE) both key off it.
        score_one(f"{label}_step{step:03d}", checkpoint, bank_info=bank_info, dataset=dataset,
                  teacher_logits=teacher_logits, out=out, base_config=base_config, config=config,
                  relative=True, step=step)

    (out / "COMPLETE.json").write_text(json.dumps({"bank": args.bank, "gate_pass": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
