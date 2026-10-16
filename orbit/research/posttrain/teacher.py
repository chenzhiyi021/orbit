"""Job-scoped SGLang prefill-temperature adapter from the sealed M6 hook."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def install_prefill_temperature(module, temperature, audit_path, model_path):
    cls = module.LogitsMetadata
    if getattr(cls, "_posttrain_temperature", None) is not None:
        if cls._posttrain_temperature != temperature:
            raise RuntimeError("Conflicting teacher temperature adapters")
        return
    original = cls.from_forward_batch.__func__
    written = False

    def converted(kls, batch):
        nonlocal written
        metadata = original(kls, batch)
        if metadata.extend_return_logprob:
            import torch

            metadata.temp_scaled_logprobs = True
            metadata.temperature = torch.full((batch.batch_size, 1), temperature, device=batch.input_ids.device)
            if not written:
                target = Path(audit_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({"temperature": temperature, "temp_scaled_logprobs": True, "model_path": str(Path(model_path).resolve()), "pid": os.getpid()}, indent=2) + "\n")
                written = True
        return metadata

    cls.from_forward_batch = classmethod(converted)
    cls._posttrain_temperature = temperature


def main():
    parser = argparse.ArgumentParser(description="Serve the fixed M6 teacher with audited T=.7 prefill logprobs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    if Path(args.audit).exists():
        raise ValueError("Use a fresh teacher audit path for this process")
    bootstrap = Path(__file__).resolve().parents[3] / "examples/posttrain_subspace/teacher_bootstrap"
    os.environ["PYTHONPATH"] = str(bootstrap) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["ORBIT_POSTTRAIN_TEACHER_PREFILL_T"] = "0.7"
    os.environ["ORBIT_POSTTRAIN_TEACHER_AUDIT"] = str(Path(args.audit).resolve())
    os.environ["ORBIT_POSTTRAIN_TEACHER_MODEL"] = str(Path(args.model).resolve())
    os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "0"
    from sglang.srt.layers import logits_processor

    install_prefill_temperature(logits_processor, 0.7, args.audit, args.model)
    os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "0"
    sys.argv = ["sglang.launch_server", "--model-path", args.model, "--port", str(args.port), "--tp-size", "1", "--mem-fraction-static", "0.18", "--max-running-requests", "16", "--max-prefill-tokens", "4096", "--attention-backend", "fa3", "--sampling-backend", "flashinfer"]
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
