"""Post-hoc explanations on native LoRA factors; never substitute for a freeze run."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .factors import endpoint_reconstruction


def explain(a0, a, b, scale=1.0):
    a0, a, b = a0.detach().double(), a.detach().double(), b.detach().double()
    delta = scale * b @ a
    reset = scale * b @ a0
    projection = delta @ torch.linalg.pinv(a0) @ a0
    return {"trained_delta": delta, "reset_a_delta": reset, "initial_space_projection": projection, "geometry": endpoint_reconstruction(a, b, delta, scale=scale), "initial_space_leakage": endpoint_reconstruction(a0, delta @ torch.linalg.pinv(a0), delta)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--coordinates", choices=["effective", "master"], default="effective")
    args = p.parse_args()
    initial = torch.load(args.initial, map_location="cpu", weights_only=False)
    end = torch.load(args.endpoint, map_location="cpu", weights_only=False)

    def factors(state):
        key = "effective_parameters" if args.coordinates == "effective" else "master_parameters"
        if key not in state:
            raise ValueError(f"Missing {key}; use the full checkpoint, not a lightweight geometry file")
        return state[key]

    a0s, final = factors(initial), factors(end)
    target = Path(args.output)
    if target.exists():
        raise ValueError("Use a fresh interpretation directory")
    target.mkdir(parents=True)
    report = {}
    for key, a in final.items():
        if ".adapter.linear_in.weight" not in key:
            continue
        bkey = key.replace(".adapter.linear_in.weight", ".adapter.linear_out.weight")
        if key not in a0s or bkey not in final:
            raise ValueError("Missing aligned initial/endpoint factor")
        result = explain(a0s[key], a, final[bkey], scale=end["adapter_scale"])
        report[key] = {"endpoint": result["geometry"], "initial_space": result["initial_space_leakage"]}
        torch.save({k: v for k, v in result.items() if torch.is_tensor(v)}, target / (key + ".pt"))
    if not report:
        raise ValueError("No native fused LoRA factors found")
    (target / "INTERPRETATION.json").write_text(json.dumps({"scope": "posthoc only; not a new trained checkpoint", "coordinates": args.coordinates, "modules": report}, indent=2) + "\n")


if __name__ == "__main__":
    main()
