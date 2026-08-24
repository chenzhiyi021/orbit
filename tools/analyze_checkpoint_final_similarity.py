#!/usr/bin/env python3
"""Compare intermediate Megatron torch-dist checkpoints against a run's final checkpoint.

For each ``--run`` spec, loads the raw (unconverted) Megatron distributed
state dict at a series of iterations and computes the global cosine
similarity of each intermediate checkpoint's weights against a designated
"final" iteration within that same run. All runs are plotted on one figure
so their similarity-to-final trajectories can be compared directly.

This intentionally skips HF conversion (no layer/expert unrolling, no
architecture-specific renaming) since we only need raw tensors to compare
checkpoints within the same run/architecture against each other.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed.checkpoint as dist_cp
from typing_extensions import override


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def resolve_iter_dir(checkpoint_root: Path, iteration: int) -> Path:
    candidate = checkpoint_root / f"iter_{iteration:07d}"
    if not (candidate / ".metadata").exists():
        raise FileNotFoundError(f"No DCP checkpoint metadata at {candidate}")
    return candidate


def load_weight_state_dict(iter_dir: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(str(iter_dir)),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}


def global_cosine_similarity(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]
) -> tuple[float, list[str]]:
    skipped: list[str] = sorted(set(a) ^ set(b))
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for key in sorted(set(a) & set(b)):
        ta = a[key].to(torch.float32).flatten()
        tb = b[key].to(torch.float32).flatten()
        if ta.shape != tb.shape:
            skipped.append(key)
            continue
        dot += torch.dot(ta, tb).item()
        norm_a_sq += torch.dot(ta, ta).item()
        norm_b_sq += torch.dot(tb, tb).item()
    denom = (norm_a_sq * norm_b_sq) ** 0.5
    return (dot / denom if denom else float("nan")), skipped


def parse_run_spec(spec: str) -> dict:
    label, root, iters, final_iter = spec.split(":")
    return {
        "label": label,
        "root": Path(root),
        "iters": sorted({int(x) for x in iters.split(",")}),
        "final_iter": int(final_iter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        dest="runs",
        help=(
            "label:checkpoint_root:iter1,iter2,...:final_iter -- repeatable, "
            "one per run to plot on the same figure"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("ckpt_final_similarity.png"))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    runs = [parse_run_spec(spec) for spec in args.runs]

    figure, axis = plt.subplots(figsize=(7, 5))
    csv_rows: list[tuple[str, int, float]] = []
    for run in runs:
        final_dir = resolve_iter_dir(run["root"], run["final_iter"])
        print(f"[{run['label']}] loading final checkpoint iter_{run['final_iter']:07d}")
        final_state = load_weight_state_dict(final_dir)

        steps: list[int] = []
        sims: list[float] = []
        for iteration in run["iters"]:
            iter_dir = resolve_iter_dir(run["root"], iteration)
            print(f"[{run['label']}] loading iter_{iteration:07d}")
            state = load_weight_state_dict(iter_dir)
            similarity, skipped = global_cosine_similarity(state, final_state)
            if skipped:
                print(f"[{run['label']}] iter_{iteration:07d}: {len(skipped)} skipped/mismatched keys")
            steps.append(iteration)
            sims.append(similarity)
            csv_rows.append((run["label"], iteration, similarity))
            del state

        axis.plot(steps, sims, marker="o", label=run["label"])
        del final_state

    axis.set_xlabel("Iteration")
    axis.set_ylabel("Cosine similarity to final checkpoint")
    axis.set_title("Checkpoint similarity to final checkpoint")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output, dpi=150)
    print(f"Saved figure to {args.output}")

    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run", "iteration", "cosine_similarity_to_final"])
            writer.writerows(csv_rows)
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
