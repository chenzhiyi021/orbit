"""Shared helpers for reading raw Megatron torch-distributed (DCP) checkpoints.

Used by the ``analyze_checkpoint_*`` post-hoc analysis scripts to load a
plain (unsharded, single-process) state dict from a Megatron ``--load``-style
checkpoint directory, without needing a distributed process group. See
``tools/convert_torch_dist_to_hf.py`` for the original recipe this is
extracted from.
"""

from __future__ import annotations

import pickle
from pathlib import Path

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


def resolve_checkpoint_dir(checkpoint_path: Path) -> Path:
    """Resolve a Megatron ``--load``-style path the same way Megatron itself does.

    Accepts either a directory that already holds ``.metadata`` directly, or
    a checkpoint root with a ``latest_checkpointed_iteration.txt`` tracker
    (or, failing that, the highest ``iter_*`` subdirectory) -- matching
    ``orbit/utils/arguments.py``'s expectations for ``--load``/``MEGATRON_LOAD``.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {checkpoint_path!r} "
            "(if this looks empty/'.', check that the corresponding env var was actually set)"
        )

    if (checkpoint_path / ".metadata").exists():
        return checkpoint_path

    tracker_path = checkpoint_path / "latest_checkpointed_iteration.txt"
    if tracker_path.exists():
        iteration = int(tracker_path.read_text().strip())
        resolved = checkpoint_path / f"iter_{iteration:07d}"
        if (resolved / ".metadata").exists():
            return resolved

    candidates = sorted(p for p in checkpoint_path.glob("iter_*") if (p / ".metadata").exists())
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(f"No DCP checkpoint metadata found under {checkpoint_path}")


def load_weight_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(str(checkpoint_dir)),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}


def compute_delta(
    state: dict[str, torch.Tensor], base: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    skipped: list[str] = sorted(set(state) ^ set(base))
    delta: dict[str, torch.Tensor] = {}
    for key in sorted(set(state) & set(base)):
        w = state[key]
        b = base[key]
        if w.shape != b.shape:
            skipped.append(key)
            continue
        delta[key] = w.to(torch.float32) - b.to(torch.float32)
    return delta, skipped


def global_cosine_similarity(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]
) -> tuple[float, list[str]]:
    skipped: list[str] = sorted(set(a) ^ set(b))
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for key in sorted(set(a) & set(b)):
        ta = a[key].flatten()
        tb = b[key].flatten()
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
