"""Shared checkpoint I/O and extrapolation math for `tools/extrapolation/*`.

Orbit's full-finetune runs save Megatron torch-distributed (DCP) shards under
``orbit_ckpts/<run>/iter_NNNNNNN/`` -- not HF safetensors. Both extrapolation
methods in this package do their tensor arithmetic in **HF-named** parameter
space (matching the released EffOPD implementation, which was written against
HF checkpoints), so a raw ``iter_*`` directory is first converted with the
existing ``tools/convert_torch_dist_to_hf.py`` before any candidate is built.
Converting first (rather than keyword-filtering on Megatron-native names) also
matters for fidelity: Megatron fuses the pre-attention LayerNorm weight into
the QKV module (``self_attention.linear_qkv.layer_norm_weight``), so an
``"attention"`` substring match in Megatron-native space would catch a
different set of LayerNorm parameters than the same match does in HF-native
space (see ``EFFOPD_KEYWORDS`` below). The released EffOPD selection rule is
only reproduced correctly by matching HF names.

``CheckpointTensorStore`` and ``EFFOPD_KEYWORDS``/``should_modify`` are ported
near-verbatim from the trl-side posthoc tooling this was asked to mirror:
``trl/scripts/opd_posthoc_common.py`` and
``trl/scripts/build_effopd_extrapolated_checkpoint.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# EffOPD's released `_should_modify_param` substring rule, copied verbatim
# from `trl/scripts/build_effopd_extrapolated_checkpoint.py`. Deliberately
# over-inclusive: "attention" also selects Qwen's post_attention_layernorm
# parameters (they contain the substring "attention"), which is a quirk of
# the released implementation, not a bug in this port.
EFFOPD_KEYWORDS = (
    "attn",
    "self_attn",
    "attention",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "mlp",
    "feed_forward",
    "ffn",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Support files an HF checkpoint directory needs besides the weights, copied
# alongside an extrapolated candidate so it is directly servable.
SUPPORT_FILES = (
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def should_modify(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in EFFOPD_KEYWORDS)


class CheckpointTensorStore:
    """Read a model's weights from an HF safetensors (sharded or single-file) dir.

    Ported from `trl/scripts/opd_posthoc_common.py`, trimmed to the
    safetensors-only path (Orbit's `convert_torch_dist_to_hf.py` always emits
    safetensors, sharded or not; the pytorch_model.bin.index.json branch from
    the trl original is not needed here).
    """

    def __init__(self, checkpoint: Path, stack: ExitStack):
        checkpoint = checkpoint.resolve()
        safetensors_index = checkpoint / "model.safetensors.index.json"
        safetensors_file = checkpoint / "model.safetensors"
        self.root = checkpoint

        if safetensors_index.is_file():
            self._weight_map = json.loads(safetensors_index.read_text())["weight_map"]
            self.files = sorted(checkpoint / shard for shard in set(self._weight_map.values()))
            self._safe_handles = {
                path.name: stack.enter_context(safe_open(path, framework="pt", device="cpu")) for path in self.files
            }
        elif safetensors_file.is_file():
            self.files = [safetensors_file]
            handle = stack.enter_context(safe_open(safetensors_file, framework="pt", device="cpu"))
            self._safe_handles = {safetensors_file.name: handle}
            self._weight_map = dict.fromkeys(handle.keys(), safetensors_file.name)
        else:
            raise FileNotFoundError(f"No model.safetensors[.index.json] under {checkpoint}")

        self.keys = set(self._weight_map)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._safe_handles[self._weight_map[name]].get_tensor(name)


# --------------------------------------------------------------------------
# Megatron torch_dist -> HF conversion (cached; reused across every candidate
# built from the same anchor/current pair).
# --------------------------------------------------------------------------


def resolve_run_iterations(run_root: Path) -> list[int]:
    """Sorted ascending iteration numbers with a DCP checkpoint under `run_root`."""
    iterations = []
    for candidate in sorted(run_root.glob("iter_*")):
        if not (candidate / ".metadata").is_file():
            continue
        try:
            iterations.append(int(candidate.name.removeprefix("iter_")))
        except ValueError:
            continue
    if not iterations:
        raise FileNotFoundError(f"No iter_*/.metadata DCP checkpoints under {run_root}")
    return sorted(iterations)


def iter_dir(run_root: Path, iteration: int) -> Path:
    path = run_root / f"iter_{iteration:07d}"
    if not (path / ".metadata").is_file():
        raise FileNotFoundError(f"No DCP checkpoint metadata at {path}")
    return path


def convert_iter_to_hf(
    checkpoint_dir: Path,
    origin_hf_dir: Path,
    output_dir: Path,
    python_bin: str = sys.executable,
    orbit_root: Path | None = None,
) -> Path:
    """Materialize `checkpoint_dir` (a Megatron DCP iter_* dir) as an HF checkpoint.

    Wraps the existing `tools/convert_torch_dist_to_hf.py` as a subprocess
    rather than importing its internals directly: that script monkeypatches
    `pickle.Unpickler` at import time and expects to run as `__main__`-style
    tooling, and shelling out is exactly what
    `examples/on_policy_distillation/eval/eval-math-evalchemy.sh`
    (`resolve_torch_dist`) already does for the same conversion. Idempotent:
    if `output_dir/config.json` already exists, the cached conversion is
    reused and nothing is re-run (same caching contract as that wrapper).
    """
    if (output_dir / "config.json").is_file():
        return output_dir

    orbit_root = orbit_root or Path(__file__).resolve().parents[2]
    script = orbit_root / "tools" / "convert_torch_dist_to_hf.py"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            python_bin,
            str(script),
            "--input-dir",
            str(checkpoint_dir),
            "--output-dir",
            str(output_dir),
            "--origin-hf-dir",
            str(origin_hf_dir),
            "--force",
        ],
        check=True,
    )
    return output_dir


# --------------------------------------------------------------------------
# Extrapolation math, shared by simple_extrapolate.py and effopd_extrapolate.py
# --------------------------------------------------------------------------


def build_candidate(
    current: CheckpointTensorStore,
    anchor: CheckpointTensorStore,
    alpha: float,
    keyword_filter: bool,
) -> tuple[dict[str, torch.Tensor], dict]:
    """candidate = current + alpha * (current - anchor).

    When `keyword_filter` is True, only tensors matching `EFFOPD_KEYWORDS`
    (EffOPD's released selection rule) are modified; every other tensor is
    copied through from `current` unchanged. When False (the "simple"
    baseline), every shared tensor is extrapolated.

    Returns (tensors, stats) where stats mirrors the metadata sidecar written
    by `trl/scripts/build_effopd_extrapolated_checkpoint.py`.
    """
    missing = sorted(current.keys - anchor.keys)
    if missing:
        raise ValueError(f"Current tensors absent from anchor: {missing[:10]}")

    tensors: dict[str, torch.Tensor] = {}
    selected_names: list[str] = []
    selected_elements = 0
    total_elements = 0
    max_abs_delta = 0.0

    for name in sorted(current.keys):
        current_tensor = current.get_tensor(name)
        total_elements += current_tensor.numel()
        modify = should_modify(name) if keyword_filter else True
        if modify:
            anchor_tensor = anchor.get_tensor(name)
            if current_tensor.shape != anchor_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: current={tuple(current_tensor.shape)} "
                    f"anchor={tuple(anchor_tensor.shape)}"
                )
            delta = current_tensor.float() - anchor_tensor.float()
            candidate = current_tensor.float().add_(delta, alpha=alpha)
            tensors[name] = candidate.to(dtype=current_tensor.dtype).contiguous()
            selected_names.append(name)
            selected_elements += current_tensor.numel()
            max_abs_delta = max(max_abs_delta, float(delta.abs().max()))
        else:
            tensors[name] = current_tensor.contiguous()

    stats = {
        "alpha": alpha,
        "equivalent_total_displacement_multiplier": 1.0 + alpha,
        "keyword_filtered": keyword_filter,
        "selection_keywords": EFFOPD_KEYWORDS if keyword_filter else None,
        "selected_tensor_count": len(selected_names),
        "total_tensor_count": len(tensors),
        "selected_elements": selected_elements,
        "total_elements": total_elements,
        "selected_element_fraction": selected_elements / total_elements if total_elements else 0.0,
        "maximum_absolute_anchor_to_current_delta_fp32": max_abs_delta,
    }
    return tensors, stats


def copy_support_file(reference_hf_dir: Path, output: Path, filename: str) -> None:
    candidates = (reference_hf_dir / filename, reference_hf_dir.parent / filename)
    source = next((path for path in candidates if path.is_file()), None)
    if source is not None:
        shutil.copy2(source, output / filename)


def write_hf_checkpoint(
    tensors: dict[str, torch.Tensor],
    reference_hf_dir: Path,
    output_dir: Path,
    metadata: dict,
) -> None:
    """Write `tensors` plus tokenizer/config assets copied from `reference_hf_dir`.

    Refuses to overwrite an existing `output_dir`, matching the released
    `build_effopd_extrapolated_checkpoint.py`'s safety contract -- candidates
    are meant to be immutable once scored, so a caller re-running a sweep
    should point at a fresh directory rather than silently clobber one.
    """
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    save_file(tensors, output_dir / "model.safetensors", metadata={"format": "pt"})
    for filename in SUPPORT_FILES:
        copy_support_file(reference_hf_dir, output_dir, filename)
    (output_dir / "extrapolation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
