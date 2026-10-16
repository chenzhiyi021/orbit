"""Exact one-sided LoRA continuation constraints.

The continuation experiments need to restore an existing Adam checkpoint while
holding one LoRA factor fixed.  Removing that factor from ``requires_grad``
before optimizer construction changes Megatron's parameter groups and makes the
saved optimizer state incompatible.  This module instead keeps the original
parameter groups intact and removes the selected factor's optimizer gradient
after Megatron has copied/reduced gradients, but before clipping and the inner
optimizer step.

For ``Delta W = (alpha / r) B A``:

* ``right`` freezes A (Megatron Bridge ``adapter.linear_in`` / HF ``lora_A``)
* ``left`` freezes B (Megatron Bridge ``adapter.linear_out`` / HF ``lora_B``)

The selected optimizer shards and every tensor in their restored optimizer
state are snapshotted and checked bitwise after each configured audit interval.
"""

from __future__ import annotations

import copy
import logging
import types
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


_RIGHT_MARKERS = (".adapter.linear_in.weight", ".lora_A.weight", ".lora_A.default.weight")
_LEFT_MARKERS = (".adapter.linear_out.weight", ".lora_B.weight", ".lora_B.default.weight")


def lora_factor_from_name(name: str) -> str | None:
    """Return the mathematical LoRA side represented by ``name``."""

    if any(marker in name for marker in _RIGHT_MARKERS):
        return "right"
    if any(marker in name for marker in _LEFT_MARKERS):
        return "left"
    return None


def _iter_named_parameters(model: Sequence[torch.nn.Module]) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for chunk_id, chunk in enumerate(model):
        for name, parameter in chunk.named_parameters():
            yield f"chunk{chunk_id}.{name}", parameter


def _iter_optimizer_leaves(optimizer: Any) -> Iterable[Any]:
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is None:
        yield optimizer
        return
    for child in chained:
        yield from _iter_optimizer_leaves(child)


def _zip_groups(model_groups: Any, optimizer_groups: Any) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    if model_groups is None or optimizer_groups is None:
        return
    for model_group, optimizer_group in zip(model_groups, optimizer_groups, strict=True):
        for model_parameter, optimizer_parameter in zip(model_group, optimizer_group, strict=True):
            if optimizer_parameter is not None:
                yield model_parameter, optimizer_parameter


def _leaf_parameter_pairs(leaf: Any) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Yield model parameter -> inner optimizer parameter/shard mappings."""

    seen: set[int] = set()

    # DistributedOptimizer.  Precision-aware mode uses the model-dtype shard;
    # ordinary BF16 mode uses an FP32 shard.
    if hasattr(leaf, "model_float16_groups"):
        precision_aware = bool(getattr(getattr(leaf, "config", None), "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False))
        float_target = getattr(leaf, "shard_float16_groups", None) if precision_aware else getattr(leaf, "shard_fp32_from_float16_groups", None)
        for model_parameter, optimizer_parameter in _zip_groups(leaf.model_float16_groups, float_target):
            seen.add(id(model_parameter))
            yield model_parameter, optimizer_parameter
        for model_parameter, optimizer_parameter in _zip_groups(getattr(leaf, "model_fp32_groups", None), getattr(leaf, "shard_fp32_groups", None)):
            seen.add(id(model_parameter))
            yield model_parameter, optimizer_parameter

    # Float16OptimizerWithFloat16Params.
    elif hasattr(leaf, "float16_groups"):
        for model_parameter, optimizer_parameter in _zip_groups(leaf.float16_groups, getattr(leaf, "fp32_from_float16_groups", None)):
            seen.add(id(model_parameter))
            yield model_parameter, optimizer_parameter
        for group in getattr(leaf, "fp32_from_fp32_groups", []):
            for parameter in group:
                seen.add(id(parameter))
                yield parameter, parameter

    # FP32/simple optimizer fallback: model and optimizer parameter identities
    # are the same.
    inner = getattr(leaf, "optimizer", None)
    if inner is not None:
        for group in inner.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in seen:
                    yield parameter, parameter


@dataclass
class _FrozenRecord:
    name: str
    model_parameter: torch.Tensor
    optimizer_parameter: torch.Tensor
    inner_optimizer: Any
    parameter_snapshot: torch.Tensor = field(init=False)
    state_snapshot: dict[Any, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.parameter_snapshot = self.optimizer_parameter.detach().clone()
        state = self.inner_optimizer.state.get(self.optimizer_parameter, {})
        self.state_snapshot = {key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in state.items()}

    @torch.no_grad()
    def clear_optimizer_grad(self) -> float:
        total_sq = 0.0
        for attr in ("grad", "decoupled_grad"):
            grad = getattr(self.optimizer_parameter, attr, None)
            if grad is not None:
                total_sq += float(torch.sum(grad.detach().float().square()).item())
                setattr(self.optimizer_parameter, attr, None)
        return total_sq

    @torch.no_grad()
    def assert_unchanged(self) -> None:
        if not torch.equal(self.optimizer_parameter.detach(), self.parameter_snapshot):
            raise RuntimeError(f"Frozen LoRA optimizer parameter changed: {self.name}")
        current_state = self.inner_optimizer.state.get(self.optimizer_parameter, {})
        if set(current_state) != set(self.state_snapshot):
            raise RuntimeError(f"Frozen LoRA optimizer-state keys changed: {self.name}")
        for key, expected in self.state_snapshot.items():
            actual = current_state[key]
            if torch.is_tensor(expected):
                if not torch.is_tensor(actual) or not torch.equal(actual.detach(), expected):
                    raise RuntimeError(f"Frozen LoRA optimizer state changed: {self.name}[{key!r}]")
            elif actual != expected:
                raise RuntimeError(f"Frozen LoRA optimizer state changed: {self.name}[{key!r}]")


@dataclass
class LoRAFactorConstraint:
    factor: str
    audit_interval: int
    records: list[_FrozenRecord]
    selected_tensor_count_local_model: int
    unowned_tensor_count_local_dp: int
    prepare_calls: int = 0
    optimizer_steps: int = 0
    verified_steps: int = 0
    last_frozen_grad_norm: float = 0.0

    @property
    def frozen_numel(self) -> int:
        return sum(record.model_parameter.numel() for record in self.records)

    @torch.no_grad()
    def clear_grads(self) -> None:
        total_sq = sum(record.clear_optimizer_grad() for record in self.records)
        self.prepare_calls += 1
        if total_sq:
            self.last_frozen_grad_norm = total_sq**0.5

    @torch.no_grad()
    def audit(self, *, force: bool = False) -> None:
        if not force and self.optimizer_steps % self.audit_interval:
            return
        for record in self.records:
            record.assert_unchanged()
        self.verified_steps += 1

    def summary(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "selected_tensor_count_local_model": self.selected_tensor_count_local_model,
            "frozen_tensor_count": len(self.records),
            "unowned_tensor_count_local_dp": self.unowned_tensor_count_local_dp,
            "frozen_numel_local": self.frozen_numel,
            "prepare_calls": self.prepare_calls,
            "optimizer_steps": self.optimizer_steps,
            "verified_steps": self.verified_steps,
            "last_frozen_grad_norm": self.last_frozen_grad_norm,
        }


def install_lora_factor_constraint(
    model: Sequence[torch.nn.Module],
    optimizer: Any,
    *,
    factor: str,
    audit_interval: int = 1,
) -> LoRAFactorConstraint | None:
    """Install an exact one-sided LoRA optimizer constraint after resume."""

    factor = str(factor).lower()
    if factor == "none":
        return None
    if factor not in {"right", "left"}:
        raise ValueError(f"Unsupported LoRA freeze factor: {factor!r}")
    if audit_interval < 1:
        raise ValueError("LoRA freeze audit interval must be at least one")
    if hasattr(optimizer, "_orbit_lora_factor_constraint"):
        raise RuntimeError("LoRA factor constraint is already installed")

    selected = {id(parameter): (name, parameter) for name, parameter in _iter_named_parameters(model) if lora_factor_from_name(name) == factor}
    if not selected:
        raise RuntimeError(f"No LoRA {factor} factor parameters were found")

    records: list[_FrozenRecord] = []
    mapped: set[int] = set()
    distributed_owned: set[int] = set()
    saw_distributed_optimizer = False
    for leaf in _iter_optimizer_leaves(optimizer):
        inner = getattr(leaf, "optimizer", None)
        if inner is None:
            continue
        # Megatron's DistributedOptimizer partitions optimizer parameters over
        # DP ranks.  Every rank still has the complete model parameter list,
        # but only the parameters in model_param_group_index_map have a local
        # optimizer shard/state.  A selected LoRA tensor absent from this map
        # is owned (and constrained) by another DP rank; requiring it to map
        # locally incorrectly rejects valid ZeRO-style partitioning.
        ownership_map = getattr(leaf, "model_param_group_index_map", None)
        if ownership_map is not None:
            saw_distributed_optimizer = True
            distributed_owned.update(id(parameter) for parameter in ownership_map)
        for model_parameter, optimizer_parameter in _leaf_parameter_pairs(leaf):
            item = selected.get(id(model_parameter))
            if item is None or id(model_parameter) in mapped:
                continue
            name, parameter = item
            records.append(_FrozenRecord(name, parameter, optimizer_parameter, inner))
            mapped.add(id(model_parameter))

    missing_ids = set(selected) - mapped
    if saw_distributed_optimizer:
        missing_owned = missing_ids & distributed_owned
        if missing_owned:
            missing = [selected[param_id][0] for param_id in missing_owned]
            raise RuntimeError(f"Could not map {len(missing)} locally-owned frozen LoRA tensors to optimizer shards: {missing[:5]!r}")
        unowned_count = len(missing_ids)
    else:
        if missing_ids:
            missing = [selected[param_id][0] for param_id in missing_ids]
            raise RuntimeError(f"Could not map {len(missing)} frozen LoRA tensors to optimizer shards: {missing[:5]!r}")
        unowned_count = 0

    constraint = LoRAFactorConstraint(
        factor=factor,
        audit_interval=audit_interval,
        records=records,
        selected_tensor_count_local_model=len(selected),
        unowned_tensor_count_local_dp=unowned_count,
    )
    original_prepare = optimizer.prepare_grads
    original_step_ready = optimizer.step_with_ready_grads

    def prepare_grads_with_constraint(self, *args, **kwargs):
        result = original_prepare(*args, **kwargs)
        constraint.clear_grads()
        return result

    def step_with_constraint(self, *args, **kwargs):
        # Defensive clearing also covers callers that bypass prepare_grads.
        constraint.clear_grads()
        result = original_step_ready(*args, **kwargs)
        constraint.optimizer_steps += 1
        constraint.audit()
        return result

    optimizer.prepare_grads = types.MethodType(prepare_grads_with_constraint, optimizer)
    optimizer.step_with_ready_grads = types.MethodType(step_with_constraint, optimizer)
    optimizer._orbit_lora_factor_constraint = constraint
    logger.warning(
        "Installed exact LoRA %s-factor constraint: owned_tensors=%d unowned_dp_tensors=%d local_numel=%d audit_interval=%d",
        factor,
        len(records),
        unowned_count,
        constraint.frozen_numel,
        audit_interval,
    )
    return constraint


def audit_lora_factor_constraint(optimizer: Any) -> dict[str, Any] | None:
    """Force a bitwise parameter/state audit and return serializable metadata."""

    constraint = getattr(optimizer, "_orbit_lora_factor_constraint", None)
    if constraint is None:
        return None
    constraint.audit(force=True)
    return constraint.summary()
