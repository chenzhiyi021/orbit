"""Keep a full fine-tune's weight update out of a fixed per-tensor subspace.

Used to ask whether the directions a previous fine-tune relied on are *necessary*: build
per-tensor bases with tools/function_space/build_exclusion_subspace.py (top-k singular
vectors of that fine-tune's Delta W, or random controls), then retrain with
--exclude-subspace-path / --exclude-subspace-k / --exclude-subspace-side.

Mechanism (a reparametrization, so it is exact and optimizer-agnostic). With W the raw
parameter the optimizer updates and P the projector that removes the excluded subspace,

    left:  P(D) = (I - U_k U_k^T) D        right: P(D) = D (I - V_k V_k^T)        both: both

the network always runs with the effective weight  W_eff = W_base + P(W - W_base).
Because P is linear and self-adjoint, dL/dW = P(dL/dW_eff), so:

  * after backward and BEFORE the data-parallel gradient reduction, every rank projects its
    local main_grad (projection commutes with the sum, so the reduced gradient is exactly
    P of the global gradient -- no full-gradient gather is needed under the distributed
    optimizer's sharded reduce-scatter);
  * after every optimizer step the (all-gathered) model parameters are overwritten with
    W_eff, so the rollout engine's weight sync, the next forward pass and every saved
    checkpoint all see the constrained weights. The fp32 master weights keep the raw W;
    whatever Adam puts into the excluded subspace there is never used.

Layout: the bases are stored per HF tensor (q/k/v_proj, gate/up_proj separately), while
Megatron fuses them into linear_qkv (rows interleaved per query group: the group's query
heads, then its key head, then its value head) and linear_fc1 (gate rows, then up rows).
Each HF block is projected with its own U/V on its own rows of the fused matrix. At setup
the fused layout is checked against the HF base checkpoint, so a layout mismatch fails
loudly instead of silently projecting the wrong rows.

Requires tensor- and pipeline-parallel size 1 (every rank holds whole matrices) and no
overlapped grad reduce / param gather (see validate()).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

SIDES = ("left", "right", "both")
_MEGATRON_RE = re.compile(
    r"decoder\.layers\.(\d+)\.(self_attention\.linear_qkv|self_attention\.linear_proj|mlp\.linear_fc1|mlp\.linear_fc2)\.weight$"
)


@dataclass
class _Block:
    hf_name: str
    rows: torch.Tensor | None  # rows of the Megatron matrix holding this HF tensor, in HF row order; None = all
    u: torch.Tensor | None = None  # (block_out, k) fp32
    v: torch.Tensor | None = None  # (in, k) fp32


@dataclass
class _Target:
    megatron_name: str
    param: torch.nn.Parameter
    base: torch.Tensor  # bf16 copy of the initial (base) weights, Megatron layout
    blocks: list[_Block] = field(default_factory=list)


def qkv_row_indices(num_heads: int, num_groups: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows of Megatron's fused linear_qkv that hold HF q_proj / k_proj / v_proj, each in HF row order."""
    rep = num_heads // num_groups
    group_rows = (rep + 2) * head_dim
    arange = torch.arange(head_dim)
    q = torch.cat([(h // rep) * group_rows + (h % rep) * head_dim + arange for h in range(num_heads)])
    k = torch.cat([g * group_rows + rep * head_dim + arange for g in range(num_groups)])
    v = torch.cat([g * group_rows + (rep + 1) * head_dim + arange for g in range(num_groups)])
    return q, k, v


def hf_blocks_for(kind: str, layer: int, num_heads: int, num_groups: int, head_dim: int, ffn: int) -> list[_Block]:
    prefix = f"model.layers.{layer}"
    if kind == "self_attention.linear_qkv":
        q, k, v = qkv_row_indices(num_heads, num_groups, head_dim)
        return [_Block(f"{prefix}.self_attn.q_proj.weight", q),
                _Block(f"{prefix}.self_attn.k_proj.weight", k),
                _Block(f"{prefix}.self_attn.v_proj.weight", v)]
    if kind == "self_attention.linear_proj":
        return [_Block(f"{prefix}.self_attn.o_proj.weight", None)]
    if kind == "mlp.linear_fc1":
        return [_Block(f"{prefix}.mlp.gate_proj.weight", torch.arange(ffn)),
                _Block(f"{prefix}.mlp.up_proj.weight", torch.arange(ffn, 2 * ffn))]
    if kind == "mlp.linear_fc2":
        return [_Block(f"{prefix}.mlp.down_proj.weight", None)]
    raise ValueError(kind)


def project_(matrix: torch.Tensor, blocks: list[_Block], side: str) -> torch.Tensor:
    """In place: remove each block's excluded subspace from its rows of `matrix` (fp32)."""
    for block in blocks:
        part = matrix if block.rows is None else matrix[block.rows]
        if side in ("left", "both"):
            part = part - block.u @ (block.u.T @ part)
        if side in ("right", "both"):
            part = part - (part @ block.v) @ block.v.T
        if block.rows is None:
            matrix.copy_(part)
        else:
            matrix[block.rows] = part
    return matrix


class SubspaceExclusion:
    def __init__(self, args, model_chunks) -> None:
        from safetensors import safe_open

        self.side = args.exclude_subspace_side
        self.k = args.exclude_subspace_k
        heads, groups = args.num_attention_heads, args.num_query_groups or args.num_attention_heads
        head_dim = args.kv_channels or args.hidden_size // args.num_attention_heads
        ffn = args.ffn_hidden_size
        device = torch.cuda.current_device()

        path = Path(args.exclude_subspace_path)
        with safe_open(str(path), framework="pt") as handle:
            meta = handle.metadata() or {}
            available = set(handle.keys())
            k_max = int(meta.get("k_max", "0") or 0)
            if self.k > k_max:
                raise ValueError(f"--exclude-subspace-k {self.k} exceeds the file's k_max {k_max} ({path})")
            self.targets: list[_Target] = []
            for chunk in model_chunks:
                for name, param in chunk.named_parameters():
                    match = _MEGATRON_RE.search(name)
                    if match is None:
                        continue
                    blocks = hf_blocks_for(match.group(2), int(match.group(1)), heads, groups, head_dim, ffn)
                    for block in blocks:
                        for side_key, attr in (("U", "u"), ("V", "v")):
                            key = f"{block.hf_name}.{side_key}"
                            if key not in available:
                                raise KeyError(f"{key} missing from {path}")
                            setattr(block, attr, handle.get_tensor(key)[:, : self.k].float().to(device).contiguous())
                        if block.rows is not None:
                            block.rows = block.rows.to(device)
                    self.targets.append(_Target(name, param, param.detach().clone(), blocks))
        if not self.targets:
            raise RuntimeError("--exclude-subspace-path set but no linear_qkv/linear_proj/linear_fc1/linear_fc2 weights found")
        self._check_layout(args.hf_checkpoint)
        logger.info("subspace exclusion: %d tensors, k=%d, side=%s, source=%s (kind=%s, k_max=%s)",
                    len(self.targets), self.k, self.side, path, meta.get("kind"), meta.get("k_max"))

    def _check_layout(self, hf_dir: str) -> None:
        """At setup the parameters are the base model: rebuild a few fused matrices from the HF
        base checkpoint with our row mapping and require them to match."""
        import json

        from safetensors import safe_open

        hf_path = Path(hf_dir)
        index = hf_path / "model.safetensors.index.json"
        if index.exists():
            locations = {name: str(hf_path / shard) for name, shard in json.loads(index.read_text())["weight_map"].items()}
        else:
            single = hf_path / "model.safetensors"
            with safe_open(str(single), framework="pt") as handle:
                locations = {name: str(single) for name in handle.keys()}
        checks = [t for t in self.targets if ".layers.0." in t.megatron_name] or self.targets[:4]
        for target in checks:
            rebuilt = torch.empty_like(target.base)
            for block in target.blocks:
                with safe_open(locations[block.hf_name], framework="pt") as handle:
                    weight = handle.get_tensor(block.hf_name).to(rebuilt.device, rebuilt.dtype)
                if block.rows is None:
                    rebuilt.copy_(weight)
                else:
                    rebuilt[block.rows] = weight
            err = (rebuilt.float() - target.base.float()).abs().max().item()
            if err > 1e-2:
                raise RuntimeError(
                    f"subspace exclusion: fused layout of {target.megatron_name} does not match the HF base "
                    f"(max abs diff {err:.3g}); training must start from the base checkpoint and the "
                    f"q/k/v / gate/up row mapping must match Megatron's")
        logger.info("subspace exclusion: fused-layout check passed on %d tensors", len(checks))

    @torch.no_grad()
    def project_grads(self) -> None:
        for target in self.targets:
            grad = getattr(target.param, "main_grad", None)
            if grad is None:
                grad = target.param.grad
            if grad is None:
                continue
            g32 = grad.float()
            project_(g32, target.blocks, self.side)
            grad.copy_(g32)

    @torch.no_grad()
    def constrain_params(self) -> None:
        for target in self.targets:
            base32 = target.base.float()
            delta = target.param.data.float() - base32
            project_(delta, target.blocks, self.side)
            target.param.data.copy_((base32 + delta).to(target.param.dtype))

    @torch.no_grad()
    def excluded_fraction(self) -> float:
        """||delta inside the excluded subspace||^2 / ||delta||^2 over all targets (should stay ~0)."""
        inside, total = 0.0, 0.0
        for target in self.targets:
            delta = target.param.data.float() - target.base.float()
            kept = project_(delta.clone(), target.blocks, self.side)
            total += float(delta.pow(2).sum())
            inside += float((delta - kept).pow(2).sum())
        return inside / total if total > 0 else 0.0


_INSTANCE: SubspaceExclusion | None = None


def validate(args) -> None:
    if not getattr(args, "exclude_subspace_path", None):
        return
    problems = []
    if args.tensor_model_parallel_size != 1:
        problems.append("--tensor-model-parallel-size must be 1")
    if args.pipeline_model_parallel_size != 1:
        problems.append("--pipeline-model-parallel-size must be 1")
    if getattr(args, "overlap_grad_reduce", False):
        problems.append("--overlap-grad-reduce must be off (grads are projected before the DP reduction)")
    if getattr(args, "overlap_param_gather", False):
        problems.append("--overlap-param-gather must be off (params are rewritten right after the step)")
    if getattr(args, "peft_method", "none") not in (None, "none"):
        problems.append("--peft-method must be none (this constrains full fine-tuning)")
    if args.exclude_subspace_side not in SIDES:
        problems.append(f"--exclude-subspace-side must be one of {SIDES}")
    if args.exclude_subspace_k < 1:
        problems.append("--exclude-subspace-k must be >= 1")
    if problems:
        raise ValueError("subspace exclusion: " + "; ".join(problems))


def get(args, model_chunks) -> SubspaceExclusion | None:
    """The run's exclusion (built on first use, while the parameters are still the base), or None."""
    global _INSTANCE
    if not getattr(args, "exclude_subspace_path", None):
        return None
    if _INSTANCE is None:
        validate(args)
        _INSTANCE = SubspaceExclusion(args, model_chunks)
        _INSTANCE.constrain_params()
    return _INSTANCE
