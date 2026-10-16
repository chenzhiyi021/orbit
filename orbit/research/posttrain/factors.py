"""Native fused shared-A layout and exact factor constraints."""

from __future__ import annotations
import hashlib
import torch

FUSED = {"linear_qkv": (4096, 2048), "linear_proj": (2048, 2048), "linear_fc1": (12288, 2048), "linear_fc2": (2048, 6144)}


def name_seed(name, initialization_seed=22):
    return int(hashlib.sha256((f"seed{initialization_seed}:" + name).encode()).hexdigest()[:8], 16)


def initial_factors(shape, rank, name, device="cpu", initialization_seed=22):
    m, n = shape
    generator = torch.Generator(device=device).manual_seed(name_seed("native:" + name, initialization_seed))
    a = torch.empty(rank, n, dtype=torch.float32, device=device).uniform_(-(n**-0.5), n**-0.5, generator=generator)
    return a, torch.zeros(m, rank, dtype=torch.float32, device=device)


def independent_parameters(rank, layers=28):
    return layers * rank * sum(m + n for m, n in FUSED.values())


def fused_modules(model):
    found = {}
    for chunk in model:
        for name, module in chunk.named_modules():
            if module.__class__.__name__ == "LoRALinear":
                key = name[name.index("decoder.layers.") :] + ".weight"
                if key in found:
                    raise ValueError("Duplicate logical fused module")
                found[key] = module
    return found


def registry(model):
    rows = []
    for chunk_id, chunk in enumerate(model):
        for name, parameter in chunk.named_parameters():
            rows.append({"name": f"chunk{chunk_id}.{name}", "shape": list(parameter.shape), "dtype": str(parameter.dtype), "requires_grad": parameter.requires_grad, "numel": parameter.numel()})
    return rows


def endpoint_reconstruction(a, b, delta, scale=1.0):
    """Evaluate the actual dense D, including leakage outside a fixed A row space."""
    a, b, delta = a.detach().double(), b.detach().double(), delta.detach().double()
    denom = torch.linalg.vector_norm(delta).clamp_min(1e-30)
    reconstructed = scale * b @ a
    row_projector = torch.linalg.pinv(a) @ a
    leak = delta - delta @ row_projector
    return {"reconstruction_relative": float(torch.linalg.vector_norm(delta - reconstructed) / denom), "fixed_a_leakage_relative": float(torch.linalg.vector_norm(leak) / denom), "delta_norm": float(torch.linalg.vector_norm(delta)), "direction_defined": bool(torch.linalg.vector_norm(delta) > 0)}


def validate_shared_export(state):
    """Count HF aliases once, and require exact shared-A export for Q/K/V and gate/up."""
    import re

    a = {}
    for key, value in state.items():
        match = re.search(r"model\.layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)\.lora_A(?:\.default)?\.weight$", key)
        if match:
            a[(int(match[1]), match[3])] = value
    if len(a) != 196:
        raise ValueError(f"Expected 196 logical A export keys, found {len(a)}")
    for layer in range(28):
        for names in [("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")]:
            if any(not torch.equal(a[layer, names[0]], a[layer, name]) for name in names[1:]):
                raise ValueError(f"Shared A was not preserved in layer {layer}: {names}")
    return {"logical_projection_count": 196, "independent_fused_a_count": 112, "shared_a_equal": True}


def endpoint_basis(delta, rank):
    delta = delta.detach().double()
    _, singular, vh = torch.linalg.svd(delta, full_matrices=False)
    if rank < 1:
        raise ValueError("Basis rank must be positive")
    tolerance = max(delta.shape) * torch.finfo(delta.dtype).eps * singular.max()
    available = int((singular > tolerance).sum())
    if available < rank:
        raise ValueError(f"Endpoint has only {available} nonzero directions; cannot invent {rank}-dimensional basis")
    return vh[:rank]


def effective_motion_squared(a0, b0, a1, b1):
    """Stable ||B1 A1 - B0 A0||_F^2 through small Gram matrices, no dense D allocation."""
    a0, b0, a1, b1 = [v.detach().double() for v in (a0, b0, a1, b1)]
    da, db = a1 - a0, b1 - b0
    first = ((db.T @ db) * (a1 @ a1.T)).sum()
    second = ((b0.T @ b0) * (da @ da.T)).sum()
    cross = 2 * ((db.T @ b0) * (a1 @ da.T)).sum()
    return (first + second + cross).clamp_min(0)
