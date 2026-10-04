"""The per-layer implicit bias a fine-tune writes along the activation mean direction.

Qwen3-style linear layers have no bias, but their input has a large, nearly constant component along
the mean direction mu (massive activations): activation_covariance_overlap.py's mean-direction
report finds e_1 of A = E[x x^T] ~ mu/||mu|| with e_1^T x ~ constant (CV ~0.15 on attn_in/mlp_in).
So the part of a weight delta D along e_1 acts on every token as one constant output vector

    b = (e_1^T mu) * D e_1                         (n_out,)

-- exactly what remove_delta_subspace.py --k 1 deletes. Shared by bias_vector_compare.py (compare
those vectors across fine-tunes) and reflection_token_logprob.py --add-bias (add them back as
steering vectors at inference).
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from subspace_overlap_profile import GROUP_OF_KIND, tensor_kind
from svd_rank_profile import CANONICAL_RE, load_tensor, tensor_locations


class ImplicitBias:
    """b = (e_1^T mu) * (W_ft - W_base) e_1 per canonical tensor, from a --save-eig file with means."""

    def __init__(self, base: Path, eig: Path, device: str | torch.device = "cpu"):
        self.base, self.device = Path(base), torch.device(device)
        self.base_locations = tensor_locations(self.base)
        self.eig = safe_open(str(eig), framework="pt")
        keys = set(self.eig.keys())
        if not any(k.endswith(".mean") for k in keys):
            raise ValueError(f"{eig} has no '.mean' entries: rerun activation_covariance_overlap.py (it now saves them)")
        self._cache: dict[str, tuple[torch.Tensor, float]] = {}

    def direction(self, name: str) -> tuple[torch.Tensor, float]:
        """(e_1, m = e_1^T mu) for this tensor's layer and input group; e_1 signed so that m >= 0."""
        if name not in self._cache:
            layer, group = int(CANONICAL_RE.match(name).group(1)), GROUP_OF_KIND[tensor_kind(name)]
            e1 = self.eig.get_tensor(f"layers.{layer}.{group}.eigvecs")[:, 0].double()
            mu = self.eig.get_tensor(f"layers.{layer}.{group}.mean").double()
            m = float(e1 @ mu)
            if m < 0:
                e1, m = -e1, -m
            self._cache[name] = (e1.to(self.device), m)
        return self._cache[name]

    def base_weight(self, name: str) -> torch.Tensor:
        return load_tensor(self.base, self.base_locations, name).to(self.device).double()

    def bias(self, name: str, finetuned_weight: torch.Tensor, base_weight: torch.Tensor | None = None) -> torch.Tensor:
        e1, m = self.direction(name)
        w0 = self.base_weight(name) if base_weight is None else base_weight
        return m * ((finetuned_weight.to(self.device).double() - w0) @ e1)

    def biases_for(self, finetuned: Path, names: list[str]) -> dict[str, torch.Tensor]:
        locations = tensor_locations(Path(finetuned))
        return {n: self.bias(n, load_tensor(Path(finetuned), locations, n)) for n in names}
