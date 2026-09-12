"""Top-k singular-subspace overlap of the QK / OV attention *circuits*.

Companion to ``subspace_overlap_profile.py``, restricted to attention and
operating on the composed, basis-invariant circuits instead of the raw
Q/K/V/O projection tensors (Elhage et al., "A Mathematical Framework for
Transformer Circuits"):

    QK circuit (decides *where* a head attends; acts on the residual stream):
        C_QK^h = (W_Q^h)^T @ W_K^{kv(h)}        shape (hidden, hidden)
    OV circuit (decides *what* gets written back once attended):
        C_OV^h = W_O^h @ W_V^{kv(h)}             shape (hidden, hidden)

Why the composed circuit and not the raw per-head Q/K weights: W_Q^h and
W_K^h individually have a GL(head_dim) gauge freedom -- right-multiplying
W_Q^h by any invertible M and W_K^h by M^{-T} leaves C_QK^h unchanged. SVD of
the raw q_proj/k_proj tensors (as svd_rank_profile.py / subspace_overlap_
profile.py do) can therefore mix genuine structure with an arbitrary
per-head basis choice; C_QK / C_OV cannot, since they are exactly what the
attention computation actually depends on.

Delta convention: delta is taken on the *composed* circuit, not the
composition of deltas, i.e.

    Delta(C_QK^h) = C_QK^h(checkpoint) - C_QK^h(base)

not (Delta W_Q^h)^T (Delta W_K^{kv(h)}). This answers "how did fine-tuning
change this circuit", which is what a subspace-overlap comparison across
checkpoints should be measuring.

GQA is handled: kv(h) = h // (num_attention_heads // num_key_value_heads),
read from the base checkpoint's config.json.

This file does not modify, and only imports from, svd_rank_profile.py
(``load_tensor``, ``tensor_locations``) and subspace_overlap_profile.py
(``top_k_bases``, ``subspace_sim``, ``empirical_chance``). Neither of those
files is touched.

CPU-only. Each circuit is (hidden_size, hidden_size) -- the same size as the
tensors svd_rank_profile.py already handles, so cost per circuit is
comparable; total SVDs = layers x heads x circuits x checkpoints (cached and
reused across all pairwise comparisons).

Usage:
    python attention_circuit_overlap_profile.py \\
        --base /mnt/L202500431/models/qwen3-1.7b \\
        --checkpoint oracle=/mnt/L202500431/models/hyy_models/m6_coord_mask_1p6pct_oracle_step300 \\
        --checkpoint random=/mnt/L202500431/models/hyy_models/m6_coord_mask_1p6pct_random_step300 \\
        --checkpoint fullft=/mnt/L202500431/models/hyy_models/m6_trl_fullft_non_thinking_step300 \\
        --layers auto --heads auto --circuits qk,ov \\
        --k 1,2,4,8,16,32,64,128,256 --random-trials 20
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from subspace_overlap_profile import empirical_chance, subspace_sim, top_k_bases
from svd_rank_profile import load_tensor, tensor_locations


def load_attn_config(base_dir: Path) -> dict:
    config = json.loads((base_dir / "config.json").read_text())
    num_heads = config["num_attention_heads"]
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    hidden_size = config["hidden_size"]
    head_dim = config.get("head_dim", hidden_size // num_heads)
    num_layers = config["num_hidden_layers"]
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_attention_heads ({num_heads}) not divisible by num_key_value_heads ({num_kv_heads})")
    return dict(num_heads=num_heads, num_kv_heads=num_kv_heads, hidden_size=hidden_size,
                head_dim=head_dim, num_layers=num_layers, n_rep=num_heads // num_kv_heads)


def pick_auto(count: int) -> list[int]:
    return sorted({0, count // 2, count - 1})


def parse_indices(spec: str, count: int) -> list[int]:
    if spec == "auto":
        return pick_auto(count)
    if spec == "all":
        return list(range(count))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def load_qkvo(checkpoint_dir: Path, locations: dict, layer: int) -> tuple[torch.Tensor, ...]:
    names = [f"model.layers.{layer}.self_attn.{proj}.weight" for proj in ("q_proj", "k_proj", "v_proj", "o_proj")]
    return tuple(load_tensor(checkpoint_dir, locations, name).float() for name in names)


def circuits_for_head(qkvo: tuple[torch.Tensor, ...], head: int, cfg: dict) -> dict[str, torch.Tensor]:
    q, k, v, o = qkvo
    d = cfg["head_dim"]
    kv_head = head // cfg["n_rep"]
    w_q = q[head * d : (head + 1) * d, :]          # (head_dim, hidden)
    w_k = k[kv_head * d : (kv_head + 1) * d, :]     # (head_dim, hidden)
    w_v = v[kv_head * d : (kv_head + 1) * d, :]     # (head_dim, hidden)
    w_o = o[:, head * d : (head + 1) * d]           # (hidden, head_dim)
    return {
        "qk": w_q.transpose(0, 1) @ w_k,   # (hidden, hidden)
        "ov": w_o @ w_v,                    # (hidden, hidden)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH",
                         help="repeat for each checkpoint; all pairs are compared")
    parser.add_argument("--layers", default="auto", help="comma-separated layer indices, 'auto' (first/mid/last), or 'all'")
    parser.add_argument("--heads", default="auto", help="comma-separated query-head indices, 'auto' (first/mid/last), or 'all'")
    parser.add_argument("--circuits", default="qk,ov", help="comma-separated subset of {qk, ov}")
    parser.add_argument("--k", default="1,2,4,8,16,32,64,128,256", help="comma-separated top-k values to test")
    parser.add_argument("--random-trials", type=int, default=20,
                         help="independent Haar-random subspace pairs sampled per (n, k) for the empirical chance baseline")
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    checkpoints = {}
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        checkpoints[name] = Path(path)
    if len(checkpoints) < 2:
        raise ValueError("Need >=2 --checkpoint entries to compare pairwise")

    circuits_wanted = [c.strip() for c in args.circuits.split(",") if c.strip()]
    assert all(c in ("qk", "ov") for c in circuits_wanted), circuits_wanted
    k_values = [int(k.strip()) for k in args.k.split(",") if k.strip()]

    cfg = load_attn_config(args.base)
    layers = parse_indices(args.layers, cfg["num_layers"])
    heads = parse_indices(args.heads, cfg["num_heads"])
    print(f"config: {cfg}", flush=True)
    print(f"layers={layers} heads={heads} circuits={circuits_wanted} k={k_values} "
          f"{args.random_trials} random-baseline trials each\n", flush=True)

    base_locations = tensor_locations(args.base)
    ckpt_locations = {name: tensor_locations(path) for name, path in checkpoints.items()}
    generator = torch.Generator().manual_seed(args.seed)
    chance_cache: dict[tuple[int, int], tuple[float, float]] = {}

    for layer in layers:
        base_qkvo = load_qkvo(args.base, base_locations, layer)
        ckpt_qkvo = {name: load_qkvo(path, ckpt_locations[name], layer) for name, path in checkpoints.items()}
        for head in heads:
            base_circuits = circuits_for_head(base_qkvo, head, cfg)
            print(f"=== layer {layer}, head {head} (kv_head {head // cfg['n_rep']}) ===")
            for circuit_name in circuits_wanted:
                base_c = base_circuits[circuit_name]
                deltas = {}
                for name, qkvo in ckpt_qkvo.items():
                    c = circuits_for_head(qkvo, head, cfg)[circuit_name]
                    deltas[name] = c - base_c
                bases = {name: top_k_bases(delta, k_values) for name, delta in deltas.items()}
                n_dim = base_c.shape[0]  # square (hidden, hidden)
                for left_name, right_name in itertools.combinations(checkpoints, 2):
                    print(f"  -- {circuit_name} {left_name} vs {right_name} --")
                    for side in ("left", "right"):
                        for k in k_values:
                            if k > n_dim:
                                continue
                            sim = subspace_sim(bases[left_name][side][k], bases[right_name][side][k])
                            cache_key = (n_dim, k)
                            if cache_key not in chance_cache:
                                chance_cache[cache_key] = empirical_chance(n_dim, k, args.random_trials, generator)
                            chance_mean, chance_std = chance_cache[cache_key]
                            asymptotic = (k / n_dim) ** 0.5
                            multiple = sim / chance_mean if chance_mean > 1e-12 else float("nan")
                            print(f"    {side:<5} k={k:<4} sim_k={sim:.4f}  "
                                  f"chance_empirical={chance_mean:.4f}+/-{chance_std:.4f}  "
                                  f"chance_asymptotic_sqrt(k/n)={asymptotic:.4f}  "
                                  f"observed/chance={multiple:.2f}x")
            print()


if __name__ == "__main__":
    main()
