"""Direct logit attribution of each fine-tune's implicit bias onto reflection tokens.

o_proj and down_proj write into the residual stream, so their implicit biases b = (e1^T mu) * dW e1
(implicit_bias.py) are added to the residual of every token. Along the direct path (ignoring every
later layer), adding a residual vector r changes the next-token logits by

    d_logit = W_U (gamma * r) / rms(h_final)

with W_U the unembedding and gamma the final RMSNorm weight of the model. The unknown 1/rms is one
positive number per position and softmax ignores constant shifts, so we report, for the vocabulary
vector d = W_U (gamma * r):

    z(token)   (d_token - mean_vocab d) / std_vocab d      how strongly r pushes that token up,
                                                         relative to the whole vocabulary
    pct        share of the vocabulary that r pushes up less than that token

for r = the bias summed over all layers' o_proj and down_proj (and per layer), for reflection
tokens (Wait / Alternatively / Hmm / ...) and control tokens (Therefore / So / Thus / Final / boxed).
It also compares fine-tunes on the reflection+control readout only: their bias vectors can be nearly
orthogonal in the 2048-d residual (bias_vector_compare.py) and still push the same tokens.

    python bias_logit_attribution.py --base /mnt/L202500431/models/qwen3-1.7b \\
        --eig /mnt/.../actcov_eig_math_t07_merged.safetensors \\
        --model opd300=/mnt/... --model rl300=/mnt/... --model rlp300=/mnt/... --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import statistics
from pathlib import Path

import torch

from implicit_bias import ImplicitBias
from svd_rank_profile import CANONICAL_RE, load_tensor, tensor_locations

REFLECTION = ("Wait", "Alternatively", "Hmm", "But", "However", "Double", "Actually", "Hold")
CONTROL = ("Therefore", "So", "Thus", "Final", "Hence", "The", "We", "\\boxed")
WRITERS = ("self_attn.o_proj", "mlp.down_proj")


def first_token_ids(tokenizer, words: tuple[str, ...]) -> dict[str, int]:
    """Each word with and without a leading space -> its first token id (deduplicated by id)."""
    ids = {}
    for word in words:
        for text in (word, " " + word):
            tid = tokenizer.encode(text, add_special_tokens=False)[0]
            ids.setdefault(tokenizer.decode([tid]), tid)
    return ids


def unembedding(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    locations = tensor_locations(path)
    name = "lm_head.weight" if "lm_head.weight" in locations else "model.embed_tokens.weight"  # tied embeddings
    return load_tensor(path, locations, name), load_tensor(path, locations, "model.norm.weight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--eig", required=True, type=Path, help="--save-eig file with '.mean' entries")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--top-layers", type=int, default=5, help="layers listed by their reflection push")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    refl, ctrl = first_token_ids(tokenizer, REFLECTION), first_token_ids(tokenizer, CONTROL)
    print(f"reflection tokens: {list(refl)}\ncontrol tokens:    {list(ctrl)}\n", flush=True)
    ib = ImplicitBias(args.base, args.eig, device)
    writers = sorted((n for n in ib.base_locations if CANONICAL_RE.match(n) and any(w in n for w in WRITERS)),
                     key=lambda n: int(CANONICAL_RE.match(n).group(1)))
    models = dict(item.split("=", 1) for item in args.model)

    readouts = {}  # model -> vocab vector of the summed bias
    for model, path in models.items():
        w_u, gamma = unembedding(Path(path))
        w_u, gamma = w_u.to(device).float(), gamma.to(device).float()
        biases = ib.biases_for(Path(path), writers)
        per_layer = {}
        for name, b in biases.items():
            layer = int(CANONICAL_RE.match(name).group(1))
            per_layer[layer] = per_layer.get(layer, 0) + b.float()
        total = sum(per_layer.values())
        d = w_u @ (gamma * total)
        readouts[model] = d
        z = (d - d.mean()) / d.std()
        rank = lambda tid: float((d < d[tid]).float().mean())  # noqa: E731
        print(f"##### {model}: summed o_proj+down_proj implicit bias, |r| = {float(total.norm()):.3f}")
        for label, group in (("reflection", refl), ("control", ctrl)):
            cells = "  ".join(f"{tok!r}:{float(z[tid]):+.2f}({rank(tid):.3f})" for tok, tid in group.items())
            print(f"  {label:<10} mean z {statistics.mean(float(z[t]) for t in group.values()):+.3f} | {cells}")
        layer_push = []
        for layer, r in per_layer.items():
            dl = w_u @ (gamma * r)
            zl = (dl - dl.mean()) / dl.std()
            layer_push.append((statistics.mean(float(zl[t]) for t in refl.values()),
                               statistics.mean(float(zl[t]) for t in ctrl.values()), layer, float(r.norm())))
        layer_push.sort(reverse=True)
        print(f"  layers pushing reflection most (mean z refl / ctrl, |r|): " + ", ".join(
            f"L{l} {a:+.2f}/{c:+.2f} ({n:.2f})" for a, c, l, n in layer_push[:args.top_layers]))
        print(f"  layers pushing reflection least: " + ", ".join(
            f"L{l} {a:+.2f}/{c:+.2f} ({n:.2f})" for a, c, l, n in layer_push[-args.top_layers:]))
        print(flush=True)
        del w_u

    print("##### comparing fine-tunes' readouts (centered vocab vectors)")
    ids = list(refl.values()) + list(ctrl.values())
    for a, c in itertools.combinations(models, 2):
        da, dc = readouts[a] - readouts[a].mean(), readouts[c] - readouts[c].mean()
        full = float(da @ dc / (da.norm() * dc.norm()))
        sa, sc = da[ids] - da[ids].mean(), dc[ids] - dc[ids].mean()
        sub = float(sa @ sc / (sa.norm() * sc.norm()))
        print(f"  {a} vs {c}: cos over whole vocab {full:+.3f}   cos over reflection+control tokens {sub:+.3f}")


if __name__ == "__main__":
    main()
