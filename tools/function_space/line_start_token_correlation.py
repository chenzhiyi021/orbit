"""Do the tokens a fine-tune's implicit bias pushes up match the tokens it actually starts lines with more?

Inputs: the whole-vocab z-scored readouts from bias_logit_attribution.py --save-readouts, and eval
completions (completions.jsonl) of the base model and of each fine-tune. For every response it takes
the first token of every line (the response start and each token right after "\\n"), counts them per
model, and over the --top-n most common line-start tokens (pooled over all models) computes

    log ratio(token) = log p_model(token at line start) - log p_base(token at line start)   (add-1 smoothed)

Then, per (readout of model A, frequency change of model B), the Spearman correlation between A's
bias readout z and B's log ratio across those tokens, with a bootstrap 95% CI over tokens. The
diagonal asks "does a model's bias push up the words it says more at line starts?"; off-diagonal
cells (A's bias vs B's words) are the specificity control.

    python line_start_token_correlation.py --tokenizer /mnt/L202500431/models/qwen3-1.7b \\
        --readouts /mnt/.../bias_readouts.safetensors \\
        --completions base=/mnt/.../base/math/math500/completions.jsonl \\
        --completions opd300=/mnt/.../opd300_full/math/math500/completions.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


def line_start_counts(path: Path, tokenizer) -> tuple[Counter, int]:
    counts, total = Counter(), 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = row.get("output", "")
        if row.get("status") != "ok" or not text:
            continue
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        for i, (tid, (start, _end)) in enumerate(zip(enc["input_ids"], enc["offset_mapping"])):
            if i == 0 or (start > 0 and text[start - 1] == "\n"):
                counts[tid] += 1
                total += 1
    return counts, total


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return cov / var if var > 0 else float("nan")


def boot(x: list[float], y: list[float], n: int = 2000) -> tuple[float, float, float]:
    rng = random.Random(0)
    idx = range(len(x))
    stats = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        stats.append(spearman([x[i] for i in s], [y[i] for i in s]))
    stats = sorted(v for v in stats if v == v)
    return spearman(x, y), stats[int(.025 * len(stats))], stats[int(.975 * len(stats))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--readouts", required=True, type=Path)
    parser.add_argument("--completions", action="append", required=True, metavar="NAME=PATH",
                        help="one must be named 'base'; the others must match readout names")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    paths = dict(item.split("=", 1) for item in args.completions)
    if "base" not in paths:
        raise ValueError("--completions needs one entry named 'base'")
    handle = safe_open(str(args.readouts), framework="pt")
    readouts = {name: handle.get_tensor(name) for name in handle.keys()}
    models = [m for m in paths if m != "base"]
    missing = [m for m in models if m not in readouts]
    if missing:
        raise ValueError(f"no readout for {missing}; readouts file has {sorted(readouts)}")

    counts = {}
    for name, path in paths.items():
        counts[name] = line_start_counts(Path(path), tokenizer)
        print(f"  {name:<10} {counts[name][1]:7d} line starts, {len(counts[name][0]):5d} distinct tokens", flush=True)
    pooled = Counter()
    for c, _ in counts.values():
        pooled.update(c)
    tokens = [t for t, _ in pooled.most_common(args.top_n)]
    vocab = len(tokens)

    def log_ratio(name: str) -> list[float]:
        (c, n), (cb, nb) = counts[name], counts["base"]
        return [math.log((c[t] + 1) / (n + vocab)) - math.log((cb[t] + 1) / (nb + vocab)) for t in tokens]

    ratios = {m: log_ratio(m) for m in models}
    zs = {m: [float(readouts[m][t]) for t in tokens] for m in models}

    print(f"\n##### Spearman(bias readout z of ROW model, line-start log freq ratio vs base of COLUMN model) "
          f"over the {vocab} most common line-start tokens [95% bootstrap CI]")
    print("  " + " " * 18 + "".join(f"{'words: ' + m:>32}" for m in models))
    for a in models:
        cells = []
        for b in models:
            r, lo, hi = boot(zs[a], ratios[b])
            cells.append(f"{r:+.3f} [{lo:+.3f},{hi:+.3f}]")
        print(f"  {'bias: ' + a:<18}" + "".join(f"{c:>32}" for c in cells))

    for m in models:
        order = sorted(range(vocab), key=lambda i: ratios[m][i], reverse=True)
        show = lambda i: f"{tokenizer.decode([tokens[i]])!r}(ratio {ratios[m][i]:+.2f}, z {zs[m][i]:+.2f})"  # noqa: E731
        print(f"\n  {m}: line-start tokens it uses MORE than base -> " + ", ".join(show(i) for i in order[:args.show]))
        print(f"  {m}: LESS than base -> " + ", ".join(show(i) for i in order[-args.show:]))
        top_z = sorted(range(vocab), key=lambda i: zs[m][i], reverse=True)
        print(f"  {m}: tokens its bias pushes MOST -> " + ", ".join(show(i) for i in top_z[:args.show]))


if __name__ == "__main__":
    main()
