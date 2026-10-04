"""Does an edited model specifically lower reflection, at fixed text? A teacher-forced token test.

Takes responses one reference model wrote (an eval run's completions.jsonl, e.g. rl300_full on
math500), rebuilds the exact eval input (run_evalchemy_math_eval.MATH_PROMPT in the chat template,
enable_thinking=False) and scores the SAME text under every --model. Because the text is fixed,
"the edit makes answers shorter" cannot explain any difference. Per model it reports

  refl   mean log p of the first token of each reflection marker (Wait / Alternatively / Hmm /
         let me check / double-check / verify ...) where it occurs in the text
  other  mean log p of every other completion token
  d_refl, d_other   the same, minus the reference model's value at the same positions (paired)
  d_refl - d_other  how much MORE the model lowers reflection tokens than ordinary ones
                    (95% bootstrap CI over responses)
  prop   reflection propensity: at every line start of the text (the token right after "\\n"),
         the probability mass on tokens that begin a reflection word ("Wait", " Wait",
         "Alternatively", "Hmm", ...), averaged over line starts; ratio = prop / reference prop

A reflection-specific edit lowers refl and prop much more than other; a generic edit (e.g. uniform
delta scaling at a similar overall change) moves both alike.

--add-bias NAME=SRC@SCALE adds, through forward hooks on model NAME's q/k/v/o/gate/up/down layers,
SCALE times the implicit bias b = (e1^T mu) * (W_SRC - W_base) e1 of fine-tune SRC (implicit_bias.py;
needs --bias-base and --bias-eig, a --save-eig file with means). E.g. topE_k1 + rl300's bias (@1)
should recover rl300; base + rl300's bias tests whether the bias alone induces reflection; rl300 @-1
removes it at inference, @+1 doubles it.

    python reflection_token_logprob.py \\
        --completions /mnt/.../eval_results/task_split_ablation/rl300_full/math/math500/completions.jsonl \\
        --model rl300_full=/mnt/.../iter_0000299_hf --model rl300_topE_k1=/mnt/.../rl300_topE_rmright_k1_bf16 \\
        --model rl300_alpha0.75=/mnt/.../rl300_alpha0.75_bf16 --model base=/mnt/L202500431/models/qwen3-1.7b \\
        --num-samples 200 --output-dir /mnt/.../reflection_logprob/rl300
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

import torch

ORBIT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ORBIT_ROOT / "examples" / "on_policy_distillation" / "eval"))
import run_evalchemy_math_eval as EVAL  # noqa: E402

MARKER_RE = re.compile(r"\b(?:wait|alternatively|hmm|let me (?:check|verify|double[- ]check|re-?examine)|"
                       r"double[- ]check|verify|on second thought)\b", re.I)
PROPENSITY_WORDS = ("Wait", "Alternatively", "Hmm", "But wait", "Let me check", "Let me verify", "Double-check")


def parse_named(item: str) -> tuple[str, str]:
    name, _, path = item.partition("=")
    if not path:
        raise ValueError(f"expected NAME=PATH, got {item!r}")
    return name, path


def load_samples(args) -> list[dict]:
    problems = {eid: problem for eid, problem, _ in EVAL.load_examples(args.task, Path(args.evalchemy_root))}
    rows = []
    for line in open(args.completions, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out = r.get("output", "")
        if r.get("status") != "ok" or not out:
            continue
        eid = r["key"].split("/", 1)[1].rsplit("/", 1)[0]
        if args.require_marker and not MARKER_RE.search(out):
            continue
        rows.append(dict(key=r["key"], eid=eid, output=out, problem=problems[eid]))
    random.Random(args.seed).shuffle(rows)
    return rows[:args.num_samples]


def encode(sample: dict, tokenizer, max_completion_tokens: int):
    """Token ids of prompt+completion, completion token range, marker-start and line-start positions."""
    messages = [{"role": "user", "content": EVAL.MATH_PROMPT.format(problem=sample["problem"])}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text = prompt + sample["output"]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    start = next(i for i, (s, _) in enumerate(offsets) if s >= len(prompt))
    end = min(len(ids), start + max_completion_tokens)
    marker_chars = [len(prompt) + m.start() for m in MARKER_RE.finditer(sample["output"])]
    markers, j = set(), start
    for c in marker_chars:
        while j < end and offsets[j][1] <= c:
            j += 1
        if j < end:
            markers.add(j)
    line_starts = {i for i in range(start + 1, end) if text[offsets[i][0] - 1:offsets[i][0]] == "\n"
                   or text[offsets[i - 1][0]:offsets[i - 1][1]].endswith("\n")}
    return ids[:end], start, markers, line_starts


@torch.no_grad()
def score(model, ids: list[int], start: int, propensity_ids: torch.Tensor, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """log p(token_i | <i) for i in [start, len) and the propensity mass predicted for each i."""
    device = next(model.parameters()).device
    x = torch.tensor([ids], device=device)
    hidden = model.model(input_ids=x).last_hidden_state[0]  # (T, d)
    targets = x[0]
    logps, props = [], []
    for a in range(start - 1, len(ids) - 1, chunk):  # position p predicts token p+1
        b = min(a + chunk, len(ids) - 1)
        logits = model.lm_head(hidden[a:b]).float()
        lp = torch.log_softmax(logits, dim=-1)
        logps.append(lp.gather(1, targets[a + 1:b + 1, None])[:, 0].cpu())
        props.append(lp[:, propensity_ids].exp().sum(-1).cpu())
    return torch.cat(logps), torch.cat(props)


def add_bias_hooks(model, implicit, src: Path, scale: float) -> None:
    """Forward hooks adding scale * b(src) to every q/k/v/o/gate/up/down output of `model`."""
    from svd_rank_profile import CANONICAL_RE
    modules = {f"{n}.weight": m for n, m in model.named_modules() if CANONICAL_RE.match(f"{n}.weight")}
    biases = implicit.biases_for(src, sorted(modules))
    for weight_name, module in modules.items():
        b = (scale * biases[weight_name]).to(module.weight.device, module.weight.dtype)
        module.register_forward_hook(lambda _m, _inp, out, b=b: out + b)


def boot_ci(values: list[float], n: int = 10000) -> tuple[float, float, float]:
    rng = random.Random(0)
    m = statistics.mean(values)
    bs = sorted(statistics.mean(values[rng.randrange(len(values))] for _ in values) for _ in range(n))
    return m, bs[int(.025 * n)], bs[int(.975 * n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", required=True, type=Path, help="reference model's completions.jsonl")
    parser.add_argument("--task", default="math500")
    parser.add_argument("--evalchemy-root", default="/mnt/L202500431/third_party/evalchemy")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH",
                        help="models to score the text with; the first is the reference unless --reference")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--require-marker", action=argparse.BooleanOptionalAction, default=True,
                        help="only use responses containing at least one reflection marker (default on)")
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=1024, help="positions per lm_head chunk (memory)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--add-bias", action="append", default=[], metavar="NAME=SRC@SCALE",
                        help="add SCALE x fine-tune SRC's implicit bias to model NAME (repeatable)")
    parser.add_argument("--bias-base", type=Path, default=None, help="base model the implicit biases are relative to")
    parser.add_argument("--bias-eig", type=Path, default=None, help="--save-eig file with '.mean' entries")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    models = [parse_named(m) for m in args.model]
    reference = args.reference or models[0][0]
    tokenizer = AutoTokenizer.from_pretrained(models[0][1])
    propensity_ids = sorted({tokenizer.encode(w, add_special_tokens=False)[0]
                             for word in PROPENSITY_WORDS for w in (word, " " + word)})
    print(f"propensity tokens: {[tokenizer.decode([i]) for i in propensity_ids]}", flush=True)
    samples = load_samples(args)
    encoded = [encode(s, tokenizer, args.max_completion_tokens) for s in samples]
    n_markers = sum(len(e[2]) for e in encoded)
    print(f"{len(samples)} responses from {args.completions}, {n_markers} reflection-marker tokens, "
          f"{sum(len(e[3]) for e in encoded)} line starts", flush=True)

    # per model, per response: (mean logp at markers, mean logp elsewhere, mean propensity at line starts)
    bias_specs: dict[str, list[tuple[str, float]]] = {}
    for item in args.add_bias:
        target, _, rest = item.partition("=")
        src, _, scale = rest.rpartition("@")
        if not src or target not in dict(models):
            raise ValueError(f"--add-bias {item!r}: expected NAME=SRC@SCALE with NAME one of the --model names")
        bias_specs.setdefault(target, []).append((src, float(scale)))
    implicit = None
    if bias_specs:
        if args.bias_base is None or args.bias_eig is None:
            raise ValueError("--add-bias needs --bias-base and --bias-eig")
        from implicit_bias import ImplicitBias
        implicit = ImplicitBias(args.bias_base, args.bias_eig, args.device)

    per_model: dict[str, list[tuple[float, float, float]]] = {}
    started = time.perf_counter()
    for name, path in models:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(args.device).eval()
        for src, scale in bias_specs.get(name, []):
            add_bias_hooks(model, implicit, Path(src), scale)
            print(f"  {name}: + {scale:g} x implicit bias of {src}", flush=True)
        pids = torch.tensor(propensity_ids, device=args.device)
        rows = []
        for ids, start, markers, line_starts in encoded:
            logp, prop = score(model, ids, start, pids, args.chunk)
            pos = torch.arange(start, len(ids))
            is_marker = torch.tensor([p in markers for p in pos.tolist()])
            is_line = torch.tensor([p in line_starts for p in pos.tolist()])
            refl = float(logp[is_marker].mean()) if is_marker.any() else float("nan")
            other = float(logp[~is_marker].mean())
            prop_line = float(prop[is_line].mean()) if is_line.any() else float("nan")
            rows.append((refl, other, prop_line))
        per_model[name] = rows
        del model
        torch.cuda.empty_cache()
        print(f"  scored {name} ({time.perf_counter() - started:.0f}s)", flush=True)

    ref = per_model[reference]
    print(f"\n##### teacher-forced on {args.completions.parent.parent.parent.name} text, reference = {reference}")
    print(f"  {'model':<26} {'refl':>7} {'other':>7} {'d_refl':>7} {'d_other':>8} {'d_refl - d_other [95% CI]':>28} "
          f"{'prop':>8} {'ratio':>6}")
    summary = {}
    for name, rows in per_model.items():
        ok = [i for i, r in enumerate(rows) if r[0] == r[0] and ref[i][0] == ref[i][0]]
        refl = statistics.mean(rows[i][0] for i in ok)
        other = statistics.mean(r[1] for r in rows)
        props = [r[2] for r in rows if r[2] == r[2]]
        prop = statistics.mean(props)
        ref_prop = statistics.mean(r[2] for r in ref if r[2] == r[2])
        d_refl = [rows[i][0] - ref[i][0] for i in ok]
        d_other = [rows[i][1] - ref[i][1] for i in ok]
        diff = boot_ci([a - b for a, b in zip(d_refl, d_other)])
        summary[name] = dict(refl=refl, other=other, d_refl=statistics.mean(d_refl), d_other=statistics.mean(d_other),
                             specificity=diff, propensity=prop, propensity_ratio=prop / ref_prop)
        print(f"  {name:<26} {refl:7.3f} {other:7.3f} {statistics.mean(d_refl):+7.3f} {statistics.mean(d_other):+8.3f} "
              f"{diff[0]:+9.3f} [{diff[1]:+.3f},{diff[2]:+.3f}] {prop:8.4f} {prop / ref_prop:6.2f}")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(dict(
            completions=str(args.completions), reference=reference, n_responses=len(samples), n_marker_tokens=n_markers,
            propensity_tokens=[tokenizer.decode([i]) for i in propensity_ids], models=dict(models), results=summary),
            indent=2))
        print(f"\nwrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
