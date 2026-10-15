"""Output-side second moments G = E[delta delta^T] of every linear layer, delta = dL/dy at its output.

K-FAC writes a linear layer's Fisher as A (x) G: A = E[x x^T] (activation_covariance_overlap.py) is the
input side, G the output side. Here G is collected per tensor (q/k/v/o/gate/up/down each have their own
output space) from the NLL of response tokens, in one of two modes:

  rollout (default)  the model samples its own responses to --data prompts (temperature --temperature);
                     NLL of those samples gives the true-Fisher output factor (on-policy methods: OPD, RL).
  --teacher-forced   rows carry an assistant turn (SFT data); prompt + that response is fed as is and
                     the loss is the SFT loss on it (off-policy methods).

--loss picks the per-token loss on response tokens (rollout mode for the OPD ones, which need --teacher):
  nll           -log p(y_t)                                  (default; SFT loss / true-Fisher probe)
  opd_sampled   -sg(A_t) log p(y_t), A_t = log p_T(y_t) - log p(y_t)
                orbit's OPD (--advantage-estimator on_policy_distillation --teacher-score-mode sampled_token)
  opd_fullkl    KL(p || p_T) over the full vocabulary at every response position (exact reverse KL)
The teacher scores the same token ids (Qwen3 student and teacher share the tokenizer).

Prompt positions get gradient through attention and are included in G (like activation_covariance_
overlap.py's token_scope=all); the loss is on response tokens only. Saves, per tensor,
'layers.L.<self_attn.q_proj|...|mlp.down_proj>.{eigvals (all), eigvecs (top --k-max), mean}' for
export_activation_random_subspace.py --geig, and prints the spectrum summary.

    python output_gradient_covariance.py --model /mnt/.../qwen3-1.7b --num-seqs 256 \\
        --save /mnt/.../grad_cov/gcov_eig_opd_rollout.safetensors
    python output_gradient_covariance.py --model /mnt/.../qwen3-1.7b --teacher-forced \\
        --data /mnt/.../openmathreasoning_nothink_25k6/train.jsonl --max-seq-tokens 4096 \\
        --save /mnt/.../grad_cov/gcov_eig_omr_tf.safetensors
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from activation_covariance_overlap import DEFAULT_DATA, eos_ids, load_prompts, read_rows, to_messages

KINDS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
REPORT_K = (1, 4, 16, 64, 256)


def rollout_sequences(args, tok, model) -> list[tuple[list[int], int]]:
    ns = SimpleNamespace(data=args.data, input_key=args.input_key, seed=args.seed, raw_prompt_template=None,
                         chat_template_kwargs='{"enable_thinking": false}', max_prompt_tokens=args.max_prompt_tokens,
                         num_prompts=args.num_seqs)
    prompts = load_prompts(ns, tok)
    stop = torch.tensor(eos_ids(tok, model), device="cuda")
    seqs = []
    for s in range(0, len(prompts), args.gen_batch):
        b = tok(prompts[s:s + args.gen_batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**b, max_new_tokens=args.gen_tokens, do_sample=True, temperature=args.temperature,
                                 top_p=1.0, pad_token_id=tok.pad_token_id)
        new = out[:, b["input_ids"].shape[1]:]
        is_stop = torch.isin(new, stop)
        keep = (is_stop.cumsum(1) - is_stop.long()) == 0
        for i in range(out.shape[0]):
            p_ids = b["input_ids"][i][b["attention_mask"][i].bool()]
            seqs.append((torch.cat([p_ids, new[i][keep[i]]]).tolist(), p_ids.numel()))
        print(f"  sampled {len(seqs)}/{len(prompts)}", flush=True)
    return seqs


def teacher_forced_sequences(args, tok) -> list[tuple[list[int], int]]:
    rows = [r for r in read_rows(args.data) if r.get(args.input_key) is not None]
    random.Random(args.seed).shuffle(rows)
    seqs = []
    for r in rows:
        msgs = to_messages(r[args.input_key])
        if not msgs or msgs[-1]["role"] != "assistant":
            continue
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(msgs[-1]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p_ids) > args.max_prompt_tokens:
            continue
        seqs.append(((p_ids + r_ids)[:args.max_seq_tokens], len(p_ids)))
        if len(seqs) == args.num_seqs:
            break
    print(f"  {len(seqs)} teacher-forced sequences from {args.data}", flush=True)
    return seqs


def main() -> None:
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="HF checkpoint whose G is measured (usually the base)")
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument("--input-key", default="messages")
    parser.add_argument("--teacher-forced", action="store_true", help="use each row's assistant turn as the response")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-seq-tokens", type=int, default=4096, help="teacher-forced: truncate prompt+response")
    parser.add_argument("--gen-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gen-batch", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2, help="sequences per backward")
    parser.add_argument("--loss", default="nll", choices=("nll", "opd_sampled", "opd_fullkl"))
    parser.add_argument("--teacher", type=Path, default=None, help="teacher HF checkpoint for --loss opd_*")
    parser.add_argument("--k-max", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", required=True, type=Path)
    args = parser.parse_args()
    if args.loss != "nll" and args.teacher is None:
        parser.error(f"--loss {args.loss} needs --teacher")

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    seqs = teacher_forced_sequences(args, tok) if args.teacher_forced else rollout_sequences(args, tok, model)
    teacher = None
    if args.loss != "nll":
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).cuda().eval()
        if teacher.config.vocab_size != model.config.vocab_size:
            raise ValueError(f"teacher vocab {teacher.config.vocab_size} != student vocab {model.config.vocab_size}")

    sums, firsts, counts = {}, {}, defaultdict(int)
    state = {"mask": None}

    def make_hook(key):
        def fwd(_module, _inputs, out):
            def bwd(grad):
                x = grad[state["mask"]].float()
                sums[key] = sums[key] + x.T @ x if key in sums else x.T @ x
                firsts[key] = firsts[key] + x.sum(0) if key in firsts else x.sum(0)
                counts[key] += x.shape[0]
            out.register_hook(bwd)
        return fwd

    hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_())]
    for layer_idx, layer in enumerate(model.model.layers):
        for kind in KINDS:
            hooks.append(layer.get_submodule(kind).register_forward_hook(make_hook((layer_idx, kind))))
    seqs.sort(key=lambda z: len(z[0]))
    for s in range(0, len(seqs), args.batch):
        chunk = seqs[s:s + args.batch]
        width = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), width), dtype=torch.long)
        tgt = torch.zeros((len(chunk), width), dtype=torch.bool)
        for i, (x, pl) in enumerate(chunk):
            ids[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
            tgt[i, pl:len(x)] = True
        ids, att, tgt = ids.cuda(), att.cuda(), tgt.cuda()
        state["mask"] = att.bool()
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        logp = torch.log_softmax(logits[:, :-1].float(), -1)
        lp = logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
        mask = tgt[:, 1:].float()
        if args.loss == "nll":
            loss = -(lp * mask).sum()
        else:
            with torch.no_grad():
                tlogp = torch.log_softmax(teacher(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1].float(), -1)
            if args.loss == "opd_sampled":
                adv = tlogp.gather(-1, ids[:, 1:, None]).squeeze(-1) - lp.detach()
                loss = -(adv * lp * mask).sum()
            else:
                loss = ((logp.exp() * (logp - tlogp)).sum(-1) * mask).sum()
            del tlogp
        loss.backward()
        del logits, logp, lp, loss
        if (s // args.batch) % 16 == 0:
            print(f"  backward {s + len(chunk)}/{len(seqs)}  {time.perf_counter() - started:.0f}s", flush=True)
    for h in hooks:
        h.remove()

    tensors, stats = {}, defaultdict(list)
    for (layer_idx, kind) in sorted(sums):
        g = (sums[(layer_idx, kind)] / counts[(layer_idx, kind)]).double()
        mu = (firsts[(layer_idx, kind)] / counts[(layer_idx, kind)]).double()
        lam, vec = torch.linalg.eigh(g)
        lam, vec = lam.flip(0).clamp(min=0), vec.flip(1)
        tr = float(lam.sum())
        p = lam / tr
        for k in REPORT_K:
            stats[(kind, f"top{k}")].append(float(lam[:k].sum()) / tr)
        stats[(kind, "mean^2/tr")].append(float(mu @ mu) / tr)
        stats[(kind, "PR/d")].append(float(1 / (p ** 2).sum()) / lam.numel())
        stats[(kind, "expH/d")].append(math.exp(float(-(p[p > 0] * p[p > 0].log()).sum())) / lam.numel())
        prefix = f"layers.{layer_idx}.{kind}"
        tensors[f"{prefix}.eigvals"] = lam.float().cpu()
        tensors[f"{prefix}.eigvecs"] = vec[:, :args.k_max].float().contiguous().cpu()
        tensors[f"{prefix}.mean"] = mu.float().cpu()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.save), metadata={
        "model": str(args.model), "data": str(args.data), "teacher_forced": str(args.teacher_forced),
        "loss": args.loss, "teacher": str(args.teacher or ""),
        "num_seqs": str(len(seqs)), "k_max": str(args.k_max)})
    print(f"\nsaved {args.save}  ({time.perf_counter() - started:.0f}s)")
    print("##### G spectrum: median over layers [min, max] #####")
    for kind in KINDS:
        print(f"=== {kind}")
        for stat in [f"top{k}" for k in REPORT_K] + ["mean^2/tr", "PR/d", "expH/d"]:
            v = stats[(kind, stat)]
            print(f"    {stat:<10} {statistics.median(v):7.3f} [{min(v):6.3f},{max(v):7.3f}]")


if __name__ == "__main__":
    main()
