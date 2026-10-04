"""Do fine-tune deltas live in the top eigenspace of their layer's input-activation covariance?

A linear layer's gradient is grad_W = E[delta x^T], so an SGD delta's right singular vectors lie in
the span of its inputs x (Adam's per-coordinate scaling makes this approximate). K-FAC writes the
layer's Fisher as A (x) G with A = E[x x^T]. This script measures A directly on the base model and
asks, per tensor, how much of each delta's input side lies in A's top-k eigenvectors E_k:

    sim_k(V_k(D), E_k) = ||V_k(D)^T E_k||_F^2 / k      (same metric as subspace_overlap_profile.py)
    energy_k(D)        = ||D E_k||_F^2 / ||D||_F^2       (all of D's input energy, not only top-k)

both with chance k/n, for every --checkpoint delta and for the base weight W itself as a control.

Steps:
  1. sample --num-prompts prompts from --data (orbit's prompt-data format: .jsonl / .parquet,
     chat template with enable_thinking=False and add_generation_prompt=True, as orbit's Dataset);
  2. generate --gen-tokens tokens per prompt with the base model (approximate on-policy rollouts;
     0 = prompts only);
  3. forward prompt+response with pre-hooks on every layer's linear INPUTS (post-RMSNorm, which is
     what grad_W sees), accumulating sum x x^T per input group:
         attn_in  -> q/k/v_proj      mlp_in  -> gate/up_proj
         o_in     -> o_proj          down_in -> down_proj   (6144 x 6144 per layer; --groups drops it)
     plus the per-coordinate mean square of the raw residual stream h (input_layernorm's input),
     where massive-activation dimensions show up before RMSNorm/gamma rescale them;
  4. eigendecompose each A and compare against the deltas' and W's right singular subspaces.

The per-tensor log uses subspace_overlap_profile.py's line format with the pair "<model> vs act_cov"
(right side only), so plot_tensor_subspace_overlap.py plots it directly; "energy" and coordinate
lines are skipped by that parser. A coordinate report then ranks --probe-coords by activation
mean square (raw residual, attn_in, mlp_in) to check which are high-variance / massive dims.

Usage:
    python activation_covariance_overlap.py \\
        --base /mnt/L202500431/models/qwen3-1.7b \\
        --checkpoint rl300=/mnt/.../iter_0000299_hf --checkpoint opd300=/mnt/.../m6_..._step300 \\
        --num-prompts 256 --gen-tokens 1024 --device cuda [--save-eig eig.safetensors]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from subspace_overlap_profile import (
    GROUP_OF_KIND,
    SVD_DTYPES,
    all_target_tensors,
    format_sim_line,
    subspace_sim,
    tensor_kind,
    top_k_bases,
)
from svd_rank_profile import CANONICAL_RE, load_tensor, tensor_locations

# Default prompt data: the M6 OPD launchers' TRAIN_JSONL default
# (examples/on_policy_distillation/unified_300step_constant/run-m6-opd-st-non-thinking-full.sh).
DEFAULT_DATA = "/mnt/L202500431/datasets/openreasoning_mixed_100k/train.parquet"
DEFAULT_INPUT_KEY = "messages"

ACT_COV_NAME = "act_cov"
BASE_W_NAME = "W_base"
ALL_GROUPS = ("attn_in", "o_in", "mlp_in", "down_in")
# Groups whose coordinates are residual-stream (hidden_size) coordinates, for the coordinate report.
RESIDUAL_GROUPS = ("resid", "attn_in", "mlp_in")


# ----------------------------------------------------------------------------- data


def read_rows(path: Path):
    """orbit.utils.data.read_file without the ray import: .jsonl or .parquet rows as dicts."""
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
    else:
        raise ValueError(f"Unsupported prompt data {path}: expected .jsonl or .parquet")


def to_messages(prompt) -> list[dict]:
    """orbit.utils.data._build_messages with apply_chat_template: str -> one user turn, list as is."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(message) for message in prompt]


def load_prompts(args, tokenizer) -> list[str]:
    rows = [row for row in read_rows(args.data) if row.get(args.input_key) is not None]
    random.Random(args.seed).shuffle(rows)
    template_kwargs = json.loads(args.chat_template_kwargs)
    prompts = []
    for row in rows:
        text = tokenizer.apply_chat_template(to_messages(row[args.input_key]), tokenize=False,
                                             add_generation_prompt=True, **template_kwargs)
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= args.max_prompt_tokens:
            prompts.append(text)
        if len(prompts) == args.num_prompts:
            break
    print(f"sampled {len(prompts)} prompts (<= {args.max_prompt_tokens} tokens) from {args.data} "
          f"[{args.input_key}], {len(rows)} rows total", flush=True)
    return prompts


# ----------------------------------------------------------------------------- activations


class CovarianceCollector:
    """Pre-hooks on each layer's linear inputs accumulating sum x x^T over the selected tokens."""

    def __init__(self, model, groups: tuple[str, ...], dtype: torch.dtype, device: torch.device):
        self.groups, self.dtype, self.device = groups, dtype, device
        self.sums: dict[tuple[int, str], torch.Tensor] = {}
        self.sums_x: dict[tuple[int, str], torch.Tensor] = {}  # sum x, for the activation mean
        self.counts: Counter = Counter()
        self.token_mask: torch.Tensor | None = None  # (B, T) bool; None = not collecting (e.g. generate)
        self.handles = []
        for index, layer in enumerate(model.model.layers):
            targets = {"resid": layer.input_layernorm, "attn_in": layer.self_attn.q_proj,
                       "o_in": layer.self_attn.o_proj, "mlp_in": layer.mlp.gate_proj,
                       "down_in": layer.mlp.down_proj}
            for group, module in targets.items():
                if group == "resid" or group in groups:
                    self.handles.append(module.register_forward_pre_hook(self._hook(index, group)))

    def _hook(self, layer: int, group: str):
        def hook(_module, inputs):
            if self.token_mask is None:
                return
            x = inputs[0][self.token_mask].to(self.dtype)  # (N, d) selected tokens
            key = (layer, group)
            if group == "resid":  # per-coordinate mean square only
                update = x.pow(2).sum(0)
            else:
                update = x.transpose(0, 1) @ x
            self.sums[key] = self.sums[key] + update if key in self.sums else update
            total = x.sum(0)
            self.sums_x[key] = self.sums_x[key] + total if key in self.sums_x else total
            self.counts[key] += x.shape[0]
        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()

    def second_moments(self) -> dict[tuple[int, str], torch.Tensor]:
        """Uncentered second moment E[x x^T] (what grad_W = E[delta x^T] spans), not covariance."""
        return {key: total / self.counts[key] for key, total in self.sums.items()}

    def means(self) -> dict[tuple[int, str], torch.Tensor]:
        """Activation mean E[x] per layer and group."""
        return {key: total / self.counts[key] for key, total in self.sums_x.items()}


def eos_ids(tokenizer, model) -> list[int]:
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return sorted(set(ids))


@torch.no_grad()
def collect(args, model, tokenizer, prompts: list[str], collector: CovarianceCollector) -> None:
    device = next(model.parameters()).device
    stop_ids = torch.tensor(eos_ids(tokenizer, model), device=device)
    n_tokens = Counter()
    for start in range(0, len(prompts), args.batch_size):
        batch = tokenizer(prompts[start:start + args.batch_size], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(device)
        prompt_mask = batch["attention_mask"].bool()
        if args.gen_tokens > 0:
            sequences = model.generate(**batch, max_new_tokens=args.gen_tokens, do_sample=args.temperature > 0,
                                       temperature=args.temperature if args.temperature > 0 else None,
                                       top_p=args.top_p if args.temperature > 0 else None,
                                       pad_token_id=tokenizer.pad_token_id)
            new = sequences[:, prompt_mask.shape[1]:]
            is_stop = torch.isin(new, stop_ids)
            # keep generated tokens up to and including the first stop token, drop the padding after it
            response_mask = (is_stop.cumsum(1) - is_stop.long()) == 0
            attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        else:
            sequences, attention_mask = batch["input_ids"], prompt_mask
            response_mask = torch.zeros_like(prompt_mask[:, :0])
        # Left padding: positions must count real tokens only, as during generation.
        position_ids = (attention_mask.long().cumsum(1) - 1).clamp(min=0)
        if args.token_scope == "response":
            collector.token_mask = torch.cat([torch.zeros_like(prompt_mask), response_mask], dim=1)
        else:
            collector.token_mask = attention_mask
        # The backbone only: the hooks need hidden states, not (B, T, vocab) logits.
        model.model(input_ids=sequences, attention_mask=attention_mask.long(), position_ids=position_ids,
                    use_cache=False)
        collector.token_mask = None
        n_tokens["prompt"] += int(prompt_mask.sum())
        n_tokens["response"] += int(response_mask.sum())
        print(f"  batch {start // args.batch_size + 1}/{-(-len(prompts) // args.batch_size)}: "
              f"{n_tokens['prompt']} prompt + {n_tokens['response']} response tokens so far", flush=True)


# ----------------------------------------------------------------------------- analysis


def energy_fraction(matrix: torch.Tensor, basis: torch.Tensor) -> float:
    """||M E||_F^2 / ||M||_F^2 for an (out, n) matrix and an (n, k) orthonormal basis."""
    return float(torch.linalg.norm(matrix @ basis) ** 2 / torch.linalg.norm(matrix) ** 2)


def format_energy_line(k: int, frac: float, n: int) -> str:
    """Matched by neither plot parser ("frac=", not "sim_k=")."""
    return f"    energy right k={k:<4} frac={frac:.4f}  chance(k/n)={k / n:.6f}  observed/chance={frac / (k / n):.2f}x"


def print_spectrum_summary(eigvals: dict[tuple[int, str], torch.Tensor], k_values: list[int]) -> None:
    """How anisotropic each group's second moment is: share of the trace in the top-k eigenvalues."""
    print("##### activation second moment: share of trace in top-k eigenvalues (isotropic = k/n) #####")
    for group in ALL_GROUPS:
        keys = [key for key in eigvals if key[1] == group]
        if not keys:
            continue
        n = eigvals[keys[0]].numel()
        shares = []
        for k in k_values:
            per_layer = [float(eigvals[key][:k].sum() / eigvals[key].sum()) for key in keys]
            shares.append(f"k={k}:{statistics.mean(per_layer):.3f}")
        print(f"  {group:<8} n={n:<5} " + "  ".join(shares) + f"  (mean over {len(keys)} layers)")
    print(flush=True)


def mean_direction_report(eigvals: dict[tuple[int, str], torch.Tensor], eigvecs: dict[tuple[int, str], torch.Tensor],
                          means: dict[tuple[int, str], torch.Tensor], top: int) -> None:
    """Is the top eigenvector of the uncentered second moment A = E[x x^T] the activation MEAN, and is the
    input's projection on it nearly constant (so delta along it acts like a bias)? With mu = E[x] and
    eigenpair (lambda_k, e_k): m_k = e_k^T mu, Var(e_k^T x) = lambda_k - m_k^2, and
        cos      |cos(e_k, mu)|                         1 = e_k is the mean direction
        meanFrac m_k^2 / lambda_k                         share of e_k's energy that is the constant mean part
        CV       sqrt(Var) / |m_k|                        relative spread of the projection; << 1 = ~constant
    plus ||mu||^2 / tr(A), the share of all input energy that is the mean."""
    print("##### mean-direction report: is e_k the activation mean, and is e_k^T x ~ constant (bias-like)? #####")
    print("  per group: median over layers [min, max]")
    stat = lambda v: f"{statistics.median(v):7.3f} [{min(v):6.3f},{max(v):7.3f}]"  # noqa: E731
    for group in ALL_GROUPS:
        keys = sorted(key for key in eigvals if key[1] == group and key in means)
        if not keys:
            continue
        mean_share = [float(means[k].double().pow(2).sum() / eigvals[k].double().sum()) for k in keys]
        print(f"  ===== {group}  ||mu||^2 / tr(A): {stat(mean_share)}")
        for r in range(min(top, eigvecs[keys[0]].shape[1])):
            cos, frac, cv = [], [], []
            for k in keys:
                e, mu, lam = eigvecs[k][:, r].double(), means[k].double().to(eigvecs[k].device), float(eigvals[k][r])
                m = float(e @ mu)
                cos.append(abs(m) / (float(mu.norm()) + 1e-30))
                frac.append(m * m / max(lam, 1e-30))
                cv.append(math.sqrt(max(lam - m * m, 0.0)) / max(abs(m), 1e-30))
            print(f"    e_{r + 1}:  cos {stat(cos)}   meanFrac {stat(frac)}   CV {stat(cv)}")
    print(flush=True)


def coordinate_report(moments: dict[tuple[int, str], torch.Tensor], probe: list[int], top_n: int = 10) -> None:
    """Rank coordinates by mean square per layer, for the raw residual and the post-norm residual inputs."""
    print("##### residual-stream coordinates by activation mean square #####")
    print(f"  rank 1 = largest in that layer; ratio = mean square / median coordinate mean square\n")
    for group in RESIDUAL_GROUPS:
        keys = sorted(key for key in moments if key[1] == group)
        if not keys:
            continue
        diags = [moments[key] if group == "resid" else torch.diagonal(moments[key]) for key in keys]
        print(f"  -- {group} ({len(keys)} layers) --")
        top_counts = Counter()
        for diag in diags:
            top_counts.update(torch.topk(diag, top_n).indices.tolist())
        shown = ", ".join(f"{coord}:{count}" for coord, count in top_counts.most_common(top_n))
        print(f"    most frequent per-layer top-{top_n} coords (coord:#layers): {shown}")
        for coord in probe:
            ranks, ratios = [], []
            for diag in diags:
                if coord >= diag.numel():
                    break
                ranks.append(int((diag > diag[coord]).sum()) + 1)
                ratios.append(float(diag[coord] / diag.median()))
            if ranks:
                print(f"    coord {coord:<5} median rank {statistics.median(ranks):>6.0f} / {diags[0].numel()}  "
                      f"best rank {min(ranks):<5} median ratio {statistics.median(ratios):8.1f}x  "
                      f"max ratio {max(ratios):10.1f}x")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                         help="fine-tuned HF checkpoint(s) whose delta from --base is compared; repeatable")
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA),
                         help=f"prompt data (.jsonl/.parquet); default: the M6 OPD launchers' {DEFAULT_DATA}")
    parser.add_argument("--input-key", default=DEFAULT_INPUT_KEY,
                         help="prompt field: a string or a list of chat messages (M9 verifier data uses 'prompt')")
    parser.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}')
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--gen-tokens", type=int, default=1024,
                         help="tokens generated per prompt by the base model; 0 = prompt activations only")
    parser.add_argument("--temperature", type=float, default=1.0, help="0 = greedy")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--token-scope", default="all", choices=("all", "response"),
                         help="tokens entering the second moment; 'all' because prompt positions also get "
                              "gradient through attention")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--groups", default=",".join(ALL_GROUPS),
                         help=f"input groups to collect, subset of {ALL_GROUPS}; down_in costs 6144^2 per layer")
    parser.add_argument("--k", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--probe-coords", default="978,505,1536,1720,542,1999,1793",
                         help="residual coordinates to rank (default: subspace_overlap_profile --localization "
                              "top hits for rl300/opd300 deltas and W_base o/down)")
    parser.add_argument("--mean-report-k", type=int, default=4,
                         help="eigenvectors e_1..e_k covered by the mean-direction report")
    parser.add_argument("--save-eig", type=Path, default=None,
                         help="save each group's eigenvalues and top-max(k) eigenvectors to this .safetensors")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cov-dtype", default="float32", choices=tuple(SVD_DTYPES),
                         help="accumulation / eigendecomposition precision of the second moments")
    parser.add_argument("--svd-dtype", default="float32", choices=tuple(SVD_DTYPES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"--groups {sorted(unknown)} not in {ALL_GROUPS}")
    k_values = [int(k) for k in args.k.split(",") if k.strip()]
    probe = [int(c) for c in args.probe_coords.split(",") if c.strip()]
    checkpoints = {}
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        if name in (BASE_W_NAME, ACT_COV_NAME):
            raise ValueError(f"--checkpoint name {name!r} is reserved")
        checkpoints[name] = Path(path)

    # 1-3: rollouts and second moments on the base model.
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(device).eval()
    prompts = load_prompts(args, tokenizer)
    collector = CovarianceCollector(model, groups, SVD_DTYPES[args.cov_dtype], device)
    print(f"collecting {groups} + resid over {args.token_scope} tokens, gen_tokens={args.gen_tokens}, "
          f"temperature={args.temperature}", flush=True)
    collect(args, model, tokenizer, prompts, collector)
    collector.remove()
    moments = collector.second_moments()
    means = collector.means()
    del model, collector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"##### collection took {time.perf_counter() - started:.1f}s #####\n", flush=True)

    # 4: eigendecompositions, descending.
    max_k = max(k_values)
    eigvals, eigvecs = {}, {}
    for key, moment in moments.items():
        if key[1] == "resid":
            continue
        values, vectors = torch.linalg.eigh(moment)
        eigvals[key] = values.flip(0)
        eigvecs[key] = vectors.flip(1)[:, :max_k].contiguous()
    print_spectrum_summary(eigvals, k_values)
    mean_direction_report(eigvals, eigvecs, means, args.mean_report_k)
    coordinate_report(moments, probe)
    if args.save_eig is not None:
        from safetensors.torch import save_file

        tensors = {}
        for (layer, group), values in eigvals.items():
            tensors[f"layers.{layer}.{group}.eigvals"] = values.float().cpu()
            tensors[f"layers.{layer}.{group}.eigvecs"] = eigvecs[(layer, group)].float().cpu()
        for (layer, group), mean in means.items():
            tensors[f"layers.{layer}.{group}.mean"] = mean.float().cpu()
        save_file(tensors, str(args.save_eig))
        print(f"saved eigendecompositions -> {args.save_eig}\n", flush=True)

    # Per-tensor comparison against every delta and W itself.
    svd_dtype = SVD_DTYPES[args.svd_dtype]
    base_locations = tensor_locations(args.base)
    ckpt_locations = {name: tensor_locations(path) for name, path in checkpoints.items()}
    tensor_names = [name for name in all_target_tensors(base_locations) if GROUP_OF_KIND[tensor_kind(name)] in groups]
    sims: dict[tuple, list[float]] = defaultdict(list)  # (kind, model, k, n) -> per layer
    energies: dict[tuple, list[float]] = defaultdict(list)
    for tensor_name in tensor_names:
        layer, kind = int(CANONICAL_RE.match(tensor_name).group(1)), tensor_kind(tensor_name)
        basis = eigvecs[(layer, GROUP_OF_KIND[kind])].to(device, svd_dtype)
        base_tensor = load_tensor(args.base, base_locations, tensor_name).to(device, svd_dtype)
        n_out, n_in = base_tensor.shape
        matrices = {name: load_tensor(path, ckpt_locations[name], tensor_name).to(device, svd_dtype) - base_tensor
                    for name, path in checkpoints.items()}
        matrices[BASE_W_NAME] = base_tensor
        print(f"=== {tensor_name}  (shape {tuple(base_tensor.shape)}) ===", flush=True)
        for model_name, matrix in matrices.items():
            right = top_k_bases(matrix, k_values, svd_dtype)["right"]
            print(f"  -- {model_name} vs {ACT_COV_NAME} --")
            for k in k_values:
                if k > min(n_out, n_in):
                    continue
                sim = subspace_sim(right[k], basis[:, :k])
                print(format_sim_line("right", k, sim, n_in))
                sims[(kind, model_name, k, n_in)].append(sim)
            for k in k_values:
                frac = energy_fraction(matrix, basis[:, :k])
                print(format_energy_line(k, frac, n_in))
                energies[(kind, model_name, k, n_in)].append(frac)
        print()

    print(f"##### total {time.perf_counter() - started:.1f}s #####\n")
    print(f"##### mean over layers, per tensor kind: sim_k(V_k, E_k) | energy_k (chance k/n for both) #####\n")
    for kind in dict.fromkeys(key[0] for key in sims):
        print(f"=== summary: {kind} (input group {GROUP_OF_KIND[kind]}) ===")
        for model_name in dict.fromkeys(key[1] for key in sims if key[0] == kind):
            print(f"  -- {model_name} vs {ACT_COV_NAME} --")
            for key in (key for key in sims if key[0] == kind and key[1] == model_name):
                _, _, k, n_in = key
                sim, frac = statistics.mean(sims[key]), statistics.mean(energies[key])
                print(f"    right k={k:<4} mean_sim_k={sim:.4f}  mean_energy={frac:.4f}  chance(k/n)={k / n_in:.6f}  "
                      f"sim/chance={sim / (k / n_in):.1f}x  energy/chance={frac / (k / n_in):.1f}x")
        print()


if __name__ == "__main__":
    main()
