# Checkpoint extrapolation (`tools/extrapolation`)

Two post-hoc extrapolation methods for OPD runs, applied to Orbit's existing
`orbit_ckpts/<run>/iter_NNNNNNN` full-finetune checkpoints:

- **`simple_extrapolate.py`** — fixed-alpha, whole-model (or optional
  keyword-subset) linear extrapolation, no validation gating. This is the
  "fixed extrapolation strategy" baseline the paper contrasts EffOPD
  against (their AlphaOPD/ExOPD comparisons in Section 4.2), not a specific
  reimplementation of either.
- **`effopd_extrapolate.py`** — **EffOPD** itself: "Learning to Foresee:
  Unveiling the Unlocking Efficiency of On-Policy Distillation" (Cai et
  al.), Section 4.1. Exponentially-spaced trigger checkpoints (t=2^n),
  attention/MLP-only parameter subset, five increasing-magnitude candidates
  (α=2,4,6,8,10) scored in order against a lightweight validation set D_v,
  accept-while-improving / stop-on-first-failure.

Both share `checkpoint_io.py` for the actual tensor math and checkpoint I/O,
and both were built to mirror the reference EffOPD port already checked
into this repo's sibling checkout, `trl/scripts/`:
`build_effopd_extrapolated_checkpoint.py` (candidate math + parameter
selection) and `opd_posthoc_common.py` (`CheckpointTensorStore`).
`analyze_m6_effopd_extrapolation_oracle.py` there is the precedent for this
tool's "always diff against the real trained checkpoint" convention (see
below).

Everything above is **post-hoc**: it reads checkpoints a finished run
already produced and never touches the trainer — no forward+backward, no
optimizer step, ever, confirmed against every function in `checkpoint_io.py`.
For an actual live-training integration (extrapolation results feed back
into a real, running orbit training job and genuinely shorten it), see
**[`live_training/`](live_training/README.md)** — a separate, much higher-
blast-radius tool that launches real multi-GPU training segments, with its
own README covering what's unverified before you run it for real.

## Default run: lr=2e-6

Per the current extrapolation study, every script here defaults
`--run-dir` to `orbit_ckpts/m6_st_full_non_thinking_lr_2e-6` — the lr sweep
where the AIME24 curve stayed monotonically increasing through step 20
(40.63) with no mid-run reversal, unlike the default lr=5e-6 run (peaks at
step 8, then oscillates). Override `--run-dir` to point at a different
sweep arm.

`--base-model` (the run's `HF_CKPT`, i.e. W₀) defaults to
`/mnt/L202500431/models/qwen3-1.7b` — confirmed against this cluster's
`../models/` layout (the launcher script's own hardcoded default,
`/mnt/L202500430/orbit/data/hf_ckpts/Qwen3-1.7B`, is a *different* data
mount and was not used). Override `--base-model` if you point `--run-dir`
at a run trained from a different base checkpoint.

## Why the math runs in HF-parameter space, not raw Megatron space

`iter_*` checkpoints are Megatron torch-distributed (DCP) shards, not HF
safetensors. Extrapolation here always converts to HF first (via a
subprocess call to the existing `tools/convert_torch_dist_to_hf.py`,
cached per-iteration) rather than working on the raw Megatron state dict,
because EffOPD's parameter-selection rule is a **name substring match**
(`checkpoint_io.EFFOPD_KEYWORDS`) and Megatron's naming doesn't line up
with it the way HF's does: Megatron fuses the pre-attention LayerNorm
weight into the QKV module name
(`self_attention.linear_qkv.layer_norm_weight`, which *would* wrongly match
`"attention"`), whereas post-conversion it's `input_layernorm.weight`
(matches nothing) and `post_attention_layernorm.weight` (matches
`"attention"` — this one **should** match; it's the released
implementation's own documented quirk, reproduced deliberately, not fixed).
Matching only works correctly on the HF names the released EffOPD code was
actually written against.

The QKV-fusion split and gate/up-fusion split Megatron→HF conversion
applies are just reshape/chunk — linear operations — so extrapolating
after conversion is numerically equivalent to extrapolating before it, for
any parameter that survives the split unmixed. Doing it after just makes
the keyword selection correct.

## Post-hoc framing (important, `effopd_extrapolate.py`)

This tool runs against a fixed, already fully-trained checkpoint sequence —
it cannot restart live training from an extrapolated checkpoint. So at
every trigger n, Δn is always computed between the two **real** trained
checkpoints W_(2^(n-1)) and W_(2^n), never between a previous n's
*accepted/extrapolated* candidate and a real checkpoint. This matches the
paper's own definition of W_t ("parameters after the t-th OPD update", the
actual trained trajectory) and mirrors the existing oracle-style convention
on the trl side (`analyze_m6_effopd_extrapolation_oracle.py`). It answers
*"how good would EffOPD's candidate have been at checkpoint 2^n"*, not
*"what would live training have produced had EffOPD been wired into the
optimizer loop"* — the latter needs the extrapolated weights fed back into
training itself, out of scope for a post-hoc analysis tool.

`t` indexes a checkpoint by its **ordinal position** in the run's saved
sequence, not the raw Megatron iteration number. `m6_st_full_non_thinking_lr_2e-6`
saves every 2 steps starting at iteration 1
(`iter_0000001, iter_0000003, ..., iter_0000019`, 10 checkpoints), so
`t=1..10` are available → `n=0,1,2,3` (t=1,2,4,8) all run; `n=4` (t=16) is
skipped, not enough saved checkpoints. (These are also the same 10
checkpoints as the "step2..step20" eval table: `step_k` ↔ `iter_(k-1)`.)

## Reference norms (`checkpoint_norms.py`)

Before spending an eval cycle on any alpha, get a sense of scale for the
deltas you'd be multiplying:

```bash
python tools/extrapolation/checkpoint_norms.py \
    --output extrapolation_results/norms/lr_2e-6.json
```

Prints, per saved checkpoint, `||W_t - W_0||` (total displacement from the
base model) and `||W_t - W_(t-1)||` (the per-save-interval step delta),
both whole-model and restricted to EffOPD's attention/MLP keyword subset.
`simple_extrapolate.py`/`effopd_extrapolate.py` build `current + alpha *
(current - anchor)`, so this directly tells you how big alpha=10 actually
makes the step (an 11x-sized perturbation relative to whatever `||delta||`
prints for your chosen anchor/current pair) before you commit a serve+eval
cycle to finding out.

## First things to try

Not run yet on this repo's checkpoints (see "Not executed end-to-end"
below), so treat this as a starting point, not a validated recommendation:

1. **Run `checkpoint_norms.py` first**, full run (defaults are fine). Look
   at `iter_0000019`'s `||W-Wprev||` (keyword subset) — that's the size of a
   single 2-step delta near convergence, and it's what `--alphas 2 4 ...`
   actually scales.
2. **Anchor/current pair**: start with the last two checkpoints,
   `--anchor-iter 17 --current-iter 19` (a single 2-step interval, matching
   the granularity EffOPD's own `n=3` step would use if it landed there).
   Also try a longer baseline, `--anchor-iter 15 --current-iter 19` (spans
   4 steps) — Property 2 in the paper says the update *direction* should be
   stable either way, but a longer interval averages out more per-step
   noise, and the lr=2e-6 AIME24 curve isn't perfectly smooth step-to-step
   even though it's monotonic overall.
3. **Alpha**: start with `--alphas 2 4` (conservative) before the full
   `2 4 6 8 10` sweep — this is the first time this pipeline (conversion +
   extrapolation + eval) will have actually run end-to-end here, so confirm
   a small alpha produces a checkpoint that loads and scores sanely before
   spending a serve+eval cycle on alpha=10.
4. **Whole-model vs `--keyword-only`**: run both for the same anchor/current
   pair once — it's just re-running `build_candidate` with a different
   flag, no extra conversion cost — to see how much of the effect (if any)
   comes from touching the embedding/lm_head/layernorms too, versus
   attention+MLP alone. `--keyword-only` also doubles as a rehearsal for
   `effopd_extrapolate.py`, which always restricts to that subset.

## Usage

```bash
# Simple baseline: sweep alpha in {2,4,6,8,10} extrapolating from the last
# two saved checkpoints, whole model. --run-dir and --base-model both default
# to this study's setup, so a bare invocation works from the orbit repo root.
python tools/extrapolation/simple_extrapolate.py \
    --anchor-iter 17 --current-iter 19 \
    --output-root extrapolation_results/simple/lr_2e-6

# EffOPD: full exponential-schedule, validation-gated search. --validator-cmd
# is required -- point it at validate_checkpoint.py (serves candidates via
# the existing eval-math-evalchemy.sh) or your own scorer.
python tools/extrapolation/effopd_extrapolate.py \
    --validator-cmd "python tools/extrapolation/validate_checkpoint.py \
        --evalchemy-root /path/to/evalchemy --task aime24 --num-samples 50" \
    --output-root extrapolation_results/effopd/lr_2e-6
```

Each candidate is written as a standalone, directly-servable HF checkpoint
directory (`model.safetensors` + tokenizer/config copied from the nearer
real checkpoint + an `extrapolation_metadata.json` sidecar recording alpha,
which parameters were touched, and displacement stats). `simple_extrapolate.py`
writes one `manifest_*.json` per anchor/current pair;
`effopd_extrapolate.py` writes one `search_manifest.json` per `n` plus a
top-level `effopd_manifest.json` with the full schedule and every accepted
alpha.

`validate_checkpoint.py`'s default D_v is 50 examples from an existing
Evalchemy task (`aime24`), not literally "sampled from the training set"
(the paper's exact Dv construction) — that needs your training jsonl, which
isn't available from this checkout. `sample_validation_set.py` builds a
proper training-set-sampled D_v file if you want that instead; see its
docstring for wiring it back into `run_evalchemy_math_eval.py`'s task table.

## Not executed end-to-end

Written and reasoned through against the paper (Section 4.1) and the two
trl-side reference scripts, and syntax-checked (`ast.parse`), but **not run**
— this authoring session has no GPU, no `torch`/`safetensors`/Megatron/orbit
installed, and no access to the cluster paths this run actually lives on
(`/mnt/L202500431/...`). Before trusting numbers out of it:

- Confirm `resolve_run_iterations(run_dir)` returns `[1, 3, 5, ..., 19]` as
  expected for this run.
- Run `simple_extrapolate.py` once for a single alpha and check
  `extrapolation_metadata.json`'s `selected_element_fraction` looks sane
  (whole-model run: 1.0; `--keyword-only`: well under 1.0, attention+MLP
  weights only).
- Confirm the written candidate directory actually loads with
  `AutoModelForCausalLM.from_pretrained` before wiring up a full
  `effopd_extrapolate.py` sweep — that script triggers a full serve+eval
  cycle per candidate `k`, so a broken conversion fails slow and expensive.
- Start `effopd_extrapolate.py` with `--max-n 0` (only the t=1 search)
  before letting it walk the whole schedule.
