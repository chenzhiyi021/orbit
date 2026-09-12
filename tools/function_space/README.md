# Function-space cosine heatmap (orbit port)

Ports the "cosine of centered-logit deltas" measurement from the trl-side
experiment
`experimental/function_space/workspace_snapshot/experiments/p4_function_space_20260907/`
(itself wrapping the framework-agnostic scorer in
`unifying_posttrain/experiments/m5_m6_param_native_trajectory/`) so it can be
run against orbit-trained checkpoints and orbit-trained checkpoints can be
mixed with trl-trained ones (e.g. the trl-side M6 FullFT run) in the same
heatmap.

Source of truth for the math: `scoring.py` (`score_bank`, `sketch_maps`) and
`relations.py` (`cosine_distance`, `bootstrap_cosine_distance`, `sketch_gate`)
are copied **unchanged** from the trl-side files. If you need to verify that,
diff them against:

- `.../unifying_posttrain/experiments/m5_m6_param_native_trajectory/score_function_space.py`
- `.../unifying_posttrain/experiments/m5_m6_param_native_trajectory/aggregate_function_space.py`
- `.../unifying_posttrain/experiments/m5_m6_param_native_trajectory/analyze_function_relations.py`

Everything else (`score_checkpoints.py`, `plot_cosine_heatmap.py`,
`config.example.json`) is new orchestration code, generalized away from the
trl workspace's hardcoded P4/M6 manifest so you can point it at any mix of
checkpoints, from either codebase.

## What changed vs the trl version (and why)

1. **Config-driven, not manifest-generated.** The trl version's `prepare.py`
   derives a `manifest.json` from a specific workspace layout (prompt banks,
   historical caches, checkpoint directories all at fixed relative paths).
   Here you write one `config.json` (see `config.example.json`) with
   explicit paths -- works regardless of which cluster/workspace you're on.
2. **No `copy_orbit_env_direct` staging.** That was a Lustre O_DIRECT
   throughput optimization for one specific cluster, not part of the scoring
   math. Dropped; add your own staging step back if you hit the same
   page-cache-thrashing problem.
3. **No hard `transformers==4.57.1` assert.** Recorded per-checkpoint in each
   `*.json` metadata sidecar (`transformers_version`, `torch_version`,
   `cuda_version`, `gpu`, `host`) instead, so a version mismatch between two
   runs you're comparing is auditable rather than either invisible or a hard
   crash.
4. **`plot_cosine_heatmap.py` auto-discovers runs** in the scored output
   directory instead of a hardcoded `RUNS` tuple; pass `--runs` to fix the
   order or restrict to a subset.
5. **Dropped the "historical vs freshly recomputed base" sensitivity block**
   from `analyze.py` (that compared P4's fresh scoring against orbit's
   pre-existing cache under two different bases). Replaced by a simpler,
   optional `base_noise_gate` in `score_checkpoints.py`: if you give a bank a
   `historical_base_npz`, it checks your freshly-scored Base against it and
   writes `baseline_parity.json` with the same 5%-of-norm threshold as the
   original gate.

Full list of trl-side dependencies this port removes:
`copy_orbit_env_direct` (cluster-specific I/O), the trl `manifest.json`
generation pipeline (`prepare.py`'s bank-provenance re-derivation, which
depends on artifacts specific to that workspace), and the hardcoded P4/M6
run identities.

## Usage

1. Copy `config.example.json` to `config.json` and fill in real paths:
   - `base_model`: the pretrained checkpoint every run's delta is measured
     against.
   - `banks.<id>.path`: a prefix bank saved as an HF `datasets` directory
     (`Dataset.save_to_disk`). Each row needs `input_ids`, `prompt_ids`,
     `selected_positions`, `selected_index`, `prompt_sha256`, `domain`. **This
     tool does not build banks** -- point it at bank_a / bank_b directories
     you already have (e.g. copied from the trl-side
     `restored/artifacts/.../state_banks/`), or build your own with the same
     schema.
   - `banks.<id>.teacher_cache` (optional): precomputed teacher logits from
     step 2 below. Omit / set `null` to skip teacher RKL/FKL scalars.
   - `checkpoints`: `{run_name: path}` for any full, mergeable HF checkpoint
     (must load via `AutoModelForCausalLM.from_pretrained` directly) --
     works for FullFT and for merged LoRA/OFT exports. **Does not work for
     raw un-merged LoRA/OFT adapter state** (the trl-side `adapter_geometry.py`
     baking step was not ported -- see "Residual issues" below).

2. (Optional, needed for teacher-RKL/FKL columns) Cache teacher logits once
   per bank:
   ```
   python cache_teacher_logits.py --bank /path/to/bank_a \
       --model /path/to/teacher --output cache/teacher/bank_a.npy
   ```
   Point `banks.bank_a.teacher_cache` at the resulting `.npy`.

3. Score every checkpoint (Base + everything in `config["checkpoints"]`) on
   one bank:
   ```
   python score_checkpoints.py --config config.json --bank bank_a
   python score_checkpoints.py --config config.json --bank bank_b
   ```
   Writes `<output_root>/<bank>/{Base,<run>}.npz` (+ `.scalars.parquet` +
   `.json` sidecar each), `baseline_parity.json` if `historical_base_npz`
   was given, and `COMPLETE.json` on success. Re-running skips any run whose
   `.npz` + `.json` already exist.

4. Plot the heatmap:
   ```
   python plot_cosine_heatmap.py --config config.json --bank bank_a \
       --runs "M6-FullFT-trl,M6-OFT-orbit,M6-LoRA-orbit"
   ```
   Writes `<output_root>/<bank>/heatmap/{cosine_matrix.csv, pairwise.csv,
   pairwise_by_domain.csv, countsketch_exact_gate.csv, VALIDATION.json,
   REPORT.md, function_cosine.png}`. Omit `--runs` to auto-discover every
   scored run (order not guaranteed then -- prefer passing it explicitly for
   a reproducible figure).

To compare a trl-trained checkpoint (e.g. your M6 FullFT run) against
orbit-trained checkpoints: just list both under `checkpoints` in the same
`config.json` and score them in the same pass, on the same bank. They will
be scored with the *same* environment (this orbit checkout's `transformers`/
`torch`/attention backend), which removes the cross-codebase-scoring
confound entirely -- you're only still exposed to whatever numerical
difference exists between the *training* stacks (see below), not the
scoring stack.

## Probability-space cosine (`--space prob`)

By default (`--space logit`, unchanged), the cosine is computed on
vocabulary-**centered logits** (`z - mean(z)`). This weights every
vocabulary token equally, including near-zero-probability tail tokens --
which can matter a lot: a method that mostly reshuffles the tail can show a
large centered-logit delta while barely moving the actual output
distribution.

Pass `--space prob` to both scripts to sketch **post-softmax probabilities**
instead (`softmax(z)`, no centering needed since probability vectors already
share the same mean, 1/V). Tail tokens are automatically down-weighted
there, since their probability is ~0.

```
python score_checkpoints.py --config config.json --bank bank_a --space prob
python plot_cosine_heatmap.py --config config.json --bank bank_a --space prob \
    --runs "M6-FullFT-trl,M6-OFT-orbit,M6-LoRA-orbit"
```

Writes to `<output_root>/<bank>/prob/...` -- a separate tree from the
default logit-space results, so running this never touches or requires
re-running anything you've already scored. **Run both** and compare: if a
clustering/gap you found in logit space also shows up in probability space,
it's a real functional finding; if it only shows up in one of them, the
logit-space version may have been (partly) a tail-weighting artifact -- see
the `REPORT.md` written by each pass for the exact caveat text.

## Residual issues / not verified

I could not execute any of this in the environment I wrote it in -- no GPU,
and no `torch`/`transformers`/`datasets`/`pandas` installed locally (checked;
none are importable here). What follows is a structural port with the
numerically-critical functions copied unchanged, syntax-checked, but **not
run end-to-end**. Before trusting numbers out of it:

- **Run `plot_cosine_heatmap.py`'s `sketch_gate` output first** and check
  `cosine_error_gate_lt_0_02` / `distance_order_gate_gt_0_98` /
  `independent_sketch_gate_gt_0_98` all pass. This is the same validation
  the trl side used; it directly tells you whether the CountSketch
  approximation is behaving here, independent of anything else.
- **Bank schema**: I inferred the required `datasets` row schema
  (`input_ids`, `prompt_ids`, `selected_positions`, `selected_index`,
  `prompt_sha256`, `domain`) from `score_bank`'s usage and `prepare.py`'s
  provenance checks, but did not have a bank file to test-load locally.
  Verify a real `load_from_disk(...)[0]` has exactly these keys before your
  first real run.
- **Un-merged LoRA/OFT adapters are not supported.** `adapter_geometry.py`
  (`bake_linear_weight`, `cayley_neumann`, `load_lora`, `load_oft`) was not
  ported. If your orbit LoRA/OFT checkpoints are raw adapter state rather
  than a merged full checkpoint, either export a merged checkpoint first
  (there's already merge tooling in `orbit/merge/`, e.g. `oft_merge.py` for
  OFT) or ask to have `adapter_geometry.py` ported too.
  `score_one`'s `AutoModelForCausalLM.from_pretrained(checkpoint)` will
  either fail to load or silently load the wrong weights on a raw adapter
  checkpoint -- I did not add a check for that distinction.
  This mirrors the two orbit-side conventions you already have: `orbit/merge/oft_merge.py` merges OFT, and
  the coordinate-mask / subspace-lock pipelines in `trl/scripts/` also
  export full safetensors checkpoints -- so this should be a non-issue for
  those, only for raw LoRA/OFT adapter dirs.
- **Training-stack confound is untouched by this port.** Scoring both
  checkpoints in the same environment removes the *scoring*-side confound
  (attention backend, transformers version at inference time), but the
  trl-trained and orbit-trained checkpoints were still produced by different
  *training* stacks (Megatron-fused vs `adamw_torch` optimizer, `flash-attn`
  vs whatever attention backend each training run used, etc. -- see the
  optimizer/attention-backend discussion from earlier). This tool cannot
  and does not eliminate that; it only removes the scoring-time part of it.
- **`--exact-positions 16` / `sketch_dimension 1024` are inherited
  defaults**, not re-validated for a different vocab size or a very
  different model family. If you score a model with a much larger
  vocabulary than the ~150k the original scorer targeted, consider whether
  1024 sketch buckets are still enough for a tight `cosine_error_gate`.
- Only syntax-checked (`ast.parse`), not executed -- see the check below.
