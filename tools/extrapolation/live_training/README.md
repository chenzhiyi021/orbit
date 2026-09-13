# Live EffOPD training (`tools/extrapolation/live_training`)

Everything one directory up (`tools/extrapolation/*.py`) is post-hoc: it
reads checkpoints an already-finished training run produced and never
touches the trainer — confirmed [earlier in this thread](.): no
forward+backward, no optimizer step, ever.

This directory is the real thing: **`run_effopd_live_training.py` actually
drives orbit's training launcher**, segment by segment, and at each
exponential trigger `t=2^n` splices the accepted (possibly extrapolated)
candidate back in as the starting point for the *next* segment of real
training. Later segments genuinely train from the extrapolated weights —
this is what "acceleration" means in the paper, and what the post-hoc tools
next door deliberately do not do (they always diff two *real* checkpoints,
oracle-style).

## Segment plan (`--total-steps 20`, matching this study's budget)

```
n=0: train step        1  (NUM_ROLLOUT=1)  -> extrapolation search at t=1
n=1: train step        2  (NUM_ROLLOUT=1)  -> extrapolation search at t=2
n=2: train steps     3-4  (NUM_ROLLOUT=2)  -> extrapolation search at t=4
n=3: train steps     5-8  (NUM_ROLLOUT=4)  -> extrapolation search at t=8
n=4: train steps    9-16  (NUM_ROLLOUT=8)  -> extrapolation search at t=16
     train steps   17-20  (NUM_ROLLOUT=4)  -> finish; no trigger (2^5=32>20)
```

`1+1+2+4+8+4 = 20` real optimizer steps, same total as the lr sweep this
study has been comparing against. Each segment is one full invocation of
`examples/on_policy_distillation/unified_300step_constant/run-m6-opd-st-non-thinking-full.sh`
(a fresh Ray cluster, `NUM_ROLLOUT` capped to the segment length,
`MEGATRON_LOAD` pointed at the previous segment's accepted checkpoint).

## How to run

```bash
python tools/extrapolation/live_training/run_effopd_live_training.py \
    --train-jsonl /path/to/openreasoning_mixed_100k/train_qa.parquet \
    --teacher-hf-ckpt /path/to/hf_ckpts/Qwen3-4B-Instruct-2507 \
    --evalchemy-root /mnt/L202500431/third_party/evalchemy \
    --validator-cmd "python tools/extrapolation/validate_checkpoint.py \
        --evalchemy-root /mnt/L202500431/third_party/evalchemy \
        --task aime24 --num-samples 50 --num-gpus 1 --eval-tp-size 1" \
    --live-root live_training_runs/effopd_lr_2e-6 \
    --total-steps 20
```

`--base-model` and `--megatron-base` default to
`/mnt/L202500431/models/qwen3-1.7b` and `/mnt/L202500431/models/megatron_ckpt`
respectively — the first is confirmed (same path you gave me for the
post-hoc tools), the second is **a guess** from `ls ../models` showing a
`megatron_ckpt` entry; **verify it actually holds a Megatron-converted
Qwen3-1.7B before trusting it** (`ls` it, check for a config/args file that
names the model). `--train-jsonl`, `--teacher-hf-ckpt`, `--evalchemy-root`
have no default — this session cannot see your `/mnt/L202500431/...` layout
well enough to guess those safely (the launcher script's own hardcoded
defaults point at a *different* mount, `L202500430`, which is why the
earlier post-hoc `--base-model` default was wrong until you corrected it).

## What actually happens, mechanically

1. **Train a segment**: `run_segment()` shells out to the existing launcher
   script with `MEGATRON_LOAD`/`NUM_ROLLOUT`/`SAVE_DIR` overridden via env
   vars (all already overridable in that script — nothing in it was
   modified). `RUN_TRAIN=1 RUN_EVAL=0`, same as a training-only invocation
   you'd run by hand.
2. **Verify the step landed**: after the segment exits, `resolve_run_iterations(segment_dir)`
   must show the checkpoint at exactly the expected iteration. If it
   doesn't, the run **aborts immediately** with a loud error rather than
   silently drifting the step budget — see the unverified-assumption list
   below for why this check exists.
3. **Extrapolate**: same `build_candidate`/`should_modify` math as
   `effopd_extrapolate.py` next door (imported directly, not reimplemented),
   except the anchor is whatever this run accepted at the *previous*
   trigger — which may itself have been extrapolated.
4. **Validate**: same `--validator-cmd` contract as the post-hoc tool.
5. **Stage the accepted checkpoint for the next segment**:
   `megatron_checkpoint_stage.py` converts the accepted HF checkpoint back to
   Megatron format (`tools/convert_hf_to_torch_dist.py`) and re-stamps it as
   `iter_{t:07d}` so the next segment's `MEGATRON_LOAD` resumes counting from
   the right step. **This is the newest, least-tested code in the whole
   `tools/extrapolation` tree** — see below.

## What is genuinely unverified (read this before running for real)

Unlike the post-hoc tools (whose only real unknown was "no GPU here to test
with"), two of this script's mechanics could not be confirmed by reading
code alone, because the deciding logic lives inside `megatron.bridge`, an
external library this checkout doesn't fully expose:

1. **Does a checkpoint from `tools/convert_hf_to_torch_dist.py`, re-stamped
   with a hand-written `iter_{t:07d}` directory name and
   `latest_checkpointed_iteration.txt`, actually make Megatron resume
   step-counting (and data-sampler shuffling, and `--save-interval`
   bookkeeping) from iteration `t`?** `convert_hf_to_torch_dist.py` wraps
   `megatron.bridge.AutoBridge.save_megatron_model`, written to produce a
   *fresh-start* `MEGATRON_LOAD` (this repo's existing recipes only ever use
   it that way, at iteration ~0) — it exposes no iteration argument, and its
   on-disk layout was never previously something this repo needed to
   inspect. `materialize_accepted_checkpoint()` handles the two layouts that
   seemed plausible (flat `.metadata`, or an `iter_*/.metadata` subdir) and
   self-checks only that the result *looks* like a valid `--load` target
   structurally. It does not and cannot confirm Megatron accepts the
   stamped iteration semantically.
2. **Does `--no-save-optim --no-save-rng` (set in the launcher recipe) mean
   every segment boundary already resets Adam's moment estimates, live run
   or not?** Believed yes, from reading the recipe's own `CKPT_ARGS` — if
   so, this is a *pre-existing* property of the recipe's checkpointing
   choice, not something this script uniquely introduces, and it applies
   uniformly whether or not extrapolation happened at a given boundary. Not
   independently confirmed against Megatron's loader behavior.

**Before trusting a full run**: let `n=0` finish (one real optimizer step —
cheap) and manually check the segment's wandb run / training log confirms
it started from iteration 0 and the *next* segment's log confirms it
resumed from iteration 1, not 0. If the second segment's log shows it
restarting from 0 (or crashes on load), the fix belongs in
`materialize_accepted_checkpoint()`'s layout-detection branch, per its
docstring — stop there rather than let the step-budget silently drift
across all five triggers.

## Cost

Five extrapolation triggers x up to 6 serve+eval cycles each (1 baseline +
5 candidates) = up to 30 SGLang serve/eval round trips, *plus* five fresh
Ray-cluster training-segment startups, *plus* five HF<->Megatron conversions
per trigger. This is meaningfully slower and more GPU-hour-expensive than
either post-hoc tool next door — budget for it accordingly, and consider
`--alphas 2 4` for a first pass (fewer candidates per trigger) before the
full `2 4 6 8 10` sweep once the mechanics above are confirmed working.
