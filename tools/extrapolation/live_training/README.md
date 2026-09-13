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
    --validator-cmd "python tools/extrapolation/validate_checkpoint.py \
        --evalchemy-root /mnt/L202500431/third_party/evalchemy \
        --task aime24 --num-samples 50 --num-gpus 2 --eval-tp-size 1" \
    --live-root live_training_runs/effopd_lr_2e-6 \
    --total-steps 20
```

Every path arg now has a default confirmed against this cluster's actual
layout (`ls ../models`, `ls ../models/megatron_ckpt`, `ls ../datasets`,
`ls ../datasets/openreasoning_mixed_100k`):

| flag | default |
|---|---|
| `--base-model` | `/mnt/L202500431/models/qwen3-1.7b` |
| `--megatron-base` | `/mnt/L202500431/models/megatron_ckpt/qwen3-1.7b` |
| `--train-jsonl` | `/mnt/L202500431/datasets/openreasoning_mixed_100k/train.parquet` |
| `--teacher-hf-ckpt` | `/mnt/L202500431/models/qwen3-4b-instruct-2507` |
| `--evalchemy-root` | `/mnt/L202500431/third_party/evalchemy` |

Override any of them (`--train-jsonl`, etc.) if you want a different
dataset/teacher for this study.

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

**Update from the first real run**: iteration numbering here is
**0-indexed**, not 1-indexed as this script originally assumed — confirmed
from an actual segment-1 log: `NUM_ROLLOUT=1` from the untouched pretrained
base saved its checkpoint as `iteration 0`, not `iteration 1`. Fixed via
`megatron_iteration_for(step_count) = step_count - 1`, applied everywhere a
real "N steps completed" count gets turned into the iteration number this
stack actually saves/loads under. This also retroactively explains the
existing lr-sweep runs' odd-numbered checkpoints (`iter_0000001,
iter_0000003, ..., iter_0000019` for a 20-step, save-every-2 run) that this
tree next door had to work around with an "ordinal position" convention
instead of raw iteration numbers.

Two things are *still* unconfirmed by anything short of watching a real
run, because the deciding logic lives inside `megatron.bridge`, an external
library this checkout doesn't fully expose:

1. **Does a checkpoint from `tools/convert_hf_to_torch_dist.py`, re-stamped
   with a hand-written `iter_{t-1:07d}` directory name and
   `latest_checkpointed_iteration.txt`, actually make Megatron resume
   step-counting (and data-sampler shuffling, and `--save-interval`
   bookkeeping) from that iteration?** The segment-1 evidence above confirms
   the *numbering convention*, but segment 1 loaded the untouched pretrained
   base, not something `materialize_accepted_checkpoint()` produced — the
   HF->Megatron round-trip and hand-written tracker file are exercised for
   the first time at the n=0 -> n=1 transition. `materialize_accepted_checkpoint()`
   self-checks only that its output *looks* like a valid `--load` target
   structurally; it cannot confirm Megatron accepts the stamped iteration
   semantically.
2. **Does `--no-save-optim --no-save-rng` (set in the launcher recipe) mean
   every segment boundary already resets Adam's moment estimates, live run
   or not?** Believed yes, from reading the recipe's own `CKPT_ARGS` — if
   so, this is a *pre-existing* property of the recipe's checkpointing
   choice, not something this script uniquely introduces, and it applies
   uniformly whether or not extrapolation happened at a given boundary. Not
   independently confirmed against Megatron's loader behavior.

**Before trusting a full run**: watch the n=0 -> n=1 transition specifically
— the first segment that resumes from a `materialize_accepted_checkpoint()`
output rather than the raw pretrained base. Confirm its training log shows
`saving checkpoint at iteration       1` (not `0` again) after that segment.
If it shows `0` again (or crashes on load), the fix belongs in
`materialize_accepted_checkpoint()`'s layout-detection branch, per its
docstring — stop there rather than let the step-budget silently drift
across the remaining triggers.

## Cost

Five extrapolation triggers x up to 6 serve+eval cycles each (1 baseline +
5 candidates) = up to 30 SGLang serve/eval round trips, *plus* five fresh
Ray-cluster training-segment startups, *plus* five HF<->Megatron conversions
per trigger. This is meaningfully slower and more GPU-hour-expensive than
either post-hoc tool next door — budget for it accordingly, and consider
`--alphas 2 4` for a first pass (fewer candidates per trigger) before the
full `2 4 6 8 10` sweep once the mechanics above are confirmed working.
