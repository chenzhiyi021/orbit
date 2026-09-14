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

**Update from a real two-segment run**: iteration counting **resets to 0 on
every `--load`** in this recipe, full stop — it is not a running total
across segments (confirmed, not assumed: segment n=0, from the untouched
pretrained base, NUM_ROLLOUT=1, saved at `iteration 0`; segment n=1, from
this script's own `materialize_accepted_checkpoint()` output, ALSO
NUM_ROLLOUT=1, saved at `iteration 0` again — a global/cumulative counter
would have predicted `1`). Almost certainly because `--no-save-optim
--no-save-rng` (set in the launcher recipe) makes every checkpoint here a
weights-only, finetune-style load, and Megatron's classic finetune
semantics ignore the stored iteration and restart counting at 0. Fixed via
`local_iteration_for(num_rollout) = num_rollout - 1`, which this script's
own `accepted_iteration` bookkeeping (a plain Python variable, never
dependent on Megatron's counter) turns into the correct real step budget
regardless. See `local_iteration_for()`'s docstring for the full account.

**A more serious, related, and still-open finding from reading orbit's own
source** (not yet observed directly, but the code is unambiguous):
`orbit/rollout/data_source.py`'s `RolloutDataSource` tracks which prompts
have been consumed (`epoch_id`, `sample_offset`) in a *separate* file,
`<save>/rollout/global_dataset_state_dict_{rollout_id}.pt`, loaded via
`orbit/ray/placement_group.py:208`'s `rollout_manager.load(...)`.
`materialize_accepted_checkpoint()` only ever writes Megatron DCP weights —
it never produces this file. Combined with the iteration-reset behavior
above, **every segment almost certainly restarts the rollout data sampler
from the same shuffled position** (`epoch_id=0, sample_offset=0`, same fixed
`--rollout-seed`) rather than continuing on to unseen prompts, meaning a
6-segment live run may repeatedly train on overlapping/identical prompt
batches rather than genuinely advancing through the corpus 20 steps' worth.
This is a **pre-existing property of orbit's stop-and-resume mechanism**,
not something this tool's extrapolation logic introduces — any workflow
that stops and restarts training via `--load` in this codebase would hit
it. It was not chased down further (would mean copying/renaming
`<segment_dir>/rollout/global_dataset_state_dict_*.pt` into
`accepted_megatron_root` with a filename `local_iteration_for()`'s finding
implies the *next* segment will look for, which is inferred, not confirmed,
from `orbit/backends/megatron_utils/actor.py:222`'s
`start_rollout_id = loaded_rollout_id + 1` combined with
`orbit/utils/arguments.py:3545`'s `args.start_rollout_id = 0` default —
plausible that `start_rollout_id` never gets updated from `loaded_rollout_id`
in this code path at all, which would be a second, independent reason for
the same symptom). **Decide explicitly whether repeated/overlapping prompt
batches are acceptable for this exploratory pass before reading too much
into a full run's final accuracy** — the step *count* will be right, the
data *diversity* behind those steps may not be.

**Before trusting a full run**: watch training logs for whether each
segment's rollout prompts actually differ (or log a hash/first-token of the
batch per segment and diff them) rather than just checking the checkpoint's
iteration label — the label is now expected to always read `0`;
back-to-back segments training on identical batches is the thing actually
worth catching.

## Cost

Five extrapolation triggers x up to 6 serve+eval cycles each (1 baseline +
5 candidates) = up to 30 SGLang serve/eval round trips, *plus* five fresh
Ray-cluster training-segment startups, *plus* five HF<->Megatron conversions
per trigger. This is meaningfully slower and more GPU-hour-expensive than
either post-hoc tool next door — budget for it accordingly, and consider
`--alphas 2 4` for a first pass (fewer candidates per trigger) before the
full `2 4 6 8 10` sweep once the mechanics above are confirmed working.
