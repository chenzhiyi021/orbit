"""HF -> Megatron staging for the live EffOPD training loop.

This is the one genuinely new (not just "reused/wrapped") piece of the live
integration, and the least-verified: converting an *accepted* (possibly
extrapolated) HF checkpoint back into a Megatron torch_dist directory that
`--load`/`MEGATRON_LOAD` will resume from **at a specific claimed iteration
number**, so the next training segment continues counting steps correctly.

`tools/convert_hf_to_torch_dist.py` (this repo) already does the HF->Megatron
weight conversion -- it wraps `megatron.bridge.AutoBridge.save_megatron_model`.
That function exists to produce a *fresh-start* `MEGATRON_LOAD` (iteration
implicitly 0-ish), not a mid-run resume checkpoint, and exposes no iteration
argument. Its on-disk layout is therefore not something this repo previously
needed to inspect or control. `materialize_accepted_checkpoint` below handles
both layouts we might plausibly see (a flat `.metadata` directory, or an
`iter_*/.metadata` subdirectory) and re-stamps whichever one it finds as
`iter_{target_iteration:07d}` with a matching `latest_checkpointed_iteration.txt`,
which is the layout `--load` (and `tools/_megatron_dcp_common.resolve_checkpoint_dir`)
actually resolve against.

**What this does NOT verify**: that Megatron/Megatron-Bridge internally
accepts an externally-stamped iteration number and resumes step-counting,
data-sampler shuffling, and `--save-interval` bookkeeping from it correctly.
The self-check here is structural only (the directory *looks like* a valid
`--load` target). See `live_training/README.md` for how to confirm this
actually worked on your first real segment before trusting the rest of a run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def materialize_accepted_checkpoint(
    hf_dir: Path,
    target_iteration: int,
    accepted_root: Path,
    python_bin: str,
    orbit_root: Path,
) -> Path:
    """Convert `hf_dir` to Megatron format, stamped as iteration `target_iteration`,
    under the shared `accepted_root` directory (one `iter_*` subdir per accepted
    checkpoint across the whole live run, so every segment's starting point stays
    on disk for audit). Returns `accepted_root` -- pass this straight through as
    the next segment's `MEGATRON_LOAD`.
    """
    accepted_root.mkdir(parents=True, exist_ok=True)
    raw_output = accepted_root / f"_raw_import_iter{target_iteration:07d}"
    if raw_output.exists():
        shutil.rmtree(raw_output)

    script = orbit_root / "tools" / "convert_hf_to_torch_dist.py"
    print(f"[megatron-stage] converting {hf_dir} -> Megatron ({raw_output})", flush=True)
    subprocess.run(
        [python_bin, str(script), "--hf-checkpoint", str(hf_dir), "--save", str(raw_output)],
        check=True,
    )

    target_dir = accepted_root / f"iter_{target_iteration:07d}"
    if target_dir.exists():
        shutil.rmtree(target_dir)

    if (raw_output / ".metadata").is_file():
        shutil.move(str(raw_output), str(target_dir))
    else:
        candidates = sorted(p for p in raw_output.glob("iter_*") if (p / ".metadata").is_file())
        if len(candidates) != 1:
            found = sorted(p.name for p in raw_output.iterdir()) if raw_output.exists() else []
            raise RuntimeError(
                f"Don't know how to locate the DCP checkpoint inside {raw_output} "
                f"(expected .metadata directly there, or exactly one iter_*/.metadata "
                f"subdir; found: {found}). This is the unverified layout-detection step "
                "documented in this file's docstring -- inspect the directory manually "
                "and extend the two branches above rather than guess further."
            )
        shutil.move(str(candidates[0]), str(target_dir))
        shutil.rmtree(raw_output, ignore_errors=True)

    (accepted_root / "latest_checkpointed_iteration.txt").write_text(f"{target_iteration}\n")

    # Structural self-check only -- does NOT confirm Megatron will actually
    # resume step-counting / data-sampler state from this correctly, only
    # that the directory has the shape `--load` expects.
    if not (target_dir / ".metadata").is_file():
        raise RuntimeError(f"Self-check failed: {target_dir}/.metadata missing after staging")
    tracker = accepted_root / "latest_checkpointed_iteration.txt"
    if tracker.read_text().strip() != str(target_iteration):
        raise RuntimeError(f"Self-check failed: {tracker} does not read back {target_iteration}")

    print(f"[megatron-stage] staged iteration {target_iteration} at {target_dir}", flush=True)
    return accepted_root
