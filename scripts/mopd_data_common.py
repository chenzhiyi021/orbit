"""Shared helpers for building per-domain / mixed MOPD training & eval data.

All scripts in the `mopd_prepare_*` / `mopd_mix_train` family write a single
common row schema so files from different sources can be concatenated
without a schema-reconciliation step:

    {"prompt": <str>, "label": <str>, "metadata": {...}}

`metadata` always carries at least `domains` and `rm_type`; per-reward-model
extra fields (e.g. `instruction_id_list`/`kwargs`/`prompt_text`/`record_id`
for the "ifbench" rm_type) are added on top.

Output is always JSONL (never parquet) specifically to avoid pyarrow struct-
schema inference clashing across domains whose `metadata` dicts have
different keys/types -- see the discussion in the MOPD data-mix design notes.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas artifacts (from parquet columns) into
    plain JSON-serializable python types.

    Parquet list/struct columns round-trip through pandas as numpy arrays of
    numpy scalars (e.g. `instruction_id_list` as `ndarray[object]`, `kwargs`
    entries as `numpy.int64`), which `json.dumps` cannot serialize directly.
    """
    if np is not None:
        if isinstance(value, np.ndarray):
            return [_to_jsonable(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            value = value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if value is None:
        return None
    if pd is not None:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and value.is_integer():
        # Parquet round-trips commonly upcast an int column to float64 once
        # any row is null (e.g. IFEval-style kwargs share one struct column
        # across instruction types -- n_start/n_end are int for
        # RepeatSpanChecker rows, null for every other instruction type in
        # the same column). A whole-number float slipping through as e.g.
        # 3.0 breaks strict-int consumers such as IFBench's
        # RepeatSpanChecker, which slices a string with it directly
        # (`str[3.0:16.0]` raises TypeError, not caught by any `is None`
        # check upstream).
        return int(value)
    return value


def iter_records(path: str) -> Iterator[dict]:
    """Yield rows from a .jsonl or .parquet file as plain dicts."""
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield _to_jsonable(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[mopd_data] JSON decode error at {path}:{line_num}: {e}")
                    continue
    elif path.endswith(".parquet"):
        if pd is None:
            raise ImportError("pandas is required to read parquet files")
        df = pd.read_parquet(path)
        for record in df.to_dict("records"):
            yield {k: _to_jsonable(v) for k, v in record.items()}
    else:
        raise ValueError(f"Unsupported input format: {path} (expected .jsonl or .parquet)")


def make_record(prompt: str, label: Any, metadata: dict) -> dict:
    return {"prompt": prompt, "label": "" if label is None else str(label), "metadata": _to_jsonable(metadata)}


def write_jsonl(records: Iterable[dict], path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def git_commit_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            check=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


TOP_LEVEL_README = """\
# orbit_mopd

Derived data for multi-teacher on-policy distillation (MOPD) experiments.
Everything under this directory is *generated* from the raw dataset dumps
that live alongside it (`../gsm8k`, `../Nemotron-RL-instruction_following`,
`../IFBench_test`, ...) -- nothing here should be hand-edited, regenerate it
with the `scripts/mopd_prepare_*` / `scripts/mopd_mix_train.py` tools instead.

## Layout

- `domains/` -- one file per (dataset, domain), converted to the common
  orbit row schema `{"prompt", "label", "metadata"}` with `metadata.domains`
  / `metadata.rm_type` set. These are reusable building blocks; regenerating
  a mix does not require re-running the source conversion.
- `mixes/` -- concatenated training files actually passed to `--prompt-data`.
  Each `<name>.jsonl` has a sibling `<name>.README.md` recording exactly
  which domain files (and row limits/commit) went into it. Never overwrite
  a mix in place -- bump the trailing `_vN` and keep the old one so past
  runs stay reproducible.

## Naming convention

`train_<domain1>-<domain2>[-...]_vN.jsonl`, where the domain names match
`metadata.domains` values used in that file and the `--mopd-teacher-configs`
`domains` field they should route to. Eval datasets are NOT mixed (each
stays its own file, referenced separately from `--eval-config`), so only
training files live under `mixes/`.
"""


def ensure_top_level_readme(orbit_mopd_dir: str) -> None:
    """Write orbit_mopd/README.md if it doesn't already exist (never clobber)."""
    path = os.path.join(orbit_mopd_dir, "README.md")
    if os.path.exists(path):
        return
    os.makedirs(orbit_mopd_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(TOP_LEVEL_README)
    print(f"[mopd_data] wrote {path}")


def write_mix_manifest(
    manifest_path: str,
    *,
    mix_name: str,
    output_path: str,
    sources: list[dict],
    total_rows: int,
    shuffled: bool,
    seed: int | None,
) -> None:
    """Write a <name>.README.md manifest describing how a mix was built."""
    lines = [
        f"# {mix_name}",
        "",
        f"- generated: {now_iso()}",
        f"- generator: `scripts/mopd_mix_train.py` @ commit `{git_commit_short()}`",
        f"- output: `{output_path}`",
        f"- total rows: {total_rows}",
        f"- shuffled on disk: {shuffled}" + (f" (seed={seed})" if shuffled else ""),
        "",
        "## Sources",
        "",
    ]
    for src in sources:
        lines.append(f"- domain `{src['domains']}` (rm_type=`{src['rm_type']}`)")
        lines.append(f"  - file: `{src['path']}`")
        lines.append(f"  - rows used: {src['rows_used']}" + (f" / {src['rows_total']} available" if src.get("rows_total") is not None else ""))
    lines.append("")
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[mopd_data] wrote {manifest_path}")
