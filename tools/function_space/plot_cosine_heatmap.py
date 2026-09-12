"""Cosine-of-centered-logit-deltas heatmap, generalized from the trl-side
``p4_function_space_20260907/analyze.py``.

Differences from the trl version:
  * ``RUNS`` is not hardcoded to P4/M6 -- pass ``--runs`` (comma-separated,
    in display order) or omit it to auto-discover every run present in the
    scored bank directory (endpoint / max step per run).
  * No "historical cache vs freshly recomputed base" sensitivity block --
    that was specific to comparing P4's fresh scores against orbit's
    pre-existing M6 sketch cache. Here every run (trl- or orbit-trained) is
    scored by the same ``score_checkpoints.py`` in the same pass, so there
    is only one base per bank, not two. If you *do* mix in a pre-existing
    historical ``.npz`` cache for some runs, point ``--extra-cache`` at its
    directory and this script will pick up any ``{run}_step*.npz`` files
    found there too.
  * Everything else (CountSketch gate, bootstrap CI, per-domain breakdown,
    the plot itself) is unchanged from analyze.py.
  * ``--space {logit,prob}`` (new, not in the trl version): logit (default)
    reproduces the original centered-logit-space cosine unchanged. prob
    sketches post-softmax probabilities instead -- run
    ``score_checkpoints.py ... --space prob`` first, then pass the same
    ``--space prob`` here. Use this to check whether a clustering found in
    logit space survives once near-zero-probability tail tokens stop being
    weighted equally with head tokens.

Usage:
    python plot_cosine_heatmap.py --config config.json --bank bank_a \\
        --runs "M6-FullFT-trl,M6-FullFT-orbit,M6-OFT,M6-LoRA"
    python plot_cosine_heatmap.py --config config.json --bank bank_a --space prob \\
        --runs "M6-FullFT-trl,M6-FullFT-orbit,M6-OFT,M6-LoRA"
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from relations import Record, bootstrap_cosine_distance, cosine_distance, seed_cos, seed_norm, sketch_gate


def load(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        return archive["sketches"].astype(np.float32)


def md_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x) for x in row) + " |")
    return "\n".join(rows)


def discover_runs(bank_dir: Path, extra_cache: Path | None) -> dict[str, Path]:
    """Latest (max-step) checkpoint file per run name found in bank_dir
    (and, if given, extra_cache too)."""
    candidates: dict[str, Path] = {}
    steps: dict[str, int] = {}
    for base_dir in (bank_dir, extra_cache) if extra_cache else (bank_dir,):
        if base_dir is None or not base_dir.exists():
            continue
        for path in base_dir.glob("*_step*.npz"):
            if path.stat().st_size == 0:
                continue
            name, _, rest = path.stem.rpartition("_step")
            if not rest.isdigit() or not name:
                continue
            step = int(rest)
            if name not in steps or step > steps[name]:
                steps[name] = step
                candidates[name] = path
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", required=True, help="key into config['banks']")
    parser.add_argument("--runs", default=None, help="comma-separated run names, display order; default: auto-discover")
    parser.add_argument("--extra-cache", type=Path, default=None, help="extra directory of {run}_step*.npz to merge in (e.g. a historical cache)")
    parser.add_argument("--space", choices=("logit", "prob"), default=None,
                         help="overrides config['space'] (default 'logit'). Must match the "
                              "--space you used with score_checkpoints.py for this bank -- "
                              "reads from <output_root>/<bank>/ for 'logit' (unchanged default "
                              "path) or <output_root>/<bank>/prob/ for 'prob'.")
    parser.add_argument("--output", type=Path, default=None, help="default: config['output_root']/<bank>[/<space>]/heatmap")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    space = args.space or config.get("space", "logit")
    bank_dir = Path(config["output_root"]) / args.bank
    if space != "logit":
        bank_dir = bank_dir / space
    assert (bank_dir / "COMPLETE.json").exists(), (
        f"Run score_checkpoints.py --bank {args.bank}"
        f"{'' if space == 'logit' else f' --space {space}'} first (looked in {bank_dir})"
    )
    results = args.output or (bank_dir / "heatmap")
    results.mkdir(parents=True, exist_ok=True)

    run_paths = discover_runs(bank_dir, args.extra_cache)
    if args.runs:
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]
        missing = [r for r in runs if r not in run_paths]
        if missing:
            raise KeyError(f"Requested runs not found in {bank_dir} (or --extra-cache): {missing}. "
                            f"Available: {sorted(run_paths)}")
    else:
        runs = sorted(run_paths)
    if len(runs) < 2:
        raise RuntimeError(f"Need >=2 scored runs to build a heatmap, found {runs} in {bank_dir}")

    arrays = {run: load(run_paths[run]) for run in runs}
    expected_shape = arrays[runs[0]].shape
    for run, array in arrays.items():
        assert array.shape == expected_shape and np.isfinite(array).all(), (run, array.shape)
    num_prompts = expected_shape[0]
    rng = np.random.default_rng(config.get("bootstrap_seed", 20260826))
    weights = rng.multinomial(num_prompts, np.full(num_prompts, 1 / num_prompts), size=1000).astype(np.float64)

    scalar_metadata = None
    scalar_path = (run_paths[runs[0]]).with_suffix(".scalars.parquet")
    if scalar_path.exists():
        scalar_metadata = pd.read_parquet(scalar_path)

    pair_rows, domain_rows = [], []
    for left, right in itertools.combinations(runs, 2):
        a, b = arrays[left], arrays[right]
        cos_d, nl2 = cosine_distance(a, b)
        boot = 1 - bootstrap_cosine_distance(a, b, weights)
        seeds = seed_cos(a, b)
        pair_rows.append(dict(bank=args.bank, left=left, right=right, cosine=1 - cos_d,
            cosine_ci_low=float(np.quantile(boot, .025)), cosine_ci_high=float(np.quantile(boot, .975)),
            min_seed_cosine=float(seeds.min()), max_seed_cosine=float(seeds.max()),
            cosine_distance=cos_d, symmetric_normalized_l2=nl2))
        if scalar_metadata is not None and "domain" in scalar_metadata.columns:
            for domain in sorted(scalar_metadata.domain.unique()):
                select = (scalar_metadata.domain == domain).to_numpy()
                distance, d_l2 = cosine_distance(a[select], b[select])
                domain_rows.append(dict(bank=args.bank, domain=domain, prompts=int(select.sum()), left=left,
                                        right=right, cosine=1 - distance, symmetric_normalized_l2=d_l2))
        print(f"PAIR {args.bank} {left} {right} cosine={1 - cos_d:.6f}", flush=True)

    matrix = pd.DataFrame(np.eye(len(runs)), index=runs, columns=runs)
    for row in pair_rows:
        matrix.loc[row["left"], row["right"]] = matrix.loc[row["right"], row["left"]] = row["cosine"]

    gate_frame, gate = sketch_gate([Record(run, 0, run_paths[run]) for run in runs])

    pd.DataFrame(pair_rows).to_csv(results / "pairwise.csv", index=False)
    matrix.to_csv(results / "cosine_matrix.csv")
    if domain_rows:
        pd.DataFrame(domain_rows).to_csv(results / "pairwise_by_domain.csv", index=False)
    gate_frame.to_csv(results / "countsketch_exact_gate.csv", index=False)
    (results / "VALIDATION.json").write_text(json.dumps({"gate": gate, "runs": runs, "space": space}, indent=2) + "\n")

    space_desc = ("Vocabulary-centered logits minus Base" if space == "logit"
                  else "Post-softmax probabilities minus Base (no centering needed -- "
                       "probability vectors already share the same mean, 1/V)")
    space_limit = (
        "- This is centered-*logit*-space cosine, not probability-space. Near-zero-probability "
        "tail tokens get equal weight to head tokens here; rerun with --space prob (and "
        "score_checkpoints.py --space prob first) to check whether a finding here survives in "
        "probability space."
        if space == "logit" else
        "- This is *probability*-space cosine: near-zero-probability tail tokens are automatically "
        "down-weighted (their contribution to any delta is ~0), unlike the centered-logit version "
        "(--space logit, the default) which weights every vocabulary token equally regardless of "
        "how likely it is. Compare both if a clustering only shows up in one of them."
    )
    report = [f"# Function-space cosine heatmap: {args.bank} ({space} space)", "",
        f"Runs: {', '.join(runs)}.", "",
        f"{space_desc}, CountSketch seeds 3407/3408/3409, dimension "
        f"{config.get('sketch_dimension', 1024)}. Similarity is the mean of three seed-wise cosine "
        "similarities of the deltas. CI: 1000 prompt-cluster bootstrap samples.", "",
        "## Cosine matrix", "", md_table(matrix.reset_index(names="run")), "",
        "## CountSketch validation gate (exact vs sketch, 16-position slice)", "",
        "```json", json.dumps(gate, indent=2), "```", "",
        "## Interpretation limits", "",
        "- Cosine measures alignment of *changes*, not checkpoint accuracy or "
        "probability-distribution identity.",
        space_limit,
        "- If mixing runs scored by different codebases/environments, check each bank's "
        "`baseline_parity.json` (written by score_checkpoints.py) before trusting cross-run cosines.",
        ""]
    (results / "REPORT.md").write_text("\n".join(report))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.6 * len(runs) + 2, 1.4 * len(runs) + 1), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(runs)), runs, rotation=35, ha="right")
    ax.set_yticks(range(len(runs)), runs)
    title_space = "centered-logit deltas" if space == "logit" else "probability deltas"
    ax.set_title(f"{args.bank}: cosine of {title_space}")
    for i in range(len(runs)):
        for j in range(len(runs)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.savefig(results / "function_cosine.png", dpi=170)
    print(json.dumps({"report": str(results / "REPORT.md"), "png": str(results / "function_cosine.png"),
                       "pairs": len(pair_rows), "complete": True}))


if __name__ == "__main__":
    main()
