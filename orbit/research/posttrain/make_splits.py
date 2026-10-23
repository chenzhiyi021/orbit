"""Question-disjoint Train/Dev/Eval identity for an SFT bank built on the sealed B_target prompts.

* Train: every B_target prompt (the SFT bank's questions).
* Dev: ``--dev-per-domain`` openreasoning_mixed_100k questions per domain that are neither in
  Train nor in any Eval benchmark (seeded sample). Used for LR calibration / held-out NLL, never
  for choosing checkpoints by benchmark score.
* Eval: the benchmark questions the formal evaluator scores (math500, aime24, aime25, amc23 from
  the Evalchemy tree; gpqa_diamond and LiveCodeBench v5 when their paths are given).

A question id is sha256 of the whitespace-collapsed, lower-cased question text. The audit lists
every exact overlap; Train/Eval overlaps are REPORTED, not removed (the bank is sealed), so the
evaluation can exclude or flag them. Writes the id lists, dev_rows.jsonl and the audit, and fills
``identity`` and ``splits`` in the manifest (``state`` is left for you to set to "ready").

    PYTHONNOUSERSITE=1 python -m orbit.research.posttrain.make_splits \\
        --b-target /mnt/.../B_target.jsonl --mixed /mnt/.../openreasoning_mixed_100k/train.parquet \\
        --evalchemy-root /mnt/.../evalchemy --gpqa-path /mnt/.../gpqa --lcb-path /mnt/.../LCBv5-v2 \\
        --manifest /mnt/.../b_target_cot/sft_data_manifest.json --output-dir /mnt/.../b_target_cot/splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from .data import read_jsonl
from .protocol import digest, load_profile
from .select_cot import user_text

MATH_FILES = {  # same files/fields as the formal evaluator (run_evalchemy_math_eval.py)
    "math500": ("eval/chat_benchmarks/MATH500/data/math500.jsonl", "problem"),
    "aime24": ("eval/chat_benchmarks/AIME24/data/aime24.json", "problem"),
    "aime25": ("eval/chat_benchmarks/AIME25/data/aime25.json", "problem"),
    "amc23": ("eval/chat_benchmarks/AMC23/data/amc23.json", "question"),
}


def qid(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode("utf-8")).hexdigest()


def eval_questions(a) -> dict[str, list[str]]:
    out = {}
    for name, (rel, field) in MATH_FILES.items():
        path = a.evalchemy_root / rel
        if not path.is_file():
            raise FileNotFoundError(f"{name}: missing {path}")
        with path.open(encoding="utf-8") as handle:  # the evaluator's reader: blank lines skipped
            out[name] = [json.loads(line)[field] for line in handle if line.strip()]
    if a.gpqa_path:
        from datasets import load_dataset

        out["gpqa_diamond"] = list(load_dataset(str(a.gpqa_path), "gpqa_diamond")["train"]["Question"])
    if a.lcb_path:
        from datasets import load_dataset

        lcb = load_dataset(str(a.lcb_path), split="test").select_columns(["question_content"])
        out["lcbv5"] = list(lcb["question_content"])
    return out


def write_ids(path: Path, ids) -> str:
    path.write_text(json.dumps(sorted(set(ids)), indent=0) + "\n")
    return digest(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--b-target", required=True, type=Path)
    p.add_argument("--mixed", required=True, type=Path)
    p.add_argument("--evalchemy-root", required=True, type=Path)
    p.add_argument("--gpqa-path", type=Path)
    p.add_argument("--lcb-path", type=Path)
    p.add_argument("--dev-per-domain", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", default="sft_btarget_official_cot")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    if a.output_dir.exists():
        raise ValueError("Split artifacts are immutable; choose a new --output-dir")

    import pandas as pd

    train_text = {qid(t): t for t in (user_text(r["prompt"]) for r in read_jsonl(a.b_target))}
    evals = eval_questions(a)
    eval_ids = {name: {qid(t) for t in qs} for name, qs in evals.items()}
    all_eval = set().union(*eval_ids.values())

    mixed = pd.read_parquet(a.mixed, columns=["messages", "domain", "source_id"])
    pool, seen = {}, set()
    for i, (msgs, domain, sid) in enumerate(zip(mixed["messages"], mixed["domain"], mixed["source_id"])):
        text = msgs[0]["content"]
        k = qid(text)
        if k in train_text or k in all_eval or k in seen:
            continue
        seen.add(k)
        pool.setdefault(domain, []).append({"mixed_row": i, "domain": domain, "source_id": sid, "question_id": k, "prompt": text})
    rng = random.Random(a.seed)
    dev = []
    for domain in sorted(pool):
        if len(pool[domain]) < a.dev_per_domain:
            raise ValueError(f"only {len(pool[domain])} unused {domain} questions for Dev")
        dev += rng.sample(pool[domain], a.dev_per_domain)
    dev_ids = {r["question_id"] for r in dev}

    a.output_dir.mkdir(parents=True)
    hashes = {"train": write_ids(a.output_dir / "train_question_ids.json", train_text),
              "dev": write_ids(a.output_dir / "dev_question_ids.json", dev_ids),
              "eval": write_ids(a.output_dir / "eval_question_ids.json", all_eval)}
    with (a.output_dir / "dev_rows.jsonl").open("x") as f:
        for r in dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    overlap = {name: sorted(ids & train_text.keys()) for name, ids in eval_ids.items()}
    audit = {
        "question_id": "sha256(lower(collapse_whitespace(question_text))); exact-match overlap only",
        "b_target_sha256": digest(a.b_target), "mixed_sha256": digest(a.mixed), "seed": a.seed,
        "counts": {"train": len(train_text), "dev": len(dev_ids), "eval": {n: len(v) for n, v in eval_ids.items()}},
        "dev_per_domain": a.dev_per_domain, "dev_rows_sha256": digest(a.output_dir / "dev_rows.jsonl"),
        "id_file_sha256": hashes,
        "train_dev_overlap": len(train_text.keys() & dev_ids), "dev_eval_overlap": len(dev_ids & all_eval),
        "train_eval_overlap": {n: len(v) for n, v in overlap.items()},
        "train_eval_overlap_examples": {n: [train_text[k][:200] for k in v[:5]] for n, v in overlap.items() if v},
        "train_eval_overlap_ids": overlap,
        "policy": "Train is the sealed bank and is not filtered; Train/Eval overlaps are reported for the evaluator to exclude or flag",
    }
    audit_path = a.output_dir / "question_disjoint_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    m = json.loads(a.manifest.read_text())
    m["identity"] = load_profile(a.profile)["data_identity"]
    m["splits"] = {"train_question_ids_sha256": hashes["train"], "dev_question_ids_sha256": hashes["dev"],
                   "eval_question_ids_sha256": hashes["eval"], "question_disjoint_audit_sha256": digest(audit_path),
                   "artifacts_dir": str(a.output_dir.resolve())}
    a.manifest.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ["counts", "train_dev_overlap", "dev_eval_overlap", "train_eval_overlap"]}, indent=2))
    print(f"manifest identity={m['identity']}; splits filled; state is still {m['state']!r}")


if __name__ == "__main__":
    main()
