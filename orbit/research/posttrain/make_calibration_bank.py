"""LR-calibration bank and Dev bank in B_target's row format, disjoint from the formal SFT questions.

The formal SFT trains on B_target's first 100x256 rows, so LR calibration must not see those
questions. This samples ``--rows`` openreasoning_mixed_100k rows (duplicate prompts kept, as in
B_target) whose question is in none of B_target / Dev / Eval, with the per-domain counts of
B_target's first ``--rows`` rows, in a seeded order (the offline SFT reader does not shuffle).
It also writes the make_splits Dev questions in the same format. Both files go through
select_cot.py and prepare_sft.py exactly like B_target.

Writes ``calib_btarget.jsonl``, ``dev_btarget.jsonl``, ``calib_train_question_ids.json``, an
overlap audit, and ``calib_sft_data_manifest.json`` (a copy of the formal manifest with the Train
split replaced; fill ``data`` after prepare_sft).

    PYTHONNOUSERSITE=1 python -m orbit.research.posttrain.make_calibration_bank \\
        --b-target /mnt/.../B_target.jsonl --mixed /mnt/.../openreasoning_mixed_100k/train.parquet \\
        --splits-dir /mnt/.../b_target_cot/splits --manifest /mnt/.../b_target_cot/sft_data_manifest.json \\
        --tokenizer /mnt/.../qwen3-1.7b --output-dir /mnt/.../b_target_cot/calibration
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .data import read_jsonl
from .make_splits import qid, write_ids
from .protocol import digest
from .select_cot import user_text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--b-target", required=True, type=Path)
    p.add_argument("--mixed", required=True, type=Path)
    p.add_argument("--splits-dir", required=True, type=Path, help="make_splits.py --output-dir")
    p.add_argument("--manifest", required=True, type=Path, help="the formal SFT data manifest (with splits)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--rows", type=int, default=25600, help="100 steps x 256")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fill-shortfall", action="store_true",
                   help="if a domain has too few unused rows, take its deficit evenly from the other domains")
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    if a.output_dir.exists():
        raise ValueError("Calibration artifacts are immutable; choose a new --output-dir")

    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    mixed = pd.read_parquet(a.mixed, columns=["messages", "domain"])
    texts = [m[0]["content"] for m in mixed["messages"]]
    domains = list(mixed["domain"])

    b_rows = list(read_jsonl(a.b_target))
    b_ids = {qid(user_text(r["prompt"])) for r in b_rows}
    target = Counter(domains[r["canonical_row"]] for r in b_rows[: a.rows])
    dev_rows = list(read_jsonl(a.splits_dir / "dev_rows.jsonl"))
    dev_ids = {r["question_id"] for r in dev_rows}
    eval_ids = set(json.loads((a.splits_dir / "eval_question_ids.json").read_text()))
    banned = b_ids | dev_ids | eval_ids

    pool = {}
    for i, (text, domain) in enumerate(zip(texts, domains)):
        if qid(text) not in banned:
            pool.setdefault(domain, []).append(i)
    rng = random.Random(a.seed)
    counts = dict(target)
    short = {d: counts[d] - len(pool.get(d, [])) for d in counts if counts[d] > len(pool.get(d, []))}
    if short and not a.fill_shortfall:
        raise ValueError(f"not enough unused rows: {short}; pass --fill-shortfall to move the deficit to other domains")
    for d, deficit in short.items():
        counts[d] -= deficit
        donors = sorted(x for x in counts if x not in short)
        for j, x in enumerate(donors):  # split the deficit evenly; the audit records the final mix
            counts[x] += deficit // len(donors) + (j < deficit % len(donors))
    picked = []
    for domain in sorted(counts):
        if len(pool.get(domain, [])) < counts[domain]:
            raise ValueError(f"{domain}: need {counts[domain]} rows, only {len(pool.get(domain, []))} unused")
        picked += rng.sample(pool[domain], counts[domain])
    rng.shuffle(picked)

    def row(i, text):
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return {"prompt": prompt, "prompt_ids": tok.encode(prompt, add_special_tokens=False),
                "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "canonical_row": i}

    a.output_dir.mkdir(parents=True)
    with (a.output_dir / "calib_btarget.jsonl").open("x") as f:
        for i in picked:
            f.write(json.dumps(row(i, texts[i]), ensure_ascii=False) + "\n")
    with (a.output_dir / "dev_btarget.jsonl").open("x") as f:
        for r in dev_rows:
            f.write(json.dumps(row(r["mixed_row"], r["prompt"]), ensure_ascii=False) + "\n")

    calib_ids = {qid(texts[i]) for i in picked}
    train_hash = write_ids(a.output_dir / "calib_train_question_ids.json", calib_ids)
    audit = {"purpose": "LR calibration bank; questions disjoint from the formal B_target Train, Dev and Eval",
             "seed": a.seed, "rows": len(picked), "unique_questions": len(calib_ids),
             "domain_rows": dict(Counter(domains[i] for i in picked)), "matched_to": f"B_target first {a.rows} rows",
             "target_domain_rows": dict(target), "shortfall_moved_to_other_domains": short,
             "overlap": {"b_target": len(calib_ids & b_ids), "dev": len(calib_ids & dev_ids), "eval": len(calib_ids & eval_ids)},
             "calib_btarget_sha256": digest(a.output_dir / "calib_btarget.jsonl"),
             "dev_btarget_sha256": digest(a.output_dir / "dev_btarget.jsonl"),
             "formal_splits_audit_sha256": digest(a.splits_dir / "question_disjoint_audit.json")}
    audit_path = a.output_dir / "calib_question_disjoint_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    m = copy.deepcopy(json.loads(a.manifest.read_text()))
    m["state"], m["data"], m["row_count"] = "unresolved", {"path": "calib_sft_tokens.jsonl", "sha256": None}, None
    m["source"]["dataset"] = m["source"]["dataset"].replace("sealed M6 B_target", "LR-calibration bank (make_calibration_bank.py)")
    m["splits"].update({"train_question_ids_sha256": train_hash, "question_disjoint_audit_sha256": digest(audit_path),
                        "artifacts_dir": str(a.output_dir.resolve())})
    (a.output_dir / "calib_sft_data_manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
