#!/usr/bin/env python3
"""Dump ChartQA items the synthesiser rejected, so the cause can be classified.

Hand-fixing rejected rows one at a time costs hours for a few percent of the
training set. Classifying thirty of them costs minutes and tells us which change
to the generator recovers hundreds. This writes a review file for that.

It also re-checks the items that DID make it into chartqa_evidence.jsonl for an
operation mismatch: the search accepts any operation that reproduces the gold
number, so a question asking for an average can be answered by a sum of two
cells that happens to equal it. Those are in the training set right now.

    python code/data/audit_chartqa_failures.py --limit 30
    python code/data/audit_chartqa_failures.py --check-accepted

Writes training_data/evidence/chartqa_audit.md (human-readable) and
chartqa_audit.json (the same rows, for a follow-up fix).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data"))

import synth_evidence_chartqa as synth  # noqa: E402

OUT = ROOT / "training_data" / "evidence"

# Words that name an operation in a question, and the operation they imply.
OPERATION_WORDS = {
    "average": "mean", "mean": "mean",
    "sum": "sum", "total": "sum", "combined": "sum", "add": "sum",
    "difference": "difference", "gap": "difference", "more than": "difference",
    "how many more": "difference", "minus": "difference",
    "ratio": "ratio", "times": "ratio", "per": "ratio",
    "highest": "max", "largest": "max", "maximum": "max", "lowest": "min",
    "smallest": "min", "minimum": "min",
    "how many": "count", "number of": "count",
    "range": "range",
}


def implied_operation(question: str):
    q = question.lower()
    hits = {op for word, op in OPERATION_WORDS.items() if word in q}
    return hits


def check_accepted(path: pathlib.Path):
    """Find accepted arithmetic rows whose operation contradicts the question."""
    mismatches, checked = [], 0
    if not path.is_file():
        return None, 0
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        meta = row.get("meta") or {}
        if meta.get("evidence_kind") != "arithmetic":
            continue
        checked += 1
        used = (meta.get("evidence_verification") or "").replace("recomputed_", "")
        implied = implied_operation(row["question"])
        if implied and used and used not in implied:
            mismatches.append({
                "id": row["id"], "question": row["question"], "answer": row["answer"],
                "operation_used": used, "operations_the_question_implies": sorted(implied),
                "reasoning": meta.get("reasoning", [""])[0],
            })
    return mismatches, checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=30, help="rejected rows to dump")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--check-accepted", action="store_true",
                        help="only re-check what is already in the evidence file")
    args = parser.parse_args()

    accepted_path = OUT / "chartqa_evidence.jsonl"
    mismatches, checked = check_accepted(accepted_path)
    if mismatches is None:
        print(f"note: {accepted_path.name} not found; skipping the accepted-row check")
    else:
        print(f"accepted arithmetic rows checked : {checked}")
        print(f"  operation contradicts question : {len(mismatches)} "
              f"({100*len(mismatches)/max(checked,1):.1f}%)")
        for m in mismatches[:5]:
            print(f"    {m['id']}")
            print(f"      Q: {m['question'][:88]}")
            print(f"      used {m['operation_used']}, question implies "
                  f"{m['operations_the_question_implies']}")
    if args.check_accepted:
        (OUT / "chartqa_accepted_mismatches.json").write_text(
            json.dumps(mismatches or [], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {OUT / 'chartqa_accepted_mismatches.json'}")
        return 0

    # --- rebuild the rejected set, keeping the table so it can be reviewed ----
    index = {}
    for kind in ("human", "augmented"):
        qa_path = synth.ORIG / "train" / f"train_{kind}.json"
        if not qa_path.is_file():
            continue
        for entry in json.loads(qa_path.read_text(encoding="utf-8")):
            key = (str(entry["query"]).strip().lower(), str(entry["label"]).strip().lower())
            index.setdefault(key, (entry["imgname"].rsplit(".", 1)[0], kind))

    records = [json.loads(l) for l in synth.CAND.open(encoding="utf-8") if l.strip()]
    tables, rejected = {}, []
    reasons = collections.Counter()
    for record in records:
        key = (str(record["question"]).strip().lower(), str(record["answer"]).strip().lower())
        link = index.get(key)
        if not link:
            reasons["no_chart_link"] += 1
            continue
        chart, qkind = link
        if chart not in tables:
            tables[chart] = synth.load_table("train", chart)
        table = tables[chart]
        if table is None:
            reasons["no_table_file"] += 1
            continue
        gold_text = str(record["answer"]).strip()
        gold_num = synth.to_number(gold_text)
        # This must mirror the generator's routing exactly. It did not once:
        # `label_extremum_evidence` was added to the generator and not here, so
        # the audit kept reporting 2,354 rejected rows after 557 of them had
        # started succeeding. Any new evidence kind goes in both places.
        question_lower = str(record["question"]).strip().lower()
        built, _ = synth.cell_lookup_evidence(record, table, gold_num, gold_text)
        if built is not None:
            continue
        built, _ = synth.label_extremum_evidence(
            record, table, gold_text, question=question_lower)
        if built is not None:
            continue
        built, why = synth.arithmetic_evidence(
            record, table, gold_num, question=question_lower)
        if built is not None:
            continue
        reasons[why] += 1
        rejected.append({
            "id": record["id"], "chart": chart, "question_kind": qkind,
            "question": record["question"], "gold": gold_text,
            "reject_reason": why,
            "operations_the_question_implies": sorted(implied_operation(record["question"])),
            "table_header": table["header"],
            "table_cells": [{"row": c["row"], "col": c["col"], "text": c["text"]}
                            for c in table["cells"][:24]],
        })

    print(f"\nrejected rows: {len(rejected):,}")
    for why, n in reasons.most_common():
        print(f"  {why:52} {n:>6,}")

    rng = random.Random(args.seed)
    rng.shuffle(rejected)
    sample = rejected[:args.limit]

    lines = [
        "# ChartQA: rejected rows, for cause classification",
        "",
        "For each row below, mark the cause with one of:",
        "",
        "- **(a) operation missing** - the right cells are in the table, but the",
        "  operation the question needs is not one the generator tries",
        "  (it tries: difference, sum, ratio, and per-column sum/mean/max/min/range/count)",
        "- **(b) label mismatch** - the right cells are there, but the question names them",
        "  differently from the table (\"Turkey\" vs \"Turkey, Rep. of\")",
        "- **(c) not derivable** - the answer cannot be got from this table at all",
        "  (the chart shows something the table does not, or the table is wrong)",
        "",
        "Thirty of these is enough. (a) and (b) are generator changes that recover",
        "hundreds of rows at once; (c) cannot be fixed and should stay rejected.",
        "",
        f"Sampled {len(sample)} of {len(rejected):,} rejected rows, seed {args.seed}.",
        "",
        "---",
        "",
    ]
    for i, row in enumerate(sample, 1):
        lines += [
            f"## {i}. `{row['id']}`  (chart {row['chart']}, {row['question_kind']})",
            "",
            f"**Question:** {row['question']}",
            "",
            f"**Gold answer:** `{row['gold']}`",
            "",
            f"**Rejected because:** `{row['reject_reason']}`",
            f"**Operation words in the question:** "
            f"{row['operations_the_question_implies'] or 'none found'}",
            "",
            f"**Table columns:** {row['table_header']}",
            "",
            "| row | column | value |",
            "|---|---|---|",
        ]
        for cell in row["table_cells"]:
            lines.append(f"| {cell['row']} | {cell['col']} | {cell['text']} |")
        if len(row["table_cells"]) >= 24:
            lines.append("| ... | (truncated at 24 cells) | |")
        lines += ["", "**Cause:** (a) / (b) / (c)  -> ", "", "---", ""]

    (OUT / "chartqa_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "chartqa_audit.json").write_text(
        json.dumps({"sampled": sample, "reject_reasons": dict(reasons.most_common()),
                    "total_rejected": len(rejected), "seed": args.seed},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT / 'chartqa_audit.md'}  ({len(sample)} rows to classify)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
