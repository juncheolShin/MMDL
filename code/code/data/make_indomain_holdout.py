#!/usr/bin/env python3
"""Carve an in-domain holdout out of the candidate pool, for the Phase 2 gate.

DECISIONS.md section 7 gates the main matrix on "도메인 내 holdout 상승": if a
trained arm does not beat the base model on held-out rows from the SAME
distribution, the matrix must not run. That holdout did not exist yet - the
candidate pool was built for training and every row in it carries
`holdout: false` (the `holdout: true` rows are ai2d/test and chartqapro, which
are transfer benchmarks, not in-domain).

The holdout is the part of the candidate pool the arm did not train on, minus
image leakage:

  * AI2D asks several questions about one diagram and ChartQA several about one
    chart. Excluding only the trained QUESTION would leave the model evaluated
    on a picture it has already seen, which is the thing section 7 Phase 1
    planned an image-hash split to avoid. So an image that appears anywhere in
    the training set disqualifies every row that uses it.
  * Base predictions for every candidate row already exist from Phase 0
    (results/phase0/indomain_split.json), so the same rows give a paired
    base-vs-arm comparison with no extra inference.

    python code/data/make_indomain_holdout.py --sft training_data/sft/arm_a
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CAND = ROOT / "training_data" / "candidates"
SPLIT = ROOT / "results" / "phase0" / "indomain_split.json"


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sft", type=pathlib.Path, required=True,
                    help="the training set to hold out against (its train.jsonl)")
    ap.add_argument("--candidates", type=pathlib.Path, default=CAND)
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "training_data" / "candidates_holdout")
    ap.add_argument("--sources", default="ai2d_original,chartqa",
                    help="only these sources; a source the arm never trained on "
                         "measures transfer, not in-domain performance")
    args = ap.parse_args()

    train_path = args.sft / "train.jsonl"
    if not train_path.is_file():
        sys.exit(f"make_indomain_holdout: {train_path} not found")

    trained_uids, trained_images = set(), set()
    for row in read_jsonl(train_path):
        trained_uids.add(row["uid"])
        if row.get("image"):
            trained_images.add(row["image"])
    print(f"training set: {len(trained_uids):,} rows over "
          f"{len(trained_images):,} distinct images")

    base = {}
    if SPLIT.is_file():
        for entry in json.loads(SPLIT.read_text(encoding="utf-8")):
            base[entry["id"]] = entry.get("correct")
        print(f"base predictions: {len(base):,} rows from {SPLIT.name}")
    else:
        print(f"WARNING: {SPLIT} missing - the holdout will carry no base scores",
              file=sys.stderr)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    report, kept_total = [], 0

    for source in sources:
        src = args.candidates / f"{source}.jsonl"
        if not src.is_file():
            sys.exit(f"make_indomain_holdout: {src} not found")
        drop = collections.Counter()
        kept, base_correct, base_scored = [], 0, 0
        for record in read_jsonl(src):
            if record["id"] in trained_uids:
                drop["trained_on_this_question"] += 1
                continue
            if record.get("image") in trained_images:
                drop["image_seen_in_training"] += 1
                continue
            kept.append(record)
            if record["id"] in base:
                base_scored += 1
                base_correct += bool(base[record["id"]])
        out = args.out / f"{source}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        acc = 100 * base_correct / base_scored if base_scored else float("nan")
        kept_total += len(kept)
        report.append({"source": source, "kept": len(kept),
                       "dropped": dict(drop), "base_scored": base_scored,
                       "base_correct": base_correct, "base_accuracy_pct": acc})
        print(f"\n{source}")
        print(f"  kept                      {len(kept):>6,}")
        for reason, n in drop.most_common():
            print(f"  dropped {reason:<25} {n:>6,}")
        print(f"  base accuracy on the kept rows: {base_correct:,}/{base_scored:,} "
              f"= {acc:.2f}%")

    manifest = args.out / "holdout_manifest.json"
    manifest.write_text(json.dumps({
        "built_from_sft": str(args.sft),
        "sources": report,
        "total_rows": kept_total,
        "rule": "a candidate row is held out when neither its question nor its "
                "image appears anywhere in the training set",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nholdout: {kept_total:,} rows -> {args.out}")
    print(f"manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
