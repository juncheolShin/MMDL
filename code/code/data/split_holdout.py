#!/usr/bin/env python3
"""Image-level 9:1 split of the candidate pool, done BEFORE any selection (D12).

The order matters and it was wrong the first time. What happened: the training
set was chosen first, wrong-first (base-model mistakes taken preferentially),
and the holdout was then whatever was left over. That leaves the holdout
enriched in rows the base model already answers correctly - AI2D base accuracy
was 75.58% across the pool but 92.33% on the leftover holdout, so an arm had
7.67pp of headroom to show anything in.

The fix is to split first and stratify second:

    candidate pool
      |- 90%  train side   <- wrong-first selection happens only in here
      '- 10%  holdout      <- untouched, same difficulty as the pool

Grouping is by IMAGE, never by question. AI2D averages several questions per
diagram, so splitting on questions would put the same picture on both sides.
Assignment is a hash of the image path, so it is deterministic, stable when the
pool grows, and independent of row order.

This writes to NEW directories and does not touch `candidates_holdout/`, which
the arm A diagnostics are being scored against. Switch to these once the data is
rebuilt.

    python code/data/split_holdout.py
    python code/data/split_holdout.py --holdout-frac 0.1 --dry-run

Reads  training_data/candidates/<source>.jsonl
Writes training_data/candidates_split/train/<source>.jsonl
       training_data/candidates_split/holdout/<source>.jsonl
       training_data/candidates_split/split_manifest.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CAND = ROOT / "training_data" / "candidates"
OUT = ROOT / "training_data" / "candidates_split"
SPLIT = ROOT / "results" / "phase0" / "indomain_split.json"


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def in_holdout(image: str, frac: float, salt: str) -> bool:
    """Deterministic, image-keyed. The same image always lands on the same side,
    whatever order the rows arrive in and however much the pool grows."""
    digest = hashlib.sha256(f"{salt}:{image}".encode("utf-8")).digest()
    # first 8 bytes as a fraction of the 64-bit range
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < frac


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--salt", default="mmdl-holdout-v1",
                    help="change only to redraw the split deliberately")
    ap.add_argument("--sources", default="ai2d_original,chartqa,scienceqa")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = {}
    if SPLIT.is_file():
        base = {e["id"]: e["correct"]
                for e in json.loads(SPLIT.read_text(encoding="utf-8"))}

    report = []
    print(f"image-level split, holdout fraction {args.holdout_frac:.0%}, "
          f"salt {args.salt!r}\n")
    header = f"{'source':<16}{'rows':>8}{'images':>8}" \
             f"{'train':>8}{'holdout':>9}{'hold %':>8}" \
             f"{'base acc train':>16}{'base acc hold':>15}"
    print(header)
    print("-" * len(header))

    for source in [s.strip() for s in args.sources.split(",") if s.strip()]:
        path = CAND / f"{source}.jsonl"
        if not path.is_file():
            print(f"{source:<16}  (no candidate file, skipped)")
            continue
        rows = list(read_jsonl(path))
        sides = {"train": [], "holdout": []}
        for record in rows:
            image = record.get("image") or record["id"]
            side = "holdout" if in_holdout(image, args.holdout_frac, args.salt) else "train"
            sides[side].append(record)

        def acc(items):
            scored = [base[r["id"]] for r in items if r["id"] in base]
            return 100 * sum(bool(x) for x in scored) / len(scored) if scored else float("nan")

        images = len({r.get("image") for r in rows})
        print(f"{source:<16}{len(rows):>8,}{images:>8,}"
              f"{len(sides['train']):>8,}{len(sides['holdout']):>9,}"
              f"{100*len(sides['holdout'])/len(rows):>7.1f}%"
              f"{acc(sides['train']):>15.2f}%{acc(sides['holdout']):>14.2f}%")

        report.append({"source": source, "rows": len(rows), "images": images,
                       "train": len(sides["train"]), "holdout": len(sides["holdout"]),
                       "base_accuracy_train_pct": acc(sides["train"]),
                       "base_accuracy_holdout_pct": acc(sides["holdout"])})

        if not args.dry_run:
            for side, items in sides.items():
                target = OUT / side
                target.mkdir(parents=True, exist_ok=True)
                with (target / f"{source}.jsonl").open("w", encoding="utf-8") as handle:
                    for record in items:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # The whole point of the change: the two sides should be equally hard.
    print("\n두 쪽의 base 정확도가 비슷해야 분할이 제대로 된 것입니다 "
          "(기존 방식은 AI2D 75.58% vs 92.33%로 벌어져 있었습니다).")

    if not args.dry_run:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "split_manifest.json").write_text(json.dumps({
            "holdout_frac": args.holdout_frac, "salt": args.salt,
            "grouping": "image", "sources": report,
            "rule": "an image, with every question on it, lands wholly on one "
                    "side; selection and stratification happen only on the "
                    "train side",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
