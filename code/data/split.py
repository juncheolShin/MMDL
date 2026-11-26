#!/usr/bin/env python3
"""Select the Phase 0 candidate pool, grouping at the image level.

Why a candidate pool and not the final 10,000 straight away: the final set has to
be stratified by whether the base model answers each question correctly
(DECISIONS.md, the 18:24 decision to raise the wrong-answer share from ~30% to
~40%), and nothing is known about that until Phase 0 inference has run. Sampling
exactly 10,000 now would leave no slack for either that stratification or the
Claude-E answer==gold filter, so we oversample and narrow later.

    10,000 final
      x1.11  image-level 9:1 split (D12)      -> 11,100
      /0.875 Claude E filter (answer == gold) -> 12,700
      40% of that must be wrong answers       ->  5,080 wrong items needed
      at a ~25% natural error rate            -> ~20,000 items must be inferred

Grouping: every question that shares an image travels together, always. A diagram
split across the pool boundary would leak between train and validation later, and
AI2D averages 3.8 questions per diagram, so this matters.

    python code/data/split.py                 # write the pool
    python code/data/split.py --dry-run       # report only
    python code/data/split.py --targets ai2d_original=7000 chartqa=7000

Reads  training_data/normalized/<dataset>/train.jsonl
Writes training_data/candidates/<dataset>.jsonl
       training_data/candidates/candidate_manifest.json
       training_data/candidates/images.txt      (one relative image path per line)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
NORM = ROOT / "training_data" / "normalized"
OUT = ROOT / "training_data" / "candidates"

# Measured 2026-11-18 (questions available / unique images / questions per image):
#   ai2d_original 12,413 / 3,246 / 3.82
#   scienceqa      6,218 / 4,985 / 1.25
#   chartqa       28,299 / 18,271 / 1.55
# ScienceQA is the binding resource: the final target is 3,000 of 6,218, so the
# whole split goes into the pool and there is no slack. If its observed error
# rate comes in under ~25% the 3,000 cannot be met at a 40% wrong share, and the
# shortfall moves to AI2D/ChartQA or that target drops. Decide after Phase 0.
DEFAULT_TARGETS = {
    "ai2d_original": 7000,
    "scienceqa": 6218,   # everything there is
    "chartqa": 7000,
}
DEFAULT_SEED = 20261118


def load(dataset: str):
    path = NORM / dataset / "train.jsonl"
    if not path.is_file():
        sys.exit(f"split: {path} not found - run normalize.py first")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_image(rows):
    groups = collections.OrderedDict()
    for row in rows:
        groups.setdefault(row["image"], []).append(row)
    return groups


def pick(groups, target, rng):
    """Take whole image groups until `target` questions are reached.

    Never splits a group. Overshoot is allowed and reported; stopping mid-group
    would put the same image on both sides of a later train/val boundary.
    """
    keys = list(groups)
    rng.shuffle(keys)
    chosen, count = [], 0
    for key in keys:
        if count >= target:
            break
        chosen.append(key)
        count += len(groups[key])
    return chosen, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--targets", nargs="*", default=[],
                        help="dataset=N overrides, e.g. chartqa=5000")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = dict(DEFAULT_TARGETS)
    for item in args.targets:
        if "=" not in item:
            sys.exit(f"split: bad --targets entry {item!r}, want dataset=N")
        name, _, value = item.partition("=")
        targets[name] = int(value)

    report = {"seed": args.seed, "targets": targets, "datasets": {}}
    all_images = []
    total_q = 0

    for dataset, target in targets.items():
        rows = load(dataset)
        groups = group_by_image(rows)
        rng = random.Random(f"{args.seed}:{dataset}")   # per-dataset stream
        keys, count = pick(groups, target, rng)
        selected = [row for key in keys for row in groups[key]]
        # stable order: by image, then by the id normalize.py assigned
        selected.sort(key=lambda r: (r["image"], r["id"]))

        kinds = collections.Counter(r["answer_type"] for r in selected)
        entry = {
            "available_questions": len(rows),
            "available_images": len(groups),
            "target_questions": target,
            "selected_questions": len(selected),
            "selected_images": len(keys),
            "questions_per_image": round(len(selected) / len(keys), 2) if keys else 0,
            "overshoot": len(selected) - target,
            "answer_types": dict(kinds),
            "fraction_of_available": round(len(selected) / len(rows), 3),
        }
        report["datasets"][dataset] = entry
        total_q += len(selected)
        all_images.extend(keys)

        print(
            f"{dataset:16} {len(selected):>6,} q  /{len(keys):>6,} img"
            f"   target {target:>6,} (+{entry['overshoot']})"
            f"   {entry['fraction_of_available']*100:5.1f}% of available",
            flush=True,
        )

        if not args.dry_run:
            OUT.mkdir(parents=True, exist_ok=True)
            with (OUT / f"{dataset}.jsonl").open("w", encoding="utf-8") as sink:
                for row in selected:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    unique_images = sorted(set(all_images))
    if len(unique_images) != len(all_images):
        # content-addressed paths, so the same picture can back two datasets
        print(f"note: {len(all_images) - len(unique_images)} image(s) shared across datasets")

    report["totals"] = {
        "questions": total_q,
        "unique_images": len(unique_images),
    }
    print(f"\n{'TOTAL':16} {total_q:>6,} q  /{len(unique_images):>6,} img")

    if args.dry_run:
        print("\ndry run - nothing written")
        return 0

    (OUT / "images.txt").write_text("\n".join(unique_images) + "\n", encoding="utf-8")

    # Fingerprint the selection so a later run can prove it used this exact pool.
    digest = hashlib.sha256("\n".join(unique_images).encode("utf-8")).hexdigest()
    report["image_list_sha256"] = digest
    (OUT / "candidate_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"image list sha256: {digest[:16]}...")
    print(f"written -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
