#!/usr/bin/env python3
"""Control training sets for diagnosing arm A's holdout regression (DECISIONS.md D51).

arm A trained on 7,000 answer-only rows and came out 4.51pp BELOW the base model
on the in-domain holdout (McNemar p = 4.7e-17). The targets are a single letter,
so evidence quality cannot be the cause. Three candidates remain, and each needs
a control that changes exactly one thing:

  lr        learning rate 1e-4 -> 2e-5          (configs/arms/a_1gpu_lr2e5.toml)
  random    wrong-first selection -> uniform    (this script, --set random)
  chartqa   both sources -> ChartQA only        (this script, --set chartqa)

Both control sets here must be comparable to arm A on the SAME holdout, so they
are drawn only from rows the holdout does not use - neither the question nor its
image. Otherwise a control would be scored on pictures it trained on and would
look better for the wrong reason.

The rendering goes through render_targets.render_user / render_assistant, so the
rows are byte-identical in format to arm_a - only the row SELECTION differs.

    python code/data/make_diagnostic_sets.py --set random
    python code/data/make_diagnostic_sets.py --set chartqa
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data"))
import render_targets  # noqa: E402

CAND = ROOT / "training_data" / "candidates"
HOLDOUT = ROOT / "training_data" / "candidates_holdout"
ARM_A = ROOT / "training_data" / "sft" / "arm_a" / "train.jsonl"
SFT = ROOT / "training_data" / "sft"


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_set(name, records):
    out = SFT / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.jsonl"
    written, skipped = 0, 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            assistant, reason = render_targets.render_assistant(record, "answer_only")
            if not assistant:
                skipped += 1
                continue
            handle.write(json.dumps({
                "image": record["image"],
                "conversations": [
                    {"from": "human", "value": render_targets.render_user(record)},
                    {"from": "gpt", "value": assistant},
                ],
                "source": record["source"],
                "uid": record["id"],
            }, ensure_ascii=False) + "\n")
            written += 1
    print(f"  wrote {written:,} rows -> {path}" + (f"  ({skipped} skipped)" if skipped else ""))
    return written


def holdout_keys():
    """Every question id and every image the holdout uses."""
    ids, images = set(), set()
    for path in HOLDOUT.glob("*.jsonl"):
        for record in read_jsonl(path):
            ids.add(record["id"])
            if record.get("image"):
                images.add(record["image"])
    return ids, images


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", required=True, choices=("random", "chartqa"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.set == "chartqa":
        # Same rows arm A used, minus AI2D. Nothing else changes, so a
        # difference is attributable to the source mix alone.
        rows = [r for r in read_jsonl(ARM_A) if r["source"] == "chartqa"]
        out = SFT / "arm_a_chartqa"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "train.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"arm_a_chartqa: {len(rows):,} rows (arm_a's ChartQA half, verbatim)")
        print(f"  wrote -> {path}")
        return 0

    # random: uniform selection from the same pool, avoiding the holdout
    held_ids, held_images = holdout_keys()
    print(f"holdout: {len(held_ids):,} questions over {len(held_images):,} images")

    want = {"ai2d_original": 3500, "chartqa": 3500}
    chosen = []
    for source, n in want.items():
        pool = [r for r in read_jsonl(CAND / f"{source}.jsonl")
                if r["id"] not in held_ids and r.get("image") not in held_images]
        if len(pool) < n:
            sys.exit(f"make_diagnostic_sets: only {len(pool):,} rows available for "
                     f"{source}, need {n:,}")
        picked = random.Random(args.seed).sample(pool, n)
        print(f"  {source:<16} pool {len(pool):>6,}  picked {n:,} uniformly")
        chosen += picked

    print("arm_a_random:")
    write_set("arm_a_random", chosen)

    # Report how the base model did on the picked rows, so the difference from
    # arm_a's wrong-first mix is on the record rather than assumed.
    split_path = ROOT / "results" / "phase0" / "indomain_split.json"
    if split_path.is_file():
        base = {e["id"]: e["correct"]
                for e in json.loads(split_path.read_text(encoding="utf-8"))}
        wrong = sum(1 for r in chosen if base.get(r["id"]) is False)
        scored = sum(1 for r in chosen if r["id"] in base)
        print(f"  base-wrong share of this set: {wrong:,}/{scored:,} "
              f"= {100*wrong/scored:.1f}%   (arm_a was 43.5% AI2D / 36.0% ChartQA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
