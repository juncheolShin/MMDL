#!/usr/bin/env python3
"""A control that keeps arm B's SHAPE and removes its CONTENT.

The baseline leaves 42 of 900 MMMU questions with nothing to extract, and all
42 are the same failure: `finish_reason == length`, exactly 512 output tokens,
every one multiple-choice. The model does not stop. On those same 42:

    a_cqa (answer-only targets)      answered  2,  2 correct
    b_cqa (evidence+reasoning)       answered 31, 12 correct

So the evidence-trained arm gained something real - the habit of producing a
bounded block and then stopping on an answer. The open question is whether that
came from the evidence CONTENT or merely from the SHAPE of the target.

This builds the control: the same rows, the same final answer, and a preamble
that is byte-identical on every row, so it carries no information at all.

    arm B   : Evidence: 2015 reads 5, the lowest ... / Reasoning: ... / 2015
    control : Looking at the image. / 2015

If the control recovers the finishing benefit, the benefit is format and can be
had for ~6 tokens with no rationale pipeline. If it does not, the content is
doing the work after all (D72).

    python code/data/make_format_only.py --source training_data/sft/a_cqa --out training_data/sft/a_cqa_stop
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Deliberately contentless and identical everywhere. It names nothing in the
# image, nothing in the question and nothing in the options, so the only thing
# it can teach is "write one short line, then answer, then stop".
PREAMBLE = "Looking at the image."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=pathlib.Path, required=True,
                    help="an answer-only sft set; its targets get the preamble")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--preamble", default=PREAMBLE)
    args = ap.parse_args()

    src = args.source / "train.jsonl"
    if not src.is_file():
        sys.exit(f"make_format_only: {src} not found")

    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / "train.jsonl"
    n = 0
    with src.open(encoding="utf-8") as handle, dst.open("w", encoding="utf-8") as out:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            answer = row["conversations"][-1]["value"].strip()
            row["conversations"][-1]["value"] = f"{args.preamble}\n{answer}"
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    try:
        shown = dst.relative_to(ROOT)
    except ValueError:
        shown = dst
    print(f"wrote {n:,} rows -> {shown}")
    print(f"preamble: {args.preamble!r}  (identical on every row)")
    sample = json.loads(dst.open(encoding='utf-8').readline())
    print("\nexample target:")
    print("  " + sample["conversations"][-1]["value"].replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
