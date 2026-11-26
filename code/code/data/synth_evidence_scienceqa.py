#!/usr/bin/env python3
"""ScienceQA evidence from the dataset's OWN `solution` field - no model call.

D34 ruled ScienceQA's text fields out as "textbook prose about the topic, not a
description of the image" and sent the source to Claude instead. That is right
about `lecture`. It is too strong about `solution`, which is a worked answer and
often points straight at the picture:

    "To find the answer, look at the compass rose. Look at which way the north
     arrow is pointing. West Virginia is farthest north."

Measured over the 6,218 training candidates: `solution` is present on 91.3% and
40.0% of those refer to something visible. That is human-written, image-grounded
reasoning we already have, for free, and it does not depend on an API account.

The solutions follow a very regular shape:

    1. "Look at the picture of the sugar glider."   <- an instruction, not an
                                                       observation; dropped
    2. "The sugar glider can spread its patagium."  <- the observation -> Evidence
    3. "This helps it glide through the air."       <- the inference  -> Reasoning
    4. "Now look at each animal. ..."               <- option enumeration; long,
                                                       and it quotes the options,
                                                       so it is dropped

Two rules carry over from the Claude pipeline unchanged: evidence may not
restate an answer option (G4), and the rendered target has a token budget (G2).

    python code/data/synth_evidence_scienceqa.py --report
    python code/data/synth_evidence_scienceqa.py --candidates training_data/candidates_split/train/scienceqa.jsonl --out training_data/evidence_v4
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
import wrong_priority  # noqa: E402

CAND = ROOT / "training_data" / "candidates" / "scienceqa.jsonl"
OUT = ROOT / "training_data" / "evidence"
DEFAULT_SEED = 20261119

# Something the writer could only say while looking at the picture.
#
# Bare position words (above, below, between, left, right) were in here at
# first and are not: they fire on ordinary prose - "limestone forms from ocean
# sediment ... made mostly of shells" came through as visual - and a wrong
# anchor here admits textbook recall as though it were an observation. What is
# left either names the picture or names a thing that only exists in one.
VISUAL = re.compile(
    r"\b(look at|see the|shown in|in the (image|picture|diagram|map|chart|graph|photo|model|figure))"
    r"|\b(this (image|picture|diagram|map|chart|model))"
    r"|\b(arrow|compass rose|legend|scale bar|x-axis|y-axis|axis)\b"
    r"|\b(colou?r(ed|s)?|shaded|circled|highlighted|dotted|dashed)\b"
    r"|\b(label(l)?ed|pictured|depicted|illustrated)\b",
    re.IGNORECASE)

# Openers that tell the reader to look rather than saying what is there. An
# instruction is not evidence, so these sentences are dropped.
IMPERATIVE = re.compile(
    r"^\s*(to find the answer|look at|now look|use the|first,? look|"
    r"start by|consider|compare|think about|remember|decide|"
    r"for each\b.{0,40}?\b(decide|look|find|try)|"
    r"to (decide|determine|find|tell|compare)\b)",
    re.IGNORECASE)

# Where the solution stops reasoning and starts walking the option list.
ENUMERATION = re.compile(
    r"^\s*(now look at each|the other|neither|none of|for each answer choice|"
    r"so,? the (only )?(answer|option)|that leaves)\b",
    re.IGNORECASE)


def sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or "").strip())
            if s.strip()]


def norm(text) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def leaks_option(text: str, options, answer_index) -> bool:
    """Evidence may not restate an option - including the right one. The rule is
    the same one the Claude pipeline applies, so the two sources stay comparable."""
    haystack = norm(text)
    for i, option in enumerate(options or []):
        phrase = norm(option)
        if len(phrase) >= 3 and phrase in haystack:
            return True
    return False


def build(record, max_words: int):
    """Split one solution into evidence + reasoning, or say why it cannot be."""
    meta = record.get("meta") or {}
    solution = str(meta.get("solution") or "").strip()
    if not solution:
        return None, "no_solution"
    if not VISUAL.search(solution):
        return None, "solution_never_refers_to_the_picture"

    options = record.get("options") or []
    answer_index = record.get("answer_index")

    body = []
    for sentence in sentences(solution):
        if ENUMERATION.match(sentence):
            break                      # the option walk-through starts here
        if IMPERATIVE.match(sentence):
            continue                   # "Look at the picture" is not an observation
        body.append(sentence)
    if not body:
        return None, "only_instructions_no_statements"

    # The visual anchor is usually in the sentence that was just dropped - the
    # solutions say "Look at the picture of the sugar glider." and then state
    # what is there: "The sugar glider can spread its patagium." Requiring the
    # statement to carry its own visual keyword threw away 1,170 of 5,533 rows
    # for no reason, so the whole-solution check above is the one that counts.
    evidence_text = body[0]
    if leaks_option(evidence_text, options, answer_index):
        return None, "evidence_sentence_restates_an_option"

    reasoning = body[1:3] or body[:1]
    words = len(evidence_text.split()) + sum(len(s.split()) for s in reasoning)
    if words > max_words:
        # try the shorter form before giving up
        reasoning = body[1:2]
        words = len(evidence_text.split()) + sum(len(s.split()) for s in reasoning)
        if not reasoning or words > max_words:
            return None, f"over_{max_words}_words"

    return {
        "kind": "solution_grounded",
        "evidence": [{"kind": "observation", "text": evidence_text}],
        "knowledge": "",
        "reasoning": reasoning,
        "verification": "dataset_authored_solution",
    }, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", type=pathlib.Path, default=CAND)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--max-words", type=int, default=34,
                    help="evidence + reasoning budget; 34 words is about the "
                         "45 tokens G2 allows")
    ap.add_argument("--prefer-wrong", type=pathlib.Path, default=None,
                    metavar="SPLIT.json",
                    help="Phase 0 split; take rows the base model got wrong first")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--report", action="store_true", help="measure, write nothing")
    args = ap.parse_args()

    if not args.candidates.is_file():
        sys.exit(f"synth_evidence_scienceqa: {args.candidates} not found")
    records = [json.loads(l) for l in args.candidates.open(encoding="utf-8") if l.strip()]
    print(f"{len(records):,} ScienceQA candidate records", flush=True)

    built, reasons = [], collections.Counter()
    for record in records:
        item, why = build(record, args.max_words)
        if item is None:
            reasons[why] += 1
            continue
        built.append((record, item))

    print(f"\n  usable {len(built):,} / {len(records):,} "
          f"({100*len(built)/max(len(records),1):.1f}%)")
    for why, n in reasons.most_common():
        print(f"    {why:<48} {n:>6,}")
    if args.report:
        print("\nreport only - nothing written")
        return 0

    split = wrong_priority.load_split(args.prefer_wrong)
    random.Random(args.seed).shuffle(built)
    chosen = wrong_priority.wrong_first(built, split)[:args.target]
    wrong_priority.report(chosen, split, label="selected          ")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "scienceqa_evidence.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record, item in chosen:
            meta = dict(record.get("meta") or {})
            meta.update({
                "evidence": item["evidence"],
                "knowledge": item["knowledge"],
                "reasoning": item["reasoning"],
                "evidence_provenance": "dataset_authored",
                "evidence_kind": item["kind"],
                "evidence_verification": item["verification"],
                # written by the dataset's authors, not by a model and not by a
                # program that can be re-checked, so it still wants sampling
                "needs_human_audit": True,
            })
            handle.write(json.dumps({**record, "meta": meta}, ensure_ascii=False) + "\n")
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"\nwrote {len(chosen):,} records -> {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
