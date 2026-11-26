#!/usr/bin/env python3
"""Sample AI2D relation_grounded rows so their sufficiency can be judged by hand.

label_grounded rows are checked in full: the cited label's value must equal the
gold answer, so if the check passes the evidence provably contains the answer.
relation_grounded rows are only checked for GROUNDING - every cited annotation id
exists in the diagram - plus a weak sufficiency tier. The reasoning chain itself
is not verified, which is why they carry needs_human_audit (DECISIONS.md D34 and
the (da) decision).

There are two tiers and they are not equally risky:

  gold_in_cited_label              1,281  the gold answer text appears in a cited
                                          label, so the evidence at least names
                                          the thing being asked for
  question_entity_in_cited_label     469  only an entity from the QUESTION
                                          appears. The evidence may describe the
                                          right part of the diagram and still not
                                          support the gold answer.

So this oversamples the weak tier: a uniform sample would put four fifths of the
reviewer's effort on the safer rows.

What the reviewer is deciding is NOT whether the gold answer is right - AI2D's
labels are the ground truth - but whether a reader seeing ONLY this evidence
could reach it. Evidence that cannot support its own answer teaches the model to
assert conclusions from nothing, which is worse than no evidence at all.

    python code/data/audit_ai2d_relations.py --limit 30
    python code/data/audit_ai2d_relations.py --limit 60 --weak-share 0.7

Writes training_data/evidence/ai2d_relation_audit.md and .json
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import pathlib
import random
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "training_data" / "evidence"
WEAK = "question_entity_in_cited_label"

# The synthesiser's sufficiency test is an exact substring match, which is too
# strict to mean what it says. AI2D's gold answers carry typos ("todpole"),
# inconsistent case ("TADPOLE "), plurals and trailing spaces, so an answer whose
# entity IS in the evidence can still fail the test. Measured over the 469 weak
# rows: 35% are exactly that - a string artefact, not missing evidence.
#
# These words name a DIRECTION or QUANTITY of change rather than a thing in the
# diagram. "more water plants" is about `water plant`; "more" is not an entity to
# look for, and treating it as one makes every multi-word answer look unsupported.
CHANGE_WORDS = set("""
more less fewer fewest most least increase increases increased decrease decreases decreased
rise rises rising fall falls falling grow grows growth decline declines reduce reduced
population populations number numbers amount level levels stay stays remain remains
higher lower bigger smaller larger same unchanged none
the a an of in to and or will would there be is are it its their they them that this these
those if then than with without for from on at as by no not
""".split())

RISK = {
    "entity_cited_under_a_typo": (
        "the gold answer's entities ARE in the cited evidence, differing only by case, "
        "plural or spelling. Low risk: the exact-match test was simply too strict."),
    "direction_only_answer": (
        "the gold answer names no entity at all (\"decrease\", \"the same\"). Whether the "
        "evidence supports it rests entirely on the cited edges."),
    "entity_not_cited": (
        "an entity in the gold answer is genuinely absent from the cited evidence. Often a "
        "CATEGORY word (\"prey\", \"carnivore\") that a diagram never labels, which the "
        "edges may still establish by position - but this is where unsupported rows will be."),
}


def _words(text):
    return [w for w in re.split(r"[^a-z0-9]+", str(text).lower()) if w]


def _singular(word):
    return word[:-1] if len(word) > 4 and word.endswith("s") else word


def _gold_entities(answer):
    return [w for w in _words(answer) if len(w) >= 4 and w not in CHANGE_WORDS]


def _cited_vocabulary(meta):
    out = set()
    for item in meta.get("evidence") or []:
        for key in ("src", "dst", "text", "target"):
            if item.get(key):
                out |= set(_words(item[key]))
    return out | {_singular(w) for w in out}


def risk_class(record):
    """Which of the three weak-tier failure modes a row is, with its worst score.

    Returns (class name, worst similarity) - worst, because one unsupported
    entity is enough to make the evidence insufficient.
    """
    meta = record.get("meta") or {}
    entities = _gold_entities(record.get("answer", ""))
    if not entities:
        return "direction_only_answer", None
    vocab = _cited_vocabulary(meta)
    worst = min(
        max((difflib.SequenceMatcher(None, _singular(e), c).ratio() for c in vocab),
            default=0.0)
        for e in entities)
    name = ("entity_cited_under_a_typo" if worst >= 0.75 else "entity_not_cited")
    return name, worst


def render_evidence(evidence) -> list[str]:
    """One readable line per evidence item, whatever kind it is."""
    lines = []
    for item in evidence:
        kind = item.get("kind")
        if kind == "edge":
            arrow = "->" if item.get("directed") else "--"
            lines.append(f"- edge: `{item.get('src')}` {arrow} `{item.get('dst')}`"
                         f"  (connector {item.get('connector')}, {item.get('category')})")
        elif kind == "label":
            lines.append(f"- label: `{item.get('text')}`"
                         + (f"  (on {item.get('target')})" if item.get("target") else ""))
        else:
            lines.append(f"- {kind}: " + json.dumps(
                {k: v for k, v in item.items() if k not in ("kind", "source_id")},
                ensure_ascii=False))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=30, help="rows to sample in total")
    ap.add_argument("--weak-share", type=float, default=0.6,
                    help="fraction drawn from the weaker sufficiency tier")
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--evidence", type=pathlib.Path,
                    default=OUT / "ai2d_evidence.jsonl")
    args = ap.parse_args()

    if not args.evidence.is_file():
        raise SystemExit(f"audit_ai2d_relations: {args.evidence} not found - "
                         f"run synth_evidence_ai2d.py first")
    rows = [json.loads(l) for l in args.evidence.open(encoding="utf-8") if l.strip()]
    rel = [r for r in rows if (r.get("meta") or {}).get("evidence_kind") == "relation_grounded"]
    if not rel:
        raise SystemExit("audit_ai2d_relations: no relation_grounded rows in the evidence file")

    by_tier = collections.defaultdict(list)
    for r in rel:
        by_tier[r["meta"].get("evidence_sufficiency", "unknown")].append(r)
    print(f"{len(rel):,} relation_grounded rows")
    for tier, items in sorted(by_tier.items(), key=lambda kv: -len(kv[1])):
        print(f"  {tier:36} {len(items):>6,}")

    # Classify the weak tier by WHY it failed the exact-match test. The three
    # causes carry very different risk, and a sample drawn without regard to them
    # spends most of its rows on the harmless one.
    weak = by_tier.get(WEAK, [])
    by_risk = collections.defaultdict(list)
    for r in weak:
        name, score = risk_class(r)
        r["meta"]["_risk_class"] = name
        r["meta"]["_risk_score"] = score
        by_risk[name].append(r)
    print()
    print(f"  weak tier ({len(weak):,}) by cause:")
    for name in ("entity_not_cited", "direction_only_answer", "entity_cited_under_a_typo"):
        n = len(by_risk.get(name, []))
        if n:
            print(f"    {name:28} {n:>4}  ({100*n/max(len(weak),1):.1f}%)")

    rng = random.Random(args.seed)
    want_weak = min(int(round(args.limit * args.weak_share)), len(weak))
    want_rest = args.limit - want_weak

    # Inside the weak tier, split the budget by risk rather than by frequency:
    # half on entity_not_cited, a third on direction_only, the rest on the typo
    # class - enough of the last to confirm it is as harmless as it looks.
    plan = {"entity_not_cited": 0.5, "direction_only_answer": 0.3,
            "entity_cited_under_a_typo": 0.2}
    sample = []
    for name, share in plan.items():
        pool = by_risk.get(name, [])[:]
        rng.shuffle(pool)
        sample += pool[:min(int(round(want_weak * share)), len(pool))]

    rest = [r for t, v in by_tier.items() if t != WEAK for r in v]
    rng.shuffle(rest)
    sample += rest[:want_rest]
    rng.shuffle(sample)
    drawn = collections.Counter(r["meta"].get("_risk_class") or "strong tier" for r in sample)
    print()
    print(f"sampling {len(sample)}: "
          + ", ".join(f"{v} {k}" for k, v in drawn.most_common()))

    lines = [
        "# AI2D relation_grounded: sufficiency check",
        "",
        "These rows are **grounded but not verified**: every cited annotation id exists in",
        "the diagram, but nothing checks that the cited relations actually imply the gold",
        "answer. That check is this file.",
        "",
        "For each row, decide **could a reader seeing only this evidence reach the gold",
        "answer?** The gold answer is not in question - AI2D's annotations are the ground",
        "truth. What is in question is whether the evidence supports it.",
        "",
        "- **(a) sufficient** - the evidence implies the gold answer",
        "- **(b) relevant but insufficient** - right part of the diagram, but a reader would",
        "  have to already know the answer, or guess, to get there",
        "- **(c) irrelevant** - the evidence does not bear on the question",
        "",
        "(b) and (c) are the harmful ones: they teach the model to state a conclusion its",
        "stated evidence does not carry. If (b)+(c) runs much above a tenth, the weak tier",
        f"(`{WEAK}`) should be dropped from the training set rather than fixed row by row.",
        "",
        f"Sampled {len(sample)} of {len(rel):,}, weighted {args.weak_share:.0%} to the weak",
        f"tier, seed {args.seed}.",
        "",
        "---",
        "",
    ]
    for i, row in enumerate(sample, 1):
        meta = row["meta"]
        tier = meta.get("evidence_sufficiency", "unknown")
        flag = " **(weak tier)**" if tier == WEAK else ""
        lines += [
            f"## {i}. `{row['id']}`{flag}",
            "",
            f"**Question:** {row['question']}",
            "",
            "**Options:** " + " / ".join(f"`{o}`" for o in (row.get("options") or [])),
            "",
            f"**Gold answer:** `{row['answer']}`",
            "",
            f"**Sufficiency tier:** `{tier}`",
            "",
        ]
        cause = meta.get("_risk_class")
        if cause:
            score = meta.get("_risk_score")
            lines += [
                f"**Why the exact-match test failed:** `{cause}`"
                + (f"  (closest cited word scores {score:.2f})" if score is not None else ""),
                "",
                f"> {RISK[cause]}",
                "",
            ]
        lines += [
            "**Evidence as the model will see it:**",
            "",
        ]
        lines += render_evidence(meta.get("evidence") or [])
        lines += ["", "**Reasoning as the model will see it:**", ""]
        for n, step in enumerate(meta.get("reasoning") or [], 1):
            lines.append(f"{n}. {step}")
        lines += ["", "**Sufficient?** (a) / (b) / (c)  -> ", "", "---", ""]

    md = OUT / "ai2d_relation_audit.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "ai2d_relation_audit.json").write_text(json.dumps({
        "sampled_ids": [r["id"] for r in sample],
        "tier_counts": {k: len(v) for k, v in by_tier.items()},
        "weak_share": args.weak_share, "seed": args.seed,
        "total_relation_grounded": len(rel),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {md}  ({len(sample)} rows to judge)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
