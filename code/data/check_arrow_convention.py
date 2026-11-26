#!/usr/bin/env python3
"""How many AI2D food webs draw arrows the other way round?

The chain templates in synth_evidence_ai2d.py assume the standard convention -
an arrow runs from the food to the eater, the direction energy flows. Two rows
in the human re-audit were marked "화살표가 반대의 의미" (the arrow means the
opposite), and both were wrong in the same way:

    Blue Shark -> Northern      template: "Northern feeds on Blue Shark"
                                truth:    the shark eats the anchovy
    ants -> Grass               template: "Grass takes its energy from ants"
                                truth:    the ants eat the grass

The gold check did not catch it. It verifies that the ENTITY the chain names is
the gold answer, and the entity was right both times; what was backwards was the
RELATION. So the convention has to be settled per diagram before a chain is
written.

The test: a producer (a plant, algae, the sun) is where energy starts. Under the
standard convention nothing points at it and arrows only leave. If a diagram's
producers have arrows coming IN and none going out, that diagram is drawn the
other way round.

    python code/data/check_arrow_convention.py
    python code/data/check_arrow_convention.py --evidence training_data/evidence_v3/ai2d_evidence.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data"))
import synth_evidence_ai2d as synth  # noqa: E402

# Words that mark the start of a food chain. Kept deliberately narrow: a word
# that is only sometimes a producer (e.g. "plankton", which may be zooplankton)
# would mislabel the diagram, and a diagram with no clear producer is reported
# as undetermined rather than guessed at.
PRODUCER = re.compile(
    r"\b(sun|sunlight|solar)\b"
    r"|\b(green\s+)?plants?\b|\bgrass(es)?\b|\btrees?\b|\bleaves\b|\bleaf\b"
    r"|\balgae?\b|\bseaweed\b|\bphytoplankton\b|\bkelp\b|\bmoss\b"
    r"|\bshrubs?\b|\bflowers?\b|\bberries\b|\bnectar\b|\bwood\b",
    re.IGNORECASE)

EDGE_CATEGORIES = getattr(synth, "EDGE_CATEGORIES", None)


def diagram_edges(ann):
    """Every labelled edge in the diagram, as (src_text, dst_text)."""
    texts = ann["text"]
    label_of_region = {}
    for rel in ann["relationships"].values():
        if rel.get("category") not in ("intraObjectLabel", "intraObjectRegionLabel"):
            continue
        src = texts.get(rel.get("origin"))
        region = rel.get("destination")
        if src and src.get("value") and region:
            label_of_region.setdefault(region, src["value"])

    def resolve(nid):
        node = texts.get(nid)
        if node and node.get("value"):
            return node["value"]
        return label_of_region.get(nid, nid)

    out = []
    for rel in ann["relationships"].values():
        if EDGE_CATEGORIES and rel.get("category") not in EDGE_CATEGORIES:
            continue
        if not rel.get("hasDirectionality"):
            continue
        src, dst = resolve(rel.get("origin")), resolve(rel.get("destination"))
        if src and dst and src != dst:
            out.append((str(src), str(dst)))
    return out


def classify(edges):
    """'standard' | 'reversed' | 'no_producer' | 'mixed'"""
    outdeg, indeg = collections.Counter(), collections.Counter()
    for s, d in edges:
        outdeg[s] += 1
        indeg[d] += 1
    producers = {n for n in set(outdeg) | set(indeg) if PRODUCER.search(n)}
    if not producers:
        return "no_producer", producers
    only_out = {p for p in producers if outdeg[p] and not indeg[p]}
    only_in = {p for p in producers if indeg[p] and not outdeg[p]}
    if only_out and not only_in:
        return "standard", producers
    if only_in and not only_out:
        return "reversed", producers
    return "mixed", producers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evidence", type=pathlib.Path,
                    default=ROOT / "training_data" / "evidence_v3" / "ai2d_evidence.jsonl")
    ap.add_argument("--show", type=int, default=4, help="examples per class")
    args = ap.parse_args()

    if not args.evidence.is_file():
        sys.exit(f"check_arrow_convention: {args.evidence} not found")

    rows = [json.loads(l) for l in args.evidence.open(encoding="utf-8") if l.strip()]
    rel = [r for r in rows if (r.get("meta") or {}).get("evidence_kind") == "relation_grounded"]
    print(f"{len(rel):,} relation_grounded rows over "
          f"{len({(r['meta'] or {}).get('ai2d_image_id') for r in rel}):,} diagrams\n")

    per_diagram, examples = {}, collections.defaultdict(list)
    rows_by_class = collections.Counter()
    for record in rel:
        image_id = (record.get("meta") or {}).get("ai2d_image_id")
        if image_id not in per_diagram:
            ann = synth.load_annotation(image_id)
            per_diagram[image_id] = classify(diagram_edges(ann)) if ann else ("no_annotation", set())
        verdict, producers = per_diagram[image_id]
        rows_by_class[verdict] += 1
        if len(examples[verdict]) < args.show:
            examples[verdict].append((record["id"], image_id, sorted(producers)[:3],
                                      record["question"][:70]))

    diagrams = collections.Counter(v for v, _ in per_diagram.values())
    print(f"{'class':<14}{'diagrams':>10}{'rows':>8}{'rows %':>9}")
    print("-" * 41)
    total = sum(rows_by_class.values())
    for name in ("standard", "reversed", "mixed", "no_producer", "no_annotation"):
        if not rows_by_class[name] and not diagrams[name]:
            continue
        print(f"{name:<14}{diagrams[name]:>10,}{rows_by_class[name]:>8,}"
              f"{100*rows_by_class[name]/total:>8.1f}%")

    for name in ("reversed", "mixed"):
        if examples[name]:
            print(f"\n--- {name} ---")
            for rid, img, prod, q in examples[name]:
                print(f"  {rid}  producers={prod}")
                print(f"    {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
