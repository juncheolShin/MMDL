#!/usr/bin/env python3
"""Synthesise visual evidence for AI2D from the original annotations. No model, no cost.

The AI2D release ships, per diagram, the text boxes with their bounding boxes and
the letter that replaces them in the image, the arrow and blob polygons, and a
relationship graph. That is exactly the "coordinates, existence of an edge"
evidence the project set out to train on, as ground truth rather than as
something a model was asked to produce - so it has none of the blind-spot problem
that made self-distillation unusable (DECISIONS.md section 4).

Two evidence kinds, kept apart so they can be compared later (D18 provenance):

  label_grounded      The question asks about a letter, or answers with one. The
                      evidence states what that letter marks, with its box.
                      Verified fully: the label's value must equal the gold answer.

  relation_grounded   The question is about how parts of the diagram connect. The
                      evidence lists the labelled parts and the edges between them.
                      Verified for grounding (every cited id exists) and weakly for
                      sufficiency (the gold answer appears in a cited label). The
                      reasoning chain itself is NOT verified - these are the ones
                      worth human audit.

    python code/data/synth_evidence_ai2d.py --report
    python code/data/synth_evidence_ai2d.py --label-target 1750 --relation-target 1750
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import re
import sys

import wrong_priority

ROOT = pathlib.Path(__file__).resolve().parents[2]
AI2D = ROOT / "training_data" / "raw_external" / "ai2d_original" / "extracted" / "ai2d"
CAND = ROOT / "training_data" / "candidates" / "ai2d_original.jsonl"
OUT = ROOT / "training_data" / "evidence"

DEFAULT_SEED = 20261119
# Relationship kinds that say something about structure. imageTitle/sectionTitle/
# imageCaption describe the page, not the diagram's parts, so they are not edges.
EDGE_CATEGORIES = {
    "intraObjectLinkage", "interObjectLinkage", "arrowHeadTail",
    "intraObjectLabel", "intraObjectRegionLabel", "arrowDescriptor",
    "intraObjectTextLinkage",
}


def _is_text(value) -> bool:
    """True when this endpoint is readable text rather than an annotation id.

    AI2D ids look like T0, A1, B0, H0, C2 - a letter and digits. Anything else
    came from a text box's value.
    """
    return bool(value) and not re.fullmatch(r"[A-Z]\d+", str(value).strip())


def norm(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def bbox_of(rect):
    """AI2D stores a rectangle as two corners; report x, y, w, h."""
    (x1, y1), (x2, y2) = rect[0], rect[1]
    return [int(min(x1, x2)), int(min(y1, y2)), int(abs(x2 - x1)), int(abs(y2 - y1))]


def load_annotation(image_id: str):
    path = AI2D / "annotations" / f"{image_id}.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    texts = {}
    for tid, entry in (doc.get("text") or {}).items():
        texts[tid] = {
            "id": tid,
            "value": entry.get("value"),
            "label": (entry.get("replacementText") or "").strip().upper() or None,
            "bbox": bbox_of(entry["rectangle"]) if entry.get("rectangle") else None,
        }
    # Every node kind a relationship can point at. Leaving any of these out
    # makes the grounding check reject valid edges: arrowHeads, containers,
    # imageConsts and meaningfulSpaces all appear as relationship endpoints.
    nodes = set(texts)
    for key in ("arrows", "arrowHeads", "blobs", "containers",
                "imageConsts", "meaningfulSpaces"):
        nodes |= set(doc.get(key) or {})
    return {
        "text": texts,
        "nodes": nodes,
        "relationships": doc.get("relationships") or {},
    }


# The label_grounded reasoning used to name the diagram's own label letter:
#   The diagram marks "CENTRAL VENT" with the letter C.
# while the multiple-choice options are ALSO lettered A-D, and the option that
# holds "C" may be B. Two letter namespaces in one target, with the wrong-looking
# letter sitting immediately before the answer.
#
# Measured on arm B's AI2D holdout (600 rows), split by whether the model's own
# response named a letter in prose:
#   names a letter   336 rows   47.0% correct   17.0% unextractable
#   no letter        264 rows   68.6% correct   14.4% unextractable
#   (base model: 75.67%)
# and 78.4% of the AI2D targets named one. `--letter-in-prose` keeps the old
# wording so the two can be compared; the default drops it (D65).

LETTERS = "ABCDEFGHI"


def _option_holding(record, wanted):
    """Which option letter holds `wanted`, or None.

    AI2D letter questions have two letter namespaces: the letters the DIAGRAM
    uses and the letters the OPTIONS use, and they are deliberately scrambled
    ("Which letter shows the adult stage?" with options A. F  B. A  C. B  D. D).
    The whole inference is that mapping, so the reasoning has to state it.
    """
    for i, text in enumerate(record.get("options") or []):
        if i < len(LETTERS) and norm(str(text)) == norm(str(wanted)):
            return LETTERS[i]
    return None


def _direction1_reasoning(record, entry, letter_in_prose):
    """The answer is a diagram letter. State the label AND which option is it.

    v2 named the diagram letter and then jumped to a different option letter
    with nothing in between; v3 removed the letter and left "that position",
    which names nothing at all. Both leave the step the question actually asks
    for unsaid, so this states the whole mapping (D69).
    """
    if letter_in_prose:
        return (f"The diagram marks \"{entry['value']}\" with the letter "
                f"{entry['label']}.")
    option = _option_holding(record, entry["label"])
    if option:
        return (f"The diagram labels \"{entry['value']}\" as {entry['label']}, "
                f"and option {option} is {entry['label']}.")
    return f"The diagram labels \"{entry['value']}\" as {entry['label']}."


def _direction2_reasoning(record, entry, letter, letter_in_prose):
    """The question names a diagram letter; the answer is the text it marks."""
    if letter_in_prose:
        return (f"Letter {letter} in the diagram marks the text "
                f"\"{entry['value']}\".")
    option = _option_holding(record, entry["value"])
    if option:
        return (f"{letter} in the diagram marks \"{entry['value']}\", "
                f"which is option {option}.")
    return f"{letter} in the diagram marks \"{entry['value']}\"."


def label_evidence(record, ann, letter_in_prose=False):
    """Letter <-> word questions. Fully verifiable."""
    by_label = {t["label"]: t for t in ann["text"].values() if t["label"]}
    by_value = {norm(t["value"]): t for t in ann["text"].values() if t["value"]}
    gold = str(record["answer"]).strip()

    # direction 1: the answer is a letter that the diagram uses as a label
    if len(gold) == 1 and gold.upper() in by_label:
        entry = by_label[gold.upper()]
        if not entry["value"]:
            return None, "label_has_no_value"
        return {
            "kind": "label_grounded",
            "evidence": [{
                "kind": "label", "text": entry["value"], "label": entry["label"],
                "bbox": entry["bbox"], "source_id": entry["id"],
            }],
            "reasoning": [_direction1_reasoning(record, entry, letter_in_prose)],
            "cited_ids": [entry["id"]],
        }, None

    # direction 2: the question names a letter, the answer is what it marks
    letters = re.findall(r"\b([A-Z])\b", record["question"])
    for letter in letters:
        entry = by_label.get(letter)
        if entry and entry["value"] and norm(entry["value"]) == norm(gold):
            return {
                "kind": "label_grounded",
                "evidence": [{
                    "kind": "label", "text": entry["value"], "label": letter,
                    "bbox": entry["bbox"], "source_id": entry["id"],
                }],
                "reasoning": [_direction2_reasoning(record, entry, letter,
                                                    letter_in_prose)],
                "cited_ids": [entry["id"]],
            }, None

    # Deliberately no "the answer appears somewhere in the diagram" fallback.
    # It matches almost every food-web question and produces evidence that is
    # true but useless ("the diagram contains the text Crayfish"), which is
    # worse than no evidence: it trains the model to state trivia and stop.
    return None, "no_label_match"


# A question that asks what a letter marks is a LABEL question. Giving it an
# edge list as evidence is the wrong evidence type, and the human audit of 30
# relation rows found exactly that: 0 of 30 were sufficient, and 42% of the
# 1,750 relation rows turned out to be questions of this shape ("What is shown
# at letter D?"). They reach relation_evidence only because label_evidence
# declined them, so the honest outcome is to drop them, not to dress them up.
LABEL_LOOKUP = re.compile(
    r"(shown at|marked|indicated|labell?ed)\s+(by\s+)?(letter|label)?\s*\b[A-Z]\b"
    r"|what\s+does\s+(the\s+)?(letter|label)\s+\b[A-Z]\b"
    r"|which\s+(letter|label)\b"
    r"|at\s+(letter|label)\s+\b[A-Z]\b",
    re.IGNORECASE)

# Families whose inference follows from the DIRECTION of the arrows alone. The
# rest get no reasoning rather than a fabricated one.
REMOVAL = re.compile(
    r"\b(if|when|should)\b.{0,80}?\b(eliminat\w*|remov\w*|die[sd]?\b|died|"
    r"disappear\w*|extinct\w*|eradicat\w*|absent|decrease\w*)", re.IGNORECASE)
PREDATION = re.compile(r"\b(eats?|eaten|feeds?\s+on|prey(s)?\s+on|predator)\b",
                       re.IGNORECASE)
ENERGY = re.compile(r"\b(source\s+of\s+(food|energy)|energy\s+(source|for)|"
                    r"food\s+source|gets?\s+(its\s+)?energy)\b", re.IGNORECASE)

# Life-cycle order. Only the "what follows / what precedes" shape: one edge,
# one direction, and the gold must be its endpoint - the same contract as
# PREDATION. "Which stage is between B and C?" needs a path search between two
# letter-labelled nodes and is left out (11 rows), as is "how many stages"
# (8 rows), which is a count and not a chain at all.
SUCCESSOR = re.compile(r"\b(after|next|follows?|following|comes?\s+after)\b",
                       re.IGNORECASE)
PREDECESSOR = re.compile(r"\b(before|precede[sd]?|preceding|prior\s+to|"
                         r"comes?\s+before)\b", re.IGNORECASE)
BETWEEN = re.compile(r"\bbetween\b", re.IGNORECASE)

# Role questions: a node's position in the web, read off its arrows.
#   producer  - nothing flows into it
#   predator  - nothing flows out of it (nothing eats it)
ROLE_PRODUCER = re.compile(r"\b(producer|autotroph|makes?\s+(its\s+)?own\s+food)\b",
                           re.IGNORECASE)
ROLE_TOP = re.compile(r"\b(only\s+a\s+predator|top\s+predator|apex|"
                      r"not\s+eaten|no\s+predators?)\b", re.IGNORECASE)


def _edges_into(edges, name):
    return [e for e in edges if norm(e["dst"]) == norm(name)]


def _edges_out_of(edges, name):
    return [e for e in edges if norm(e["src"]) == norm(name)]


def _quote_list(names):
    names = [f"\"{n}\"" for n in dict.fromkeys(names)]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


# Some AI2D answers carry the whole option list inside the answer field, e.g.
#   "diving beetle, only (A) diving beetle, only (B) water weed, only (C) ..."
# Every verification here asks whether a derived entity appears in the gold
# string, and a gold like that contains half the diagram, so the check passes
# on rows the arrows do not actually settle. Those golds are refused outright.
OPTION_LIST = re.compile(r"\(\s*[A-Da-d]\s*\)")


def gold_is_checkable(gold: str) -> bool:
    text = str(gold).strip()
    if len(OPTION_LIST.findall(text)) >= 2:
        return False
    # a gold long enough to mention several entities cannot be verified by a
    # substring test either
    return len(text) <= 60


def arrow_convention(edges):
    """'standard' | 'reversed' | 'unknown' for this diagram's food-web arrows.

    The food-web chains all read a direction off the arrow: an arrow from the
    food to the eater. Some AI2D diagrams draw it the other way, and some mix
    the two in one picture - 1977.png has `the sun! -> Grass` (energy) beside
    `ants -> Grass` (eating). The human re-audit caught two such rows and both
    chains were confidently backwards.

    The gold check does not catch this. It verifies the ENTITY the chain names,
    and in both cases the entity was right; what was backwards was the RELATION.

    A producer is where energy starts, so under the standard convention nothing
    points at it. That anchors the diagram. When no producer is named, or the
    producers disagree with each other, the convention is unknown and the
    direction-dependent chains are refused rather than guessed (D68).
    """
    outdeg, indeg = collections.Counter(), collections.Counter()
    for edge in edges:
        outdeg[norm(edge["src"])] += 1
        indeg[norm(edge["dst"])] += 1
    producers = {n for n in set(outdeg) | set(indeg) if PRODUCER_WORD.search(n)}
    if not producers:
        return "unknown"
    only_out = {p for p in producers if outdeg[p] and not indeg[p]}
    only_in = {p for p in producers if indeg[p] and not outdeg[p]}
    if only_out and not only_in:
        return "standard"
    if only_in and not only_out:
        return "reversed"
    return "unknown"


# Narrow on purpose: a word that is only sometimes a producer ("plankton", which
# may be zooplankton) would mislabel the diagram, and a wrong anchor is worse
# than no anchor - it would flip every chain in that picture.
PRODUCER_WORD = re.compile(
    r"\b(sun|sunlight|solar)\b"
    r"|\b(green\s+)?plants?\b|\bgrass(es)?\b|\btrees?\b|\bleaves\b|\bleaf\b"
    r"|\balgae?\b|\bseaweed\b|\bphytoplankton\b|\bkelp\b|\bmoss\b"
    r"|\bshrubs?\b|\bflowers?\b|\bberries\b|\bnectar\b|\bwood\b",
    re.IGNORECASE)


def relation_chain(question, gold, edges):
    """A reasoning chain the arrows actually license, or (None, why).

    Only the three families below are derivable from edge direction. Everything
    else used to get "In the diagram, X is connected to Y." followed by the
    answer - a recitation with no inferential step, which the audit judged
    insufficient in 23 of 30 rows and a recitation in 30 of 30.
    """
    if not gold_is_checkable(gold):
        return None, "gold_embeds_the_option_list_so_nothing_can_be_verified"
    # Every food-web chain below reads a direction off the arrow. Life-cycle
    # order does not - there the arrow means "next stage" and has no energy
    # convention to get backwards - so the gate is applied per family, not to
    # the whole function (D68).
    directed_ok = arrow_convention(edges) == "standard"
    gold_n = norm(gold)
    # the question entity is the named thing the question talks about: the one
    # whose name appears in the question but is not the answer
    q_entities = [e for e in edges
                  if norm(e["src"]) != gold_n and len(norm(e["src"])) > 3
                  and norm(e["src"]) in norm(question)]
    q_entities += [e for e in edges
                   if norm(e["dst"]) != gold_n and len(norm(e["dst"])) > 3
                   and norm(e["dst"]) in norm(question)]

    if directed_ok and REMOVAL.search(question):
        # Removing X has two consequences, and which one the question asks for
        # is not in the arrows:
        #   what X ate (arrows INTO X)      -> increases
        #   what ate X (arrows OUT OF X)    -> decreases
        # The first version of this template always told the "increases" story
        # and was wrong on 3 of the first 4 rows: it answered "who benefits"
        # when the question asked "who suffers", and once it removed the wrong
        # animal entirely. So the chain is now DERIVED and then CHECKED against
        # the gold answer, the same contract ChartQA's arithmetic has - if the
        # conclusion the arrows license is not the gold answer, there is no
        # chain and the row is dropped.
        q_norm = norm(question)
        named = {norm(e[k]): e[k] for e in edges for k in ("src", "dst")}
        candidates = [name for key, name in named.items()
                      if len(key) > 3 and key in q_norm and key not in gold_n]
        for removed in candidates:
            prey = [e["src"] for e in _edges_into(edges, removed)
                    if norm(e["src"]) != norm(removed)]
            eaters = [e["dst"] for e in _edges_out_of(edges, removed)
                      if norm(e["dst"]) != norm(removed)]

            hit_prey = [p for p in prey if norm(p) in gold_n or gold_n in norm(p)]
            hit_eater = [p for p in eaters if norm(p) in gold_n or gold_n in norm(p)]
            says_up = any(w in gold_n for w in ("increase", "more", "rise", "grow"))
            says_down = any(w in gold_n for w in ("decrease", "less", "fewer",
                                                  "decline", "suffer", "die"))

            # increases: the gold names something the removed animal ate, and
            # does not contradict the direction
            if hit_prey and not says_down:
                return ([
                    f"Arrows run from {_quote_list(hit_prey)} to \"{removed}\", "
                    f"so \"{removed}\" consumes them.",
                    f"With \"{removed}\" gone, nothing consumes "
                    f"{_quote_list(hit_prey)} along that arrow, so they increase.",
                ], [e for e in _edges_into(edges, removed)
                    if e["src"] in hit_prey][:3])

            # decreases: the gold names something that ate the removed animal
            if hit_eater and not says_up:
                return ([
                    f"An arrow runs from \"{removed}\" to "
                    f"{_quote_list(hit_eater)}, so {_quote_list(hit_eater)} "
                    f"feeds on \"{removed}\".",
                    f"With \"{removed}\" gone, that food is gone, so "
                    f"{_quote_list(hit_eater)} decreases.",
                ], [e for e in _edges_out_of(edges, removed)
                    if e["dst"] in hit_eater][:3])

    if directed_ok and PREDATION.search(question):
        for edge in q_entities:
            name = edge["src"] if norm(edge["src"]) in norm(question) else edge["dst"]
            out = [e for e in _edges_out_of(edges, name)
                   if norm(e["dst"]) in gold_n or gold_n in norm(e["dst"])]
            if out:
                return ([
                    f"An arrow runs from \"{name}\" to \"{out[0]['dst']}\".",
                    f"The arrow points to the consumer, so \"{out[0]['dst']}\" "
                    f"feeds on \"{name}\".",
                ], out[:2])

    if directed_ok and ENERGY.search(question):
        for edge in q_entities:
            name = edge["dst"] if norm(edge["dst"]) in norm(question) else edge["src"]
            into = [e for e in _edges_into(edges, name)
                    if norm(e["src"]) in gold_n or gold_n in norm(e["src"])]
            if into:
                return ([
                    f"An arrow runs from \"{into[0]['src']}\" to \"{name}\".",
                    f"Energy travels along the arrow, so \"{name}\" takes its "
                    f"energy from \"{into[0]['src']}\".",
                ], into[:2])

    # --- life-cycle order: what follows / what precedes -------------------
    # "between X and Y" needs a path search, so it is refused rather than
    # answered with a single edge that happens to touch the gold.
    if (SUCCESSOR.search(question) or PREDECESSOR.search(question)) \
            and not BETWEEN.search(question):
        forward = bool(SUCCESSOR.search(question))
        q_norm = norm(question)
        named = {norm(e[k]): e[k] for e in edges for k in ("src", "dst")}
        for key, name in named.items():
            if len(key) <= 3 or key not in q_norm or key in gold_n:
                continue
            step = _edges_out_of(edges, name) if forward else _edges_into(edges, name)
            others = [e["dst"] if forward else e["src"] for e in step]
            others = [o for o in dict.fromkeys(others) if norm(o) != norm(name)]
            if not others:
                continue
            # Every stage the arrows reach must appear in the gold answer, and
            # the gold must not name a stage the arrows do not reach. A cycle
            # has a well-defined next step, so this is checkable.
            if not others or not all(norm(o) in gold_n for o in others):
                continue
            if forward:
                chain = [
                    f"An arrow runs from \"{name}\" to {_quote_list(others)}.",
                    f"The arrow gives the order of the cycle, so "
                    f"{_quote_list(others)} follows \"{name}\".",
                ]
            else:
                chain = [
                    f"An arrow runs from {_quote_list(others)} to \"{name}\".",
                    f"The arrow gives the order of the cycle, so "
                    f"{_quote_list(others)} comes before \"{name}\".",
                ]
            picked = step[:3]
            return chain, picked

    # --- role in the web: read off a node's in/out arrows ------------------
    # A role is a property of the WHOLE web, so these use every edge, not just
    # the ones near the question.
    if directed_ok and (ROLE_PRODUCER.search(question)
                        or ROLE_TOP.search(question)):
        incoming = collections.Counter(norm(e["dst"]) for e in edges)
        outgoing = collections.Counter(norm(e["src"]) for e in edges)
        named = {norm(e[k]): e[k] for e in edges for k in ("src", "dst")}
        target = next((name for key, name in named.items()
                       if len(key) > 3 and (key in gold_n or gold_n in key)), None)
        if target:
            key = norm(target)
            if ROLE_PRODUCER.search(question) and incoming[key] == 0 and outgoing[key]:
                eaten_by = [e["dst"] for e in _edges_out_of(edges, target)][:2]
                return ([
                    f"No arrow in the diagram points to \"{target}\"; arrows only "
                    f"leave it, towards {_quote_list(eaten_by)}.",
                    f"Nothing feeds \"{target}\", so it makes its own food and is "
                    f"the producer.",
                ], _edges_out_of(edges, target)[:3])
            if ROLE_TOP.search(question) and outgoing[key] == 0 and incoming[key]:
                eats = [e["src"] for e in _edges_into(edges, target)][:2]
                return ([
                    f"Arrows reach \"{target}\" from {_quote_list(eats)}, and no "
                    f"arrow leaves it.",
                    f"Nothing feeds on \"{target}\", so it is only a predator.",
                ], _edges_into(edges, target)[:3])

    if not directed_ok:
        return None, "arrow_convention_unconfirmed_for_this_diagram"
    return None, "no_derivable_chain"


def relation_evidence(record, ann, max_edges=3, max_label_chars=32):
    """Structure questions. Grounded, and checked weakly for sufficiency."""
    if LABEL_LOOKUP.search(record["question"]):
        return None, "label_lookup_question_not_a_relation_question"
    texts = ann["text"]

    # In these diagrams an arrow usually joins two regions, and a separate
    # relationship says which text labels each region. Without composing the two,
    # every edge reads "a region is connected to a region". Resolve regions to
    # their label first.
    label_of_region = {}
    for rel in ann["relationships"].values():
        if rel.get("category") not in ("intraObjectLabel", "intraObjectRegionLabel"):
            continue
        src_text = texts.get(rel.get("origin"))
        region = rel.get("destination")
        if src_text and src_text.get("value") and region:
            label_of_region.setdefault(region, src_text["value"])

    def resolve(nid):
        node = texts.get(nid)
        if node and node.get("value"):
            return node["value"]
        return label_of_region.get(nid, nid)

    edges, cited = [], set()
    for rid, rel in ann["relationships"].items():
        if rel.get("category") not in EDGE_CATEGORIES:
            continue
        origin, dest, conn = rel.get("origin"), rel.get("destination"), rel.get("connector")
        edges.append({
            "kind": "edge",
            "category": rel["category"],
            "src": resolve(origin),
            "dst": resolve(dest),
            "connector": conn,
            "directed": bool(rel.get("hasDirectionality")),
            "source_id": rid,
        })
        for nid in (origin, dest, conn):
            if nid:
                cited.add(nid)
    if not edges:
        return None, "no_edges"

    # Grounding: every cited id must exist in this diagram's annotation.
    unknown = [nid for nid in cited if nid not in ann["nodes"]]
    if unknown:
        return None, "ungrounded_id"

    # Weak sufficiency, in two tiers. Requiring the gold answer to appear in a
    # cited label works for "which part is X" questions but throws away every
    # inference question ("what happens if X disappears"), whose answer is not
    # diagram text - and those are the ones worth training on. So a second tier
    # accepts evidence about the entities the QUESTION names. Which tier passed
    # is recorded per record, so the weaker one can be ablated later.
    gold = norm(record["answer"])
    question = norm(record["question"])
    mentioned = [e for e in edges
                 if gold and (gold in norm(e["src"]) or gold in norm(e["dst"]))]
    tier = "gold_in_cited_label"
    if not mentioned:
        mentioned = [e for e in edges
                     if (len(norm(e["src"])) > 3 and norm(e["src"]) in question)
                     or (len(norm(e["dst"])) > 3 and norm(e["dst"]) in question)]
        tier = "question_entity_in_cited_label"
    if not mentioned:
        return None, "neither_gold_nor_question_entity_cited"

    # Put the edges that mention the answer first, then fill up to max_edges.
    ordered = mentioned + [e for e in edges if e not in mentioned]

    # Only keep edges that say something a reader could check in the picture.
    # An edge between two internal ids ("A4 -- H0") carries no information once
    # the annotation is gone, and training on it teaches the model to emit
    # meaningless tokens.
    # AI2D's text annotations are not all entity names. Some are whole caption
    # paragraphs ("Eggs (0.20-0.50 mm long) Laid in a mass of 200-500 eggs. Laid
    # in or on with flowing water ..."), and citing one as an edge endpoint
    # produced targets of up to 1,106 tokens against a 45-token budget (G2).
    # Anything this long is a caption, not a thing the arrow connects.
    def short_enough(edge):
        return all(len(str(edge.get(k, ""))) <= max_label_chars for k in ("src", "dst"))

    ordered = [e for e in ordered if short_enough(e)]
    if not ordered:
        return None, "every_candidate_edge_cites_a_caption_length_label"

    both = [e for e in ordered if _is_text(e["src"]) and _is_text(e["dst"])
            and norm(e["src"]) != norm(e["dst"])]
    if not both:
        return None, "no_edge_between_two_named_things"

    # The chain decides which edges are cited, not the other way round. The old
    # code filled up to max_edges from whatever the diagram had, and 35.6% of
    # the cited edges touched neither the question nor the answer - 309 rows
    # (17.7%) cited nothing but filler. Padding teaches the model to emit edges
    # it has not checked, which at generation time is hallucination.
    chain, chain_edges = relation_chain(record["question"], record["answer"], both)
    if chain is None:
        return None, chain_edges        # the reason string

    chosen = [e for e in chain_edges if short_enough(e)][:max_edges]
    if not chosen:
        return None, "chain_edges_all_cite_a_caption_length_label"

    cited = set()
    for edge in chosen:
        for part in str(edge.get("source_id", "")).split("+"):
            if part:
                cited.add(part)
    unknown = [nid for nid in cited if nid not in ann["nodes"]]
    if unknown:
        return None, "ungrounded_id"

    return {
        "kind": "relation_grounded",
        "evidence": chosen,
        "reasoning": chain,
        "cited_ids": sorted(cited),
        "sufficiency_tier": tier,
    }, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prefer-wrong", type=pathlib.Path, default=None,
                        metavar="SPLIT.json",
                        help="Phase 0 split; select rows the base model got wrong first")
    parser.add_argument("--report", action="store_true", help="measure coverage, write nothing")
    parser.add_argument("--label-target", type=int, default=1750)
    parser.add_argument("--relation-target", type=int, default=1750)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--letter-in-prose", action="store_true",
                        help="keep the old wording that names the diagram's label "
                             "letter in the reasoning (D65); the default drops it")
    parser.add_argument("--candidates", type=pathlib.Path, default=CAND,
                          help="candidate pool to read; point this at "
                               "training_data/candidates_split/train/ so the "
                               "holdout is never used for training (D51)")
    args = parser.parse_args()
    split = wrong_priority.load_split(args.prefer_wrong)

    # The candidate pool is now split 9:1 at the image level BEFORE any
    # selection (D51), so evidence must be generated from the TRAIN side
    # only. Reading the whole pool would put holdout rows in the training
    # set and make the Phase 2 gate meaningless.
    cand = args.candidates
    if not cand.is_file():
        sys.exit(f"synth_evidence_ai2d: {cand} not found - run split.py first")

    records = [json.loads(line) for line in cand.open(encoding="utf-8") if line.strip()]
    print(f"{len(records):,} AI2D candidate records", flush=True)

    ann_cache: dict[str, dict] = {}
    pools = {"label_grounded": [], "relation_grounded": []}
    reasons = collections.Counter()

    for record in records:
        image_id = (record.get("meta") or {}).get("ai2d_image_id")
        if not image_id:
            reasons["no_image_id"] += 1
            continue
        if image_id not in ann_cache:
            ann_cache[image_id] = load_annotation(image_id)
        ann = ann_cache[image_id]
        if ann is None:
            reasons["no_annotation_file"] += 1
            continue

        built, why = label_evidence(record, ann,
                                    letter_in_prose=args.letter_in_prose)
        if built is None:
            built, why2 = relation_evidence(record, ann)
            if built is None:
                reasons[f"{why}+{why2}"] += 1
                continue
        pools[built["kind"]].append((record, built))

    print()
    for kind, items in pools.items():
        print(f"  {kind:20} {len(items):>6,}")
    print(f"  {'unusable':20} {sum(reasons.values()):>6,}")
    if reasons:
        print("  reasons:")
        for why, n in reasons.most_common(6):
            print(f"    {why:46} {n:>6,}")

    targets = {"label_grounded": args.label_target, "relation_grounded": args.relation_target}
    short = {k: targets[k] - len(v) for k, v in pools.items() if len(v) < targets[k]}
    if short:
        print("\n  SHORT OF TARGET: " + ", ".join(f"{k} by {v:,}" for k, v in short.items()))

    if args.report:
        print("\nreport only - nothing written")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    written = collections.Counter()
    out_path = args.out / "ai2d_evidence.jsonl"
    with out_path.open("w", encoding="utf-8") as sink:
        for kind, items in pools.items():
            rng = random.Random(f"{args.seed}:{kind}")
            rng.shuffle(items)
            # Spend the headroom (1,885 + 2,943 usable for 1,750 + 1,750) on the
            # rows Phase 0 got wrong, rather than on a random draw. Stable, so
            # the shuffle above still decides the order inside each group.
            items = wrong_priority.wrong_first(items, split)
            chosen = items[:targets[kind]]
            wrong_priority.report(chosen, split, label=f"{kind:18}")
            for record, built in chosen:
                meta = dict(record.get("meta") or {})
                meta.update({
                    "evidence": built["evidence"],
                    "knowledge": "",
                    "reasoning": built["reasoning"],
                    "evidence_provenance": "program_synthesized",
                    "evidence_kind": built["kind"],
                    "evidence_cited_ids": built["cited_ids"],
                    "evidence_sufficiency": built.get("sufficiency_tier", "exact_label_match"),
                    # relation_grounded is grounded but its reasoning is not
                    # verified, so it is the queue humans should audit
                    "needs_human_audit": built["kind"] == "relation_grounded",
                })
                sink.write(json.dumps({**record, "meta": meta}, ensure_ascii=False) + "\n")
                written[kind] += 1

    manifest = {
        "seed": args.seed,
        "source": str(cand).replace("\\", "/"),
        "targets": targets,
        "available": {k: len(v) for k, v in pools.items()},
        "written": dict(written),
        "written_total": sum(written.values()),
        "unusable": sum(reasons.values()),
        "unusable_reasons": dict(reasons.most_common()),
        "verification": {
            "label_grounded": "label value equals the gold answer (full check)",
            "relation_grounded": "every cited annotation id exists, and the gold "
                                 "answer appears in a cited label (grounding + weak "
                                 "sufficiency); the reasoning chain is NOT verified",
        },
    }
    (args.out / "ai2d_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # --out may sit outside the repo (a scratch directory, when sampling for a
    # quality check), and relative_to raises on that. The write is already done.
    try:
        shown = out_path.relative_to(ROOT)
    except ValueError:
        shown = out_path
    print(f"\nwrote {sum(written.values()):,} records -> {shown}")
    for kind, n in sorted(written.items()):
        print(f"  {kind:20} {n:>6,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
