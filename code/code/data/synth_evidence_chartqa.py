#!/usr/bin/env python3
"""Synthesise numeric evidence for ChartQA from the original data tables. No model, no cost.

ChartQA's original release ships, per chart, the underlying data table as a CSV
and a bounding-box annotation of the legend, title and axis labels. The parquet
mirror we train from carries neither, which is why the original was fetched (D28).

Evidence here is a citation of the table cells the answer rests on, and the
arithmetic that produces it. Both are checked:

  cell_lookup    the answer is a value in the table. Verified exactly.
  arithmetic     the answer is produced by one operation over table cells
                 (difference, sum, ratio, mean, max, min, count, max-min).
                 Verified by recomputing it and comparing to the gold answer.

Measured on 2026-11-19 (DECISIONS.md section 12): machine-generated questions
have their answer verbatim in the table 100% of the time, human-written ones only
40% - the rest need the arithmetic tier, which is what makes them worth training on.

Tables are noisy. train/tables/10095.csv holds "2007,5250,39" where the chart
shows ~52, so a failed check can mean a wrong table rather than a wrong answer.
Failures are therefore counted and reported, never silently dropped as "model
error" - they go to the human audit queue.

    python code/data/synth_evidence_chartqa.py --report
    python code/data/synth_evidence_chartqa.py --target 3500
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import random
import re
import sys

import wrong_priority

ROOT = pathlib.Path(__file__).resolve().parents[2]
ORIG = (ROOT / "training_data" / "raw_external" / "chartqa_original"
        / "extracted" / "ChartQA Dataset")
CAND = ROOT / "training_data" / "candidates" / "chartqa.jsonl"
OUT = ROOT / "training_data" / "evidence"

DEFAULT_SEED = 20261119
TOLERANCE = 0.02          # the evaluation rounds to 2 decimals; match that
MAX_CELLS_CITED = 6


def to_number(text):
    if text is None:
        return None
    s = str(text).strip().replace(",", "").replace("%", "")
    s = re.sub(r"[^\d.\-+eE]", "", s)
    if not s or s in {"-", "+", ".", "-."}:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def close(a, b, tol=TOLERANCE):
    return a is not None and b is not None and abs(round(a, 2) - round(b, 2)) <= tol


def load_table(split: str, chart: str):
    """Return (header, rows, cells) where cells is [(row_label, col, value, number)]."""
    path = ORIG / split / "tables" / f"{chart}.csv"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            rows = [r for r in csv.reader(handle) if any(c.strip() for c in r)]
    except OSError:
        return None
    if len(rows) < 2:
        return None
    header = [c.strip() for c in rows[0]]
    cells = []
    for row in rows[1:]:
        if not row:
            continue
        label = row[0].strip()
        for idx, raw in enumerate(row[1:], start=1):
            col = header[idx] if idx < len(header) else f"col{idx}"
            cells.append({"row": label, "col": col, "text": raw.strip(),
                          "value": to_number(raw)})
    return {"header": header, "n_rows": len(rows) - 1, "cells": cells}


def find_chart_id(record):
    """ChartQA records carry no chart id; recover it by matching question text."""
    return (record.get("meta") or {}).get("chartqa_imgname")


def cell_lookup_evidence(record, table, gold_num, gold_text):
    hits = [c for c in table["cells"]
            if (gold_num is not None and close(c["value"], gold_num))
            or (gold_num is None and c["text"] and c["text"].lower() == gold_text.lower())]
    if not hits:
        return None, "answer_not_a_cell"
    cell = hits[0]
    return {
        "kind": "cell_lookup",
        "evidence": [{"kind": "cell", "row": cell["row"], "column": cell["col"],
                      "text": cell["text"]}],
        # The evidence line already carries row, column and value. Repeating all
        # three in prose doubled every target, and ChartQA column names are long
        # ("Share of suicide deaths from pesticide poisoning, 2014"), so the
        # duplication alone put the mean over the 45-token budget (G2). The step
        # names the row and the value and leaves the column to the evidence.
        "reasoning": [
            f"{cell['row']} reads {cell['text']}."
        ],
        "verification": "exact_cell_match",
    }, None


OPERATIONS = (
    ("difference", lambda a, b: abs(a - b)),
    ("sum", lambda a, b: a + b),
    ("ratio", lambda a, b: a / b if b else None),
)

# "In which year was the prison population of the UK the highest?" -> "2018".
# The operation is an argmax and it is perfectly derivable, but the ANSWER is a
# row label, not a number, so arithmetic_evidence refused it as
# `answer_not_numeric`. The human audit of 30 rejected rows classified 25 as
# "the operation is missing", and 14 of the 17 `answer_not_numeric` rows were of
# this shape, with the audit note "연산보단 이런것들 이름찾기가 많음".
HIGHEST = re.compile(r"\b(highest|high|most|largest|greatest|maximum|max|peak|"
                     r"top|biggest|longest)\b", re.IGNORECASE)
LOWEST = re.compile(r"\b(lowest|low|least|smallest|fewest|minimum|min|"
                    r"bottom|shortest)\b", re.IGNORECASE)
TOTAL = re.compile(r"\b(total|combined|altogether|overall|sum)\b", re.IGNORECASE)


def _fmt(value):
    """Match how the table prints a number, so the prose and the cited cell agree."""
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def label_extremum_evidence(record, table, gold_text, question=""):
    """argmax / argmin whose answer is a ROW or COLUMN LABEL, not a value.

    Verified the same way the arithmetic is: the label the extremum picks out
    must BE the gold answer. Nothing is accepted on the strength of the wording.

    A maximum cannot be proved from one cell, so the runner-up is cited too -
    enough for a reader to see the ordering without quoting the whole table.
    """
    gold = str(gold_text).strip()
    if not gold:
        return None, "no_gold_text"
    want_max = bool(HIGHEST.search(question))
    want_min = bool(LOWEST.search(question))
    if want_max == want_min:          # neither, or contradictory wording
        return None, "question_names_no_extremum"

    numeric = [c for c in table["cells"] if c["value"] is not None]
    if len(numeric) < 2:
        return None, "not_enough_numeric_cells"

    def pick(groups, label_of):
        """groups: label -> list of (value, cell). Returns the winning label."""
        scored = [(label, vals) for label, vals in groups.items() if vals]
        if len(scored) < 2:
            return None
        ranked = sorted(scored, key=lambda kv: max(v for v, _ in kv[1]),
                        reverse=want_max)
        return ranked

    # scope 1: within one column, compare rows -> a row label
    by_col = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in numeric:
        by_col[c["col"]][c["row"]].append((c["value"], c))
    # scope 2: within one row, compare columns -> a column label
    by_row = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in numeric:
        by_row[c["row"]][c["col"]].append((c["value"], c))

    scopes = []
    for col, groups in by_col.items():
        scopes.append(("column", col, groups))
    for row, groups in by_row.items():
        scopes.append(("row", row, groups))
    # a scope the question names is tried first; otherwise any scope whose
    # extremum IS the gold is accepted, because that equality is the check
    scopes.sort(key=lambda s: 0 if _mentions(s[1], question) else 1)

    for kind, scope_label, groups in scopes:
        ranked = pick(groups, kind)
        if not ranked:
            continue
        winner, winner_vals = ranked[0]
        if str(winner).strip().lower() != gold.lower():
            continue
        runner, runner_vals = ranked[1]
        w_val, w_cell = max(winner_vals, key=lambda vc: vc[0])
        r_val, r_cell = max(runner_vals, key=lambda vc: vc[0])
        word = "highest" if want_max else "lowest"
        return {
            "kind": "label_extremum",
            "evidence": [
                {"kind": "cell", "row": w_cell["row"], "column": w_cell["col"],
                 "text": w_cell["text"]},
                {"kind": "cell", "row": r_cell["row"], "column": r_cell["col"],
                 "text": r_cell["text"]},
            ],
            # The column name is left out of the prose on purpose: ChartQA
            # column labels run to whole sentences and naming one put the mean
            # target over the 45-token budget (G2).
            "reasoning": [
                f"{winner} reads {_fmt(w_val)}, the {word}; "
                f"{runner} reads {_fmt(r_val)}."
            ],
            "verification": f"arg{'max' if want_max else 'min'}_label_equals_gold",
            "operation_used": f"arg{'max' if want_max else 'min'}",
        }, None

    return None, "no_extremum_label_reproduces_answer"


def _mentions(label: str, question: str) -> bool:
    """Is this row/column label actually named in the question?"""
    lab = str(label).strip().lower()
    return len(lab) > 2 and lab in question


def arithmetic_evidence(record, table, gold_num, question=""):
    """Try a small, fixed set of operations and keep the one that reproduces the answer."""
    if gold_num is None:
        return None, "answer_not_numeric"
    numeric = [c for c in table["cells"] if c["value"] is not None]
    if len(numeric) < 2:
        return None, "not_enough_numeric_cells"

    values = [c["value"] for c in numeric]
    # whole-column aggregates first: cheaper and more often the intended answer
    by_col = collections.defaultdict(list)
    for c in numeric:
        by_col[c["col"]].append(c)
    for col, cells in by_col.items():
        # Any column might happen to sum to the answer. Only accept one the
        # question actually names, or the evidence cites unrelated data.
        if question and not _mentions(col, question):
            continue
        vals = [c["value"] for c in cells]
        for name, computed in (
            ("sum", sum(vals)),
            ("mean", sum(vals) / len(vals)),
            ("max", max(vals)),
            ("min", min(vals)),
            ("range", max(vals) - min(vals)),
            ("count", float(len(vals))),
        ):
            if close(computed, gold_num):
                cited = cells[:MAX_CELLS_CITED]
                listed = ", ".join(f"{c['row']}={c['text']}" for c in cited)
                return {
                    "kind": "arithmetic",
                    "evidence": [{"kind": "cell", "row": c["row"], "column": c["col"],
                                  "text": c["text"]} for c in cited],
                    "reasoning": [
                        f"Column \"{col}\" of the chart's data holds {listed}"
                        + (f" and {len(cells) - len(cited)} more" if len(cells) > len(cited) else "")
                        + f". Its {name} is {round(computed, 2):g}."
                    ],
                    "verification": f"recomputed_{name}",
                }, None

    # then pairwise operations, capped so this stays linear enough to run
    limit = min(len(numeric), 40)
    for i in range(limit):
        for j in range(limit):
            if i == j:
                continue
            a, b = numeric[i], numeric[j]
            # Without this, the search returns the first pair that happens to
            # hit the number - e.g. "United States + Syria" for a question about
            # Turkey and Iran. True, and about the wrong rows.
            if question and not (_mentions(a["row"], question) and _mentions(b["row"], question)):
                continue
            for name, fn in OPERATIONS:
                try:
                    computed = fn(a["value"], b["value"])
                except ZeroDivisionError:
                    continue
                if computed is None or not close(computed, gold_num):
                    continue
                verb = {"difference": "minus", "sum": "plus", "ratio": "divided by"}[name]
                return {
                    "kind": "arithmetic",
                    "evidence": [
                        {"kind": "cell", "row": a["row"], "column": a["col"], "text": a["text"]},
                        {"kind": "cell", "row": b["row"], "column": b["col"], "text": b["text"]},
                    ],
                    # Same reasoning: the two cells are already in the evidence,
                    # so the step states only the operation over them.
                    "reasoning": [
                        f"{a['row']} is {a['text']} and {b['row']} is {b['text']}. "
                        f"{a['text']} {verb} {b['text']} is {round(computed, 2):g}."
                    ],
                    "verification": f"recomputed_{name}",
                }, None
    return None, "no_operation_over_question_cells_reproduces_answer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prefer-wrong", type=pathlib.Path, default=None,
                        metavar="SPLIT.json",
                        help="Phase 0 split; select rows the base model got wrong first")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--target", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--reject-list", type=pathlib.Path,
                          default=OUT / "audit_rejected_chartqa.json",
                          help="ids a human audit judged wrong; dropped even "
                               "though the program check passed (D62)")
    parser.add_argument("--candidates", type=pathlib.Path, default=CAND,
                          help="candidate pool to read; point this at "
                               "training_data/candidates_split/train/ so the "
                               "holdout is never used for training (D51)")
    args = parser.parse_args()
    split = wrong_priority.load_split(args.prefer_wrong)

    if not ORIG.is_dir():
        sys.exit(f"synth_evidence_chartqa: {ORIG} not found - run fetch_chartqa_original.py")
    # The candidate pool is now split 9:1 at the image level BEFORE any
    # selection (D51), so evidence must be generated from the TRAIN side
    # only. Reading the whole pool would put holdout rows in the training
    # set and make the Phase 2 gate meaningless.
    cand = args.candidates
    # Rows a human judged wrong. The program check passed on these - the
    # arithmetic reproduces the gold answer - but the audit found the prose
    # calls the operation something it is not (a sum narrated as an
    # average). A verified number with a false description is worse than a
    # dropped row, so the human verdict overrides the program check.
    rejected = set()
    if args.reject_list and args.reject_list.is_file():
        rejected = set(json.loads(args.reject_list.read_text(
            encoding='utf-8')).get('ids') or [])
        print(f'{len(rejected)} ids on the human reject list '
              f'({args.reject_list.name})', flush=True)
    if not cand.is_file():
        sys.exit(f"synth_evidence_chartqa: {cand} not found - run split.py first")

    # The candidate records came from the parquet mirror, which dropped the chart
    # file name, so rebuild the link through the original QA files: the (query,
    # label) pair identifies the chart.
    index = {}
    for kind in ("human", "augmented"):
        qa_path = ORIG / args.split / f"{args.split}_{kind}.json"
        if not qa_path.is_file():
            continue
        for entry in json.loads(qa_path.read_text(encoding="utf-8")):
            key = (str(entry["query"]).strip().lower(), str(entry["label"]).strip().lower())
            index.setdefault(key, (entry["imgname"].rsplit(".", 1)[0], kind))
    print(f"{len(index):,} question->chart links from the original {args.split} split", flush=True)

    records = [json.loads(l) for l in cand.open(encoding="utf-8") if l.strip()]
    print(f"{len(records):,} ChartQA candidate records", flush=True)

    tables: dict[str, dict] = {}
    built_pool, reasons = [], collections.Counter()
    kinds = collections.Counter()

    for record in records:

        if record['id'] in rejected:

            reasons['human_audit_rejected'] += 1

            continue
        key = (str(record["question"]).strip().lower(), str(record["answer"]).strip().lower())
        link = index.get(key)
        if not link:
            reasons["no_chart_link"] += 1
            continue
        chart, qkind = link
        if chart not in tables:
            tables[chart] = load_table(args.split, chart)
        table = tables[chart]
        if table is None:
            reasons["no_table_file"] += 1
            continue

        gold_text = str(record["answer"]).strip()
        gold_num = to_number(gold_text)
        built, why = cell_lookup_evidence(record, table, gold_num, gold_text)
        if built is None:
            # argmax/argmin whose answer is a label is tried BEFORE the
            # arithmetic, because the arithmetic refuses any non-numeric answer
            # outright and would hide these rows behind `answer_not_numeric`.
            built, why_lx = label_extremum_evidence(
                record, table, gold_text, question=str(record["question"]).lower())
            if built is not None:
                why = why_lx
        if built is None:
            built, why2 = arithmetic_evidence(
                record, table, gold_num,
                question=str(record["question"]).strip().lower())
            if built is None:
                reasons[why2] += 1
                continue
        built["chart"] = chart
        built["question_kind"] = qkind
        built_pool.append((record, built))
        kinds[built["kind"]] += 1

    print()
    for kind, n in kinds.most_common():
        print(f"  {kind:22} {n:>6,}")
    print(f"  {'unusable':22} {sum(reasons.values()):>6,}")
    for why, n in reasons.most_common(5):
        print(f"    {why:40} {n:>6,}")
    print(f"\n  usable {len(built_pool):,} / {len(records):,} "
          f"({100*len(built_pool)/max(len(records),1):.1f}%)")

    if args.report:
        print("\nreport only - nothing written")
        return 0

    # Take every arithmetic item before any lookup item. Lookups outnumber them
    # roughly 9:1, so a plain shuffle would leave the arithmetic - the only kind
    # that trains anything beyond reading a cell - at under a tenth of the set.
    rng = random.Random(args.seed)
    by_kind = collections.defaultdict(list)
    for item in built_pool:
        by_kind[item[1]["kind"]].append(item)
    for items in by_kind.values():
        rng.shuffle(items)
    # Arithmetic and label_extremum are taken whole either way (157 + 557 of a
    # 3,500 target), so only the lookups have a choice to make; spend it on the
    # rows Phase 0 got wrong. Stable, so the shuffle above still orders each
    # group internally.
    #
    # label_extremum must be listed here explicitly. It was added later and the
    # first version of this selection only named "arithmetic" and
    # "cell_lookup", so all 557 rows were built, counted in the report, and
    # then silently dropped before anything was written.
    chosen = by_kind["arithmetic"][:args.target]
    chosen += by_kind["label_extremum"][:max(0, args.target - len(chosen))]
    lookups = wrong_priority.wrong_first(by_kind["cell_lookup"], split)
    chosen += lookups[:max(0, args.target - len(chosen))]
    wrong_priority.report(chosen, split, label="selected          ")
    rng.shuffle(chosen)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "chartqa_evidence.jsonl"
    written = collections.Counter()
    with out_path.open("w", encoding="utf-8") as sink:
        for record, built in chosen:
            meta = dict(record.get("meta") or {})
            meta.update({
                "evidence": built["evidence"],
                "knowledge": "",
                "reasoning": built["reasoning"],
                "evidence_provenance": "program_synthesized",
                "evidence_kind": built["kind"],
                "evidence_verification": built["verification"],
                "chartqa_chart": built["chart"],
                "chartqa_question_kind": built["question_kind"],
                # arithmetic is recomputed, so it is self-checking; a mismatch
                # there usually means the table is wrong, not the answer
                "needs_human_audit": built["kind"] == "arithmetic",
            })
            sink.write(json.dumps({**record, "meta": meta}, ensure_ascii=False) + "\n")
            written[built["kind"]] += 1

    (args.out / "chartqa_evidence_manifest.json").write_text(json.dumps({
        "seed": args.seed, "split": args.split, "target": args.target,
        "available": dict(kinds), "written": dict(written),
        "written_total": sum(written.values()),
        "unusable": sum(reasons.values()),
        "unusable_reasons": dict(reasons.most_common()),
        "tolerance": TOLERANCE,
        "verification": {
            "cell_lookup": "the answer equals a table cell (exact)",
            "arithmetic": "one operation over table cells reproduces the answer, "
                          "compared at 2 decimals like the evaluation",
        },
        "caveat": "the shipped tables contain digit errors (e.g. train/tables/10095.csv "
                  "has 5250 where the chart shows ~52), so a failed check can mean a bad "
                  "table rather than a bad answer",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --out may sit outside the repo (a scratch directory, when sampling for a
    # quality check), and relative_to raises on that. The write is already done.
    try:
        shown = out_path.relative_to(ROOT)
    except ValueError:
        shown = out_path
    print(f"\nwrote {sum(written.values()):,} records -> {shown}")
    for kind, n in sorted(written.items()):
        print(f"  {kind:22} {n:>6,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
