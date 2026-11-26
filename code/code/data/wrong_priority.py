#!/usr/bin/env python3
"""Order an evidence pool so Phase-0 wrong answers are selected first.

D16 asks the training set to over-weight questions the base model gets wrong,
on the reasoning that a question it already answers teaches less. Phase 0
(DECISIONS.md section 14) measured which those are: 4,542 of 20,222.

Both synthesisers produce more usable rows than their target - AI2D 4,828 for
3,500, ChartQA 4,647 for 3,500 - so which 3,500 is a free choice. Taking them at
random leaves the wrong-answer share at roughly the base rate. This orders the
pool wrong-first instead, which spends the headroom on the harder rows and costs
nothing.

The order stays deterministic: the caller's existing seeded shuffle runs first,
then a STABLE sort moves the wrong rows ahead without disturbing order inside
either group. Same seed, same set.

A row missing from the split is treated as correct, i.e. deprioritised, so a
stale or partial split file can never promote an unmeasured row.
"""
from __future__ import annotations

import json
import pathlib


def load_split(path) -> dict:
    """{question id: was the base model correct} from the Phase 0 predictions."""
    if path is None:
        return {}
    path = pathlib.Path(path)
    if not path.is_file():
        raise SystemExit(f"wrong_priority: {path} not found - run the Phase 0 "
                         f"in-domain job and pull its predictions first")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):           # tolerate {id: correct} as well
        return {str(k): bool(v) for k, v in rows.items()}
    return {str(r["id"]): bool(r["correct"]) for r in rows}


def wrong_first(items, split, id_of=lambda item: item[0]["id"]):
    """Stable-sort `items` so rows the base model got WRONG come first.

    `items` is whatever the caller holds - here (record, built) pairs - and
    `id_of` pulls the question id out of one. False sorts before True, so wrong
    leads; rows absent from the split sort last with the correct ones.
    """
    if not split:
        return list(items)
    return sorted(items, key=lambda item: split.get(str(id_of(item)), True))


def report(chosen, split, id_of=lambda item: item[0]["id"], label="selected"):
    """Print the achieved wrong share, so the number lands in the job log."""
    if not split:
        return None
    flags = [split.get(str(id_of(item))) for item in chosen]
    wrong = sum(1 for f in flags if f is False)
    unknown = sum(1 for f in flags if f is None)
    total = max(len(chosen), 1)
    print(f"  {label}: wrong {wrong:,}/{len(chosen):,} ({100*wrong/total:.1f}%)"
          + (f", {unknown:,} not in the split" if unknown else ""))
    return wrong
