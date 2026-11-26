#!/usr/bin/env python3
"""Re-score saved predictions with the current extractor. No GPU, no inference.

`indomain_predictions.jsonl` keeps the model's raw `response` for every row, so
a change to answer_extraction can be applied to runs that already happened. That
matters: the `final_line_letter` rule (D66) moved arm B's AI2D score by 9.67pp,
and without re-scoring every arm the same way the comparison would mix two
extractor versions.

    python code/eval/rescore_predictions.py results/diag-*.jsonl
    python code/eval/rescore_predictions.py results/diag-v3b-predictions.jsonl --write
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "eval"))
import answer_extraction as AX  # noqa: E402

LETTERS = "ABCDEFGHI"
HOLDOUTS = (ROOT / "training_data" / "holdout_v3",
            ROOT / "training_data" / "holdout_v2",
            ROOT / "training_data" / "candidates_holdout")


def load_options():
    """id -> option list, from whichever holdout the row came from."""
    out = {}
    for folder in HOLDOUTS:
        if not folder.is_dir():
            continue
        for path in folder.glob("*.jsonl"):
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    record = json.loads(line)
                    out.setdefault(record["id"], record.get("options") or [])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions", nargs="+", type=pathlib.Path)
    ap.add_argument("--write", action="store_true",
                    help="overwrite the files with the re-scored verdicts")
    args = ap.parse_args()

    options_of = load_options()
    if not options_of:
        sys.exit("rescore: no holdout files found; cannot recover option lists")

    for path in args.predictions:
        if not path.is_file():
            print(f"{path}: not found")
            continue
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        changed = collections.Counter()
        per = collections.defaultdict(collections.Counter)

        for row in rows:
            options = options_of.get(row["id"]) or []
            gold = str(row.get("gold", "")).strip()
            before_ok = bool(row.get("correct"))
            if options:
                omap = {LETTERS[i]: t for i, t in enumerate(options)}
                got = AX.extract_choice(row.get("response") or "", omap,
                                        truncated=row.get("finish_reason") == "length")
                predicted, rule = got.pred, got.method
                after_ok = bool(predicted) and predicted.upper() == gold.upper()
            else:
                # open questions are untouched by this change
                predicted, rule = row.get("predicted"), row.get("extraction_rule")
                after_ok = before_ok

            src = row.get("source", "?")
            per[src]["n"] += 1
            per[src]["before"] += before_ok
            per[src]["after"] += after_ok
            per[src]["unparsed_before"] += row.get("predicted") in (None, "")
            per[src]["unparsed_after"] += predicted in (None, "")
            if after_ok != before_ok:
                changed[f"{src}: {'gained' if after_ok else 'lost'}"] += 1

            row["predicted"], row["extraction_rule"], row["correct"] = \
                predicted, rule, after_ok

        print(f"\n=== {path.name} ===")
        header = (f"{'source':<16}{'n':>7}{'before':>9}{'after':>9}{'Δpp':>8}"
                  f"{'unparsed':>11}{'→':>3}{'':>6}")
        print(header)
        print("-" * len(header))
        tot = collections.Counter()
        for src, c in per.items():
            for k, v in c.items():
                tot[k] += v
            print(f"{src:<16}{c['n']:>7,}{100*c['before']/c['n']:>8.2f}%"
                  f"{100*c['after']/c['n']:>8.2f}%"
                  f"{100*(c['after']-c['before'])/c['n']:>+8.2f}"
                  f"{c['unparsed_before']:>11,}{'→':>3}{c['unparsed_after']:>6,}")
        print(f"{'ALL':<16}{tot['n']:>7,}{100*tot['before']/tot['n']:>8.2f}%"
              f"{100*tot['after']/tot['n']:>8.2f}%"
              f"{100*(tot['after']-tot['before'])/tot['n']:>+8.2f}"
              f"{tot['unparsed_before']:>11,}{'→':>3}{tot['unparsed_after']:>6,}")
        if changed:
            print("  " + ", ".join(f"{k} {v}" for k, v in changed.most_common()))

        if args.write:
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  rewrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
