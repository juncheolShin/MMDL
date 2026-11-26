#!/usr/bin/env python3
"""Paired comparison of trained arms against the base model, on the same rows.

Every arm is scored on the in-domain holdout, and Phase 0 already recorded the
base model's verdict for every candidate row (results/phase0/indomain_split.json),
so each arm gets a PAIRED comparison with no extra inference: how many rows it
recovered (base wrong -> arm right) and how many it lost (base right -> arm
wrong), with McNemar's exact test on the discordant pairs.

Unpaired accuracy differences hide this. arm A at lr 1e-4 moved the overall
figure by -4.51pp, which is 90 recoveries against 241 regressions - two numbers
that say different things about what training did, and only the pair shows it.

    python code/eval/compare_arms.py \\
        --run "arm A=results/diag-v2a-predictions.jsonl" \\
        --run "arm B=results/diag-v2b-predictions.jsonl"
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPLIT = ROOT / "results" / "phase0" / "indomain_split.json"


def mcnemar_exact(recovered: int, regressed: int) -> float:
    """Two-sided exact binomial on the discordant pairs. The concordant pairs
    carry no information about a difference, so they are not in the test."""
    n = recovered + regressed
    if n == 0:
        return float("nan")
    k = min(recovered, regressed)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2 ** n))


def load_predictions(path: pathlib.Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                    help="repeatable; a run's predictions and what to call it")
    ap.add_argument("--split", type=pathlib.Path, default=SPLIT)
    args = ap.parse_args()
    if not args.run:
        sys.exit("compare_arms: at least one --run LABEL=PATH is required")
    if not args.split.is_file():
        sys.exit(f"compare_arms: {args.split} not found (Phase 0 base verdicts)")

    base = {entry["id"]: bool(entry["correct"])
            for entry in json.loads(args.split.read_text(encoding="utf-8"))}

    for spec in args.run:
        if "=" not in spec:
            sys.exit(f"compare_arms: --run wants LABEL=PATH, got {spec!r}")
        label, raw = spec.split("=", 1)
        path = pathlib.Path(raw)
        if not path.is_file():
            print(f"\n=== {label} ===\n  (no predictions at {path})")
            continue
        rows = load_predictions(path)

        per = collections.defaultdict(collections.Counter)
        unscored = 0
        for row in rows:
            rid = row.get("id")
            if rid not in base:
                unscored += 1
                continue
            cell = (base[rid], bool(row["correct"]))
            per[row.get("source", "?")][cell] += 1
            per["ALL"][cell] += 1

        print(f"\n=== {label} ===")
        if unscored:
            print(f"  note: {unscored} rows had no Phase 0 base verdict and are "
                  f"left out of the pairing")
        header = (f"{'source':<16}{'n':>7}{'base':>9}{'arm':>9}{'Δpp':>9}"
                  f"{'recovered':>11}{'regressed':>11}{'McNemar p':>12}")
        print(header)
        print("-" * len(header))
        for source in [s for s in per if s != "ALL"] + ["ALL"]:
            counts = per[source]
            n = sum(counts.values())
            if not n:
                continue
            base_ok = counts[(True, True)] + counts[(True, False)]
            arm_ok = counts[(True, True)] + counts[(False, True)]
            recovered = counts[(False, True)]
            regressed = counts[(True, False)]
            p = mcnemar_exact(recovered, regressed)
            print(f"{source:<16}{n:>7,}{100*base_ok/n:>8.2f}%{100*arm_ok/n:>8.2f}%"
                  f"{100*(arm_ok-base_ok)/n:>+9.2f}{recovered:>11,}{regressed:>11,}"
                  f"{p:>12.3g}")

        base_wrong = per["ALL"][(False, True)] + per["ALL"][(False, False)]
        if base_wrong:
            rate = 100 * per["ALL"][(False, True)] / base_wrong
            print(f"\n  recovery rate on the rows the base model got wrong: "
                  f"{per['ALL'][(False, True)]:,}/{base_wrong:,} = {rate:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
