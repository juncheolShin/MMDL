#!/usr/bin/env python3
"""Collect arm results into one table, with paired significance tests.

MMMU validation has 900 questions, so at around 50% accuracy the standard error
of a single run is about 1.67 percentage points (DECISIONS.md section 6). A 1-2
point gap between two arms is inside that, and ranking arms on it would be
reading noise. This reports:

  * per-arm accuracy, averaged over seeds, with the spread
  * recovery and regression against the baseline, per question
  * McNemar's exact test on each arm-vs-baseline pair
  * the runaway rate - responses cut off at the token limit with no answer
    stated. The base model does this on 42 of 900 (4.7%), all scored wrong, and
    stopping it is the mechanism the 45-token answer format is for, so it moves
    before accuracy does and belongs next to it (DECISIONS.md section 14, G4).

McNemar is the right test here because the arms answer the SAME questions: it
looks only at the questions where the two runs disagree, which is far more
sensitive than comparing two independent accuracy figures.

It also prints the trainable-parameter count per arm, because arms B and C are
not capacity-matched (132.12M vs 109.11M) and a difference between them may be
capacity rather than pathway - that has to travel with the numbers, not be
remembered separately.

    python code/eval/aggregate.py --results results --baseline evaluation_20260928_173159
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def binom_two_sided(b: int, c: int) -> float:
    """Exact McNemar: P(|X - n/2| >= |b - n/2|) for X ~ Binomial(b + c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    observed = min(b, c)
    tail = sum(math.comb(n, k) for k in range(observed + 1))
    p = 2.0 * tail / (2 ** n)
    return min(1.0, p)


def load_per_sample(run_dir: pathlib.Path):
    """question id -> correct, from score_mmmu.py's per_sample.csv."""
    path = run_dir / "per_sample.csv"
    if not path.is_file():
        return None
    import csv
    out = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        id_key = next((k for k in (reader.fieldnames or []) if k.lower() in ("id", "question_id")), None)
        ok_key = next((k for k in (reader.fieldnames or [])
                       if k.lower() in ("correct", "is_correct", "structured_correct")), None)
        if not id_key or not ok_key:
            return None
        for row in reader:
            value = str(row[ok_key]).strip().lower()
            out[row[id_key]] = value in ("1", "true", "yes", "correct")
    return out


def load_runaway(run_dir: pathlib.Path):
    """How often the model talked past the token limit without stating an answer.

    Phase 0 found all 42 unparsed multiple-choice responses were finish_reason
    "length" at exactly 512 tokens: the base model did not misread the image, it
    ran away generating and never got to an answer. They are 4.7% of MMMU and all
    counted wrong (DECISIONS.md section 14, G4).

    That is precisely what the evidence -> reasoning -> answer format under a
    45-token budget is meant to stop, so the runaway rate is the most direct
    intermediate measure of whether an arm's training took. An arm can move it
    without moving accuracy - the model answers concisely but still wrongly -
    and that is worth seeing rather than hiding inside the accuracy column.

    Returns (runaway, scored, mean_output_tokens) or None if the CSV lacks the
    columns, which is the case for runs scored before this was added.
    """
    path = run_dir / "per_sample.csv"
    if not path.is_file():
        return None
    import csv
    runaway = scored = 0
    tokens = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "finish_reason" not in fields:
            return None
        for row in reader:
            scored += 1
            if "n_output_tokens" in fields:
                try:
                    tokens.append(int(row["n_output_tokens"]))
                except (TypeError, ValueError):
                    pass
            # cut off AND no answer extracted: the two together are the failure.
            # A response cut off after stating its answer still counts.
            if row.get("finish_reason") == "length" and not str(row.get("pred", "")).strip():
                runaway += 1
    mean_tokens = sum(tokens) / len(tokens) if tokens else None
    return runaway, scored, mean_tokens


def load_scores(run_dir: pathlib.Path):
    path = run_dir / "scores.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def arm_and_seed(name: str):
    """Split a run directory name like 'a_prime-seed1' or 'c1-seed0-1119'."""
    m = re.match(r"^(?:yeunsuh-)?(?P<arm>[a-z0-9_]+?)[-_]seed(?P<seed>\d+)", name)
    if m:
        return m.group("arm"), int(m.group("seed"))
    return name, None


def trainable_of(run_dir: pathlib.Path):
    for candidate in (run_dir / "trainable_params.json",
                      run_dir.parent / run_dir.name / "trainable_params.json"):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8")).get("trainable_parameters")
            except json.JSONDecodeError:
                return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=pathlib.Path, default=ROOT / "results")
    parser.add_argument("--baseline", required=True, help="run directory name of the baseline")
    parser.add_argument("--checkpoints", type=pathlib.Path, default=None,
                        help="where trainable_params.json lives, if not beside the results")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if not args.results.is_dir():
        sys.exit(f"aggregate: {args.results} is not a directory")
    base_dir = args.results / args.baseline
    base = load_per_sample(base_dir)
    if base is None:
        sys.exit(f"aggregate: no usable per_sample.csv in {base_dir}")
    print(f"baseline: {args.baseline}  ({len(base)} questions, "
          f"{sum(base.values())} correct, {100*sum(base.values())/len(base):.2f}%)")

    runs = collections.defaultdict(list)
    for run_dir in sorted(p for p in args.results.iterdir() if p.is_dir()):
        if run_dir.name == args.baseline:
            continue
        per = load_per_sample(run_dir)
        if per is None:
            continue
        arm, seed = arm_and_seed(run_dir.name)
        scores = load_scores(run_dir) or {}
        trainable = trainable_of(run_dir)
        if trainable is None and args.checkpoints:
            trainable = trainable_of(args.checkpoints / run_dir.name)
        runs[arm].append({"dir": run_dir.name, "seed": seed, "per_sample": per,
                          "runaway": load_runaway(run_dir),
                          "scores": scores, "trainable": trainable})

    if not runs:
        sys.exit(f"aggregate: no arm runs with per_sample.csv found under {args.results}")

    rows = []
    for arm, entries in sorted(runs.items()):
        accs, recov, regr, pvals = [], [], [], []
        for entry in entries:
            per = entry["per_sample"]
            shared = [q for q in base if q in per]
            if not shared:
                continue
            accs.append(sum(per[q] for q in shared) / len(shared))
            # b: baseline wrong -> arm right. c: baseline right -> arm wrong.
            b = sum(1 for q in shared if not base[q] and per[q])
            c = sum(1 for q in shared if base[q] and not per[q])
            base_wrong = sum(1 for q in shared if not base[q])
            base_right = len(shared) - base_wrong
            recov.append(b / base_wrong if base_wrong else 0.0)
            regr.append(c / base_right if base_right else 0.0)
            pvals.append((b, c, binom_two_sided(b, c)))
        if not accs:
            continue
        trainables = {e["trainable"] for e in entries if e["trainable"]}
        runaways = [e["runaway"] for e in entries if e.get("runaway")]
        runaway_rate = (sum(r / max(n, 1) for r, n, _ in runaways) / len(runaways)
                        if runaways else None)
        tok = [m for _, _, m in runaways if m is not None]
        mean_tokens = sum(tok) / len(tok) if tok else None
        rows.append({
            "arm": arm, "seeds": len(accs),
            "accuracy_mean": sum(accs) / len(accs),
            "accuracy_min": min(accs), "accuracy_max": max(accs),
            "recovery_mean": sum(recov) / len(recov),
            "regression_mean": sum(regr) / len(regr),
            "mcnemar": pvals,
            "trainable_parameters": sorted(trainables)[0] if trainables else None,
            "runaway_rate": runaway_rate,
            "mean_output_tokens": mean_tokens,
        })

    base_acc = sum(base.values()) / len(base)
    base_runaway = load_runaway(base_dir)
    print()
    def runaway_cell(rate):
        return f"{100*rate:.1f}%" if rate is not None else "-"

    def tokens_cell(mean):
        return f"{mean:.1f}" if mean is not None else "-"

    header = (f"{'arm':14}{'seeds':>6}{'acc':>9}{'range':>14}{'vs base':>10}"
              f"{'recov':>8}{'regr':>8}{'McNemar p':>11}{'trainable':>12}"
              f"{'runaway':>10}{'out tok':>9}")
    print(header)
    print("-" * len(header))
    if base_runaway:
        r, n, m = base_runaway
        print(f"{'(baseline)':14}{'':>6}{100*base_acc:>8.2f}%{'':>14}{'':>10}"
              f"{'':>8}{'':>8}{'':>11}{'':>12}"
              f"{runaway_cell(r / max(n, 1)):>10}{tokens_cell(m):>9}")
    for row in rows:
        ps = [p for _, _, p in row["mcnemar"]]
        p_shown = min(ps) if ps else float("nan")
        rng = f"{100*row['accuracy_min']:.2f}-{100*row['accuracy_max']:.2f}"
        trn = f"{row['trainable_parameters']/1e6:.1f}M" if row["trainable_parameters"] else "-"
        print(f"{row['arm']:14}{row['seeds']:>6}{100*row['accuracy_mean']:>8.2f}%"
              f"{rng:>14}{100*(row['accuracy_mean']-base_acc):>+9.2f}p"
              f"{100*row['recovery_mean']:>7.1f}%{100*row['regression_mean']:>7.1f}%"
              f"{p_shown:>11.4f}{trn:>12}"
              f"{runaway_cell(row['runaway_rate']):>10}"
              f"{tokens_cell(row['mean_output_tokens']):>9}")

    # The standard error that decides whether any of the above means anything.
    n = len(base)
    se = math.sqrt(base_acc * (1 - base_acc) / n)
    print(f"\nbaseline accuracy {100*base_acc:.2f}% on {n} questions; "
          f"1 SE = {100*se:.2f}pp, so a gap under {100*2*se:.2f}pp is within two "
          f"standard errors of noise.")
    print("McNemar p is the smallest across that arm's seeds; it tests the paired "
          "disagreements, not the accuracy gap.")
    if any(r["trainable_parameters"] for r in rows):
        print("Trainable-parameter counts differ across arms by design - a B-vs-C "
              "difference may be capacity rather than pathway (DECISIONS.md section 2).")

    out = args.out or (args.results / "registry.json")
    out.write_text(json.dumps({
        "baseline": args.baseline,
        "baseline_accuracy": round(base_acc, 4),
        "baseline_runaway": (
            {"n": base_runaway[0], "of": base_runaway[1],
             "rate": round(base_runaway[0] / max(base_runaway[1], 1), 4),
             "mean_output_tokens": (round(base_runaway[2], 2)
                                    if base_runaway[2] is not None else None)}
            if base_runaway else None),
        "n_questions": n,
        "standard_error_pp": round(100 * se, 3),
        "arms": [{k: v for k, v in r.items() if k != "mcnemar"} | {
            "mcnemar": [{"baseline_wrong_arm_right": b, "baseline_right_arm_wrong": c,
                         "p_value": round(p, 6)} for b, c, p in r["mcnemar"]]
        } for r in rows],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
