#!/usr/bin/env python3
"""Compact, checkable evidence for this experiment's two headline findings.

`results/` is gitignored, so a fresh clone cannot re-run the analyses. This
script distils the two claims that are easy to get wrong into small CSVs that
ARE tracked, and then reads them back with no GPU and no downloads.

    python experiments/evidence_supervision/make_evidence.py --build   # needs results/
    python experiments/evidence_supervision/make_evidence.py           # check only

Finding 1 (official_vs_structured.csv)
    The official MMMU scorer guesses a random option when it cannot parse, so
    it pays a bonus to models that fail to answer. Comparing models on
    `official_correct` therefore rewards the opposite of what we trained for.

Finding 2 (truncated_42.csv)
    The baseline leaves 42 questions with nothing to extract - all 42 are the
    same failure, `finish_reason == length` at 512 tokens. This records what
    each arm did on those same 42 rows, which is where the "finishing" benefit
    of structured targets shows up.

Run from the repository root.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

# label -> per_sample.csv, relative to the repository root
RUNS = {
    "baseline": "results/pilot-a-mmmu/base_per_sample.csv",
    "a_cqa": "results/mmmu-a_cqa/per_sample.csv",
    "b_cqa": "results/mmmu-b_cqa/per_sample.csv",
    "b_cqa_shape": "results/mmmu-bshape/per_sample.csv",
    "b_cqa_shape_s1": "results/mmmu-bshape1/per_sample.csv",
    "b_cqa_shape_s2": "results/mmmu-bshape2/per_sample.csv",
    "t_stop": "results/mmmu-tstop/per_sample.csv",
}

# `Reasoning: Read the answer from the image.` - our own control wording, which
# the official open-answer parser mistakes for the answer itself.
WORDING_ARTIFACT = ("from the im", "see the im", "read the answer")

GAP_TABLE = "official_vs_structured.csv"
TRUNC_TABLE = "truncated_42.csv"


def load(root: pathlib.Path, rel: str) -> list[dict]:
    path = root / rel
    if not path.is_file():
        raise SystemExit(
            f"make_evidence: {rel} not found.\n"
            "  --build needs the local results/ tree; without it, drop --build "
            "to read the tracked evidence/ tables instead."
        )
    with path.open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("id")]


def build(root: pathlib.Path) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    loaded = {name: load(root, rel) for name, rel in RUNS.items()}

    gap_rows = []
    for name, rows in loaded.items():
        structured = sum(1 for r in rows if r["correct"] == "1")
        official = sum(1 for r in rows if r["official_correct"] == "1")
        fallback = [r for r in rows if r.get("official_random_fallback") == "1"]
        # disagreements that are NOT a random guess = genuine parser divergence
        diverged = [r for r in rows
                    if r["correct"] != r["official_correct"]
                    and r.get("official_random_fallback") != "1"]
        loss = [r for r in diverged if r["correct"] == "1"]
        gap_rows.append({
            "run": name,
            "structured": structured,
            "official": official,
            "diff": official - structured,
            "random_fallback": len(fallback),
            "fallback_hit": sum(1 for r in fallback if r["official_correct"] == "1"),
            "misparse_loss": len(loss),
            "misparse_gain": sum(1 for r in diverged if r["official_correct"] == "1"),
            "wording_artifact": sum(
                1 for r in loss
                if any(k in (r.get("official_parsed") or "").lower()
                       for k in WORDING_ARTIFACT)),
        })

    write(EVIDENCE / GAP_TABLE, gap_rows)

    base = loaded["baseline"]
    unparsed = [r for r in base if not (r["pred"] or "").strip()]
    others = {n: {r["id"]: r for r in rows}
              for n, rows in loaded.items() if n != "baseline"}
    trunc_rows = []
    for row in unparsed:
        out = {
            "id": row["id"],
            "subject": row["subject"],
            "gt": row["gt"],
            "baseline_finish_reason": row["finish_reason"],
            "baseline_output_tokens": row["n_output_tokens"],
        }
        for name, table in others.items():
            hit = table.get(row["id"])
            answered = bool(hit and (hit["pred"] or "").strip())
            out[f"{name}_answered"] = int(answered)
            out[f"{name}_correct"] = int(bool(answered and hit["correct"] == "1"))
        trunc_rows.append(out)

    write(EVIDENCE / TRUNC_TABLE, trunc_rows)
    print(f"wrote {len(gap_rows)} rows -> evidence/{GAP_TABLE}")
    print(f"wrote {len(trunc_rows)} rows -> evidence/{TRUNC_TABLE}")


def write(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit(f"make_evidence: refusing to write an empty {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check() -> int:
    missing = [n for n in (GAP_TABLE, TRUNC_TABLE) if not (EVIDENCE / n).is_file()]
    if missing:
        raise SystemExit(
            "make_evidence: missing " + ", ".join(f"evidence/{m}" for m in missing)
            + "\n  rebuild them from the local results/ tree with --build.")

    with (EVIDENCE / GAP_TABLE).open(encoding="utf-8") as handle:
        gap = list(csv.DictReader(handle))
    with (EVIDENCE / TRUNC_TABLE).open(encoding="utf-8") as handle:
        trunc = list(csv.DictReader(handle))

    print("공식 채점기는 파싱 실패 시 무작위로 찍습니다 — 못 푸는 모델이 보너스를 받습니다")
    head = f"{'run':<16}{'구조화':>7}{'공식':>6}{'차이':>6}{'무작위대체':>11}{'대체적중':>9}{'문구artifact':>13}"
    print(head)
    print("-" * len(head))
    for row in gap:
        print(f"{row['run']:<16}{row['structured']:>7}{row['official']:>6}"
              f"{int(row['diff']):>+6}{row['random_fallback']:>11}"
              f"{row['fallback_hit']:>9}{row['wording_artifact']:>13}")

    print()
    print(f"기준선이 답을 못 낸 {len(trunc)}문항 — 각 arm이 어떻게 했나")
    arms = sorted({k[:-9] for k in trunc[0] if k.endswith("_answered")})
    for arm in arms:
        answered = sum(int(r[f"{arm}_answered"]) for r in trunc)
        correct = sum(int(r[f"{arm}_correct"]) for r in trunc)
        rate = f"{correct / answered:.0%}" if answered else "—"
        print(f"  {arm:<16} 답함 {answered:>3} · 정답 {correct:>3} ({rate})")

    reasons = {r["baseline_finish_reason"] for r in trunc}
    print(f"\n기준선 실패 원인: {', '.join(sorted(reasons))} "
          f"({len(trunc)}/{len(trunc)} 동일)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true",
                    help="regenerate the tables from the local results/ tree")
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."),
                    help="repository root that holds results/ (default: cwd)")
    args = ap.parse_args()

    if args.build:
        build(args.root.resolve())
        print()
    return check()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
