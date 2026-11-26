"""Score our saved MMMU predictions with the ICoT team's scorer, side by side.

Their protocol (icot/evaluate.sh) runs the SAME inference script and the SAME
generation config we do; the only difference is the scorer:

    ours   code/eval/answer_extraction.py   + code/eval/score_mmmu.py
    theirs icot/scoring/answer_extraction.py + icot/scoring/score_mmmu.py

So the comparison needs no GPU and no re-inference - just re-score the saved
`predictions.jsonl`. Two obstacles, both handled here:

1. Their scorer OVERWRITES scores.json / per_sample.csv in the run directory.
   We never point it at a real run: inputs are copied to a scratch dir first.

2. Their scorer needs `evaluation_data.json` (gold + options), which our runs
   did not save. We rebuild it from predictions.jsonl - gold is the `answer`
   field, options are parsed out of the rendered prompt - and VALIDATE the
   reconstruction before using it. If a single row fails validation the run is
   skipped rather than scored on a guess.

What it shows: on arm A, whose output is a single letter and never truncates,
the two scorers agree EXACTLY (485 = 485). Every disagreement elsewhere sits in
responses that were cut off at 512 tokens or that print a reasoning block before
the answer - the two rules their lineage does not have yet.

Usage (from the repository root):
    python experiments/evidence_supervision/cross_score_icot.py
    python experiments/evidence_supervision/cross_score_icot.py mmmu-bshape ...
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(".").resolve()
RESULTS = ROOT / "results"
LETTERS = "ABCDEFGHIJ"

# The prompt renders options between these two fixed strings (code/eval/prompting.py).
OPTIONS_BLOCK = re.compile(
    r"\nOptions:\n(.*?)\nAnswer with the option letter only\.", re.DOTALL)
OPTION_LINE = re.compile(r"^([A-J])\.[ \t](.*)$")


def parse_options(prompt: str, sample_id: str) -> dict[str, str]:
    """Letter -> option text, exactly as the prompt listed them.

    Their extract_choice() wants a mapping, not a sequence (icot/scoring/
    answer_extraction.py:145), so the shape matters.
    """
    block = OPTIONS_BLOCK.search(prompt)
    if not block:
        raise ValueError(f"{sample_id}: no Options block in the prompt")
    options, expected = [], 0
    for line in block.group(1).split("\n"):
        hit = OPTION_LINE.match(line)
        if not hit:
            # a wrapped option would land here; refuse rather than mis-split
            raise ValueError(f"{sample_id}: unparsable option line {line!r}")
        if hit.group(1) != LETTERS[expected]:
            raise ValueError(
                f"{sample_id}: options jump to {hit.group(1)}, expected {LETTERS[expected]}")
        options.append(hit.group(2))
        expected += 1
    if len(options) < 2:
        raise ValueError(f"{sample_id}: only {len(options)} option(s)")
    return {LETTERS[i]: text for i, text in enumerate(options)}


def rebuild_metadata(preds: list[dict]) -> dict:
    samples = {}
    for p in preds:
        sid, gold = p["id"], p["answer"]
        if p["question_type"] == "multiple-choice":
            options = parse_options(p["prompt"], sid)
            if gold not in LETTERS[:len(options)]:
                raise ValueError(
                    f"{sid}: gold {gold!r} outside A-{LETTERS[len(options) - 1]}")
        else:
            options = {}
        samples[sid] = {"question_type": p["question_type"],
                        "options": options, "ground_truth": gold}
    if len(samples) != 900:
        raise ValueError(f"{len(samples)} samples, expected 900")
    return {"samples": samples}


def build_protocol(dest: pathlib.Path) -> pathlib.Path:
    """The layout icot/evaluate.sh creates, so their scorer's imports resolve."""
    code = dest / "code"
    (code / "eval").mkdir(parents=True)
    shutil.copy(ROOT / "code" / "configuration.py", code)
    shutil.copytree(ROOT / "code" / "third_party", code / "third_party")
    for name in ("answer_extraction", "score_mmmu"):
        shutil.copy(ROOT / "icot" / "scoring" / f"{name}.py", code / "eval")
    return code / "eval" / "score_mmmu.py"


def ours(run: pathlib.Path) -> tuple[int | None, int | None]:
    """Our own (structured, official) count for this run.

    scores.json is the normal source. `pilot-a-mmmu` predates it and keeps two
    scorings side by side - base_per_sample.csv is the BASELINE carried over for
    paired comparison, not this run - so the per-sample fallback must skip it.
    """
    path = run / "scores.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return (data.get("structured", {}).get("overall", {}).get("correct"),
                data.get("official", {}).get("overall", {}).get("correct"))

    csvs = [p for p in run.glob("*per_sample.csv") if p.name != "base_per_sample.csv"]
    if len(csvs) != 1:
        return (None, None)
    with csvs[0].open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("id")]
    return (sum(1 for r in rows if r.get("correct") == "1"),
            sum(1 for r in rows if r.get("official_correct") == "1"))


def main() -> int:
    wanted = sys.argv[1:]
    runs = sorted(d for d in RESULTS.iterdir()
                  if d.is_dir() and (d / "predictions.jsonl").is_file()
                  and (not wanted or d.name in wanted))
    if not runs:
        raise SystemExit("cross_score: no run has predictions.jsonl")

    work = pathlib.Path(tempfile.mkdtemp(prefix="cross_score_"))
    scorer = build_protocol(work)

    rows, skipped = [], []
    for run in runs:
        preds = [json.loads(l) for l in (run / "predictions.jsonl").open(encoding="utf-8")]
        try:
            metadata = rebuild_metadata(preds)
        except ValueError as exc:
            skipped.append(f"{run.name}: {exc}")
            continue

        stage = work / run.name
        stage.mkdir()
        shutil.copy(run / "predictions.jsonl", stage)
        (stage / "evaluation_data.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

        done = subprocess.run([sys.executable, str(scorer), str(stage)],
                              capture_output=True, text=True, encoding="utf-8")
        if done.returncode != 0:
            skipped.append(f"{run.name}: their scorer failed - "
                           f"{(done.stderr or '').strip().splitlines()[-1:]}")
            continue

        theirs = json.loads((stage / "scores.json").read_text(encoding="utf-8"))
        our_structured, our_official = ours(run)
        rows.append({
            "run": run.name,
            "ours_structured": our_structured,
            "theirs_structured": theirs["structured"]["overall"]["correct"],
            "ours_official": our_official,
            "theirs_official": theirs["official"]["overall"]["correct"],
            "theirs_unparsed_mc": theirs["structured"]["unparsed"]["multiple_choice"],
            "theirs_unparsed_open": theirs["structured"]["unparsed"]["open"],
            "truncated": theirs["generation"]["finish_reason"].get("length", 0),
        })

    head = (f"{'run':<16}{'우리구조화':>10}{'그들구조화':>10}{'차이':>6}"
            f"{'우리공식':>9}{'그들공식':>9}{'그들미파싱MC':>13}{'잘림':>6}")
    print("같은 예측, 같은 추론 — 채점기만 다름 (구조화 채점 기준)")
    print(head)
    print("-" * len(head))
    for r in sorted(rows, key=lambda x: x["run"]):
        diff = (r["theirs_structured"] - r["ours_structured"]
                if r["ours_structured"] is not None else None)
        print(f"{r['run']:<16}"
              f"{str(r['ours_structured'] or '—'):>10}"
              f"{r['theirs_structured']:>10}"
              f"{(f'{diff:+d}' if diff is not None else '—'):>6}"
              f"{str(r['ours_official'] or '—'):>9}"
              f"{r['theirs_official']:>9}"
              f"{r['theirs_unparsed_mc']:>13}"
              f"{r['truncated']:>6}")

    out = pathlib.Path(__file__).resolve().parent / "evidence" / "cross_scorer.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: x["run"]))
    print(f"\n{len(rows)} runs -> {out.relative_to(ROOT)}")
    if skipped:
        print("\n건너뜀:")
        for line in skipped:
            print("  " + line)
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
