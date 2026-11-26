"""How much is left on the table by parse failure, and who would collect it?

Two things get called "the parser misbehaving" and they are worth very
different amounts:

  (a) our control wording confusing the official OPEN-answer parser
      -> official score only, structured score already correct
  (b) the model not producing a parseable answer at all
      -> real points under BOTH scorers

This prices both.
"""
import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")

RUNS = {
    "기준선": "results/pilot-a-mmmu/base_per_sample.csv",
    "a_cqa": "results/mmmu-a_cqa/per_sample.csv",
    "b_cqa": "results/mmmu-b_cqa/per_sample.csv",
    "bshape": "results/mmmu-bshape/per_sample.csv",
    "t_stop": "results/mmmu-tstop/per_sample.csv",
}


def load(path):
    return [r for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("id")]


base = load(RUNS["기준선"])
by_id = {r["id"]: r for r in base}

# structured scoring counts an unextractable row as wrong; which rows are they?
unparsed = [r for r in base if not (r["pred"] or "").strip()]
mc = [r for r in unparsed if r["question_type"] == "multiple-choice"]
print(f"기준선: 구조화 추출 실패 {len(unparsed)}건 (객관식 {len(mc)}건)")

# the model's own hit rate on rows it DID answer, same question type
def rate(rows, qtype):
    sub = [r for r in rows if r["question_type"] == qtype and (r["pred"] or "").strip()]
    return sum(1 for r in sub if r["correct"] == "1") / max(len(sub), 1), len(sub)

for qtype in ("multiple-choice", "open"):
    acc, n = rate(base, qtype)
    miss = sum(1 for r in unparsed if r["question_type"] == qtype)
    print(f"  {qtype:<16} 답한 문항 정확도 {acc:.1%} (n={n}) · 실패 {miss}건"
          f" → 같은 실력으로 답했다면 기대 +{acc * miss:.1f}문항")

print()
print("무작위 대체의 기대값 (공식 채점기가 공짜로 주는 양):")
for name, path in RUNS.items():
    rows = load(path)
    fb = [r for r in rows if r.get("official_random_fallback") == "1"]
    hit = sum(1 for r in fb if r["official_correct"] == "1")
    fb_mc = sum(1 for r in fb if r["question_type"] == "multiple-choice")
    print(f"  {name:<8} 대체 {len(fb):>3}건 (객관식 {fb_mc:>3}) · 적중 {hit:>3} = {hit/max(len(fb),1):.0%}")

print()
print("기준선이 구조화에서 답을 못 낸 문항을, 각 arm은 어떻게 했나:")
ids = [r["id"] for r in unparsed]
for name, path in RUNS.items():
    if name == "기준선":
        continue
    rows = {r["id"]: r for r in load(path)}
    sub = [rows[i] for i in ids if i in rows]
    answered = [r for r in sub if (r["pred"] or "").strip()]
    good = sum(1 for r in answered if r["correct"] == "1")
    print(f"  {name:<8} {len(sub)}건 중 답함 {len(answered):>2} · 정답 {good:>2}"
          f" ({good/max(len(answered),1):.0%})")
