"""Where does the official-vs-structured MMMU gap come from?

official = structured + (random guesses that happened to hit)
                      - (rows the official parser read differently and got wrong)
                      + (rows the official parser read differently and got right)

The official MMMU scorer assigns a RANDOM option when it cannot parse, so a model
that fails to answer collects free points. That makes `official_correct`
non-comparable across models whose unparsed rates differ, which is exactly what
this project varies.
"""
import csv
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

RUNS = {
    "기준선": "results/pilot-a-mmmu/base_per_sample.csv",
    "a_cqa": "results/mmmu-a_cqa/per_sample.csv",
    "b_cqa": "results/mmmu-b_cqa/per_sample.csv",
    "bshape": "results/mmmu-bshape/per_sample.csv",
    "bshape1": "results/mmmu-bshape1/per_sample.csv",
    "bshape2": "results/mmmu-bshape2/per_sample.csv",
    "t_stop": "results/mmmu-tstop/per_sample.csv",
}

IMAGE_ARTIFACT = re.compile(r"from the im|see the im|read the answer", re.IGNORECASE)

hdr = (f"{'run':<9}{'구조화':>7}{'공식':>6}{'차이':>6}"
       f"{'대체':>6}{'대체적중':>9}{'오파싱손실':>11}{'오파싱이득':>11}{'문구artifact':>12}")
print(hdr)
print("-" * len(hdr))
for name, path in RUNS.items():
    try:
        rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("id")]
    except FileNotFoundError:
        continue
    s = sum(1 for r in rows if r["correct"] == "1")
    o = sum(1 for r in rows if r["official_correct"] == "1")
    fb = [r for r in rows if r.get("official_random_fallback") == "1"]
    fb_hit = sum(1 for r in fb if r["official_correct"] == "1")
    # disagreements that are NOT random fallback = genuine parser divergence
    div = [r for r in rows if r["correct"] != r["official_correct"]
           and r.get("official_random_fallback") != "1"]
    loss = [r for r in div if r["correct"] == "1"]
    gain = [r for r in div if r["official_correct"] == "1"]
    artifact = sum(1 for r in loss if IMAGE_ARTIFACT.search(r.get("official_parsed") or ""))
    print(f"{name:<9}{s:>7}{o:>6}{o-s:>+6}{len(fb):>6}{fb_hit:>9}"
          f"{len(loss):>11}{len(gain):>11}{artifact:>12}")

print()
print("무작위 대체가 공식 점수에 더해준 양 (대체적중):")
print("  구조화 점수는 파싱 실패를 오답으로 셉니다. 공식은 찍습니다.")
print("  → 파싱 실패가 많은 모델일수록 공식 점수가 부풀려집니다.")
