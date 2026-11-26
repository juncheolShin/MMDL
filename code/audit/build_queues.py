#!/usr/bin/env python3
"""Build self-contained review queues for the local audit UI (code/audit/serve_audit.py).

A queue is one JSON file holding everything the reviewer needs - the image path,
the question, the gold answer, whatever evidence the generator produced, and the
choices to pick from. Nothing in the UI reaches back into the dataset, so a queue
can be copied to another machine and reviewed there.

Three queues are built from the audit samples drawn on 2026-11-19:

  chartqa_mismatch  33 rows that were ACCEPTED although the operation the
                    generator used is not the one the question's wording implies.
                    Pre-triaged: rows that are mechanically harmless are badged
                    as such and sorted last, so the real suspects come first.
  chartqa_rejected  30 of 2,354 rejected rows, for cause classification. (a) and
                    (b) are generator changes that recover rows in bulk; (c) is
                    not recoverable.
  ai2d_relation     30 of 1,750 relation_grounded rows, for sufficiency. Asks a
                    second question as well: is the written reasoning a chain, or
                    a recitation of the edge list? (DECISIONS.md 17)

Any evidence JSONL can also be turned into a queue, for reviewing the training
set once it is built:

    python code/audit/build_queues.py --from-jsonl training_data/evidence/ai2d_evidence.jsonl \\
        --name ai2d_full --sample 100 --seed 1

Usage:
    python code/audit/build_queues.py            # build the three audit queues
    python code/audit/serve_audit.py             # then review them in a browser
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVID = ROOT / "training_data" / "evidence"
NORM = ROOT / "training_data" / "normalized"
OUT = EVID / "audit_queues"

# The queues show the ACTUAL trained strings, produced by the same renderer the
# training set is built with, so the reviewer is never judging a prettier
# paraphrase of the data. See WHAT_THE_MODEL_SEES below.
sys.path.insert(0, str(ROOT / "code" / "data"))
import render_targets  # noqa: E402

# This is the single most important thing for a reviewer to have right. The
# model is a VLM: it ALWAYS receives the image. The evidence is not an input,
# not a hint - it is the TARGET, the string we are teaching the model to emit.
# (The clue-injection probe from the earlier presentation is the opposite
# arrangement, and is a diagnostic, not a training arm. DECISIONS.md section 2
# and section 6.)
WHAT_THE_MODEL_SEES = (
    "**모델은 이미지를 봅니다.** 증거는 모델에게 주는 입력이나 힌트가 아니라, "
    "모델이 **생성하도록 가르치는 목표 문자열**입니다.\n\n"
    "입력 = `<image>` + 질문 + 선택지 (arm A와 arm B가 완전히 동일)\n"
    "목표 = arm A는 `A` 한 글자, arm B는 `Evidence: ... / Reasoning: ... / A`\n\n"
    "따라서 판정 기준은 '이 증거만 보고 풀 수 있나'가 아니라 **'이 이미지를 보는 "
    "모델이 정답에 이르는 길로 이 문장을 말하도록 가르쳐도 되는가'** 입니다. "
    "증거가 이미지에 대해 **사실이어야 하고**, 정답을 **실제로 뒷받침해야** 합니다."
)

# Image paths inside the records are relative to this directory. The server
# refuses to serve anything outside it.
IMAGE_ROOT = NORM


# --------------------------------------------------------------------------
# loading helpers
# --------------------------------------------------------------------------

def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def index_jsonl(path: pathlib.Path, keys=None):
    """id -> record, optionally only for the ids in `keys` (saves memory)."""
    out = {}
    for record in read_jsonl(path):
        rid = record.get("id")
        if keys is None or rid in keys:
            out[rid] = record
    return out


def normalized_path(record_id: str) -> pathlib.Path:
    """`chartqa/train/001070` -> training_data/normalized/chartqa/train.jsonl"""
    parts = record_id.split("/")
    if len(parts) < 3:
        raise ValueError(f"unexpected id shape: {record_id}")
    return NORM / parts[0] / f"{parts[1]}.jsonl"


def load_by_id(ids):
    """Fetch normalized records for ids that may span several source/split files."""
    wanted = {}
    for rid in ids:
        wanted.setdefault(normalized_path(rid), set()).add(rid)
    found = {}
    for path, keys in wanted.items():
        if not path.is_file():
            print(f"  WARNING: {path} missing; {len(keys)} rows will have no image",
                  file=sys.stderr)
            continue
        found.update(index_jsonl(path, keys))
    return found


# --------------------------------------------------------------------------
# panel builders - the blocks of context shown beside the image
# --------------------------------------------------------------------------

def lines_panel(title, rows, note=""):
    return {"title": title, "kind": "lines", "rows": [str(r) for r in rows],
            "note": note}


def kv_panel(title, pairs):
    return {"title": title, "kind": "kv",
            "rows": [[str(k), str(v)] for k, v in pairs]}


def table_panel(title, header, rows, note=""):
    return {"title": title, "kind": "table", "header": [str(h) for h in header],
            "rows": [[str(c) for c in row] for row in rows], "note": note}


def trained_panels(record):
    """The exact two strings training uses, straight from render_targets.py.

    Showing a paraphrase here would let the reviewer approve something the
    trainer never sees, so this calls the real renderer and reports when it
    refuses a row."""
    panels = []
    try:
        panels.append(lines_panel(
            "① 모델이 받는 입력",
            render_targets.render_user(record).split("\n"),
            note="arm A와 arm B가 완전히 같습니다. <image>는 실제 이미지로 치환됩니다."))
    except Exception as exc:                       # a renderer change, not data
        panels.append(lines_panel("① 모델이 받는 입력", [f"(렌더 실패: {exc})"]))
    try:
        text, reason = render_targets.render_assistant(record, "ekra")
        rows = text.split("\n") if text else [f"(이 행은 학습셋에서 제외됩니다: {reason})"]
        panels.append(lines_panel(
            "② 학습 목표 — 모델이 생성하도록 가르칠 문자열", rows,
            note="판정 대상은 이 문자열입니다. 이미지를 보는 모델이 이렇게 "
                 "말하도록 가르쳐도 되는지 보세요."))
    except Exception as exc:
        panels.append(lines_panel("② 학습 목표", [f"(렌더 실패: {exc})"]))
    return panels


def evidence_lines(items):
    """Render evidence items the way render_targets.py does, so the reviewer
    sees what the model will see - not a prettier version of it."""
    out = []
    for item in items or []:
        kind = item.get("kind")
        if kind == "edge":
            arrow = "->" if item.get("directed") else "--"
            tail = f"  (connector {item['connector']}, {item.get('category','')})" \
                if item.get("connector") else ""
            out.append(f"edge: {item.get('src','?')} {arrow} {item.get('dst','?')}{tail}")
        elif kind == "cell":
            out.append(f"cell: row \"{item.get('row','')}\" / column "
                       f"\"{item.get('column','')}\" = {item.get('text','')}")
        elif kind == "label":
            out.append(f"label: \"{item.get('text','')}\" marked "
                       f"{item.get('label','?')}")
        else:
            out.append(f"{kind}: {item.get('text','')}")
    return out or ["(no evidence items)"]


# --------------------------------------------------------------------------
# queue 1: ChartQA accepted-but-mismatched operations
# --------------------------------------------------------------------------

HARMLESS_RANGE = {"difference", "max", "min", "range"}


def mismatch_triage(question, used, implied):
    """Pre-classify the 33 rows so the reviewer's eyes go where they matter.

    Returns (severity, why). Severity orders the queue: suspect first.
    """
    imp = set(implied or [])

    # "How many points moved from 2010 to 2017?" is a difference. The word
    # matcher sees "how many" and records `count`; that is the matcher being
    # literal, not the generator being wrong.
    if question.strip().lower().startswith("how many"):
        imp = imp - {"count"}

    if not imp or used in imp:
        return "ok", "질문이 함의하는 연산과 사용한 연산이 일치"
    if used == "range" and imp <= HARMLESS_RANGE:
        return "ok", "동의어: range = max - min = 최고와 최저의 difference"
    if used == "sum" and "mean" in imp:
        return "suspect", "합을 '평균'이라고 서술 - 실제 오류일 가능성이 높음"
    return "check", f"교차: {used} 사용, 질문은 {sorted(imp)} 함의 - 눈으로 확인 필요"


SEVERITY_ORDER = {"suspect": 0, "check": 1, "ok": 2}


def build_chartqa_mismatch():
    src = EVID / "chartqa_accepted_mismatches.json"
    rows = json.loads(src.read_text(encoding="utf-8"))
    ids = {r["id"] for r in rows}
    norm = load_by_id(ids)
    evid = index_jsonl(EVID / "chartqa_evidence.jsonl", ids)

    stale = sorted(ids - set(evid))
    if stale:
        raise SystemExit(
            f"build_queues: {len(stale)} of {len(ids)} mismatch rows are no longer "
            f"in chartqa_evidence.jsonl - the sample is stale.\n"
            f"  first missing: {stale[0]}\n"
            f"  redraw it:     python code/data/synth_evidence_chartqa.py "
            f"(it rewrites chartqa_accepted_mismatches.json)")

    items = []
    for row in rows:
        rid = row["id"]
        base = norm.get(rid, {})
        ev_record = evid[rid]
        ev = ev_record.get("meta") or {}
        severity, why = mismatch_triage(row["question"], row["operation_used"],
                                        row["operations_the_question_implies"])
        panels = trained_panels(ev_record) + [
            kv_panel("판정 근거", [
                ("생성기가 사용한 연산", row["operation_used"]),
                ("질문이 함의하는 연산", ", ".join(
                    row["operations_the_question_implies"]) or "(없음)"),
                ("사전 분류", why),
            ]),
            lines_panel("인용된 표 셀", evidence_lines(ev.get("evidence")),
                        note="이 셀 값들이 차트와 맞는지, 그리고 서술이 자기가 한 "
                             "계산을 정직하게 부르는지가 핵심입니다."),
        ]
        items.append({
            "id": rid,
            "image": base.get("image"),
            "question": row["question"],
            "options": base.get("options"),
            "gold": row["answer"],
            "badges": ([{"text": "의심", "tone": "bad"}] if severity == "suspect"
                       else [{"text": "확인 필요", "tone": "warn"}] if severity == "check"
                       else [{"text": "무해 추정", "tone": "ok"}]),
            "sort_key": SEVERITY_ORDER[severity],
            "panels": panels,
        })

    items.sort(key=lambda it: it["sort_key"])
    counts = {}
    for it in items:
        counts[it["badges"][0]["text"]] = counts.get(it["badges"][0]["text"], 0) + 1

    return {
        "name": "chartqa_mismatch",
        "title": "ChartQA - 승인됐지만 연산이 어긋난 행",
        "subtitle": f"33행 (사전 분류: {counts})",
        "instructions": (
            WHAT_THE_MODEL_SEES + "\n\n"
            "생성기가 **사용한 연산**과 질문이 **함의하는 연산**이 다른데도 정답이 맞아 "
            "통과한 행입니다. 정답이 맞다는 사실이 오히려 위험합니다 - 틀린 서술이 "
            "정답과 함께 학습되기 때문입니다.\n\n"
            "판단 기준은 하나입니다: **서술이 자기가 한 계산을 정직하게 부르고 있는가?**\n\n"
            "예: 질문이 '세 지역의 평균'인데 서술이 두 지역을 더하고 '평균'이라 부르면 (c)입니다."
        ),
        "choices": [
            {"key": "a", "label": "문제 없음", "tone": "ok",
             "hint": "연산 이름만 다를 뿐 수학적으로 같음 (range = max-min 등)"},
            {"key": "b", "label": "서술만 어색", "tone": "warn",
             "hint": "계산은 맞지만 표현이 질문과 어긋남 - 서술 템플릿 수정 대상"},
            {"key": "c", "label": "실제 오류, 제거", "tone": "bad",
             "hint": "계산 자체가 질문과 다름 (합을 평균이라 부르는 등)"},
        ],
        "items": items,
    }


# --------------------------------------------------------------------------
# queue 2: ChartQA rejected rows, cause classification
# --------------------------------------------------------------------------

def build_chartqa_rejected():
    blob = json.loads((EVID / "chartqa_audit.json").read_text(encoding="utf-8"))
    rows = blob["sampled"]
    ids = {r["id"] for r in rows}
    norm = load_by_id(ids)

    items = []
    for row in rows:
        base = norm.get(row["id"], {})
        cells = row.get("table_cells") or []
        panels = [
            kv_panel("거부 사유", [
                ("사유 코드", row["reject_reason"]),
                ("질문이 함의하는 연산", ", ".join(
                    row.get("operations_the_question_implies") or []) or "(없음)"),
                ("차트", row.get("chart", "")),
                ("질문 출처", row.get("question_kind", "")),
            ]),
            table_panel(
                f"생성기가 읽은 표 ({len(cells)} 셀)",
                ["row", "column", "value"],
                [[c.get("row", ""), c.get("col", c.get("column", "")), c.get("text", "")]
                 for c in cells],
                note="정답을 만들 셀이 이 표 안에 있는지부터 보세요. 없으면 (c)입니다.",
            ),
        ]
        items.append({
            "id": row["id"],
            "image": base.get("image"),
            "question": row["question"],
            "options": base.get("options"),
            "gold": row.get("gold"),
            "badges": [{"text": row["reject_reason"], "tone": "warn"}],
            "panels": panels,
        })

    return {
        "name": "chartqa_rejected",
        "title": "ChartQA - 거부된 행의 원인 분류",
        "subtitle": f"{len(items)}행 (전체 거부 {blob['total_rejected']:,}행에서 표집, "
                    f"seed {blob['seed']})",
        "instructions": (
            WHAT_THE_MODEL_SEES + "\n\n"
            "**이 분류는 수백 행을 한 번에 회수하는 레버입니다.** (a)와 (b)는 생성기를 "
            "고치면 대량 회수되고, (c)는 영구 포기입니다.\n\n"
            "표를 먼저 보고, 정답을 만들 수 있는 셀이 표 안에 있는지 확인하세요. "
            "있는데 거부됐다면 (a) 또는 (b), 없다면 (c)입니다."
        ),
        "choices": [
            {"key": "a", "label": "연산 누락", "tone": "warn",
             "hint": "셀은 표에 있는데 필요한 연산을 생성기가 시도하지 않음 "
                     "(예: 행별 합의 max)"},
            {"key": "b", "label": "라벨 불일치", "tone": "warn",
             "hint": "셀은 있는데 질문이 다른 이름으로 부름 ('Turkey' vs 'Turkey, Rep. of')"},
            {"key": "c", "label": "유도 불가", "tone": "bad",
             "hint": "이 표에서는 정답을 얻을 수 없음 (차트에만 있거나 표가 틀림)"},
        ],
        "items": items,
    }


# --------------------------------------------------------------------------
# queue 3: AI2D relation sufficiency (+ the reasoning-quality question)
# --------------------------------------------------------------------------

TIER_NOTE = {
    "question_entity_in_cited_label":
        "약한 티어 - 정답에 등장하는 개체가 인용된 증거에 없습니다. 지지받지 못하는 "
        "행이 여기 모여 있을 가능성이 높습니다.",
    "gold_in_cited_label":
        "정답 문자열이 인용된 라벨 안에 있습니다.",
}


def build_ai2d_relation(evidence=None, name="ai2d_relation"):
    # A regenerated evidence file needs a freshly drawn sample and its own
    # queue name, so the earlier audit and its verdicts stay addressable and
    # the two rounds can be compared (D57).
    evidence = evidence or (EVID / "ai2d_evidence.jsonl")
    blob = json.loads((EVID / "ai2d_relation_audit.json").read_text(encoding="utf-8"))
    ids = set(blob["sampled_ids"])
    evid = index_jsonl(evidence, ids)

    # The sample is drawn from the evidence file, so every id must still be in
    # it. When it is not, the evidence was regenerated after the sample was
    # drawn (this happened once: the 07:15 sample against the 07:23 rebuild, and
    # 16 of 30 rows had vanished). Reviewing rows that are no longer in the
    # training set wastes the reviewer's time, so this refuses rather than warns.
    missing = sorted(ids - set(evid))
    if missing:
        raise SystemExit(
            f"build_queues: {len(missing)} of {len(ids)} sampled AI2D rows are no "
            f"longer in ai2d_evidence.jsonl - the sample is stale.\n"
            f"  first missing: {missing[0]}\n"
            f"  redraw it:     python code/data/audit_ai2d_relations.py --limit 30")

    items = []
    for rid in blob["sampled_ids"]:
        record = evid.get(rid)
        if record is None:
            continue
        meta = record.get("meta") or {}
        tier = meta.get("evidence_sufficiency", "")
        weak = tier == "question_entity_in_cited_label"
        panels = trained_panels(record) + [
            lines_panel("인용된 주석 (증거의 원재료)", evidence_lines(meta.get("evidence")),
                        note="AI2D 주석에서 온 것이라 이미지에 대해 사실입니다. "
                             "문제는 사실 여부가 아니라 정답을 뒷받침하는지입니다."),
            kv_panel("티어", [
                ("sufficiency tier", tier),
                ("설명", TIER_NOTE.get(tier, "")),
                ("인용된 주석 id", ", ".join(meta.get("evidence_cited_ids") or [])),
            ]),
        ]
        items.append({
            "id": rid,
            "image": record.get("image"),
            "question": record.get("question"),
            "options": record.get("options"),
            "gold": record.get("answer"),
            "badges": ([{"text": "약한 티어", "tone": "bad"}] if weak
                       else [{"text": "gold in label", "tone": "ok"}]),
            "panels": panels,
        })

    return {
        "name": name,
        "title": "AI2D - 관계 증거의 충분성",
        "subtitle": f"{len(items)}행 (전체 relation_grounded {blob['total_relation_grounded']:,}행에서 "
                    f"약한 티어 {int(blob['weak_share']*100)}% 가중 표집, seed {blob['seed']})",
        "instructions": (
            WHAT_THE_MODEL_SEES + "\n\n"
            "**정답은 의심 대상이 아닙니다** - AI2D 주석이 정답입니다. 의심 대상은 "
            "증거가 그 정답을 뒷받침하는가입니다.\n\n"
            "(b)와 (c)가 해로운 쪽입니다: 자기가 댄 증거가 지지하지 않는 결론을 "
            "말하도록 가르칩니다. **(b)+(c)가 10%를 크게 넘으면 약한 티어 전체를 "
            "버리는 쪽이 낫습니다.**\n\n"
            "**두 번째 질문도 같이 봐주세요.** 밤샘 작업 중 발견한 문제입니다: 서술이 "
            "추론 사슬이 아니라 간선 목록 낭독인 것으로 보입니다. 충분하다고 판정되는 "
            "행조차 '왜 그 답인지'를 말하지 않으면, 티어를 버리는 것보다 **서술 템플릿을 "
            "고치는 것이 우선**입니다 - 1,750행을 살릴 수 있습니다."
        ),
        "choices": [
            {"key": "a", "label": "충분", "tone": "ok",
             "hint": "이 목표를 그대로 가르쳐도 됨 - 증거가 정답을 실제로 뒷받침"},
            {"key": "b", "label": "관련 있으나 불충분", "tone": "warn",
             "hint": "맞는 부분을 가리키지만, 댄 증거만으로는 정답이 따라나오지 않음 "
                     "- 근거 없는 확신을 가르치게 됨"},
            {"key": "c", "label": "무관", "tone": "bad",
             "hint": "증거가 질문과 상관이 없음"},
        ],
        "extra_questions": [{
            "name": "chain",
            "label": "서술이 추론인가, 낭독인가?",
            "hint": "'왜 그 답인지'를 말하면 추론, 간선을 나열만 하면 낭독입니다.",
            "choices": [
                {"key": "r", "label": "추론 (사슬)", "tone": "ok"},
                {"key": "l", "label": "낭독 (나열)", "tone": "bad"},
            ],
        }],
        "items": items,
    }


# --------------------------------------------------------------------------
# generic: any evidence JSONL -> a review queue
# --------------------------------------------------------------------------

def build_from_jsonl(path: pathlib.Path, name: str, sample: int, seed: int):
    rows = list(read_jsonl(path))
    if sample and len(rows) > sample:
        rows = random.Random(seed).sample(rows, sample)
    items = []
    for record in rows:
        meta = record.get("meta") or {}
        panels = [
            lines_panel("모델이 보게 될 증거", evidence_lines(meta.get("evidence"))),
            lines_panel("모델이 보게 될 서술", meta.get("reasoning") or ["(없음)"]),
        ]
        if meta.get("knowledge"):
            panels.append(lines_panel("배경 지식", [meta["knowledge"]]))
        provenance = [(k, v) for k, v in meta.items()
                      if k.startswith("evidence_") and k != "evidence"]
        if provenance:
            panels.append(kv_panel("출처", provenance))
        items.append({
            "id": record.get("id"),
            "image": record.get("image"),
            "question": record.get("question"),
            "options": record.get("options"),
            "gold": record.get("answer"),
            "badges": [{"text": meta.get("evidence_kind", "?"), "tone": "neutral"}],
            "panels": panels,
        })
    return {
        "name": name,
        "title": f"{path.name} 검토",
        "subtitle": f"{len(items)}행" + (f" (전체 {len(list(read_jsonl(path))):,}행에서 "
                                         f"표집, seed {seed})" if sample else ""),
        "instructions": (
            WHAT_THE_MODEL_SEES + "\n\n"
            "학습 데이터 품질 점검입니다. 목표 문자열이 이미지에 대해 사실인지, "
            "그리고 댄 증거가 정답을 실제로 뒷받침하는지 보세요."
        ),
        "choices": [
            {"key": "a", "label": "좋음", "tone": "ok", "hint": "그대로 학습해도 됨"},
            {"key": "b", "label": "약함", "tone": "warn", "hint": "맞지만 서술이 부실함"},
            {"key": "c", "label": "버림", "tone": "bad", "hint": "증거가 정답을 지지하지 않음"},
        ],
        "items": items,
    }


# --------------------------------------------------------------------------

BUILDERS = {
    "chartqa_mismatch": build_chartqa_mismatch,
    "chartqa_rejected": build_chartqa_rejected,
    "ai2d_relation": build_ai2d_relation,
}


def write_queue(queue):
    OUT.mkdir(parents=True, exist_ok=True)
    queue["image_root"] = str(IMAGE_ROOT)
    path = OUT / f"{queue['name']}.json"
    path.write_text(json.dumps(queue, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    missing = sum(1 for it in queue["items"] if not it.get("image"))
    note = f", {missing} without an image" if missing else ""
    print(f"  {queue['name']:<18} {len(queue['items']):>4} items{note}  -> {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--from-jsonl", type=pathlib.Path, default=None,
                        help="build a review queue from any evidence JSONL")
    parser.add_argument("--name", default=None, help="queue name for --from-jsonl")
    parser.add_argument("--sample", type=int, default=0,
                        help="sample N rows (0 = all) for --from-jsonl")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--ai2d-evidence", type=pathlib.Path, default=None,
                        help="regenerated AI2D evidence to build the relation "
                             "queue from (redraw the sample first with "
                             "code/data/audit_ai2d_relations.py --evidence ...)")
    parser.add_argument("--queue-name", default=None,
                        help="name for the queue built with --ai2d-evidence")
    parser.add_argument("--only", action="append", choices=sorted(BUILDERS),
                        help="build only this queue (repeatable)")
    args = parser.parse_args()

    print("building review queues")
    if args.from_jsonl:
        if not args.from_jsonl.is_file():
            sys.exit(f"build_queues: {args.from_jsonl} not found")
        name = args.name or args.from_jsonl.stem
        write_queue(build_from_jsonl(args.from_jsonl, name, args.sample, args.seed))
        return 0

    if args.ai2d_evidence:
        write_queue(build_ai2d_relation(
            args.ai2d_evidence, args.queue_name or "ai2d_relation_v2"))
        return 0

    for name in (args.only or sorted(BUILDERS)):
        write_queue(BUILDERS[name]())
    print("\nreview them with:  python code/audit/serve_audit.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
