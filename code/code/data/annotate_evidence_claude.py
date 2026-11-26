#!/usr/bin/env python3
"""Write visual evidence for ScienceQA with Claude, through the Batch API.

AI2D and ChartQA take their evidence from their own annotations - free, and
checked against ground truth. ScienceQA ships no annotation of its pictures at
all: its `lecture` and `solution` fields are textbook prose about the topic, not
a description of the image. So its evidence has to come from a model that can
see the image (DECISIONS.md D34).

Two rules carry over from DECISIONS.md section 4, and both are load-bearing:

  * The gold answer is not shown. Asked to justify a known answer, a model
    reasons backwards from it and the "evidence" ends up restating it. Here
    Claude answers independently and only rows whose answer matches gold are
    kept, which doubles as the first quality filter.
  * Evidence must state what is visible, not what follows from it. The prompt
    says so, and a check here drops any row whose evidence contains the answer
    text - an instruction alone is not a guarantee.

Cost at current prices, for ~3,800 requests at roughly 1,250 input and 400
output tokens, through the Batch API: about $0.60 on Claude Haiku 5.5, about $12
on Claude Sonnet 5.5. Start with Haiku and a small --limit, audit it, and move up
only if the audit says to (D35).

    python code/data/annotate_evidence_claude.py --dry-run --limit 5
    python code/data/annotate_evidence_claude.py --limit 200 --submit
    python code/data/annotate_evidence_claude.py --collect msgbatch_xxx
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CAND = ROOT / "training_data" / "candidates" / "scienceqa.jsonl"
IMAGES = ROOT / "training_data" / "normalized"
OUT = ROOT / "training_data" / "evidence"

LETTERS = "ABCDEFGHI"
DEFAULT_MODEL = "claude-haiku-5-5"
MAX_TOKENS = 1024

# The first pilot came back at 141.8 tokens a row against a 45-token budget,
# with every single row over it (D50). The item COUNTS were obeyed; nothing
# bounded the length of an item, and "or an empty string" was filled on 163 of
# 163 rows. A permission that costs the model nothing to ignore is not a bound,
# so the caps below are stated as hard limits, in words, and the collector
# enforces them afterwards.
MAX_WORDS_PER_ITEM = 10

SYSTEM = (
    "You describe what is visible in a science image, for use as training data.\n\n"
    "You are given an image and a multiple-choice question about it. You are not "
    "told which option is correct.\n\n"
    "WRITE AS FEW WORDS AS POSSIBLE. The whole answer is a training target with "
    "a hard budget of 30 WORDS IN TOTAL across all fields, and it is discarded "
    "if it runs over. Terse note form, not "
    "sentences. No hedging, no restating the question.\n\n"
    "Return four fields:\n"
    "  evidence   ONE statement of what is VISIBLE and bears on the "
    "question, AT MOST 12 WORDS. Use a second one only if a single "
    "statement truly cannot carry the question. Observations only: shapes, labels, "
    "colours, counts, positions, text in the image. Do not name or quote any "
    "answer option. Do not state a conclusion.\n"
    "  knowledge  LEAVE THIS EMPTY (\"\"). Fill it only if the question is "
    "unanswerable without a fact that is not in the image, and then AT MOST 10 "
    "WORDS.\n"
    "  reasoning  ONE step from the evidence to an answer, AT MOST 12 WORDS. "
    "Use a second step only if one truly cannot get there.\n"
    "  answer     the letter of the option you judge correct, from the image and "
    "your own reasoning.\n\n"
    "Good evidence: \"Left magnet S faces right magnet N; gap 0.4 cm.\"\n"
    "Bad evidence: \"The image shows two magnets positioned with their poles "
    "facing each other across a gap of 0.4 cm.\"\n\n"
    "If the image does not show what the question asks about, say so in four "
    "words and still give your best answer."
)

# `minItems` / `maxItems` are NOT supported on arrays in a structured-output
# schema: the API rejects the whole request with
#   output_config.format.schema: For 'array' type, property 'maxItems' is not supported
# A 200-row pilot failed this way on every row before it was found (D49), which
# is what the pilot was for. The 1-to-3 bound is stated in SYSTEM instead, and
# enforced here at collect time - the model is asked, the code decides.
MAX_ITEMS = 2          # a second item is allowed, but one is what is asked for
MAX_TOTAL_WORDS = 30   # about 40 tokens of English, inside the G2 budget of 45

SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "array", "items": {"type": "string"}},
        "knowledge": {"type": "string"},
        "reasoning": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "string", "enum": list(LETTERS)},
    },
    "required": ["evidence", "knowledge", "reasoning", "answer"],
    "additionalProperties": False,
}


def clamp(values, limit=MAX_ITEMS):
    """Drop blanks and keep at most `limit`. The schema can no longer bound the
    list, so this is the only thing standing between a chatty model and a
    target that blows past the token budget."""
    return [str(v).strip() for v in (values or []) if str(v).strip()][:limit]

MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".gif": "image/gif", ".webp": "image/webp"}


def load_records(limit: int, offset: int = 0):
    """Records [offset, offset+limit). Offset exists because one batch cannot hold
    the whole pool: at ~102 KiB per request the 6,218 ScienceQA candidates come to
    roughly 620 MiB against a 256 MB per-batch limit, so they go in three parts.
    Without an offset, --limit could only ever resubmit the same first N rows."""
    if not CAND.is_file():
        sys.exit(f"annotate: {CAND} not found - run split.py first")
    rows = []
    with CAND.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if i < offset:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def question_block(record) -> str:
    options = record["options"] or []
    lines = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    hint = (record.get("meta") or {}).get("hint") or ""
    parts = [f"Question: {str(record['question']).strip()}"]
    if hint.strip():
        parts.append(f"Context given with the question: {hint.strip()}")
    parts.append("Options:\n" + lines)
    return "\n\n".join(parts)


def build_request(record):
    """One Batch API request. Returns None when the image cannot be read."""
    path = IMAGES / record["image"]
    media = MEDIA.get(path.suffix.lower())
    if media is None or not path.is_file():
        return None
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {
        "custom_id": record["id"].replace("/", "__"),
        "params": {
            "model": DEFAULT_MODEL,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM,
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": media, "data": data}},
                    {"type": "text", "text": question_block(record)},
                ],
            }],
        },
    }


def leaks_answer(evidence, options, gold_index) -> bool:
    """True when the evidence quotes an answer option - a leak, not evidence."""
    text = " ".join(evidence).lower()
    for i, option in enumerate(options or []):
        opt = str(option).strip().lower()
        if len(opt) > 3 and opt in text:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=0, help="at most N records")
    parser.add_argument("--offset", type=int, default=0,
                        help="skip the first N records; with --limit, submits one part "
                             "of a pool too large for a single batch")
    parser.add_argument("--target", type=int, default=0,
                        help="at --collect: keep only this many rows")
    parser.add_argument("--prefer-wrong", type=pathlib.Path, default=None,
                        metavar="SPLIT.json",
                        help="at --collect, with --target: choose the kept rows "
                             "wrong-first, as the other two sources do")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-words", type=int, default=MAX_TOTAL_WORDS,
                        help="drop a row whose evidence+knowledge+reasoning "
                             "runs past this many words (the 45-token budget)")
    parser.add_argument("--submit", action="store_true", help="create the batch")
    parser.add_argument("--collect", metavar="BATCH_ID", help="fetch a finished batch")
    parser.add_argument("--dry-run", action="store_true",
                        help="build requests, print one, send nothing")
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.limit, args.offset)
    by_id = {r["id"].replace("/", "__"): r for r in records}
    print(f"{len(records):,} ScienceQA candidate records", flush=True)

    # ---------------------------------------------------------------- collect
    if args.collect:
        import anthropic
        client = anthropic.Anthropic()
        batch = client.messages.batches.retrieve(args.collect)
        print(f"batch {args.collect}: {batch.processing_status}")
        if batch.processing_status != "ended":
            counts = batch.request_counts
            print(f"  processing {counts.processing}, succeeded {counts.succeeded}, "
                  f"errored {counts.errored}")
            print("not finished; run --collect again later")
            return 0

        kept, reasons = [], collections.Counter()
        for result in client.messages.batches.results(args.collect):
            record = by_id.get(result.custom_id)
            if record is None:
                reasons["unknown_custom_id"] += 1
                continue
            if result.result.type != "succeeded":
                reasons[f"result_{result.result.type}"] += 1
                continue
            message = result.result.message
            if message.stop_reason == "refusal":
                reasons["refusal"] += 1
                continue
            text = next((b.text for b in message.content if b.type == "text"), "")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                reasons["unparsable_json"] += 1
                continue

            options = record["options"] or []
            gold = LETTERS[record["answer_index"]]
            # The schema cannot bound the list lengths, so bound them here.
            evidence = clamp(parsed.get("evidence"))
            reasoning = clamp(parsed.get("reasoning"))
            if not evidence or not reasoning:
                reasons["empty_evidence_or_reasoning"] += 1
                continue
            # The length budget is enforced here, not hoped for in the prompt.
            # Rejecting is affordable: the pilot kept 81.5%, which is about
            # 5,068 rows against a target of 3,000 (D50).
            knowledge = " ".join(clamp([parsed.get("knowledge", "")], 1))
            words = sum(len(t.split()) for t in evidence + reasoning) \
                + len(knowledge.split())
            if words > args.max_words:
                reasons[f"over_{args.max_words}_words"] += 1
                continue
            # Rule 1: the model answered on its own; keep only where it agrees.
            if parsed.get("answer") != gold:
                reasons["answer_mismatch"] += 1
                continue
            # Rule 2: evidence must not restate an option.
            if leaks_answer(evidence, options, record["answer_index"]):
                reasons["evidence_quotes_an_option"] += 1
                continue
            meta = dict(record.get("meta") or {})
            meta.update({
                "evidence": [{"kind": "observation", "text": e} for e in evidence],
                "knowledge": knowledge,
                "reasoning": reasoning,
                "evidence_provenance": "claude_authored",
                "evidence_kind": "model_observation",
                "evidence_verification": "independent_answer_matches_gold",
                "evidence_model": args.model,
                # nothing here is checked against an annotation, so all of it
                # belongs in the human audit queue
                "needs_human_audit": True,
            })
            kept.append({**record, "meta": meta})

        # Selection happens HERE, not at submission. Which rows survive is not
        # known until Claude has answered - a row the base model got wrong may
        # well be one Claude gets wrong too - so sending the whole pool and
        # choosing afterwards is both cheaper ($0.41 on Haiku for all 6,218) and
        # strictly better informed than picking 3,806 up front and hoping enough
        # of them pass.
        if args.target and len(kept) > args.target:
            before = len(kept)
            if args.prefer_wrong:
                sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
                import wrong_priority
                split = wrong_priority.load_split(args.prefer_wrong)
                kept = wrong_priority.wrong_first(kept, split, id_of=lambda r: r["id"])
                wrong_priority.report(kept[:args.target], split,
                                      id_of=lambda r: r["id"], label="selected")
            kept = kept[:args.target]
            print(f"  selected {len(kept):,} of {before:,} that passed the filters")
        elif args.target:
            print(f"  WARNING: only {len(kept):,} rows passed the filters, "
                  f"short of the --target {args.target:,}")

        out_path = args.out / "scienceqa_evidence.jsonl"
        with out_path.open("w", encoding="utf-8") as sink:
            for row in kept:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        total = len(kept) + sum(reasons.values())
        (args.out / "scienceqa_evidence_manifest.json").write_text(json.dumps({
            "batch_id": args.collect, "model": args.model,
            "kept": len(kept), "rejected": dict(reasons.most_common()),
            "keep_rate": round(len(kept) / max(total, 1), 4),
            "rules": {
                "answer_hidden": "the gold answer was not in the prompt",
                "kept_only_if": "the model's own answer equals gold",
                "dropped_if": "the evidence quotes an answer option",
            },
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nkept {len(kept):,} of {total:,} ({100*len(kept)/max(total,1):.1f}%)")
        for why, n in reasons.most_common():
            print(f"  {why:34} {n:>6,}")
        print(f"wrote {out_path}")
        return 0

    # ---------------------------------------------------------------- build
    requests, skipped = [], 0
    for record in records:
        built = build_request(record)
        if built is None:
            skipped += 1
            continue
        built["params"]["model"] = args.model
        requests.append(built)
    print(f"built {len(requests):,} requests ({skipped} skipped for a missing image)")

    payload_mib = sum(len(json.dumps(r)) for r in requests) / 2 ** 20
    print(f"payload: {payload_mib:.1f} MiB (the Batch API limit is 256 MB per batch)")
    if payload_mib > 240 and requests:
        per = payload_mib / len(requests)
        chunk = max(1, int(240 / per))
        parts = -(-len(requests) // chunk)
        print(f"  too large for one batch ({per*1024:.0f} KiB per request). "
              f"Submit in {parts} parts of {chunk:,}:")
        for i in range(parts):
            print(f"    python {pathlib.Path(__file__).name} --submit "
                  f"--offset {args.offset + i * chunk} --limit {chunk}")
        if args.submit and not args.dry_run:
            sys.exit("annotate: refusing to submit a batch over the size limit")

    if args.dry_run or not args.submit:
        sample = dict(requests[0]) if requests else {}
        if sample:
            shown = json.loads(json.dumps(sample))
            img = shown["params"]["messages"][0]["content"][0]["source"]
            img["data"] = f"<{len(sample['params']['messages'][0]['content'][0]['source']['data'])} base64 chars>"
            print("\n--- one request, image elided ---")
            print(json.dumps(shown, indent=2, ensure_ascii=False)[:2200])
        print("\nnothing sent. add --submit to create the batch.")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("annotate: ANTHROPIC_API_KEY is not set. On the cluster it comes "
                 "from the yeunsuh-anthropic Secret; locally, export it for this "
                 "shell only - do not write it into a file in the repo.")

    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    state = args.out / "scienceqa_batch.json"
    state.write_text(json.dumps({
        "batch_id": batch.id, "model": args.model, "requests": len(requests),
        "processing_status": batch.processing_status,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nbatch {batch.id} created ({batch.processing_status})")
    print(f"  state -> {state}")
    print(f"  collect with: python {pathlib.Path(__file__).name} --collect {batch.id}"
          + (f" --limit {args.limit}" if args.limit else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
