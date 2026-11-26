#!/usr/bin/env python3
"""Render normalized records into Qwen conversation JSONL, one file per arm.

Step 4 of the pipeline. Normalization (step 2) deliberately kept the data neutral
so this step owns every prompt decision, and so the same pool can feed arms that
want different supervision.

Target format, from README.md lines 92-94:
    {"image": "...", "conversations": [{"from":"human","value":"<image>\\n..."},
                                       {"from":"gpt","value":"..."}],
     "source": "..."}

Two target styles, matching DECISIONS.md section 2:

  answer_only   the assistant says only the answer, in the exact shape the
                evaluation prompt asks for. This is what arm A and arm A' train on.
  ekra          the assistant states evidence, knowledge and reasoning before the
                answer. Arms B, C1, C2 and D train on this.

The user turn is identical in both styles, and is byte-for-byte the evaluation
prompt from code/eval/prompting.py. That is deliberate: arm A then measures pure
format adaptation, and nothing in the comparison varies except supervision.

Evidence/knowledge/reasoning are read from each record's `meta`; they are written
by a later step (program synthesis for AI2D/ChartQA, Claude for ScienceQA - D34).
Records that lack them are skipped for the ekra style and counted, so a half-built
annotation set cannot silently become a half-sized training run.

    python code/data/render_targets.py --style answer_only --out training_data/sft/a
    python code/data/render_targets.py --style ekra --out training_data/sft/b
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CAND = ROOT / "training_data" / "candidates"

LETTERS = "ABCDEFGHI"

# Must stay identical to code/eval/prompting.py. If that file changes, this
# breaks the "same prompt before and after fine-tuning" guarantee (D4).
MC_TEMPLATE = "Question: {question}\nOptions:\n{options}\nAnswer with the option letter only."
OPEN_TEMPLATE = (
    "{question}\n\nSolve the question using the image(s) when relevant."
    "Answer with exactly one line in this format: Answer: <your short answer>."
    "You must not show reasoning."
)


def assert_matches_eval_prompt() -> None:
    """Fail if code/eval/prompting.py has drifted from the templates above.

    The whole value of arm A as a format-adaptation control depends on the
    training prompt being the evaluation prompt. A silent divergence here would
    invalidate every arm, so this is checked on every render rather than trusted.
    """
    source = (ROOT / "code" / "eval" / "prompting.py").read_text(encoding="utf-8")
    for fragment in (
        "f'Question: {question}\\nOptions:\\n{option_lines}\\n{instruction}'",
        "instruction = 'Answer with the option letter only.'",
        "'Solve the question using the image(s) when relevant.'",
        "'Answer with exactly one line in this format: Answer: <your short answer>.'",
        "'You must not show reasoning.'",
    ):
        if fragment not in source:
            raise RuntimeError(
                "render_targets: code/eval/prompting.py no longer contains\n"
                f"  {fragment}\n"
                "The training prompt and the evaluation prompt must stay identical "
                "(DECISIONS.md D4). Update both together, deliberately."
            )


def render_user(record: dict) -> str:
    if record["answer_type"] == "multiple_choice":
        options = record["options"] or []
        if len(options) > len(LETTERS):
            raise ValueError(f"{record['id']}: {len(options)} options exceeds A-I")
        lines = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
        body = MC_TEMPLATE.format(question=str(record["question"]).strip(), options=lines)
    else:
        body = OPEN_TEMPLATE.format(question=str(record["question"]).strip())
    # one <image> tag, images before text, as the evaluation does
    return "<image>\n" + body


def answer_text(record: dict) -> str:
    """What the evaluation's parser is built to read."""
    if record["answer_type"] == "multiple_choice":
        return LETTERS[record["answer_index"]]
    return f"Answer: {record['answer']}"


def evidence_item_text(item) -> str:
    """One evidence item as the text the model will be trained to produce.

    Evidence arrives in three shapes and only one of them is a string:

      str                       Claude's ScienceQA observations
      {"kind": "label", ...}    an AI2D label or a ChartQA cell, carrying "text"
      {"kind": "edge", ...}     an AI2D diagram arrow, carrying src/dst and no
                                "text" at all

    Reading "text" off an edge yields "", which is how all 1,750
    relation_grounded rows came to be dropped as empty_evidence. An edge is
    rendered as `src -> dst` instead: compact matters here, because the whole
    assistant turn has a 45-token budget (G2) and a food web cites six of them.
    """
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return str(item).strip()
    if item.get("kind") == "edge":
        src, dst = str(item.get("src", "")).strip(), str(item.get("dst", "")).strip()
        if not src or not dst:
            return ""
        arrow = "->" if item.get("directed", True) else "--"
        return f"{src} {arrow} {dst}"
    text = str(item.get("text", "")).strip()
    if text:
        return text
    # A dict of some other shape: take the one field that reads like content
    # rather than silently contributing nothing.
    for key in ("value", "label", "statement", "observation"):
        if str(item.get(key, "")).strip():
            return str(item[key]).strip()
    return ""


def render_assistant(record: dict, style: str):
    if style == "answer_only":
        return answer_text(record), None

    meta = record.get("meta") or {}
    evidence = meta.get("evidence")
    knowledge = (meta.get("knowledge") or "").strip()
    reasoning = meta.get("reasoning")
    if not evidence:
        return None, "no_evidence"
    if not reasoning:
        return None, "no_reasoning"

    parts = []
    if isinstance(evidence, list):
        pieces = [t for t in (evidence_item_text(e) for e in evidence) if t]
        rendered = "; ".join(pieces)
    else:
        rendered = str(evidence).strip()
    if not rendered:
        return None, "empty_evidence"
    parts.append(f"Evidence: {rendered}")
    if knowledge:
        parts.append(f"Knowledge: {knowledge}")
    steps = reasoning if isinstance(reasoning, list) else [str(reasoning)]
    steps = [s.strip() for s in steps if str(s).strip()]
    if not steps:
        return None, "empty_reasoning"
    parts.append("Reasoning: " + " ".join(steps))
    # The last line is byte-identical to the answer_only target, so the answer
    # extractor reads both styles with the same rule.
    parts.append(answer_text(record))
    return "\n".join(parts), None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--style", required=True, choices=("answer_only", "ekra"))
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="directory; train.jsonl is written inside it")
    parser.add_argument("--source", type=pathlib.Path, default=CAND,
                        help="directory of normalized *.jsonl (default: candidates)")
    parser.add_argument("--image-prefix", default="images",
                        help="path prefix the JSONL should use for images")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    assert_matches_eval_prompt()

    files = sorted(p for p in args.source.glob("*.jsonl"))
    if not files:
        sys.exit(f"render_targets: no *.jsonl in {args.source}")

    kept = collections.Counter()
    skipped = collections.Counter()
    rows = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("holdout"):
                    skipped["holdout"] += 1
                    continue
                assistant, reason = render_assistant(record, args.style)
                if assistant is None:
                    skipped[reason] += 1
                    continue
                image = record["image"]
                if args.image_prefix != "images" and image.startswith("images/"):
                    image = args.image_prefix + image[len("images"):]
                rows.append({
                    "image": image,
                    "conversations": [
                        {"from": "human", "value": render_user(record)},
                        {"from": "gpt", "value": assistant},
                    ],
                    "source": record["source"],
                    "uid": record["id"],
                })
                kept[record["source"]] += 1

    print(f"style: {args.style}")
    for source, n in sorted(kept.items()):
        print(f"  {source:18} {n:>7,}")
    print(f"  {'TOTAL':18} {sum(kept.values()):>7,}")
    if skipped:
        print(f"skipped: {json.dumps(dict(skipped))}")
    if args.style == "ekra" and skipped.get("no_evidence"):
        print("\n  NOTE: evidence annotations are not in place yet, so the ekra style\n"
              "  cannot be rendered for those records. Run the annotation step first.")

    if not rows:
        print("\nnothing to write")
        return 1
    if args.dry_run:
        print("\ndry run - nothing written")
        print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:600])
        return 0

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "train.jsonl"
    with target.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "render_manifest.json").write_text(json.dumps({
        "style": args.style,
        "source": args.source.resolve().as_posix(),
        "kept": dict(kept),
        "kept_total": sum(kept.values()),
        "skipped": dict(skipped),
        "mc_template": MC_TEMPLATE,
        "open_template": OPEN_TEMPLATE,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # --out may point outside the repo (a scratch directory, when rendering a
    # pilot for measurement), and relative_to raises on that. The work is
    # already done by here, so a path that cannot be shortened is printed whole
    # rather than crashing after a successful write.
    try:
        shown = target.relative_to(ROOT)
    except ValueError:
        shown = target
    print(f"\nwrote {len(rows):,} rows -> {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
