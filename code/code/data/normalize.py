#!/usr/bin/env python3
"""Normalize every downloaded source into one schema.

Reads  : training_data/raw/<dataset>/**.parquet   (HF snapshots, pinned in configs/data_pins.json)
         training_data/raw_external/ai2d_original/extracted/ai2d  (original AI2D)
Writes : training_data/normalized/<dataset>/<split>.jsonl
         training_data/normalized/images/<ab>/<sha256>.<ext>   (content-addressed, deduped)
         training_data/normalized/normalize_manifest.json

One JSONL record:
  id            "<dataset>/<split>/<n>"      stable within a run
  source        dataset name
  split         train | validation | test
  image         path relative to training_data/normalized/
  question      str
  options       list[str] | null             present iff answer_type == "multiple_choice"
  answer        str                          the answer text (for MC: options[answer_index])
  answer_index  int | null
  answer_type   multiple_choice | short_text
  holdout       bool                         true = never train on this (it is eval data).
                                             true for: ai2d (the lmms-lab test split) and
                                             chartqapro (a secondary eval set per
                                             configs/data_sources.toml).
  meta          dict                         source-specific extras

Deliberate exclusions, each counted in the manifest under "dropped":
  * scienceqa rows with no image      - this is a multimodal mix; ~51% of ScienceQA is text-only
  * textvqa  test split               - answers are all empty strings (unlabeled)
  * ai2d_original questions whose image id appears in the lmms-lab ai2d test split
                                      - that split is our eval set; training on it would leak

This step does NOT build prompts. It only unifies the data. Prompt rendering
(which must match the evaluation prompt in assignment/assignment1.md section 2)
is a separate, later step.

Run:  .venv/Scripts/python.exe -I code/data/normalize.py
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import sys

import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "training_data" / "raw"
RAW_EXT = ROOT / "training_data" / "raw_external"
OUT = ROOT / "training_data" / "normalized"
IMG_DIR = OUT / "images"

SPLIT_ALIASES = {"val": "validation", "valid": "validation", "dev": "validation"}


# ---------------------------------------------------------------- image store


class ImageStore:
    """Content-addressed image store; identical bytes are written once."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        self.seen = {}
        self.written = 0
        self.deduped = 0

    @staticmethod
    def _ext(data: bytes) -> str:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        if data[:2] == b"\xff\xd8":
            return "jpg"
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return "gif"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "webp"
        if data[:2] == b"BM":
            return "bmp"
        return "bin"

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        if digest in self.seen:
            self.deduped += 1
            return self.seen[digest]
        rel = "images/{}/{}.{}".format(digest[:2], digest, self._ext(data))
        dest = self.root.parent / rel
        if dest.exists():
            self.deduped += 1
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            self.written += 1
        self.seen[digest] = rel
        return rel


def image_bytes(value):
    """HF image columns arrive either as a {bytes, path} dict or as raw bytes."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value) or None
    if isinstance(value, dict):
        data = value.get("bytes")
        return bytes(data) if data else None
    return None


def image_path_hint(value):
    """The original file name, where the source kept it (AI2D needs it)."""
    if isinstance(value, dict):
        return value.get("path")
    return None


def first(value):
    """Several sources wrap a scalar in a one-element list."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def split_of(path: pathlib.Path) -> str:
    name = path.stem.split("-")[0].lower()
    return SPLIT_ALIASES.get(name, name)


def parquet_files(dataset: str):
    return sorted(p for p in (RAW / dataset).rglob("*.parquet") if ".cache" not in p.parts)


def iter_rows(dataset: str, columns=None):
    for path in parquet_files(dataset):
        split = split_of(path)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                yield split, row


# ---------------------------------------------------------------- normalizers
# Each yields (record, drop_reason); exactly one of the two is None.


def norm_ai2d_hf(store: ImageStore):
    """lmms-lab ai2d: test split only. This is our eval set, so holdout=True."""
    for split, row in iter_rows("ai2d"):
        data = image_bytes(row.get("image"))
        if data is None:
            yield None, "no_image"
            continue
        options = [str(o) for o in (row.get("options") or [])]
        try:
            idx = int(str(row.get("answer")).strip())
        except (TypeError, ValueError):
            yield None, "unparsable_answer"
            continue
        if not 0 <= idx < len(options):
            yield None, "answer_index_out_of_range"
            continue
        yield {
            "source": "ai2d",
            "split": split,
            "image": store.put(data),
            "question": row["question"],
            "options": options,
            "answer": options[idx],
            "answer_index": idx,
            "answer_type": "multiple_choice",
            "holdout": True,
            "meta": {"ai2d_image_id": image_path_hint(row.get("image"))},
        }, None


def ai2d_test_image_ids():
    ids = set()
    for path in parquet_files("ai2d"):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["image"]):
            for value in batch.column(0).to_pylist():
                hint = image_path_hint(value)
                if hint:
                    ids.add(hint)
    return ids


def norm_ai2d_original(store: ImageStore, blocked):
    """Original AI2D (15,501 Q over 4,903 diagrams) minus anything in the eval split."""
    base = RAW_EXT / "ai2d_original" / "extracted" / "ai2d"
    qdir, idir = base / "questions", base / "images"
    for qfile in sorted(qdir.glob("*.json")):
        doc = json.loads(qfile.read_text(encoding="utf-8"))
        image_id = doc.get("imageName") or qfile.name[: -len(".json")]
        ipath = idir / image_id
        if not ipath.exists():
            yield None, "missing_image_file"
            continue
        questions = doc.get("questions", {})
        if image_id in blocked:
            # every question on this diagram would leak into the ai2d eval split
            for _ in questions:
                yield None, "eval_overlap"
            continue
        rel = store.put(ipath.read_bytes())
        for question, spec in questions.items():
            options = [str(o) for o in spec.get("answerTexts", [])]
            try:
                idx = int(spec.get("correctAnswer"))
            except (TypeError, ValueError):
                yield None, "unparsable_answer"
                continue
            if not 0 <= idx < len(options):
                yield None, "answer_index_out_of_range"
                continue
            yield {
                "source": "ai2d_original",
                "split": "train",
                "image": rel,
                "question": question,
                "options": options,
                "answer": options[idx],
                "answer_index": idx,
                "answer_type": "multiple_choice",
                "holdout": False,
                "meta": {
                    "ai2d_image_id": image_id,
                    "question_id": spec.get("questionId"),
                    # abcLabel questions ask about letters drawn inside the diagram
                    "abc_label": bool(spec.get("abcLabel")),
                },
            }, None


def norm_chartqa(store: ImageStore):
    for split, row in iter_rows("chartqa"):
        data = image_bytes(row.get("image"))
        if data is None:
            yield None, "no_image"
            continue
        answer = first(row.get("label"))
        if answer is None or not str(answer).strip():
            yield None, "empty_answer"
            continue
        yield {
            "source": "chartqa",
            "split": split,
            "image": store.put(data),
            "question": row["query"],
            "options": None,
            "answer": str(answer).strip(),
            "answer_index": None,
            "answer_type": "short_text",
            "holdout": False,
            # 0 = human-written question, 1 = machine-generated
            "meta": {"human_or_machine": row.get("human_or_machine")},
        }, None


def norm_chartqapro(store: ImageStore):
    """configs/data_sources.toml marks ChartQAPro a secondary EVAL set, not training
    data, so every row is holdout."""
    for split, row in iter_rows("chartqapro"):
        data = image_bytes(row.get("image"))
        if data is None:
            yield None, "no_image"
            continue
        question = first(row.get("Question"))
        answer = first(row.get("Answer"))
        if not question or answer is None or not str(answer).strip():
            yield None, "empty_question_or_answer"
            continue
        yield {
            "source": "chartqapro",
            "split": split,
            "image": store.put(data),
            "question": str(question).strip(),
            "options": None,
            "answer": str(answer).strip(),
            "answer_index": None,
            "answer_type": "short_text",
            "holdout": True,
            "meta": {
                "question_type": row.get("Question Type"),
                "year": first(row.get("Year")),
                "paragraph": row.get("Paragraph") or "",
            },
        }, None


def norm_scienceqa(store: ImageStore):
    for split, row in iter_rows("scienceqa"):
        data = image_bytes(row.get("image"))
        if data is None:
            yield None, "no_image"  # ~51% of ScienceQA is text-only
            continue
        options = [str(c) for c in (row.get("choices") or [])]
        idx = row.get("answer")
        if not isinstance(idx, int) or not 0 <= idx < len(options):
            yield None, "answer_index_out_of_range"
            continue
        yield {
            "source": "scienceqa",
            "split": split,
            "image": store.put(data),
            "question": row["question"],
            "options": options,
            "answer": options[idx],
            "answer_index": idx,
            "answer_type": "multiple_choice",
            "holdout": False,
            "meta": {
                "hint": row.get("hint") or "",
                "subject": row.get("subject"),
                "topic": row.get("topic"),
                "grade": row.get("grade"),
                # kept for a later step; this step puts neither into a prompt
                "lecture": row.get("lecture") or "",
                "solution": row.get("solution") or "",
            },
        }, None


def norm_textvqa(store: ImageStore):
    """TextVQA ships 10 human answers per question; take the majority."""
    for split, row in iter_rows("textvqa"):
        if split == "test":
            yield None, "unlabeled_split"  # every answer on the test split is ""
            continue
        data = image_bytes(row.get("image"))
        if data is None:
            yield None, "no_image"
            continue
        answers = [str(a).strip() for a in (row.get("answers") or []) if str(a).strip()]
        if not answers:
            yield None, "empty_answer"
            continue
        counts = collections.Counter(answers)
        top = max(counts.values())
        # tie-break on first appearance, so the result does not depend on dict order
        answer = next(a for a in answers if counts[a] == top)
        yield {
            "source": "textvqa",
            "split": split,
            "image": store.put(data),
            "question": row["question"],
            "options": None,
            "answer": answer,
            "answer_index": None,
            "answer_type": "short_text",
            "holdout": False,
            "meta": {
                "image_id": row.get("image_id"),
                "question_id": row.get("question_id"),
                "answer_agreement": round(top / len(answers), 3),
                "all_answers": answers,
                "ocr_tokens": row.get("ocr_tokens") or [],
            },
        }, None


# ---------------------------------------------------------------------- driver

ALL_DATASETS = ["ai2d", "ai2d_original", "chartqa", "chartqapro", "scienceqa", "textvqa"]


def run(datasets):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    store = ImageStore(IMG_DIR)

    blocked = ai2d_test_image_ids() if "ai2d_original" in datasets else set()
    if blocked:
        print("ai2d eval split covers {} diagrams; blocked from training".format(len(blocked)), flush=True)

    builders = {
        "ai2d": lambda: norm_ai2d_hf(store),
        "ai2d_original": lambda: norm_ai2d_original(store, blocked),
        "chartqa": lambda: norm_chartqa(store),
        "chartqapro": lambda: norm_chartqapro(store),
        "scienceqa": lambda: norm_scienceqa(store),
        "textvqa": lambda: norm_textvqa(store),
    }

    report = {}
    for name in datasets:
        handles = {}
        kept = collections.Counter()
        dropped = collections.Counter()
        try:
            for record, reason in builders[name]():
                if record is None:
                    dropped[reason] += 1
                    continue
                split = record["split"]
                kept[split] += 1
                record = dict(id="{}/{}/{:06d}".format(name, split, kept[split]), **record)
                if split not in handles:
                    path = OUT / name / (split + ".jsonl")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handles[split] = path.open("w", encoding="utf-8")
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            for handle in handles.values():
                handle.close()

        report[name] = {
            "kept": dict(kept),
            "kept_total": sum(kept.values()),
            "dropped": dict(dropped),
            "dropped_total": sum(dropped.values()),
        }
        line = "{:16} kept {:>6}  {}".format(
            name, sum(kept.values()), "  ".join("{}={}".format(k, v) for k, v in sorted(kept.items()))
        )
        if dropped:
            line += "   dropped " + json.dumps(dict(dropped))
        print(line, flush=True)

    report["_images"] = {
        "written": store.written,
        "reused": store.deduped,
        "unique": len(store.seen),
    }
    print("images: {} unique written, {} reuses".format(store.written, store.deduped), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Normalize downloaded sources into one schema.")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report = run(args.datasets)

    # Merge into any existing manifest, so re-running one dataset does not erase
    # the stats of the others. Each entry carries the time it was last written.
    manifest = OUT / "normalize_manifest.json"
    merged = {}
    if manifest.exists():
        try:
            merged = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("existing manifest is not valid JSON; replacing it", file=sys.stderr)
            merged = {}
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    images = report.pop("_images")
    for name, entry in report.items():
        merged[name] = dict(entry, written_utc=stamp)
    # Image stats describe this invocation only, not the whole tree.
    merged["_images_last_run"] = dict(images, datasets=args.datasets, written_utc=stamp)
    merged.pop("_images", None)
    manifest.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nmanifest -> {}".format(manifest.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
