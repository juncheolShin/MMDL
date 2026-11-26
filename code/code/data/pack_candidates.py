#!/usr/bin/env python3
"""Stage the candidate pool for upload: the JSONL files plus only their images.

training_data/normalized/ holds 59,556 images (9.0 GiB) for every dataset.
The Phase 0 candidate pool needs 11,317 of them (0.82 GiB). Copying the whole
tree to the cluster would put five times more small files on a shared Lustre
filesystem than the run actually reads.

This builds training_data/cluster_upload/ containing exactly what the pool
references, with image paths unchanged so the JSONL keeps working:

    cluster_upload/
      candidates/{ai2d_original,scienceqa,chartqa}.jsonl
      candidates/{candidate_manifest.json,images.txt}
      images/<ab>/<sha256>.<ext>      <- only the referenced ones

Hard links are used where the filesystem allows it, so staging costs no extra
disk; it falls back to copying.

    python code/data/pack_candidates.py
    python code/data/pack_candidates.py --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
NORM = ROOT / "training_data" / "normalized"
CAND = ROOT / "training_data" / "candidates"
OUT = ROOT / "training_data" / "cluster_upload"


def referenced_images() -> list[str]:
    """Read image paths from the JSONL itself, not just images.txt.

    images.txt is written by split.py; reading the records keeps this honest if
    the two ever drift.
    """
    seen: dict[str, None] = {}
    for jsonl in sorted(CAND.glob("*.jsonl")):
        with jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    seen.setdefault(json.loads(line)["image"], None)
    return list(seen)


def link_or_copy(src: pathlib.Path, dest: pathlib.Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return "present"
    try:
        os.link(src, dest)
        return "linked"
    except OSError:
        shutil.copy2(src, dest)
        return "copied"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not CAND.is_dir():
        sys.exit("pack_candidates: run code/data/split.py first")

    images = referenced_images()
    print(f"candidate pool references {len(images):,} unique images")

    missing = [rel for rel in images if not (NORM / rel).is_file()]
    if missing:
        print(f"MISSING {len(missing)} image(s), first few: {missing[:3]}", file=sys.stderr)
        return 1
    total = sum((NORM / rel).stat().st_size for rel in images)
    print(f"total image bytes: {total/2**30:.2f} GiB")

    # cross-check against split.py's own list
    listed = (CAND / "images.txt").read_text(encoding="utf-8").split()
    if set(listed) != set(images):
        print(f"WARNING: images.txt has {len(listed):,} entries but the JSONL "
              f"references {len(images):,}; using the JSONL", file=sys.stderr)

    if args.verify_only:
        print("verify only - nothing staged")
        return 0

    counts = {"linked": 0, "copied": 0, "present": 0}
    for i, rel in enumerate(images, 1):
        counts[link_or_copy(NORM / rel, OUT / rel)] += 1
        if i % 2000 == 0:
            print(f"  {i:,}/{len(images):,}", flush=True)

    dest_cand = OUT / "candidates"
    dest_cand.mkdir(parents=True, exist_ok=True)
    for name in sorted(CAND.iterdir()):
        if name.is_file():
            link_or_copy(name, dest_cand / name.name)

    digest = hashlib.sha256("\n".join(sorted(images)).encode("utf-8")).hexdigest()
    (OUT / "UPLOAD_MANIFEST.json").write_text(json.dumps({
        "images": len(images),
        "image_bytes": total,
        "image_list_sha256": digest,
        "staged_from": str(NORM.relative_to(ROOT)).replace("\\", "/"),
        "note": "paths inside candidates/*.jsonl are relative to this directory",
    }, indent=2) + "\n", encoding="utf-8")

    staged = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"\nstaged {staged:,} files into {OUT.relative_to(ROOT)}")
    print(f"  {counts['linked']:,} hard-linked, {counts['copied']:,} copied, "
          f"{counts['present']:,} already there")
    print(f"image list sha256: {digest[:16]}...")
    print(f"\nupload with:\n  bash code/cluster/push_to_pod.sh "
          f"training_data/cluster_upload /workspace/MMDL/training_data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
