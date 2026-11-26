#!/usr/bin/env python3
"""Content hash and perceptual hash for every image under a folder.

    python code/data/image_hash.py --root /workspace/data/images --out mmmu_hashes.tsv

Columns: path, sha256, dhash (64-bit, hex). The dhash is what catches an image that
was re-encoded or resized; sha256 only catches byte-identical copies. prepare_geoqa.py
imports dhash64 so both sides of an overlap check use the same function.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import pathlib

from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def dhash64(image: Image.Image) -> int:
    """Difference hash: 9x8 grayscale, one bit per left>right comparison."""
    small = image.convert("L").resize((9, 8), Image.LANCZOS)
    px = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if px[row * 9 + col] > px[row * 9 + col + 1] else 0)
    return bits


def hash_bytes(data: bytes) -> tuple[str, int]:
    return hashlib.sha256(data).hexdigest(), dhash64(Image.open(io.BytesIO(data)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    n = bad = 0
    with args.out.open("w", encoding="utf-8", newline="\n") as out:
        out.write("path\tsha256\tdhash\n")
        for path in sorted(args.root.rglob("*")):
            if path.suffix.lower() not in EXTS:
                continue
            try:
                sha, dh = hash_bytes(path.read_bytes())
            except Exception as exc:  # an unreadable image must be visible, not skipped silently
                bad += 1
                print(f"unreadable: {path}: {exc}")
                continue
            out.write(f"{path.relative_to(args.root).as_posix()}\t{sha}\t{dh:016x}\n")
            n += 1
    print(f"hashed {n:,} images ({bad} unreadable) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
