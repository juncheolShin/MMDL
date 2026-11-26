#!/usr/bin/env python3
"""Find out why training is 30x slower than the arithmetic says it should be.

The smoke test did 2 optimizer steps in 362 seconds - 183s per step. For 32
sequences of 2048 tokens on a 4B model that is about 1.6 PFLOP per step, which
an H200 should finish in roughly 5 seconds even at 30% utilisation. Something
other than arithmetic is taking the time, and resolution is not it: the training
profile runs at min_pixels 65,536 / max_pixels 524,288, far below the evaluation
settings.

The leading suspect is image loading. Every sample opens one image file, and
/workspace is Lustre - a shared parallel filesystem whose per-file latency is
high and whose metadata server is shared with the rest of the lab. Four
dataloader workers cannot hide that.

This measures each stage separately, so the answer is a number rather than a
guess. It needs a GPU only for the last stage, and the first three say most of it.

    python code/train/diagnose_throughput.py --images /workspace/MMDL/training_data/cluster_upload
    python code/train/diagnose_throughput.py --images ... --copy-to /dev/shm/imgtest
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import shutil
import statistics
import sys
import time


def timed(fn, n):
    """Return (total_seconds, per_item_ms) for n calls of fn(i)."""
    start = time.perf_counter()
    for i in range(n):
        fn(i)
    total = time.perf_counter() - start
    return total, 1000 * total / max(n, 1)


def stage_stat(paths, label, open_only=False):
    """Read, or read and decode, a list of image paths."""
    from PIL import Image
    sizes = []

    def one(i):
        path = paths[i % len(paths)]
        data = path.read_bytes()
        sizes.append(len(data))
        if not open_only:
            with Image.open(path) as img:
                img.load()

    total, per = timed(one, len(paths))
    mib = sum(sizes) / 2 ** 20
    print(f"  {label:44} {total:7.2f}s  {per:7.2f} ms/img  "
          f"{mib / max(total, 1e-9):7.1f} MiB/s")
    return per


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images", type=pathlib.Path, required=True,
                        help="directory holding images/<ab>/<sha>.<ext>")
    parser.add_argument("--n", type=int, default=200, help="images to sample")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--copy-to", type=pathlib.Path, default=None,
                        help="also time the same reads from here (e.g. /dev/shm/x)")
    parser.add_argument("--workers", type=int, nargs="*", default=[0, 4, 8, 16],
                        help="dataloader worker counts to compare")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    root = args.images
    pool = sorted((root / "images").rglob("*")) if (root / "images").is_dir() else []
    pool = [p for p in pool if p.is_file()]
    if not pool:
        sys.exit(f"diagnose_throughput: no images under {root / 'images'}")
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sample = pool[:args.n]
    print(f"{len(pool):,} images available; timing {len(sample)}")

    report = {"n": len(sample), "images_root": str(root)}

    # ---- 1. where the bytes live -------------------------------------------
    print("\n[1] single-threaded read from the training filesystem")
    report["read_only_ms"] = stage_stat(sample, "read bytes only", open_only=True)
    report["read_decode_ms"] = stage_stat(sample, "read + PIL decode")

    # second pass: if this is much faster, the first pass measured cold cache
    print("\n[2] same files again (page cache now warm)")
    report["read_decode_warm_ms"] = stage_stat(sample, "read + PIL decode, warm")

    # ---- 3. the same files from memory, for comparison ----------------------
    if args.copy_to:
        print(f"\n[3] copied to {args.copy_to} first")
        args.copy_to.mkdir(parents=True, exist_ok=True)
        local = []
        t0 = time.perf_counter()
        for src in sample:
            dst = args.copy_to / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            local.append(dst)
        print(f"  {'copy time':44} {time.perf_counter() - t0:7.2f}s")
        report["local_read_decode_ms"] = stage_stat(local, "read + PIL decode, local")
        if report.get("read_decode_ms"):
            speedup = report["read_decode_ms"] / max(report["local_read_decode_ms"], 1e-9)
            report["local_speedup"] = round(speedup, 2)
            print(f"  {'=> local is this many times faster':44} {speedup:7.2f}x")

    # ---- 4. parallel reads, which is what the dataloader does --------------
    print("\n[4] parallel reads, as dataloader workers would")
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image

        class ImageSet(Dataset):
            def __init__(self, paths):
                self.paths = paths

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                with Image.open(self.paths[idx]) as img:
                    img.load()
                    return torch.tensor(img.size)

        per_worker = {}
        for workers in args.workers:
            loader = DataLoader(ImageSet(sample), batch_size=8, num_workers=workers,
                                shuffle=False, persistent_workers=False)
            t0 = time.perf_counter()
            for _ in loader:
                pass
            total = time.perf_counter() - t0
            per = 1000 * total / len(sample)
            per_worker[workers] = round(per, 2)
            print(f"  num_workers={workers:<3}{'':30} {total:7.2f}s  {per:7.2f} ms/img")
        report["per_worker_ms"] = per_worker
        best = min(per_worker, key=per_worker.get)
        print(f"  => fastest at num_workers={best} ({per_worker[best]} ms/img)")
        report["best_workers"] = best
    except ImportError as exc:
        print(f"  skipped: {exc}")

    # ---- 5. what that implies for a training run ---------------------------
    print("\n[5] implied training cost")
    per_img = report.get("per_worker_ms", {}).get(report.get("best_workers"),
                                                  report.get("read_decode_ms", 0))
    for n_samples, label in ((10_000, "10,000-sample epoch"), (20_222, "full candidate pool")):
        seconds = per_img / 1000 * n_samples
        print(f"  image loading alone for a {label:26} {seconds/60:8.1f} min")
    report["loading_minutes_10k"] = round(per_img / 1000 * 10_000 / 60, 1)
    print("\n  Compare with the 183 s/step the smoke test measured at 32 samples/step:")
    print(f"  loading 32 images at {per_img:.1f} ms each is "
          f"{32 * per_img / 1000:.1f}s, i.e. "
          f"{100 * 32 * per_img / 1000 / 183:.1f}% of a step.")
    print("  If that share is small, the time is going somewhere else - look at")
    print("  gradient_checkpointing, padding to model_max_length, and whether the")
    print("  visual tower is recomputing per sample.")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
