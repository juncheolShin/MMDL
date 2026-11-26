#!/usr/bin/env python3
"""Name the operation that is eating the training step.

Measured on h200-03-w-7271 (DECISIONS.md section 16):

  step time          174.5 s steady state (not the first step - compile excluded)
  GPU                54% "utilization", SM at full 1,980 MHz, but only 120 W of
                     ~700 W, and 71 GB resident
  CPU                main Python thread at 98.2%, i.e. 1.00 of 8 cores, every
                     other thread idle
  I/O                no D-state threads, io.pressure 0, memory.pressure 0
  ruled out          the missing C compiler (installed, step time unchanged),
                     Lustre reads (0.93 ms/img), PIL decode (3.4 ms/img),
                     the Qwen image processor (10.2 ms/img -> 0.3 s per step),
                     CPU throttling (nr_throttled 0)

One core pegged in Python with the GPU clocked up but drawing no power means the
step is DISPATCH-bound: the main thread cannot issue work fast enough, and each
kernel finishes before the next arrives. 174 CPU-seconds of Python per step is
something on the order of ten million op dispatches, which no amount of LoRA
bookkeeping explains - 252 adapters are a few thousand ops. Something is looping
in Python at per-token or per-patch granularity.

torch.profiler counts dispatches per operator, so it answers this directly rather
than by elimination. Run it on one micro-batch, sorted by self CPU time: whatever
sits at the top with a call count in the millions is the loop.

    python code/train/profile_step.py --annotation .../train.jsonl --image-root ...
    python code/train/profile_step.py ... --steps 2 --rows 60
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="/workspace/models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--rows", type=int, default=8, help="samples in the micro-batch")
    ap.add_argument("--min-pixels", type=int, default=65536)
    ap.add_argument("--max-pixels", type=int, default=524288)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=2, help="timed fwd+bwd passes")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image

    # ---- GPU sanity check, before anything model-specific ----------------
    # 120 W turned out to be this H200's IDLE draw (measured at 0% utilisation),
    # so "120 W during training" means the GPU was idle, not lightly loaded. The
    # question that splits the remaining causes in two is whether PyTorch can
    # reach a sane FLOP rate here AT ALL. A plain bf16 matmul answers it: near
    # peak means the stack is fine and the fault is in the model's code path;
    # far below means the fault is in the image, the build or the driver, and no
    # amount of modelling changes will help.
    size, iters = 8192, 20
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = 2 * size ** 3 * iters / dt / 1e12
    print(f"[sanity] bf16 {size}^3 matmul: {tflops:,.0f} TFLOP/s "
          f"(H200 dense peak is about 990; healthy is 400+)", flush=True)
    print(f"[sanity] power under that load: ", end="", flush=True)
    import subprocess
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3:
        a @ b
    out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True)
    torch.cuda.synchronize()
    print(out.stdout.strip(), flush=True)
    del a, b
    torch.cuda.empty_cache()

    print()
    print("loading model (bf16, sdpa - the same as training)", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_disable()
    model.train()
    proc = AutoProcessor.from_pretrained(args.model)

    rows = []
    with open(args.annotation, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.rows:
                break
    print(f"{len(rows)} samples from {args.annotation}", flush=True)

    image_root = pathlib.Path(args.image_root)
    texts, images = [], []
    for r in rows:
        human = r["conversations"][0]["value"].replace("<image>", "").strip()
        gpt = r["conversations"][1]["value"]
        texts.append(proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": human}]},
             {"role": "assistant", "content": [{"type": "text", "text": gpt}]}],
            tokenize=False))
        with Image.open(image_root / r["image"]) as im:
            images.append(im.convert("RGB"))

    # padding="max_length" to 2048 makes the logits 8 x 2048 x 151,936, which
    # cross_entropy upcasts to about 10 GiB of fp32 and OOMs. The upstream
    # collator pads to the LONGEST sequence in the batch (pad_sequence in
    # qwenvl/data/data_processor.py), so pad the same way or this benchmark is
    # not measuring the same step.
    batch = proc(text=texts, images=images, return_tensors="pt", padding=True,
                 truncation=True, max_length=args.max_length,
                 min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    batch = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in batch.items()}
    batch["labels"] = batch["input_ids"].clone()
    grid = batch.get("image_grid_thw")
    b, t = batch["input_ids"].shape
    vocab = model.config.text_config.vocab_size
    print(f"logits will be {b} x {t} x {vocab:,} = "
          f"{b*t*vocab*2/2**30:.1f} GiB in bf16 "
          f"({b*t*vocab*4/2**30:.1f} GiB once cross_entropy upcasts to fp32)")
    print(f"input_ids {tuple(batch['input_ids'].shape)}"
          + (f"  image patches {int(grid.prod(dim=1).sum())}" if grid is not None else ""),
          flush=True)

    def one_pass():
        out = model(**batch)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        return float(out.loss)

    print("\nwarmup pass (compilation and allocator growth land here)", flush=True)
    t0 = time.perf_counter()
    loss = one_pass()
    torch.cuda.synchronize()
    print(f"  warmup {time.perf_counter() - t0:.1f}s, loss {loss:.4f}", flush=True)

    print(f"\n{args.steps} timed passes", flush=True)
    for i in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_pass()
        torch.cuda.synchronize()
        print(f"  pass {i+1}: {time.perf_counter() - t0:.2f}s "
              f"(x4 micro-batches = {4*(time.perf_counter()-t0):.0f}s per optimizer step)",
              flush=True)

    # ---- localize: vision tower vs language model -------------------------
    # The GPU does 783 TFLOP/s at 693 W in this very pod, so the stack is fine
    # and the 10 s belongs to this model's code. Qwen3-VL has two halves; run a
    # batch with NO images through the same model and the slow half names itself.
    print()
    print("text-only pass (same shapes, no images at all)", flush=True)
    text_batch = {k: v for k, v in batch.items()
                  if k not in ("pixel_values", "image_grid_thw",
                               "pixel_values_videos", "video_grid_thw")}
    img_id = getattr(model.config, "image_token_id", None)
    if img_id is not None:
        ids = text_batch["input_ids"].clone()
        ids[ids == img_id] = proc.tokenizer.pad_token_id or 0
        text_batch["input_ids"] = ids
        text_batch["labels"] = ids.clone()

    def text_pass():
        out = model(**text_batch)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    try:
        text_pass()
        torch.cuda.synchronize()
        for i in range(2):
            t0 = time.perf_counter()
            text_pass()
            torch.cuda.synchronize()
            print(f"  text-only pass {i+1}: {time.perf_counter() - t0:.2f}s", flush=True)
    except Exception as exc:
        print(f"  text-only pass failed: {type(exc).__name__}: {exc}", flush=True)

    # ---- the vision tower, alone ------------------------------------------
    # Measured: full pass 10.34 s, the SAME batch with no images 0.10 s. The
    # language model is fine and the entire cost is the vision path, so profile
    # just that - a far smaller graph than the whole step, which lets the
    # profiler table actually finish.
    print()
    print("vision tower alone", flush=True)
    visual = None
    for path in ("model.visual", "visual", "model.model.visual"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            visual = obj
            print(f"  found the vision tower at model.{path}", flush=True)
            break
        except AttributeError:
            continue

    if visual is None:
        print("  could not locate the vision tower; skipping", flush=True)
    else:
        pv, grid = batch.get("pixel_values"), batch.get("image_grid_thw")
        n_patches = int(grid.prod(dim=1).sum()) if grid is not None else -1

        def vision_pass(pixel_values, grid_thw):
            with torch.no_grad():
                return visual(pixel_values, grid_thw=grid_thw)

        try:
            vision_pass(pv, grid)
            torch.cuda.synchronize()
            for i2 in range(2):
                t0 = time.perf_counter()
                vision_pass(pv, grid)
                torch.cuda.synchronize()
                print(f"  forward only, {n_patches} patches, 8 images: "
                      f"{time.perf_counter() - t0:.2f}s", flush=True)

            # Is the cost per image or per patch? One image answers it.
            rows1 = int(grid[0].prod())
            t0 = time.perf_counter()
            vision_pass(pv[:rows1], grid[:1])
            torch.cuda.synchronize()
            one = time.perf_counter() - t0
            print(f"  forward only, {rows1} patches, 1 image:  {one:.2f}s", flush=True)
            print(f"  => {'roughly linear in images' if one > 0 else ''}"
                  f"  per-image cost about {one:.2f}s", flush=True)

            # Stage-by-stage, because torch.profiler cannot finish a table on
            # this graph - it hangs, which is itself a sign of the event count.
            # Cost is linear in PATCHES (352 -> 1.51 s, 2368 -> 10.0 s), so the
            # per-image split loop in the vision attention and its 24 .tolist()
            # syncs are ruled out: one image pays those too and costs 1.51 s.
            print("  stage timings (2368 patches, forward, no_grad):", flush=True)

            def tm(label, fn, reps=3):
                with torch.no_grad():
                    fn()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        out = fn()
                    torch.cuda.synchronize()
                    dt = (time.perf_counter() - t0) / reps
                print(f"    {label:38} {1000*dt:9.2f} ms", flush=True)
                return out, dt

            embedded = None
            try:
                embedded, _ = tm("patch_embed", lambda: visual.patch_embed(pv))
            except Exception as exc:
                print(f"    patch_embed failed: {type(exc).__name__}: {exc}", flush=True)
            try:
                tm("fast_pos_embed_interpolate",
                   lambda: visual.fast_pos_embed_interpolate(grid))
            except Exception as exc:
                print(f"    pos_embed_interpolate failed: {type(exc).__name__}", flush=True)
            try:
                tm("rot_pos_emb", lambda: visual.rot_pos_emb(grid))
            except Exception as exc:
                print(f"    rot_pos_emb failed: {type(exc).__name__}", flush=True)

            if embedded is not None:
                lengths = (grid[:, 1] * grid[:, 2]).repeat_interleave(grid[:, 0])
                cu = torch.nn.functional.pad(
                    lengths.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)
                rpe = visual.rot_pos_emb(grid).reshape(embedded.shape[0], -1)
                emb2 = torch.cat((rpe, rpe), dim=-1)
                pe = (emb2.cos(), emb2.sin())
                blk0 = visual.blocks[0]
                try:
                    tm("ONE vision block",
                       lambda: blk0(embedded, cu_seqlens=cu, position_embeddings=pe))
                except Exception as exc:
                    print(f"    one block failed: {type(exc).__name__}: {exc}", flush=True)

                def all_blocks():
                    h = embedded
                    for b in visual.blocks:
                        h = b(h, cu_seqlens=cu, position_embeddings=pe)
                    return h
                try:
                    h_out, _ = tm(f"all {len(visual.blocks)} vision blocks", all_blocks, reps=1)
                    tm("merger", lambda: visual.merger(h_out), reps=3)
                except Exception as exc:
                    print(f"    block stack failed: {type(exc).__name__}: {exc}", flush=True)

                print(f"    dtypes: pixel_values {pv.dtype}, "
                      f"patch_embed weight "
                      f"{next(visual.patch_embed.parameters()).dtype}, "
                      f"block0 weight {next(blk0.parameters()).dtype}", flush=True)
                print(f"    attn impl: {visual.config._attn_implementation}", flush=True)
        except Exception as exc:
            print(f"  vision-only pass failed: {type(exc).__name__}: {exc}", flush=True)


    # ---- which SDPA backend is actually usable at these shapes? -----------
    print()
    print("sdpa backends at this sequence length", flush=True)
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        q = torch.randn(8, 16, batch["input_ids"].shape[1], 128,
                        device="cuda", dtype=torch.bfloat16)
        for name, backend in (("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
                              ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
                              ("MATH", SDPBackend.MATH)):
            try:
                with sdpa_kernel(backend):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(20):
                        torch.nn.functional.scaled_dot_product_attention(q, q, q)
                    torch.cuda.synchronize()
                    per = 1000 * (time.perf_counter() - t0) / 20
                    print(f"  {name:22} {per:7.2f} ms/call", flush=True)
            except Exception as exc:
                print(f"  {name:22} unavailable: {type(exc).__name__}", flush=True)
        del q
    except ImportError:
        print("  torch.nn.attention not available", flush=True)

    if args.top == 0:
        print()
        print("--top 0: skipping the profiler table")
        return 0

    print()
    print("\nprofiling one pass", flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        one_pass()
        torch.cuda.synchronize()

    table = prof.key_averages()
    print("\n--- by SELF CPU time (dispatch cost lives here) ---")
    print(table.table(sort_by="self_cpu_time_total", row_limit=args.top))
    print("\n--- by CALL COUNT (a loop shows up here) ---")
    by_count = sorted(table, key=lambda e: -e.count)[:args.top]
    print(f"{'op':52}{'calls':>12}{'self cpu ms':>14}{'cuda ms':>12}")
    for e in by_count:
        print(f"{e.key[:52]:52}{e.count:>12,}"
              f"{e.self_cpu_time_total/1000:>14.1f}{e.self_device_time_total/1000:>12.1f}")
    total_calls = sum(e.count for e in table)
    print(f"\ntotal op dispatches in ONE forward+backward: {total_calls:,}")
    print(f"x4 micro-batches per optimizer step: {4*total_calls:,}")

    if args.out:
        prof.export_chrome_trace(str(args.out))
        print(f"\nwrote chrome trace {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
