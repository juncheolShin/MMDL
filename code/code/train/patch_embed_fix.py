#!/usr/bin/env python3
"""Measure, verify and fix the Qwen3-VL vision patch embedding.

Measured on h200-03-w-7271 (DECISIONS.md section 16), forward only, 2,368 patches:

    patch_embed                 10852.00 ms     <- 99.8% of the vision tower
    fast_pos_embed_interpolate      3.16 ms
    rot_pos_emb                     1.69 ms
    all 24 vision blocks           14.39 ms
    merger                          0.08 ms

One operator is the entire problem. It is an nn.Conv3d whose kernel and stride
both equal the input's spatial extent:

    nn.Conv3d(3, embed_dim, kernel_size=(2,16,16), stride=(2,16,16))
    input (num_patches, 3, 2, 16, 16)  ->  output (num_patches, embed_dim, 1, 1, 1)

Every output is one dot product of the whole input volume with the whole kernel,
so this convolution IS a matmul - (N, 1536) @ (1536, embed_dim) - written as a
convolution. cuDNN has to pick an algorithm for a 3D convolution with 2,368
batch entries and a 1x1x1 output, and it picks a catastrophically bad one. The
arithmetic is 8.4 GFLOP, which the same GPU does in 0.01 ms at the 783 TFLOP/s
this pod measures, so the measured 10,852 ms is about a millionfold overhead,
not slow math.

This is also why vLLM is fast on the very same weights (900 MMMU images in 91 s):
it does not use Conv3d here.

This module does three things:
  * times the convolution and the equivalent matmul side by side
  * proves they agree numerically before anything is swapped
  * provides apply(), which replaces the forward in place

    python code/train/patch_embed_fix.py --model /workspace/models/Qwen3-VL-4B-Instruct
"""
from __future__ import annotations

import argparse
import sys
import time


def matmul_forward(module, hidden_states):
    """nn.Conv3d with kernel == input, rewritten as the matmul it already is."""
    weight = module.proj.weight
    target_dtype = weight.dtype
    # (embed_dim, in_channels, T, P, P) -> (embed_dim, in_channels*T*P*P)
    flat_weight = weight.reshape(weight.shape[0], -1)
    x = hidden_states.to(dtype=target_dtype).reshape(-1, flat_weight.shape[1])
    out = x @ flat_weight.t()
    if module.proj.bias is not None:
        out = out + module.proj.bias
    return out


def apply(model) -> bool:
    """Swap the patch embedding's forward for the matmul. Returns True if done."""
    visual = None
    for path in ("model.visual", "visual", "model.model.visual"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            visual = obj
            break
        except AttributeError:
            continue
    if visual is None or not hasattr(visual, "patch_embed"):
        return False
    pe = visual.patch_embed
    if getattr(pe, "_mmdl_patched", False):
        return True
    pe.forward = lambda hidden_states: matmul_forward(pe, hidden_states)
    pe._mmdl_patched = True
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="/workspace/models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--patches", type=int, default=2368)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    visual = model.model.visual
    pe = visual.patch_embed
    c, t, p = pe.in_channels, pe.temporal_patch_size, pe.patch_size
    print(f"patch_embed: Conv3d({c}, {pe.embed_dim}, kernel=({t},{p},{p}), "
          f"stride=({t},{p},{p})), weight {tuple(pe.proj.weight.shape)}")

    x = torch.randn(args.patches, c * t * p * p, device="cuda", dtype=torch.bfloat16)
    flops = 2 * args.patches * (c * t * p * p) * pe.embed_dim

    def timeit(fn, reps):
        with torch.no_grad():
            out = fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                out = fn()
            torch.cuda.synchronize()
        return out, (time.perf_counter() - t0) / reps

    conv_out, conv_dt = timeit(lambda: pe(x), args.reps)
    mm_out, mm_dt = timeit(lambda: matmul_forward(pe, x), args.reps)

    print(f"\n{'implementation':28}{'time':>12}{'TFLOP/s':>12}")
    print(f"{'nn.Conv3d (current)':28}{1000*conv_dt:>10.2f}ms{flops/conv_dt/1e12:>12.3f}")
    print(f"{'matmul (equivalent)':28}{1000*mm_dt:>10.2f}ms{flops/mm_dt/1e12:>12.1f}")
    print(f"{'speedup':28}{conv_dt/max(mm_dt,1e-12):>11.0f}x")

    # Agreement. bf16 matmul and bf16 conv accumulate differently, so compare
    # with a bf16-sized tolerance, and report the real numbers either way.
    a, b = conv_out.float(), mm_out.reshape(conv_out.shape).float()
    diff = (a - b).abs()
    denom = a.abs().clamp(min=1e-6)
    print(f"\nagreement: max abs diff {diff.max():.3e}, "
          f"max rel diff {(diff/denom).max():.3e}, "
          f"mean abs {diff.mean():.3e}")
    ok = torch.allclose(a, b, rtol=2e-2, atol=2e-2)
    print(f"allclose(rtol=2e-2, atol=2e-2): {ok}")
    if not ok:
        print("REFUSING to recommend the swap - the two do not agree")
        return 1

    # And through the real module, end to end.
    assert apply(model)
    _, patched_dt = timeit(lambda: pe(x), args.reps)
    print(f"\nafter apply(): {1000*patched_dt:.2f} ms "
          f"({conv_dt/max(patched_dt,1e-12):.0f}x faster than the Conv3d)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
