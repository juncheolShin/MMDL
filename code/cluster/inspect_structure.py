#!/usr/bin/env python3
"""Report Qwen3-VL's parameter layout: encoder blocks, adapters, LLM.

Runs on the meta device, so it reads the config and builds the module tree
without loading weights. No GPU and no real memory needed - it runs fine on the
notebook pod.

    python code/cluster/inspect_structure.py /workspace/models/Qwen3-VL-4B-Instruct
"""
import re
import sys

import torch
from transformers import AutoConfig

try:
    from transformers import Qwen3VLForConditionalGeneration as Cls
except ImportError:  # pragma: no cover - older transformers
    from transformers import AutoModel as Cls


def main() -> int:
    md = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/Qwen3-VL-4B-Instruct"
    cfg = AutoConfig.from_pretrained(md)
    with torch.device("meta"):
        model = Cls(cfg) if hasattr(Cls, "_from_config") else Cls.from_config(cfg)

    named = list(model.named_parameters())

    def params(prefix: str) -> int:
        dotted = prefix + "."
        return sum(p.numel() for n, p in named if n == prefix or n.startswith(dotted))

    vcfg = getattr(cfg, "vision_config", None)
    taps = list(getattr(vcfg, "deepstack_visual_indexes", []) or [])
    depth = getattr(vcfg, "depth", None)

    print(f"vision depth = {depth} blocks (0..{depth - 1})" if depth else "vision depth unknown")
    print(f"deepstack taps = {taps}")
    print()

    print("=== adapters (the connector) ===")
    rows = []
    for name, module in model.named_modules():
        if re.fullmatch(r".*\.visual\.(merger|deepstack_merger_list\.\d+)", name):
            subs = ", ".join(c for c, _ in module.named_children())
            rows.append((name, params(name), subs))
    for name, n, subs in rows:
        print(f"  {name:48} {n/1e6:7.2f}M  [{subs}]")
    adapter_total = sum(n for _, n, _ in rows)
    print(f"  {'TOTAL adapters':48} {adapter_total/1e6:7.2f}M")
    print()

    print("=== which encoder block feeds which adapter ===")
    for i, tap in enumerate(taps):
        print(f"  visual.blocks.{tap:<2} -> deepstack_merger_list.{i} -> added to LLM layer {i}")
    if depth:
        print(f"  visual.blocks.{depth - 1:<2} -> merger                 -> image input tokens")
    print()

    print("=== one encoder block, for scale ===")
    probe = taps[0] if taps else 0
    base = None
    for cand in (f"model.visual.blocks.{probe}", f"visual.blocks.{probe}"):
        if params(cand):
            base = cand
            break
    if base:
        qkv = params(base + ".attn.qkv")
        proj = params(base + ".attn.proj")
        print(f"  {base:48} {params(base)/1e6:7.2f}M")
        print(f"  {base + '.attn.qkv + .attn.proj':48} {(qkv+proj)/1e6:7.2f}M  <- LoRA target")
    print()

    print("=== totals ===")
    for label, prefix in (("vision tower", "visual"), ("language model", "language_model")):
        for cand in (f"model.{prefix}", prefix):
            n = params(cand)
            if n:
                print(f"  {label:20} {cand:30} {n/1e6:9.2f}M")
                break
    print(f"  {'whole model':20} {'':30} {sum(p.numel() for p in model.parameters())/1e6:9.2f}M")
    return 0


if __name__ == "__main__":
    sys.exit(main())
