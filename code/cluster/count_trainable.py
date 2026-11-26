#!/usr/bin/env python3
"""Count trainable parameters per experiment arm, on the meta device.

Why this exists: arm C trains the adapters in full (a fixed ~109M), while arms
A and B train LoRA adapters whose size depends on the rank. If those numbers are
far apart, a win for one arm may be measuring capacity rather than which pathway
matters. This prints them side by side so the ranks can be chosen deliberately.

LoRA on a Linear(in_features, out_features) at rank r adds r*(in+out) weights
(A: r x in, B: out x r), bias-free.

    python code/cluster/count_trainable.py /workspace/models/Qwen3-VL-4B-Instruct
"""
import re
import sys

import torch
import torch.nn as nn
from transformers import AutoConfig

try:
    from transformers import Qwen3VLForConditionalGeneration as Cls
except ImportError:  # pragma: no cover
    from transformers import AutoModel as Cls

RANKS = (8, 16, 32, 64)


def main() -> int:
    md = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/Qwen3-VL-4B-Instruct"
    cfg = AutoConfig.from_pretrained(md)
    with torch.device("meta"):
        model = Cls(cfg)

    linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    named = list(model.named_parameters())

    def full(prefix: str) -> int:
        dotted = prefix + "."
        return sum(p.numel() for n, p in named if n == prefix or n.startswith(dotted))

    def lora(pattern: str, rank: int):
        hits = [(n, m) for n, m in linears.items() if re.search(pattern, n)]
        total = sum(rank * (m.in_features + m.out_features) for _, m in hits)
        return total, len(hits)

    # --- the target sets, written exactly as trainable_spec will use them -----
    LLM_ATTN = r"language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$"
    LLM_MLP = r"language_model\.layers\.\d+\.mlp\.(gate|up|down)_proj$"
    VIS_TOP4 = r"visual\.blocks\.(20|21|22|23)\.attn\.(qkv|proj)$"
    VIS_TAPS = r"visual\.blocks\.(5|11|17)\.attn\.(qkv|proj)$"
    VIS_ALL = r"visual\.blocks\.\d+\.attn\.(qkv|proj)$"

    adapters = full("model.visual.merger") + full("model.visual.deepstack_merger_list")

    print(f"model: {md}")
    print(f"total parameters: {sum(p.numel() for p in model.parameters())/1e6:,.2f}M")
    print()

    print("=== full-training targets (no LoRA) ===")
    print(f"  adapters: merger + deepstack_merger_list x3   {adapters/1e6:8.2f}M   <- arm C1")
    print()

    print("=== LoRA parameter counts by rank ===")
    ADAPTER_LIN = r"visual\.(merger|deepstack_merger_list\.\d+)\.linear_fc[12]$"
    VIS_MLP = r"visual\.blocks\.\d+\.mlp\.(linear_fc1|linear_fc2|gate_proj|up_proj|down_proj)$"

    sets = [
        ("adapters linear_fc1,2   (C1 as LoRA instead of full)", ADAPTER_LIN),
        ("vision blocks mlp       (all blocks)", VIS_MLP),
        ("LLM attention q,k,v,o   (arms A/B, upstream default set)", LLM_ATTN),
        ("LLM mlp gate,up,down    (NOT in upstream's list)", LLM_MLP),
        ("vision blocks 20-23     qkv+proj", VIS_TOP4),
        ("vision blocks 5,11,17   qkv+proj (deepstack taps)", VIS_TAPS),
        ("vision blocks 0-23      qkv+proj (all)", VIS_ALL),
    ]
    header = "  " + " " * 58 + "".join(f"{('r=' + str(r)):>11}" for r in RANKS)
    print(header)
    for label, pattern in sets:
        counts = [lora(pattern, r)[0] for r in RANKS]
        n_mod = lora(pattern, RANKS[0])[1]
        cells = "".join(f"{c/1e6:10.2f}M" for c in counts)
        print(f"  {label:<52} ({n_mod:>3}){cells}")
    print()

    print("=== per-arm trainable totals ===")
    for r in RANKS:
        a = lora(LLM_ATTN, r)[0]
        c2 = adapters + lora(VIS_TOP4, r)[0]
        d = a + c2
        print(f"  rank {r:<3}  A/B(언어 LoRA) {a/1e6:7.2f}M   "
              f"C1(어댑터) {adapters/1e6:7.2f}M   "
              f"C2(+비전LoRA) {c2/1e6:7.2f}M   D {d/1e6:7.2f}M")
    print()

    # What rank would make the language LoRA match the adapter budget?
    per_rank = lora(LLM_ATTN, 1)[0]
    if per_rank:
        print(f"=== capacity matching ===")
        print(f"  LLM attention LoRA costs {per_rank/1e6:.3f}M per unit of rank.")
        print(f"  Matching the {adapters/1e6:.2f}M adapter budget needs rank "
              f"~{round(adapters/per_rank)} (impractically large).")
        print("  => arms are NOT capacity-matched at any sane rank; report the")
        print("     trainable-parameter count per arm alongside the scores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
