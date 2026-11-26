#!/usr/bin/env python3
"""Check every arm config against the real module names, before any GPU time.

mmdl_trainable_spec.apply() raises when its target regex matches nothing, but
that happens after the model has loaded inside a GPU job. This does the same
check on the meta device, so a typo in a config costs seconds instead of a
queued job.

For each configs/arms/*.toml it reports how many modules the arm would train and
how many parameters that is, so the per-arm capacity in DECISIONS.md section 2
can be confirmed rather than trusted.

    python code/train/verify_arms.py /workspace/models/Qwen3-VL-4B-Instruct
"""
import json
import pathlib
import re
import sys

import torch
import torch.nn as nn
from transformers import AutoConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from configuration import TRAINABLE_TYPES, TRAINING_TYPES, load_section  # noqa: E402
from mmdl_trainable_spec import (adapter_module_names, build_target_regex,  # noqa: E402
                                 load_spec)

try:
    from transformers import Qwen3VLForConditionalGeneration as Cls
except ImportError:  # pragma: no cover
    from transformers import AutoModel as Cls

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    md = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/Qwen3-VL-4B-Instruct"
    cfg = AutoConfig.from_pretrained(md)
    with torch.device("meta"):
        model = Cls(cfg)

    names = [n for n, _ in model.named_modules()]
    linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    named = list(model.named_parameters())
    total = sum(p.numel() for p in model.parameters())

    def params_under(prefix: str) -> int:
        dotted = prefix + "."
        return sum(p.numel() for n, p in named if n == prefix or n.startswith(dotted))

    arms_dir = ROOT / "configs" / "arms"
    files = sorted(arms_dir.glob("*.toml"))
    if not files:
        sys.exit(f"no arm configs in {arms_dir}")

    failures = []
    print(f"model: {md}  ({total/1e6:,.2f}M parameters)")
    print(f"{'arm':18} {'rank':>5} {'LoRA mods':>10} {'LoRA par':>11} "
          f"{'adapters':>11} {'TRAINABLE':>12} {'share':>8}")
    print("-" * 92)

    for path in files:
        try:
            training = load_section(path, "training", TRAINING_TYPES)
            spec = load_spec(json.dumps(load_section(path, "trainable", TRAINABLE_TYPES)))
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue

        rank = training["lora_r"]
        regex = build_target_regex(spec)
        lora_mods, lora_par = 0, 0
        if regex:
            hits = [n for n in names if re.fullmatch(regex, n)]
            missing_linear = [n for n in hits if n not in linears]
            if not hits:
                failures.append(f"{path.name}: target regex matched NO module: {regex}")
            if missing_linear:
                failures.append(f"{path.name}: matched non-Linear module(s): {missing_linear[:3]}")
            lora_mods = len(hits)
            lora_par = sum(rank * (linears[n].in_features + linears[n].out_features)
                           for n in hits if n in linears)

        adapter_par = 0
        if spec["train_adapters"]:
            mods = adapter_module_names(model)
            if not mods:
                failures.append(f"{path.name}: train_adapters is true but no adapter module found")
            adapter_par = sum(params_under(m) for m in mods)

        trainable = lora_par + adapter_par
        if trainable == 0:
            failures.append(f"{path.name}: would train 0 parameters")
        print(f"{spec['arm']:18} {rank:>5} {lora_mods:>10} {lora_par/1e6:>10.2f}M "
              f"{adapter_par/1e6:>10.2f}M {trainable/1e6:>11.2f}M "
              f"{100*trainable/total:>7.3f}%")

    print()
    if failures:
        print("FAILURES:")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"all {len(files)} arm config(s) resolve to a non-empty, Linear-only target set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
