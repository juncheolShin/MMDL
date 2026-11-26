"""Decide what is trainable, for the arms in DECISIONS.md section 2.

Upstream's LoRA branch hardcodes `target_modules=["q_proj","k_proj","v_proj","o_proj"]`
with no layer selection, no vision path and no `modules_to_save`, so only arm A and
arm B (language attention only) are reachable through it. `patch_lora_branch.py`
replaces that branch with a call to `apply()` here.

The spec arrives as JSON in MMDL_TRAINABLE_SPEC_JSON (train_lora.py sets it from
the TOML's [trainable] section) because a patched call site cannot take arguments.

Module names are the measured ones (probe job, 2026-11-19):
    model.language_model.layers.{0..35}.self_attn.{q,k,v,o}_proj
    model.language_model.layers.{0..35}.mlp.{gate,up,down}_proj
    model.visual.blocks.{0..23}.attn.{qkv,proj}
    model.visual.merger.{norm,linear_fc1,linear_fc2}
    model.visual.deepstack_merger_list.{0,1,2}.{norm,linear_fc1,linear_fc2}
"""
import json
import os
import re

SPEC_ENV = "MMDL_TRAINABLE_SPEC_JSON"

# Keys and types, checked strictly like configuration.load_section does.
SPEC_TYPES = {
    "arm": str,
    "llm_lora": bool,
    "llm_lora_targets": list,
    "vision_lora_blocks": list,
    "vision_lora_targets": list,
    "train_adapters": bool,
}

LLM_PREFIX = r".*language_model\.layers\.\d+\."
VIS_PREFIX = r".*visual\.blocks\."
# Suffix-matched by PEFT when given as plain names; we build one regex instead
# because the vision side needs a layer-index filter and PEFT cannot mix the two.
ADAPTER_MODULES = ("merger", "deepstack_merger_list")


def load_spec(raw=None):
    raw = raw if raw is not None else os.environ.get(SPEC_ENV)
    if not raw:
        raise RuntimeError(
            f"{SPEC_ENV} is not set. The patched upstream LoRA branch needs it; "
            "run training through code/train/train_lora.py."
        )
    spec = json.loads(raw)
    extra = set(spec) - set(SPEC_TYPES)
    missing = set(SPEC_TYPES) - set(spec)
    if extra or missing:
        raise ValueError(f"[trainable]: extra={sorted(extra)}, missing={sorted(missing)}")
    for key, expected in SPEC_TYPES.items():
        if type(spec[key]) is not expected:
            raise TypeError(f"[trainable].{key} must be {expected.__name__}")
    if not (spec["llm_lora"] or spec["vision_lora_blocks"] or spec["train_adapters"]):
        raise ValueError("[trainable]: nothing would be trained")
    if spec["llm_lora"] and not spec["llm_lora_targets"]:
        raise ValueError("[trainable]: llm_lora is true but llm_lora_targets is empty")
    if spec["vision_lora_blocks"] and not spec["vision_lora_targets"]:
        raise ValueError("[trainable]: vision_lora_blocks set but vision_lora_targets is empty")
    for block in spec["vision_lora_blocks"]:
        if not isinstance(block, int) or not 0 <= block <= 23:
            raise ValueError(f"[trainable]: vision block {block!r} outside 0..23 (depth is 24)")
    return spec


def build_target_regex(spec):
    """One full-match regex for every LoRA target, or None when there are none."""
    parts = []
    if spec["llm_lora"]:
        names = "|".join(re.escape(t) for t in spec["llm_lora_targets"])
        # the targets span two parents (self_attn.*_proj and mlp.*_proj)
        parts.append(LLM_PREFIX + r"(?:self_attn|mlp)\.(?:" + names + r")")
    if spec["vision_lora_blocks"]:
        blocks = "|".join(str(b) for b in sorted(spec["vision_lora_blocks"]))
        names = "|".join(re.escape(t) for t in spec["vision_lora_targets"])
        parts.append(VIS_PREFIX + r"(?:" + blocks + r")\.attn\.(?:" + names + r")")
    if not parts:
        return None
    return "(?:" + "|".join(parts) + ")"


def adapter_module_names(model):
    """Leaf-ish names PEFT can wrap with modules_to_save.

    deepstack_merger_list is a ModuleList, so its entries are listed one by one;
    handing PEFT the list itself is not a documented target.
    """
    found = []
    for name, _ in model.named_modules():
        if re.fullmatch(r".*\.visual\.merger", name):
            found.append(name)
        elif re.fullmatch(r".*\.visual\.deepstack_merger_list\.\d+", name):
            found.append(name)
    return found


def apply(model, model_args, training_args, spec_json=None):
    """Replace upstream's LoRA branch. Returns the model to hand to Trainer."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    spec = load_spec(spec_json)
    print(f"[trainable_spec] arm={spec['arm']}", flush=True)

    # 1. everything off, then put back exactly what the spec asks for
    for param in model.parameters():
        param.requires_grad = False

    target_regex = build_target_regex(spec)
    save_modules = adapter_module_names(model) if spec["train_adapters"] else []

    if target_regex:
        matched = [n for n, m in model.named_modules()
                   if re.fullmatch(target_regex, n) and hasattr(m, "weight")]
        if not matched:
            raise RuntimeError(
                f"[trainable_spec] target regex matched no module: {target_regex}\n"
                "Module names changed, or the spec is wrong. Refusing to train nothing."
            )
        print(f"[trainable_spec] LoRA targets: {len(matched)} modules", flush=True)
        for name in matched[:6]:
            print(f"    {name}", flush=True)
        if len(matched) > 6:
            print(f"    ... and {len(matched) - 6} more", flush=True)

        # upstream does `lora_dropout or 0.05`, which silently turns 0.0 into 0.05.
        # Read the value directly so a configured 0.0 means 0.0.
        dropout = float(getattr(training_args, "lora_dropout", 0.0) or 0.0)
        config = LoraConfig(
            r=int(training_args.lora_r),
            lora_alpha=int(training_args.lora_alpha),
            lora_dropout=dropout,
            target_modules=target_regex,
            modules_to_save=save_modules or None,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, config)
    elif save_modules:
        # arm C1: no LoRA at all, only the adapters train. PEFT is not involved,
        # so Trainer saves a plain checkpoint and merge_lora.py is not needed.
        for name, param in model.named_parameters():
            if any(name.startswith(m + ".") for m in save_modules):
                param.requires_grad = True
        print(f"[trainable_spec] full-training {len(save_modules)} adapter module(s), no LoRA",
              flush=True)

    # 2. With the LLM frozen, gradients still have to reach the vision side
    #    through it. Without this the graph is detached at the embeddings and the
    #    run finishes with zero gradient - silently. (DECISIONS.md section 5-1.)
    base = getattr(model, "get_base_model", None)
    base = base() if callable(base) else model
    for candidate in (base, getattr(base, "model", None)):
        if candidate is not None and hasattr(candidate, "enable_input_require_grads"):
            candidate.enable_input_require_grads()
            print("[trainable_spec] enable_input_require_grads() applied", flush=True)
            break

    # 3. Prove what is actually trainable, and fail loudly if it is nothing.
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    tn = sum(n for _, n in trainable)
    if tn == 0:
        raise RuntimeError("[trainable_spec] nothing is trainable after setup")
    print(f"[trainable_spec] trainable {tn/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100*tn/total:.3f}%)", flush=True)

    report = {
        "arm": spec["arm"],
        "spec": spec,
        "target_regex": target_regex,
        "modules_to_save": save_modules,
        "lora_r": int(training_args.lora_r),
        "lora_alpha": int(training_args.lora_alpha),
        "lora_dropout": float(getattr(training_args, "lora_dropout", 0.0) or 0.0),
        "seed": int(getattr(training_args, "seed", -1)),
        "trainable_parameters": tn,
        "total_parameters": total,
        "trainable_names": [n for n, _ in trainable],
    }
    out = getattr(training_args, "output_dir", None)
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "trainable_params.json"), "w", encoding="utf-8") as sink:
            json.dump(report, sink, indent=2)
        print(f"[trainable_spec] wrote {out}/trainable_params.json", flush=True)

    # The vision patch embedding is an nn.Conv3d whose kernel and stride equal
    # its input's spatial extent, so it is a matmul written as a convolution -
    # and cuDNN picks a catastrophic algorithm for it. Measured on this node:
    # 10,145 ms versus 0.17 ms for the equivalent matmul, a 60,762x difference,
    # with output that is BIT-IDENTICAL (max abs diff 0.0). It was 99.8% of the
    # vision tower and essentially all of the 174.5 s training step.
    # DECISIONS.md section 16.5. This swaps the kernel, never the weights, so
    # the model computes the same numbers.
    try:
        import patch_embed_fix
        if patch_embed_fix.apply(model):
            print("[trainable_spec] vision patch_embed: Conv3d -> matmul "
                  "(bit-identical, ~60,000x faster; DECISIONS.md 16.5)", flush=True)
        else:
            print("[trainable_spec] WARNING: could not locate the vision "
                  "patch_embed; training will run at the slow Conv3d speed",
                  flush=True)
    except ImportError as exc:
        print(f"[trainable_spec] WARNING: patch_embed_fix not importable ({exc}); "
              f"training will run at the slow Conv3d speed", flush=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    elif hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return model
