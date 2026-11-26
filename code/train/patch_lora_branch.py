"""Replace upstream's hardcoded LoRA branch with a call into mmdl_trainable_spec.

Same approach and same safety rule as patch_upstream.py: match the pinned
commit's source text exactly, and refuse to touch anything else. If Qwen's code
ever moves off commit 96588727e4..., this raises instead of mis-patching.

What it removes (train_qwen.py, lines 163-178 at the pinned commit): the branch
that freezes every parameter and then builds a LoraConfig with
`target_modules=["q_proj","k_proj","v_proj","o_proj"]` and `lora_dropout or 0.05`.
That branch supports neither layer selection, nor the vision path, nor
modules_to_save, so arms A', C1, C2 and D cannot be expressed through it
(DECISIONS.md section 5).

    QWEN_TRAIN_ROOT=/workspace/qwen3-vl/qwen-vl-finetune \
        python code/train/patch_lora_branch.py
    python code/train/patch_lora_branch.py --check     # report, change nothing
"""
import argparse
import os
import sys
from pathlib import Path

UPSTREAM_COMMIT = "96588727e44c78b25ba03ea03b8e12f7e64fd0da"

ORIGINAL = '''    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
'''

REPLACEMENT = '''    if training_args.lora_enable:
        # PATCHED by code/train/patch_lora_branch.py - see DECISIONS.md section 5.
        # Upstream hardcoded the LoRA targets; the experiment arms need per-layer
        # vision targets and modules_to_save, so the whole branch is delegated.
        import mmdl_trainable_spec
        model = mmdl_trainable_spec.apply(model, model_args, training_args)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="report only")
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("QWEN_TRAIN_ROOT",
                                                    "/opt/qwen3-vl/qwen-vl-finetune")))
    args = parser.parse_args()

    target = args.root / "qwenvl" / "train" / "train_qwen.py"
    if not target.is_file():
        sys.exit(f"patch_lora_branch: not found: {target}\n"
                 "Set QWEN_TRAIN_ROOT to the pinned checkout.")

    source = target.read_text(encoding="utf-8")

    if REPLACEMENT.strip() in source:
        print(f"already patched: {target}")
        return 0

    found = source.count(ORIGINAL)
    if found != 1:
        sys.exit(
            f"patch_lora_branch: expected the pinned LoRA branch exactly once, found {found}.\n"
            f"  file   : {target}\n"
            f"  pinned : {UPSTREAM_COMMIT}\n"
            "Qwen's source differs from the reviewed commit. Re-read the branch and\n"
            "update ORIGINAL in this file rather than loosening the match."
        )

    if args.check:
        print(f"would patch {target} (pinned branch found exactly once)")
        return 0

    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
        print(f"backup: {backup}")

    target.write_text(source.replace(ORIGINAL, REPLACEMENT), encoding="utf-8")
    print(f"patched: {target}")
    print("  the LoRA branch now calls mmdl_trainable_spec.apply()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
