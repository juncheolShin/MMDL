"""Merge a trained LoRA adapter into a standalone Qwen3-VL checkpoint for vLLM."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    import torch

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map='cpu',
        attn_implementation='sdpa')
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size='5GB')
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(args.output)
    print(f'merged model saved to {args.output}')


if __name__ == '__main__':
    main()
