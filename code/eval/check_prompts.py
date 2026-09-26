"""Render the MMMU prompts exactly as infer_mmmu.py builds them and count their tokens, without loading
the model (processor only, CPU). Use it to check the prompt function p and that
max_model_len >= longest prompt + max_new_tokens.

    CUDA_VISIBLE_DEVICES= python code/eval/check_prompts.py --model models/Qwen3-VL-4B-Instruct --show validation_Art_8
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

from tqdm import tqdm

import mmmu_data
from prompting import build_mmmu_prompt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration import DEFAULT_CONFIG, EVALUATION_TYPES, load_section  # noqa: E402


def collapse_image_pads(prompt):
    return re.sub(r'(?:<\|image_pad\|>)+', lambda m: f'<|image_pad|>×{m.group(0).count("<|image_pad|>")}', prompt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    ap.add_argument('--data-root', default=str(mmmu_data.DATA_DIR))
    ap.add_argument('--split', default='validation')
    ap.add_argument('--show', nargs='*', default=['validation_Art_8'], help='ids whose rendered prompt is printed')
    ap.add_argument('--out', default=str(mmmu_data.RESULTS_DIR / 'prompt_check' / 'prompt_tokens.csv'))
    args = ap.parse_args()
    for key, value in load_section(args.config, 'evaluation', EVALUATION_TYPES).items():
        setattr(args, key, value)
    mmmu_data.configure_data_root(args.data_root)

    from run_mmmu import prepare_inputs_for_vllm
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    data = mmmu_data.load_mmmu(args.split)
    rows = []
    for _, line in tqdm(data.iterrows(), total=len(data), desc='rendering'):
        messages = build_mmmu_prompt(
            line, mmmu_data.dump_image, args.min_pixels, args.max_pixels)
        inp = prepare_inputs_for_vllm(messages, processor)
        images = inp['multi_modal_data'].get('image')
        enc = processor(text=[inp['prompt']], images=images, return_tensors='pt')
        n_tokens = int(enc['input_ids'].shape[1])
        n_image_tokens = int((enc['input_ids'] == processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')).sum())
        sizes = [f'{im.width}x{im.height}' for im in images or []]
        rows.append({'id': line['id'], 'n_images': len(sizes), 'image_sizes': ' '.join(sizes),
                     'n_image_tokens': n_image_tokens, 'n_prompt_tokens': n_tokens})
        if line['id'] in args.show:
            print(f'\n===== {line["id"]} (images {sizes}, prompt tokens {n_tokens}) =====')
            print(collapse_image_pads(processor.decode(enc['input_ids'][0])))
            print('=' * 60)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tokens = [r['n_prompt_tokens'] for r in rows]
    print(f'\n{len(rows)} prompts: min {min(tokens)}, median {statistics.median(tokens):.0f}, '
          f'mean {statistics.mean(tokens):.0f}, max {max(tokens)}')
    for k in sorted({r['n_images'] for r in rows}):
        t = [r['n_prompt_tokens'] for r in rows if r['n_images'] == k]
        print(f'  {k} image(s): n={len(t):3d}  median {statistics.median(t):.0f}  max {max(t)}')
    longest = sorted(rows, key=lambda r: -r['n_prompt_tokens'])[:5]
    print('longest:', ', '.join(f"{r['id']} ({r['n_prompt_tokens']}, {r['n_images']} img)" for r in longest))
    budget = args.max_model_len - args.max_new_tokens
    over = [r['id'] for r in rows if r['n_prompt_tokens'] > budget]
    print(f'prompts longer than max_model_len - max_new_tokens = {budget}: {len(over)} {over[:10]}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
