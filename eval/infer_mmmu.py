"""Run a Qwen3-VL model on MMMU with vLLM, using the official Qwen3-VL prompt: r_n = f_theta(p(q_n, i_n)).

Writes runs/<run-name>/predictions.jsonl and run_config.json. Scoring is done by score_mmmu.py.

    python eval/infer_mmmu.py --model models/Qwen3-VL-4B-Instruct --run-name base_greedy --decoding greedy
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from tqdm import tqdm

import mmmu_data

# Sampling values are Qwen's infer_instruct.sh. `greedy` changes only the temperature, so the two
# presets differ in exactly one setting.
DECODING_PRESETS = {
    'qwen_official': dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0, presence_penalty=1.5),
    'greedy': dict(temperature=0.0, top_p=0.8, top_k=20, repetition_penalty=1.0, presence_penalty=1.5),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='local model dir (base or merged fine-tuned checkpoint)')
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--decoding', choices=sorted(DECODING_PRESETS), required=True)
    ap.add_argument('--split', default='validation', choices=['validation', 'dev', 'all'])
    ap.add_argument('--ids', nargs='*', help='only these MMMU ids, e.g. validation_Art_8')
    ap.add_argument('--limit', type=int, help='only the first N rows (smoke test)')
    ap.add_argument('--max-new-tokens', type=int, default=32768)
    # Qwen uses 128000, which needs ~18 GiB of KV cache on its own. check_prompts.py measured the
    # longest val prompt at 5627 tokens, so 40960 keeps the full 32768-token output budget.
    ap.add_argument('--max-model-len', type=int, default=40960)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    ap.add_argument('--tensor-parallel-size', type=int, default=1)
    ap.add_argument('--max-images-per-prompt', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-root', default=str(mmmu_data.ROOT / 'runs'))
    ap.add_argument('--overwrite', action='store_true')
    return ap.parse_args()


def to_jsonable(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    return v


def model_fingerprint(model_dir):
    files = sorted(Path(model_dir).glob('*.safetensors')) + sorted(Path(model_dir).glob('*.json'))
    return {p.name: mmmu_data.file_digest(p) for p in files}


def main():
    args = parse_args()
    out_dir = Path(args.out_root) / args.run_name
    pred_path = out_dir / 'predictions.jsonl'
    if pred_path.exists() and not args.overwrite:
        sys.exit(f'{pred_path} exists; pass --overwrite or pick another --run-name')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imports vllm/transformers/qwen_vl_utils and sets VLLM_WORKER_MULTIPROC_METHOD=spawn.
    from run_mmmu import build_mmmu_prompt, prepare_inputs_for_vllm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    data = mmmu_data.load_mmmu(args.split)
    if args.ids:
        missing = set(args.ids) - set(data['id'])
        if missing:
            sys.exit(f'unknown ids: {sorted(missing)}')
        data = data[data['id'].isin(args.ids)]
    if args.limit:
        data = data.head(args.limit)
    print(f'{len(data)} samples ({args.split})')

    processor = AutoProcessor.from_pretrained(args.model)
    rows, inputs = [], []
    for _, line in tqdm(data.iterrows(), total=len(data), desc='building prompts'):
        messages = build_mmmu_prompt(line, mmmu_data.dump_image, 'MMMU_DEV_VAL')
        vllm_input = prepare_inputs_for_vllm(messages, processor)
        inputs.append(vllm_input)
        rows.append({
            'id': line['id'],
            'index': to_jsonable(line['index']),
            'split': line['split'],
            'question_type': line['question_type'],
            'answer': line['answer'],
            'messages': messages,
            'prompt': vllm_input['prompt'],
        })

    sampling = DECODING_PRESETS[args.decoding]
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, stop_token_ids=[], **sampling)
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={'image': args.max_images_per_prompt},
        seed=args.seed,
    )
    llm = LLM(**llm_kwargs)

    t0 = time.time()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    gen_seconds = time.time() - t0

    with open(pred_path, 'w') as f:
        for row, out in zip(rows, outputs):
            o = out.outputs[0]
            row.update({
                'response_raw': o.text,
                # Qwen scores the text after </think>; for Instruct models the two are identical.
                'response': o.text.split('</think>')[-1].strip(),
                'finish_reason': o.finish_reason,
                'n_prompt_tokens': len(out.prompt_token_ids),
                'n_output_tokens': len(o.token_ids),
            })
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    config = {
        'run_name': args.run_name,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'argv': sys.argv,
        'args': vars(args),
        'decoding_preset': args.decoding,
        'sampling_params': {**sampling, 'max_tokens': args.max_new_tokens},
        'llm_kwargs': llm_kwargs,
        'n_samples': len(rows),
        'generation_seconds': round(gen_seconds, 1),
        'model_fingerprint_sha256': model_fingerprint(args.model),
        'data_and_vendored_sha256': mmmu_data.vendored_digests(),
        'restored_option_cells': mmmu_data.RESTORED_OPTION_CELLS,
        'env': {
            'python': platform.python_version(),
            **{pkg: version(pkg) for pkg in ['vllm', 'torch', 'transformers', 'qwen-vl-utils']},
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'host': platform.node(),
        },
    }
    (out_dir / 'run_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f'wrote {pred_path} ({len(rows)} rows, generation {gen_seconds:.0f}s)')


if __name__ == '__main__':
    main()
