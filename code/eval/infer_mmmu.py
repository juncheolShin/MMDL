"""Run Qwen3-VL on MMMU with vLLM and the team's agreed MC/open prompts.

Writes results/<run-name>/run_config.json and predictions.jsonl (git-ignored: it embeds the MMMU questions).
Scoring is done by score_mmmu.py.

    python code/eval/infer_mmmu.py --model models/Qwen3-VL-4B-Instruct --run-name base_greedy
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
from prompting import build_mmmu_prompt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration import DEFAULT_CONFIG, EVALUATION_TYPES, file_sha256, load_section  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='local model dir (base or merged fine-tuned checkpoint)')
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    ap.add_argument('--split', default='validation', choices=['validation', 'dev', 'all'])
    ap.add_argument('--ids', nargs='*', help='only these MMMU ids, e.g. validation_Art_8')
    ap.add_argument('--limit', type=int, help='only the first N rows (smoke test)')
    ap.add_argument('--data-root', default=str(mmmu_data.DATA_DIR))
    ap.add_argument('--out-root', default=str(mmmu_data.RESULTS_DIR))
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    for key, value in load_section(args.config, 'evaluation', EVALUATION_TYPES).items():
        setattr(args, key, value)
    return args


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
    if not 0 < args.min_pixels <= args.max_pixels:
        sys.exit('expected 0 < min-pixels <= max-pixels')
    if args.max_new_tokens >= args.max_model_len:
        sys.exit('max-new-tokens must be smaller than max-model-len')
    if not 0 <= args.temperature or not 0 < args.top_p <= 1:
        sys.exit('invalid temperature or top_p in evaluation config')
    mmmu_data.configure_data_root(args.data_root)
    out_dir = Path(args.out_root) / args.run_name
    pred_path = out_dir / 'predictions.jsonl'
    if pred_path.exists() and not args.overwrite:
        sys.exit(f'{pred_path} exists; pass --overwrite or pick another --run-name')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imports vllm/transformers/qwen_vl_utils and sets VLLM_WORKER_MULTIPROC_METHOD=spawn.
    from run_mmmu import prepare_inputs_for_vllm
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
        messages = build_mmmu_prompt(
            line, mmmu_data.dump_image, args.min_pixels, args.max_pixels)
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

    sampling = dict(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
    )
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, stop_token_ids=[], **sampling)
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
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
        'argv': [mmmu_data.repo_relative(a) for a in sys.argv],
        'args': {k: mmmu_data.repo_relative(v) if isinstance(v, (str, Path)) else v
                 for k, v in vars(args).items()},
        'decoding_preset': 'greedy' if args.temperature == 0 else 'sampling',
        'config_sha256': file_sha256(args.config),
        'prompt_style': 'notion_2026_09_25_mc_letter_open_final_answer',
        'sampling_params': {**sampling, 'max_tokens': args.max_new_tokens},
        'llm_kwargs': {**llm_kwargs, 'model': mmmu_data.repo_relative(args.model)},
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
