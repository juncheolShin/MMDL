#!/usr/bin/env python3
"""Measure per-question latency and peak memory. The only way to check the cost constraints.

The project's targets are "mean output <= 45 tokens" and "latency <= 1.1x the
baseline" (DECISIONS.md section 2). The ordinary evaluation cannot measure the
second one: it runs vLLM with max_num_seqs=2, so its 343.7s is the wall clock for
a whole batched sweep, not the time one question takes. Dividing it by 900 would
report a throughput figure as if it were a latency figure.

So this runs a fixed subset with max_num_seqs=1, one question at a time, repeats
it, and reports the median - plus prefill/decode token counts and GPU memory
sampled during the run.

Numbers are only comparable between runs of this script on the same hardware.
The baseline figure in assignment1.md was measured on an RTX 3090; the 1.1x
constraint has to be judged against an H200 baseline produced here.

    python code/eval/measure_cost.py --model <dir> --run-name cost-base \\
        --data-path /workspace/data --limit 100 --repeats 3
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'code'))
sys.path.insert(0, str(REPO_ROOT / 'code' / 'eval'))

import mmmu_data  # noqa: E402
from configuration import (DEFAULT_EVALUATION_CONFIG, EVALUATION_TYPES,  # noqa: E402
                           file_sha256, load_section)
from prompting import build_mmmu_prompt  # noqa: E402


class GpuSampler(threading.Thread):
    """Poll nvidia-smi once a second for peak memory.

    torch.cuda.max_memory_allocated() would only see this process's allocator,
    and vLLM reserves a KV cache pool outside it, so the number that matters has
    to come from the driver.
    """

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        # Deliberately not named _stop: threading.Thread already uses that name
        # for an internal method, and join() calls it. Shadowing it with an Event
        # makes join() raise TypeError after the measurement has already
        # succeeded - which is exactly what cost run 1119-0704 did.
        self._done = threading.Event()

    def run(self):
        while not self._done.is_set():
            try:
                out = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5)
                for line in out.stdout.strip().splitlines():
                    if line.strip().isdigit():
                        self.samples.append(int(line.strip()))
            except (OSError, subprocess.SubprocessError):
                pass
            self._done.wait(self.interval)

    def stop(self):
        self._done.set()

    @property
    def peak_mib(self):
        return max(self.samples) if self.samples else None


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--config', type=Path, default=DEFAULT_EVALUATION_CONFIG)
    ap.add_argument('--data-path', type=Path, default=mmmu_data.DATA_DIR)
    ap.add_argument('--out-root', type=Path, default=REPO_ROOT / 'results')
    ap.add_argument('--limit', type=int, default=100, help='questions in the fixed subset')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    for key, value in load_section(args.config, 'evaluation', EVALUATION_TYPES).items():
        setattr(args, key, value)
    return args


def main():
    args = parse_args()
    if args.repeats < 1 or args.limit < 1:
        sys.exit('--repeats and --limit must be >= 1')
    out_dir = Path(args.out_root) / args.run_name
    report_path = out_dir / 'cost_report.json'
    if report_path.exists() and not args.overwrite:
        sys.exit(f'{report_path} exists; pass --overwrite or pick another --run-name')
    out_dir.mkdir(parents=True, exist_ok=True)

    data = mmmu_data.load_mmmu(args.data_path)
    # A fixed, deterministic subset: the first `limit` rows in the loader's order.
    # Sampling at random would make runs incomparable, which defeats the purpose.
    subset = data.head(args.limit)
    print(f'subset: {len(subset)} of {len(data)} MMMU validation questions', flush=True)

    from third_party.qwen3vl_mmmu.input_utils import prepare_inputs_for_vllm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model)
    prompts, meta = [], []
    for _, line in subset.iterrows():
        messages = build_mmmu_prompt(
            line, mmmu_data.dump_image(line, args.data_path), args.min_pixels, args.max_pixels)
        prompts.append(prepare_inputs_for_vllm(messages, processor))
        meta.append({'id': line['id'], 'question_type': line['question_type']})

    sampling = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    presence_penalty=args.presence_penalty)
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, stop_token_ids=[], **sampling)

    sampler = GpuSampler()
    sampler.start()
    load_start = time.time()
    llm = LLM(model=args.model,
              tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              trust_remote_code=True,
              max_model_len=args.max_model_len,
              # the whole point: one sequence at a time, so the clock measures
              # one question instead of a batch
              max_num_seqs=1,
              limit_mm_per_prompt={'image': args.max_images_per_prompt},
              seed=args.seed)
    load_seconds = time.time() - load_start
    print(f'model loaded in {load_seconds:.1f}s', flush=True)

    # One untimed pass so cache warmup does not land in the first measurement.
    llm.generate(prompts[:1], sampling_params=sampling_params)

    per_question = {m['id']: [] for m in meta}
    rounds = []
    for r in range(args.repeats):
        round_start = time.time()
        for prompt, m in zip(prompts, meta):
            t0 = time.perf_counter()
            out = llm.generate([prompt], sampling_params=sampling_params)[0]
            elapsed = time.perf_counter() - t0
            o = out.outputs[0]
            per_question[m['id']].append({
                'seconds': elapsed,
                'n_prompt_tokens': len(out.prompt_token_ids),
                'n_output_tokens': len(o.token_ids),
                'finish_reason': o.finish_reason,
            })
        rounds.append(round(time.time() - round_start, 2))
        print(f'  round {r + 1}/{args.repeats}: {rounds[-1]}s', flush=True)

    sampler.stop()
    sampler.join(timeout=3)

    # Median over repeats per question, then aggregate across questions, so one
    # slow question cannot dominate and neither can one slow round.
    medians, out_tokens, prompt_tokens, truncated = [], [], [], 0
    rows = []
    for m in meta:
        runs = per_question[m['id']]
        med = statistics.median(x['seconds'] for x in runs)
        medians.append(med)
        out_tokens.append(statistics.median(x['n_output_tokens'] for x in runs))
        prompt_tokens.append(runs[0]['n_prompt_tokens'])
        truncated += sum(1 for x in runs if x['finish_reason'] == 'length')
        rows.append({**m, 'median_seconds': round(med, 4),
                     'seconds_per_run': [round(x['seconds'], 4) for x in runs],
                     'n_prompt_tokens': runs[0]['n_prompt_tokens'],
                     'n_output_tokens': [x['n_output_tokens'] for x in runs]})

    def pct(values, q):
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
        return round(ordered[idx], 4)

    report = {
        'run_name': args.run_name,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'model': str(args.model),
        'config_sha256': file_sha256(args.config),
        'note': 'max_num_seqs forced to 1; comparable only with other runs of '
                'this script on the same hardware',
        'subset_size': len(meta),
        'repeats': args.repeats,
        'decoding_preset': 'greedy' if args.temperature == 0 else 'sampling',
        'sampling_params': {**sampling, 'max_tokens': args.max_new_tokens},
        'model_load_seconds': round(load_seconds, 1),
        'round_seconds': rounds,
        'latency_seconds': {
            'median': round(statistics.median(medians), 4),
            'mean': round(statistics.fmean(medians), 4),
            'p90': pct(medians, 0.90),
            'p99': pct(medians, 0.99),
            'min': round(min(medians), 4),
            'max': round(max(medians), 4),
        },
        'output_tokens': {
            'mean': round(statistics.fmean(out_tokens), 2),
            'median': round(statistics.median(out_tokens), 1),
            'max': max(out_tokens),
        },
        'prompt_tokens': {
            'mean': round(statistics.fmean(prompt_tokens), 2),
            'max': max(prompt_tokens),
        },
        'truncated_responses': truncated,
        'truncated_fraction': round(truncated / (len(meta) * args.repeats), 4),
        'gpu_peak_memory_mib': sampler.peak_mib,
        'gpu_samples': len(sampler.samples),
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'cost_per_question.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')

    lat = report['latency_seconds']
    print(f"\nmedian latency   {lat['median']:.3f}s   p90 {lat['p90']:.3f}s   p99 {lat['p99']:.3f}s")
    print(f"output tokens    mean {report['output_tokens']['mean']} "
          f"(constraint: <= 45)  max {report['output_tokens']['max']}")
    print(f"truncated        {truncated} of {len(meta) * args.repeats} "
          f"({100 * report['truncated_fraction']:.1f}%)")
    print(f"peak GPU memory  {report['gpu_peak_memory_mib']} MiB "
          f"({sampler.samples and 'sampled ' + str(len(sampler.samples)) + 'x' or 'no samples'})")
    print(f"model load       {report['model_load_seconds']}s (excluded from latency)")
    print(f"\nwrote {report_path}")
    print('\nTo check the 1.1x constraint, run this once on the base model and once\n'
          'on the merged checkpoint, then compare the median figures.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
