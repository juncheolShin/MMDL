#!/usr/bin/env python3
"""Score the base model on the in-domain candidate pool. Phase 0 of the plan.

Three things depend on this and nothing else produces them:

  1. Which candidate questions the base model gets wrong. The evidence-annotation
     step deliberately over-weights wrong answers (natural ~30% -> ~40%), and that
     cannot be done without knowing which ones they are.
  2. An in-domain baseline. DECISIONS.md section 13 is explicit that without it, a
     flat MMMU score cannot be told apart from a transfer failure - the experiment
     would not be interpretable either way.
  3. The Phase 0 gate: extraction-failure rate, greedy determinism, and whether
     multiple choice beats chance.

Everything that could drift from the evaluation is imported rather than restated:
the prompt templates come from code/data/render_targets.py (which refuses to run
if code/eval/prompting.py has changed), the decoding settings come from the same
[evaluation] TOML, and the answers are read with code/eval/answer_extraction.py.
vLLM is used because the evaluation uses it (D45) - a correct/incorrect split
produced by another engine would not line up with the scores it feeds.

    python code/eval/infer_indomain.py --model /workspace/models/Qwen3-VL-4B-Instruct \\
        --candidates /workspace/MMDL/training_data/cluster_upload/candidates \\
        --image-root /workspace/MMDL/training_data/cluster_upload \\
        --run-name phase0-base

    # a cheap determinism check: run a slice twice and diff
    python code/eval/infer_indomain.py ... --limit 100 --run-name phase0-det-1
"""
import argparse
import collections
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'code'))
sys.path.insert(0, str(REPO_ROOT / 'code' / 'eval'))
sys.path.insert(0, str(REPO_ROOT / 'code' / 'data'))

from configuration import (DEFAULT_EVALUATION_CONFIG, EVALUATION_TYPES,  # noqa: E402
                           file_sha256, load_section)
import answer_extraction as ae  # noqa: E402
from render_targets import (LETTERS, MC_TEMPLATE, OPEN_TEMPLATE,  # noqa: E402
                            assert_matches_eval_prompt)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--candidates', type=Path, required=True,
                    help='directory of normalized candidate *.jsonl')
    ap.add_argument('--image-root', type=Path, required=True,
                    help='directory the records\' image paths are relative to')
    ap.add_argument('--config', type=Path, default=DEFAULT_EVALUATION_CONFIG)
    ap.add_argument('--out-root', type=Path, default=REPO_ROOT / 'results')
    ap.add_argument('--limit', type=int, default=0, help='first N records only (for probes)')
    # Building every prompt first means holding every decoded image at once.
    # At 20,222 records that exceeded the pod's 32Gi limit and the job was
    # OOMKilled, so prompts are built, generated and written one chunk at a time.
    ap.add_argument('--chunk', type=int, default=512,
                    help='records to build and generate at a time')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    for key, value in load_section(args.config, 'evaluation', EVALUATION_TYPES).items():
        setattr(args, key, value)
    return args


def load_records(directory: Path, limit: int):
    files = sorted(directory.glob('*.jsonl'))
    if not files:
        sys.exit(f'infer_indomain: no *.jsonl in {directory}')
    records = []
    for path in files:
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    return records, files
    return records, files


def build_messages(record, image_path, min_pixels, max_pixels):
    """The evaluation's prompt, with this record's fields."""
    question = str(record['question']).strip()
    if record['answer_type'] == 'multiple_choice':
        options = record['options'] or []
        if len(options) > len(LETTERS):
            raise ValueError(f"{record['id']}: {len(options)} options exceeds A-I")
        lines = '\n'.join(f'{LETTERS[i]}. {opt}' for i, opt in enumerate(options))
        text = MC_TEMPLATE.format(question=question, options=lines)
    else:
        text = OPEN_TEMPLATE.format(question=question)
    content = [{'type': 'image', 'image': str(image_path),
                'min_pixels': min_pixels, 'max_pixels': max_pixels},
               {'type': 'text', 'text': text}]
    return [{'role': 'user', 'content': content}], text


def grade(record, response, truncated):
    """Score with the evaluation's own extractor. Returns (correct, pred, rule)."""
    if record['answer_type'] == 'multiple_choice':
        options = {LETTERS[i]: opt for i, opt in enumerate(record['options'] or [])}
        gold = LETTERS[record['answer_index']]
        extraction = ae.extract_choice(response, options, truncated=truncated)
        pred = getattr(extraction, 'pred', None)
        rule = getattr(extraction, 'method', None)
        return (pred == gold), pred, rule, gold

    gold_raw = ae.parse_ground_truth(record['answer'])
    answer_type = ae.answer_type_of(gold_raw)
    extraction = ae.extract_open(response, answer_type, truncated=truncated)
    correct = ae.open_is_correct(extraction, gold_raw, answer_type)
    pred = getattr(extraction, 'pred', None)
    if isinstance(pred, tuple):
        pred = pred[0]
    return correct, pred, getattr(extraction, 'method', None), record['answer']


def main():
    pipeline_start = time.time()
    args = parse_args()
    assert_matches_eval_prompt()   # refuses if the evaluation prompt has drifted

    if args.max_new_tokens >= args.max_model_len:
        sys.exit('max-new-tokens must be smaller than max-model-len')
    out_dir = Path(args.out_root) / args.run_name
    pred_path = out_dir / 'indomain_predictions.jsonl'
    if pred_path.exists() and not args.overwrite:
        sys.exit(f'{pred_path} exists; pass --overwrite or pick another --run-name')
    out_dir.mkdir(parents=True, exist_ok=True)

    records, source_files = load_records(args.candidates, args.limit)
    print(f'{len(records):,} candidate records from {len(source_files)} file(s)', flush=True)

    from third_party.qwen3vl_mmmu.input_utils import prepare_inputs_for_vllm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model)

    # Check every image exists before loading the model, so a bad upload costs
    # seconds instead of a queued GPU job.
    missing = [r['id'] for r in records
               if not (args.image_root / r['image']).is_file()]
    if missing:
        sys.exit(f'infer_indomain: {len(missing)} record(s) point at a missing image '
                 f'under {args.image_root} (first: {missing[0]}); fix the upload '
                 f'before spending GPU time')
    print(f'{len(records):,} images present', flush=True)

    sampling = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    presence_penalty=args.presence_penalty)
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, stop_token_ids=[], **sampling)
    llm_kwargs = dict(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
                      gpu_memory_utilization=args.gpu_memory_utilization,
                      trust_remote_code=True, max_model_len=args.max_model_len,
                      max_num_seqs=args.max_num_seqs,
                      limit_mm_per_prompt={'image': args.max_images_per_prompt},
                      seed=args.seed)
    llm = LLM(**llm_kwargs)

    stats = collections.defaultdict(lambda: collections.Counter())
    tokens_out = []
    gen_seconds = 0.0
    n_chunks = (len(records) + args.chunk - 1) // args.chunk
    with pred_path.open('w', encoding='utf-8') as sink:
      for chunk_no in range(n_chunks):
        batch = records[chunk_no * args.chunk:(chunk_no + 1) * args.chunk]
        rows, inputs = [], []
        for record in batch:
            image = (args.image_root / record['image']).resolve()
            messages, text = build_messages(record, image, args.min_pixels, args.max_pixels)
            inputs.append(prepare_inputs_for_vllm(messages, processor))
            rows.append({'id': record['id'], 'source': record['source'],
                         'answer_type': record['answer_type'], 'prompt_text': text,
                         'record': record})
        t0 = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        gen_seconds += time.time() - t0
        done = sum(s['n'] for s in stats.values()) + len(rows)
        print(f'  chunk {chunk_no + 1}/{n_chunks}: {done:,}/{len(records):,} '
              f'({gen_seconds:.0f}s generating so far)', flush=True)

        for row, out in zip(rows, outputs):
            o = out.outputs[0]
            response = o.text.split('</think>')[-1].strip()
            truncated = o.finish_reason == 'length'
            record = row.pop('record')
            correct, pred, rule, gold = grade(record, response, truncated)

            key = row['source']
            stats[key]['n'] += 1
            stats[key]['correct'] += int(correct)
            stats[key]['truncated'] += int(truncated)
            stats[key]['unparsed'] += int(pred is None)
            tokens_out.append(len(o.token_ids))

            sink.write(json.dumps({
                **row,
                'gold': gold,
                'response': response,
                'predicted': pred if not isinstance(pred, float) else round(pred, 6),
                'extraction_rule': rule,
                # the field Phase 1 stratifies on
                'correct': bool(correct),
                'finish_reason': o.finish_reason,
                'n_prompt_tokens': len(out.prompt_token_ids),
                'n_output_tokens': len(o.token_ids),
            }, ensure_ascii=False) + '\n')

    total = sum(s['n'] for s in stats.values())
    total_correct = sum(s['correct'] for s in stats.values())
    total_unparsed = sum(s['unparsed'] for s in stats.values())
    summary = {
        'run_name': args.run_name,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'model': str(args.model),
        'candidates': str(args.candidates),
        'source_files': [p.name for p in source_files],
        'config_sha256': file_sha256(args.config),
        'decoding_preset': 'greedy' if args.temperature == 0 else 'sampling',
        'sampling_params': {**sampling, 'max_tokens': args.max_new_tokens},
        'n': total,
        'accuracy': round(total_correct / total, 4) if total else None,
        'wrong': total - total_correct,
        'wrong_fraction': round((total - total_correct) / total, 4) if total else None,
        'unparsed': total_unparsed,
        'unparsed_fraction': round(total_unparsed / total, 4) if total else None,
        'mean_output_tokens': round(sum(tokens_out) / len(tokens_out), 2) if tokens_out else None,
        'generation_seconds': round(gen_seconds, 1),
        'pipeline_seconds': round(time.time() - pipeline_start, 1),
        'by_source': {k: dict(v) for k, v in sorted(stats.items())},
    }
    (out_dir / 'indomain_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print()
    print(f"{'source':18} {'n':>7} {'acc':>8} {'wrong':>7} {'unparsed':>9} {'trunc':>7}")
    for source, s in sorted(stats.items()):
        print(f"{source:18} {s['n']:>7,} {100*s['correct']/s['n']:>7.2f}% "
              f"{s['n']-s['correct']:>7,} {s['unparsed']:>9,} {s['truncated']:>7,}")
    print(f"\ntotal {total:,}  accuracy {100*total_correct/total:.2f}%  "
          f"wrong {total-total_correct:,} ({100*(total-total_correct)/total:.1f}%)")
    print(f"unparsed {total_unparsed:,} ({100*total_unparsed/total:.2f}%)  "
          f"mean output {summary['mean_output_tokens']} tokens")
    print(f"generation {gen_seconds:.0f}s")

    # Phase 0 gate, stated rather than left to the reader (DECISIONS.md section 7)
    print('\nPhase 0 gate')
    gate_ok = True
    if summary['unparsed_fraction'] is not None:
        ok = summary['unparsed_fraction'] < 0.05
        gate_ok &= ok
        print(f"  extraction failure < 5%          {'PASS' if ok else 'FAIL'} "
              f"({100*summary['unparsed_fraction']:.2f}%)")
    mc = [s for k, s in stats.items() if s['n']]
    print(f"  wrong answers available for Phase 1: {total-total_correct:,} "
          f"(need ~5,080 for a 40% share of 12,700)")
    print(f"  determinism: re-run with --limit 100 and diff the predictions")
    print(f"\nwrote {pred_path}")
    return 0 if gate_ok else 2


if __name__ == '__main__':
    sys.exit(main())
