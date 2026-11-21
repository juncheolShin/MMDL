"""Score external train/dev once with the frozen ICoT at the student's answer position."""
import argparse
import json
import math
import os
import sys
import string
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import RECIPE, check_data, file_sha256, load_settings, main_rank, read_json, require, verified_source, write_json
from training.data import encode_example, verify_prepared_images
from training.model import PackedPatchEmbedding

PROTOCOL = 'training template and resize; first answer-token softmax over available options; T=1'
# Preserve the exact BF16 padded batches used by the reported four-GPU run.
SCORING_SHARDS = 4


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def letter_ids(tokenizer):
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in string.ascii_uppercase[:5]]
    require(all(len(v) == 1 for v in ids) and len({v[0] for v in ids}) == len(ids),
            'A through E must each be a distinct single answer token')
    return [v[0] for v in ids]


def prompt_prefix(encoded, gold_token):
    ids, labels = encoded['input_ids'][0], encoded['labels'][0]
    start = int((labels != -100).nonzero(as_tuple=True)[0][0])
    require(start > 0 and int(labels[start]) == int(ids[start]) == gold_token, 'wrong answer-token position')
    # encode_example checks this prefix against the generation template. No assistant tokens reach the model.
    return ids[:start]


class PromptDataset(Dataset):
    def __init__(self, data, items, processor, settings, tokens):
        self.items = items
        self.data, self.processor, self.settings, self.tokens = data, processor, settings, tokens

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row = self.items[i]
        gold = row['answer']
        opts = self.tokens[:len(row['choices'])]
        require(2 <= len(row['choices']) <= len(self.tokens) and gold in string.ascii_uppercase[:len(opts)]
                and len(gold) == 1 and row['conversations'][1]['value'] == gold
                and len(row['conversations']) == 2, f"invalid closed-choice row: {row['id']}")
        example = encode_example(row, self.processor, self.data, self.settings['min_pixels'],
                                 self.settings['max_pixels'], self.settings['model_max_length'])
        prefix = prompt_prefix(example, opts[string.ascii_uppercase.index(gold)])
        return (prefix, example['pixel_values'], example['image_grid_thw'],
                {'id': row['id'], 'gold': gold, 'source': row['source'], 'option_token_ids': opts})


class PromptBatch:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, samples):
        ids, pixels, grids, metadata = zip(*samples, strict=True)
        raw = {'input_ids': pad_sequence(ids, batch_first=True, padding_value=self.pad, padding_side='left'),
               'attention_mask': pad_sequence([torch.ones_like(seq) for seq in ids], batch_first=True,
                                              padding_side='left'),
               'pixel_values': torch.cat(pixels), 'image_grid_thw': torch.cat(grids)}
        return raw, metadata


def scored(row, logits):
    count = len(row['option_token_ids'])
    probs = logits[:count].float().softmax(dim=0).tolist()
    require(all(math.isfinite(p) for p in probs), f"nonfinite teacher scores: {row['id']}")
    best, second = sorted(range(count), key=probs.__getitem__, reverse=True)[:2]
    answer = string.ascii_uppercase[best]
    return {**row, 'probabilities': probs, 'prediction': answer, 'correct': row['gold'] == answer,
            'margin': probs[best] - probs[second]}


def verify_row(score, row, tokens):
    probs = score['probabilities']
    n = len(row['choices'])
    require((score['id'], score['gold'], score['source']) == (row['id'], row['answer'], row['source'])
            and score['option_token_ids'] == tokens[:n] and 2 <= n <= len(tokens)
            and len(probs) == n and all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
            and abs(sum(probs) - 1) < 1e-5, f"invalid teacher distribution/metadata: {row['id']}")
    best, second = sorted(range(n), key=probs.__getitem__, reverse=True)[:2]
    answer = string.ascii_uppercase[best]
    require(score['prediction'] == answer and score['correct'] is (answer == row['answer'])
            and math.isclose(score['margin'], probs[best] - probs[second], abs_tol=1e-8),
            f"invalid teacher prediction: {row['id']}")


@torch.inference_mode()
def predict(model, loader, tokens, device, rank):
    indices = torch.tensor(tokens, device=device)
    predictions = []
    for step, (batch, metadata) in enumerate(loader, 1):
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
        choices = logits.index_select(1, indices).float().cpu()
        predictions.extend(scored(row, values) for row, values in zip(metadata, choices, strict=True))
        if step == 1 or step % 100 == 0:
            print(f'Rank {rank}: {len(predictions)} examples scored', flush=True)
    return predictions


def provenance(args):
    if args.output_dir is not None:
        require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),
                f'use an empty teacher cache directory: {args.output_dir}')
        check_data(args.data_dir)
        verify_prepared_images((args.data_dir / 'train.jsonl', args.data_dir / 'dev.jsonl'), args.data_dir)
        return {'model_sha256': verified_source(args.model), 'config_sha256': file_sha256(args.config),
                'data_manifest_sha256': file_sha256(args.data_dir / 'data_manifest.json'),
                'input_sha256': {s: file_sha256(args.data_dir / f'{s}.jsonl') for s in ('train', 'dev')}}

    require(not args.expert_output.exists(), f'expert output already exists: {args.expert_output}')
    manifest = read_json(args.data_dir / 'data_manifest.json')
    data_sha = file_sha256(args.data_dir / 'dev.jsonl')
    require(data_sha == manifest['artifacts']['dev']['sha256'], 'changed expert rows')
    verify_prepared_images((args.data_dir / 'dev.jsonl',), args.data_dir)
    files = set(read_json(args.model / 'model.safetensors.index.json')['weight_map'].values())
    return {'model_sha256': {name: file_sha256(args.model / name) for name in sorted(files)},
            'input_sha256': {'dev': data_sha}}


def score_split(args, split, settings, model, processor, tokens, device):
    rank, world = dist.get_rank(), dist.get_world_size()
    source = rows(args.data_dir / f'{split}.jsonl')
    local = {}
    for shard in range(rank, SCORING_SHARDS, world):
        dataset = PromptDataset(args.data_dir, source[shard::SCORING_SHARDS], processor, settings, tokens)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                            collate_fn=PromptBatch(processor.tokenizer.pad_token_id), pin_memory=True,
                            **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if args.workers else {}))
        local[shard] = predict(model, loader, tokens, device, rank)
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)

    def assemble():
        shards = {index: items for result in gathered for index, items in result.items()}
        require(set(shards) == set(range(SCORING_SHARDS)) and
                all(len(items) == len(range(i, len(source), SCORING_SHARDS)) for i, items in shards.items()),
                f'incomplete prediction coverage for {split}')
        ordered = [shards[i % SCORING_SHARDS][i // SCORING_SHARDS] for i in range(len(source))]
        require(len({row['id'] for row in source}) == len(source), f'duplicate ID in {split}')
        for item, row in zip(ordered, source, strict=True):
            verify_row(item, row, tokens)
        summary = {'count': len(source), 'correct': sum(item['correct'] for item in ordered)}
        if args.expert_output is not None:
            domains = {}
            for row, item in zip(source, ordered, strict=True):
                group = domains.setdefault(row['domain'], {'count': 0, 'correct': 0})
                group['count'] += 1
                group['correct'] += int(item['correct'])
            summary['domains'] = domains
            require(len(domains) == 4 and all(d['count'] == 80 for d in domains.values()),
                    'expected four expert domains with 80 questions each')
            args.expert_output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.expert_output, {'items': ordered, 'summary': summary, 'letter_ids': tokens,
                       'scoring_shards': SCORING_SHARDS, 'batch_size': args.batch_size,
                       'bindings': {'shards_sha256': args.bindings['model_sha256'],
                                    'data_sha256': args.bindings['input_sha256']['dev']},
                       'model_fingerprint': args.bindings['model_sha256'], 'data_sha256': args.bindings['input_sha256']['dev']})
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            path = args.output_dir / f'{split}.jsonl'
            temp = path.with_suffix('.tmp')
            with temp.open('w') as stream:
                for item in ordered:
                    stream.write(json.dumps(item, allow_nan=False) + '\n')
            os.replace(temp, path)
            summary['sha256'] = file_sha256(path)
        print(f"{split}: {summary['correct']}/{summary['count']}", flush=True)
        return summary

    return main_rank(assemble)


def main():
    parser = argparse.ArgumentParser(description='Score public train/dev with the frozen ICoT or score EMMA dev.')
    parser.add_argument('--model', type=Path, required=True, help='Merged ICoT teacher or exported student')
    parser.add_argument('--data-dir', type=Path, required=True, help='Prepared public or EMMA directory')
    parser.add_argument('--config', type=Path, default=RECIPE)
    parser.add_argument('--batch-size', type=int, default=8, help='Examples per logical shard batch (reported run: 8)')
    parser.add_argument('--workers', type=int, default=2)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output-dir', type=Path, help='Write train/dev teacher scores and manifest')
    destination.add_argument('--expert-output', type=Path, help='Write EMMA dev scores to a JSON file')
    args = parser.parse_args()
    require(args.batch_size > 0 and args.workers >= 0, 'invalid batch size or worker count')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', device_id=device)
    try:
        args.bindings = main_rank(lambda: provenance(args))
        settings = load_settings(args.config)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        tokens = letter_ids(processor.tokenizer)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
        model.model.visual.patch_embed = PackedPatchEmbedding(model.model.visual.patch_embed)
        model.requires_grad_(False).eval().to(device)
        splits = ('train', 'dev') if args.output_dir is not None else ('dev',)
        outputs = {s: score_split(args, s, settings, model, processor, tokens, device) for s in splits}
        if args.output_dir is not None:
            main_rank(lambda: write_json(args.output_dir / 'manifest.json',
                      {'protocol': PROTOCOL, 'temperature': 1.0, 'world_size': dist.get_world_size(),
                       'scoring_shards': SCORING_SHARDS, 'batch_size': args.batch_size,
                       'bindings': args.bindings, 'outputs': outputs}))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
