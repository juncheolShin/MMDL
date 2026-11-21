"""Shared data/recipe validation and single-node distributed utilities."""
import hashlib
import json
import os
import tomllib
from pathlib import Path

import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[3]
RECIPE = Path(__file__).resolve().parents[1] / 'recipe.toml'
PUBLISHED = REPO / 'icot/evidence/evaluation_498.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, record):
    path = Path(path)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(temp, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_source(model_dir):
    expected = read_json(PUBLISHED)['model_fingerprint_sha256']
    for name, digest in expected.items():
        require(file_sha256(model_dir / name) == digest, f'ICoT-498 model fingerprint mismatch: {name}')
    # merges.txt is not part of the team's JSON/safetensors-only fingerprint.
    require(file_sha256(model_dir / 'merges.txt') == '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
            'merges.txt differs from the pinned tokenizer')
    return expected


def main_rank(fn):
    outcome = [None]
    if dist.get_rank() == 0:
        try:
            outcome[0] = {'result': fn()}
        except Exception as err:
            outcome[0] = {'error': f'{type(err).__name__}: {err}'}
    dist.broadcast_object_list(outcome, src=0, device=torch.device('cuda', torch.cuda.current_device()))
    require('error' not in outcome[0], outcome[0].get('error', ''))
    return outcome[0]['result']


def load_settings(path):
    with Path(path).open('rb') as stream:
        return tomllib.load(stream)['training']


def check_data(directory):
    manifest = read_json(directory / 'data_manifest.json')
    for split in ('train', 'dev'):
        path = directory / (split + '.jsonl')
        receipt = manifest['artifacts'][split]
        require(file_sha256(path) == receipt['sha256'], f'Changed {split} rows')
        with path.open() as stream:
            count = sum(1 for line in stream if line.strip())
        require(count == receipt['count'] > 0, f'Invalid {split} count')
