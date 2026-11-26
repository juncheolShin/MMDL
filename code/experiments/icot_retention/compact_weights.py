"""Pack the evaluated visual weights and rank-eight attention updates into one lossless ZIP per model."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np

import lora_codec
import vision_codec


FORMAT = 'icot-bf16-lora-v2'
ATTENTION = re.compile(r'model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj\.weight')
DTYPES = {'BF16': '<u2', 'F32': '<f4'}


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_kind(name):
    if name.startswith('model.visual.'):
        return 'vision'
    if ATTENTION.fullmatch(name):
        return 'attention'
    return 'frozen'


class TensorFile:
    """Views on the original safetensors layout, including raw BF16 bit patterns."""

    def __init__(self, path):
        self.path = path
        with path.open('rb') as stream:
            length = int.from_bytes(stream.read(8), 'little')
            if length < 2 or length > min(path.stat().st_size - 8, 100_000_000):
                raise ValueError(f'Invalid safetensors header in {path}')
            self.header = stream.read(length)
        self.data_start = 8 + length
        self.items = json.loads(self.header)
        self.items.pop('__metadata__', None)
        end = 0
        for info in sorted(self.items.values(), key=lambda t: t['data_offsets']):
            start, stop = info['data_offsets']
            if start != end or stop < start:
                raise ValueError(f'Invalid tensor offsets in {path}')
            end = stop
        if self.data_start + end != path.stat().st_size:
            raise ValueError(f'Invalid tensor data size in {path}')

    def offset(self, name):
        return self.data_start + self.items[name]['data_offsets'][0]

    def view(self, name, dtype):
        item = self.items[name]
        start, stop = item['data_offsets']
        dt = np.dtype(DTYPES[dtype])
        if item['dtype'] != dtype or math.prod(item['shape']) * dt.itemsize != stop - start:
            raise ValueError(f'Unexpected tensor type or shape: {name}')
        return np.memmap(self.path, dtype=dt, mode='r', offset=self.data_start + start, shape=tuple(item['shape']))

    def check_frozen(self, other, name):
        start, stop = self.items[name]['data_offsets']
        with self.path.open('rb') as left, other.path.open('rb') as right:
            left.seek(self.data_start + start)
            right.seek(other.data_start + start)
            for offset in range(start, stop, 4 << 20):
                size = min(4 << 20, stop - offset)
                if left.read(size) != right.read(size):
                    raise ValueError(f'Frozen tensor changed: {name}')


def restore_shard(base_file, destination, archive):
    base = TensorFile(base_file)
    consumed = set()
    with destination.open('r+b') as out:
        for name in base.items:
            kind = tensor_kind(name)
            if kind == 'frozen':
                continue
            words = base.view(name, 'BF16')
            payload = archive.read(name)
            restored = vision_codec.decode(words, payload) if kind == 'vision' else lora_codec.decode(words, payload)
            if restored.shape != words.shape or restored.dtype != words.dtype:
                raise ValueError(f'Invalid decoded tensor: {name}')
            out.seek(base.offset(name))
            restored.tofile(out)
            consumed.add(name)
    return consumed


def pack(base, student, checkpoint, output):
    """Publish only a complete archive, even if compression fails or two jobs race."""
    if output.exists():
        raise FileExistsError(output)
    shards = sorted(base.glob('*.safetensors'))
    if not shards or [p.name for p in shards] != sorted(p.name for p in student.glob('*.safetensors')):
        raise ValueError('Teacher and student must have matching inference shards')
    inference_assets = json.loads((Path(__file__).resolve().parent / 'weights/manifest.json').read_text())['assets']
    for name in inference_assets:
        source, target = base / name, student / name
        if source.exists() or target.exists():
            if not (source.is_file() and target.is_file()) or file_sha256(source) != file_sha256(target):
                raise ValueError(f'Inference asset changed: {name}')
    factors = TensorFile(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pack-icot-', dir=output.parent) as tmp:
        staged = Path(tmp) / 'student.zip'
        with zipfile.ZipFile(staged, 'x', compression=zipfile.ZIP_STORED) as archive:
            for base_file in shards:
                before, after = TensorFile(base_file), TensorFile(student / base_file.name)
                if before.header != after.header:
                    raise ValueError(f'Incompatible safetensors layout: {base_file.name}')
                for name in sorted(before.items):
                    kind = tensor_kind(name)
                    if kind == 'frozen':
                        before.check_frozen(after, name)
                        continue
                    source, target = before.view(name, 'BF16'), after.view(name, 'BF16')
                    if kind == 'vision':
                        payload = vision_codec.encode(source, target)
                    else:
                        stem = name.removesuffix('.weight')
                        a = factors.view(stem + '.lora_A', 'F32')
                        b = factors.view(stem + '.lora_B', 'F32')
                        payload = lora_codec.encode(source, target, a, b)
                    # A fixed timestamp and stable tensor order make the ZIP container deterministic.
                    archive.writestr(zipfile.ZipInfo(name), payload)
        os.link(staged, output)
    return file_sha256(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path, help='Merged ICoT teacher directory')
    parser.add_argument('--student', required=True, type=Path, help='Original evaluated student export')
    parser.add_argument('--checkpoint', required=True, type=Path, help='Selected trainable.safetensors with FP32 LoRA factors')
    parser.add_argument('--output', required=True, type=Path, help='New ZIP filename')
    args = parser.parse_args()
    print(pack(args.base, args.student, args.checkpoint, args.output), args.output)
