"""The model pack must reproduce exact shards and refuse any unrepresented training change."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compact_weights as compact
import model


def write_tensors(path, tensors):
    offset = 0
    header = {}
    for name, value in tensors.items():
        header[name] = {'dtype': 'BF16' if value.dtype == np.dtype('<u2') else 'F32',
                        'shape': value.shape, 'data_offsets': [offset, offset + value.nbytes]}
        offset += value.nbytes
    data = json.dumps(header).encode()
    data += b' ' * (-len(data) % 8)
    with path.open('wb') as stream:
        stream.write(len(data).to_bytes(8, 'little') + data)
        for value in tensors.values():
            value.tofile(stream)


class CompactReleaseTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.base, self.student, self.weights = [self.root / n for n in ('teacher', 'student', 'weights')]
        for p in (self.base, self.student, self.weights):
            p.mkdir()
        self.frozen = 'model.language_model.layers.0.mlp.up_proj.weight'
        self.vis = 'model.visual.merger.weight'
        self.proj = 'model.language_model.layers.0.self_attn.q_proj'
        self.shard = 'model.safetensors'
        self.teacher = {self.frozen: np.ones(9, '<u2'), self.vis: np.arange(12, dtype='<u2'),
                        self.proj + '.weight': np.full((3, 4), 0x3f80, '<u2')}
        self.target = {k: v.copy() for k, v in self.teacher.items()}
        self.target[self.vis][2] = 0x8000
        self.target[self.proj + '.weight'][0, 1] = 0x3f90
        write_tensors(self.base / self.shard, self.teacher)
        write_tensors(self.student / self.shard, self.target)
        self.checkpoint = self.root / 'trainable.safetensors'
        write_tensors(self.checkpoint, {self.proj + '.lora_A': np.ones((8, 4), '<f4') / 64,
                                        self.proj + '.lora_B': np.ones((3, 8), '<f4') / 16})
        for directory in (self.base, self.student):
            (directory / 'config.json').write_text('{}')
        self.patch = self.weights / 'lambda-0.5.zip'
        self.output = self.root / 'restored'

    def manifest(self):
        arm = {'step': 2370, 'patch': self.patch.name, 'patch_sha256': compact.file_sha256(self.patch),
               'target_sha256': {self.shard: compact.file_sha256(self.student / self.shard)}}
        doc = {'format': compact.FORMAT, 'assets': {'config.json': compact.file_sha256(self.base / 'config.json')},
               'shards': {self.shard: {'size': (self.base / self.shard).stat().st_size,
                                       'sha256': compact.file_sha256(self.base / self.shard)}},
               'candidates': {'0.5': arm}}
        (self.weights / 'manifest.json').write_text(json.dumps(doc))

    def test_round_trip_from_teacher_and_deterministic_pack(self):
        digest = compact.pack(self.base, self.student, self.checkpoint, self.patch)
        self.assertEqual(digest, compact.pack(self.base, self.student, self.checkpoint, self.root / 'repeat.zip'))
        self.manifest()
        with zipfile.ZipFile(self.patch) as z:
            self.assertEqual(set(z.namelist()), {self.vis, self.proj + '.weight'})
        model.restore(self.base, self.output, '0.5', self.weights)
        self.assertEqual((self.output / self.shard).read_bytes(), (self.student / self.shard).read_bytes())
        self.assertEqual((self.output / 'config.json').read_text(), '{}')

    def test_changed_tokenizer_or_config_cannot_be_dropped(self):
        (self.student / 'config.json').write_text('{"different": true}')
        with self.assertRaisesRegex(ValueError, 'Inference asset changed'):
            compact.pack(self.base, self.student, self.checkpoint, self.patch)
        self.assertFalse(self.patch.exists())

    def test_unexpected_member_and_frozen_changes(self):
        compact.pack(self.base, self.student, self.checkpoint, self.patch)
        with zipfile.ZipFile(self.patch, 'a') as z:
            z.writestr('unrelated', b'')
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'Unexpected or duplicate'):
            model.restore(self.base, self.output, '0.5', self.weights)
        self.assertFalse(self.output.exists())
        self.target[self.frozen][0] = 2
        write_tensors(self.student / self.shard, self.target)
        new = self.root / 'invalid.zip'
        with self.assertRaisesRegex(ValueError, 'Frozen tensor changed'):
            compact.pack(self.base, self.student, self.checkpoint, new)
        self.assertFalse(new.exists())


if __name__ == '__main__':
    unittest.main()
