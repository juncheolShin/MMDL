import struct
import sys
import unittest
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lora_codec as codec


class LoRACodecTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.base = np.arange(65 * 12, dtype='<u2').reshape(65, 12) + np.uint16(0x3e80)
        self.a = rng.normal(0, .01, (8, 12)).astype('<f4')
        self.b = rng.normal(0, .01, (65, 8)).astype('<f4')

    def test_original_merged_bits_and_evaluator_corrections(self):
        base_float = (self.base.astype('<u4') << 16).view('<f4')
        oracle = (base_float + 2 * (self.b @ self.a)).astype('<f4').view('<u4')
        target = ((oracle + np.uint32(0x7fff) + ((oracle >> 16) & 1)) >> 16).astype('<u2')
        target[0, 2] ^= np.uint16(1)
        target[-1, -1] ^= np.uint16(1)
        packet = codec.encode(self.base, target, self.a, self.b)
        self.assertTrue(np.array_equal(codec.decode(self.base, packet), target))

    def test_bfloat16_ties_round_to_even_without_corrections(self):
        base = np.array([[0x3f80, 0x3f81]], dtype='<u2')
        expected = np.array([[0x3f80, 0x3f82]], dtype='<u2')
        a = np.zeros((8, 2), dtype='<f4')
        b = np.zeros((1, 8), dtype='<f4')
        a[0] = 1
        b[0, 0] = 2**-9  # Scaling by two gives exactly half a BF16 unit at 1.0.
        packet = codec.encode(base, expected, a, b)
        self.assertEqual(codec.HEADER.unpack_from(packet)[-1], 0)
        self.assertTrue(np.array_equal(codec.decode(base, packet), expected))

    def test_zero_delta_preserves_base_and_checks_payload(self):
        self.b.fill(0)
        packet = codec.encode(self.base, self.base, self.a, self.b)
        self.assertTrue(np.array_equal(codec.decode(self.base, packet), self.base))
        for bad in (packet[:4], packet[:-1], packet + b'x', b'bad!' + packet[4:]):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                codec.decode(self.base, bad)
        with self.assertRaises(ValueError):
            codec.encode(self.base, self.base, self.a.T, self.b)
        with self.assertRaises(ValueError):
            codec.decode(self.base[:-1], packet)

    def test_rejects_invalid_correction(self):
        self.b.fill(0)
        target = self.base.copy()
        target[0, 0] ^= np.uint16(1)
        packet = codec.encode(self.base, target, self.a, self.b)
        raw = bytearray(zlib.decompress(packet[codec.HEADER.size:]))
        struct.pack_into('<I', raw, len(raw) - codec.CORRECTION.itemsize, self.base.size)
        with self.assertRaisesRegex(ValueError, 'indices'):
            codec.decode(self.base, packet[:codec.HEADER.size] + zlib.compress(raw))


if __name__ == '__main__':
    unittest.main()
