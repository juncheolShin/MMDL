from pathlib import Path
import struct
import sys
import unittest
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vision_codec


class VisualDeltaTest(unittest.TestCase):
    def test_exact_bit_patterns_including_nans_sign_flips_and_wraparound(self):
        base = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        for target in (np.flip(base), np.roll(base, 32768), base, base ^ np.uint16(0xbeef)):
            encoded = vision_codec.encode(base, target)
            restored = vision_codec.decode(base, encoded)
            self.assertEqual(restored.dtype, np.dtype("<u2"))
            np.testing.assert_array_equal(restored, target)
        transposed = base.T.astype(">u2")
        self.assertEqual(vision_codec.encode(transposed, transposed), vision_codec.encode(base.T, base.T))

    def test_multiple_blocks_odd_tail_and_empty_shape(self):
        size = vision_codec._BLOCK_WORDS + 7
        rng = np.random.default_rng(2026)
        base = rng.integers(0, 65536, size=size, dtype=np.uint16)
        target = base.copy()
        target[3::17] += np.uint16(3)
        for source, expected in ((base, target), (base[:0].reshape(0, 2), target[:0].reshape(0, 2))):
            np.testing.assert_array_equal(vision_codec.decode(source, vision_codec.encode(source, expected)), expected)

    def test_malformed_input_is_rejected(self):
        base = np.array([0x8000, 0x7fff, 3], dtype=np.uint16)
        encoded = vision_codec.encode(base, base)
        for value in (b"", encoded[:-1], encoded + b"x", encoded[:7], encoded[:8] + b"?" * (len(encoded) - 8)):
            with self.assertRaises(ValueError):
                vision_codec.decode(base, value)
        too_long = zlib.compress(bytes(2000))
        with self.assertRaises(ValueError):
            vision_codec.decode(base, encoded[:4] + struct.pack("<I", len(too_long)) + too_long)
        with self.assertRaises(ValueError):
            vision_codec.encode(base, base[::-2])
        with self.assertRaises(ValueError):
            vision_codec.encode(base.view(np.int16), base)
