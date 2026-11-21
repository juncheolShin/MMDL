"""Regression checks for byte-identical model restore and failed/partial outputs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model


def digest(data):
    return hashlib.sha256(data).hexdigest()


@unittest.skipUnless(shutil.which("zstd"), "requires the zstd executable")
class ModelRestoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.base, self.weights, self.output = root / "base", root / "weights", root / "output"
        self.base.mkdir()
        self.weights.mkdir()
        self.target = {"first": bytes([0, 255, 90, 2]), "last": bytes([25, 0])}
        original = {"first": bytes([0, 2, 15, 0]), "last": bytes([25, 254])}
        data = bytearray()
        meta = {}
        for name, value in original.items():
            (self.base / name).write_bytes(value)
            meta[name] = {"size": len(value), "sha256": digest(value)}
            data.extend(a ^ b for a, b in zip(value, self.target[name], strict=True))
        (self.base / "config.json").write_bytes(b"{}")
        patch = subprocess.run(["zstd", "-q", "-c"], input=data, capture_output=True, check=True).stdout
        (self.weights / "student.xor.zst").write_bytes(patch)
        arm = {"step": 2370, "patch": "student.xor.zst", "patch_sha256": digest(patch),
               "target_sha256": {name: digest(value) for name, value in self.target.items()}}
        self.manifest = {"format": "xor-zstd-v1", "assets": {"config.json": digest(b"{}")},
                         "shards": meta, "candidates": {"0.5": arm}}
        (self.weights / "manifest.json").write_text(json.dumps(self.manifest))

    def run_restore(self):
        model.restore(self.base, self.output, "0.5", self.weights)

    def test_roundtrip_and_existing_output(self):
        self.run_restore()
        for name, expected in self.target.items():
            self.assertEqual((self.output / name).read_bytes(), expected)
        self.assertEqual((self.output / "config.json").read_bytes(), b"{}")
        with self.assertRaises(FileExistsError):
            self.run_restore()

    def test_wrong_base_cleans_partial_output(self):
        (self.base / "last").write_bytes(b"xx")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch for merged ICoT base"):
            self.run_restore()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".restore-icot-*")), [])

    def test_corrupt_model_package_and_lfs_pointer(self):
        patch = self.weights / "student.xor.zst"
        for data, cause in [(b"x", "SHA-256"), (b"version https://git-lfs.github.com/spec/v1\n", "Git LFS")]:
            patch.write_bytes(data)
            with self.assertRaisesRegex(ValueError, cause):
                self.run_restore()
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
