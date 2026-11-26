"""Restore a selected specialist-data student from its lossless binary difference to ICoT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

WEIGHTS = Path(__file__).resolve().parent / "weights"
BLOCK_SIZE = 4 * 1024 * 1024


def check_hash(actual, expected, description):
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {description}: expected {expected}, got {actual}")


def copy_asset(source, destination, expected):
    digest = hashlib.sha256()
    with source.open("rb") as inp, destination.open("xb") as out:
        while data := inp.read(BLOCK_SIZE):
            digest.update(data)
            out.write(data)
    check_hash(digest.hexdigest(), expected, source.name)


def restore_xor_shard(source, destination, patch_stream, metadata, expected):
    if source.stat().st_size != metadata["size"]:
        raise ValueError(f"Wrong base shard size: {source.name}. Use the merged ICoT base.")
    base_digest = hashlib.sha256()
    target_digest = hashlib.sha256()
    with source.open("rb") as inp, destination.open("xb") as out:
        while original := inp.read(BLOCK_SIZE):
            delta = patch_stream.read(len(original))
            if len(delta) != len(original):
                raise ValueError(f"Incomplete binary difference at {source.name}")
            # Bytewise XOR is its own inverse; little endian preserves leading/trailing zeroes.
            result = (int.from_bytes(original, "little") ^ int.from_bytes(delta, "little")).to_bytes(
                len(original), "little")
            base_digest.update(original)
            target_digest.update(result)
            out.write(result)
    check_hash(base_digest.hexdigest(), metadata["sha256"], f"merged ICoT base {source.name}")
    check_hash(target_digest.hexdigest(), expected, f"restored {source.name}")


def restore(base, output, strength, weights=WEIGHTS):
    manifest = json.loads((weights / "manifest.json").read_text())
    compact = manifest["format"] == "icot-bf16-lora-v2"
    if not compact and manifest["format"] != "xor-zstd-v1":
        raise ValueError("Unknown model package format")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output}")
    arm = manifest["candidates"][strength]
    patch = weights / arm["patch"]
    with patch.open("rb") as inp:
        if inp.read(64).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError("This is a Git LFS pointer. Run git lfs pull for the selected weights first.")
        inp.seek(0)
        digest = hashlib.sha256()
        while data := inp.read(BLOCK_SIZE):
            digest.update(data)
        check_hash(digest.hexdigest(), arm["patch_sha256"], patch.name)
        if not compact and shutil.which("zstd") is None:
            raise RuntimeError("The zstd executable is required on PATH.")
        inp.seek(0)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".restore-icot-", dir=output.parent) as tmp:
            staged = Path(tmp) / "model"
            staged.mkdir()
            for name, sha256 in manifest["assets"].items():
                copy_asset(base / name, staged / name, sha256)
            if compact:
                from compact_weights import file_sha256, restore_shard
                consumed = set()
                with zipfile.ZipFile(inp) as archive:
                    for name, metadata in sorted(manifest["shards"].items()):
                        dest = staged / name
                        copy_asset(base / name, dest, metadata["sha256"])
                        consumed.update(restore_shard(base / name, dest, archive))
                        check_hash(file_sha256(dest), arm["target_sha256"][name], f"restored {name}")
                        print(f"Verified {name}", flush=True)
                    if sorted(consumed) != sorted(archive.namelist()):
                        raise ValueError("Unexpected or duplicate members in model package")
            else:
                # The frame contains the XOR bytes of every shard in sorted filename order.
                with subprocess.Popen(["zstd", "-q", "-d", "-c"], stdin=inp, stdout=subprocess.PIPE) as process:
                    with process.stdout as stream:
                        for name, metadata in sorted(manifest["shards"].items()):
                            restore_xor_shard(base / name, staged / name, stream, metadata, arm["target_sha256"][name])
                            print(f"Verified {name}", flush=True)
                        if stream.read(1):
                            raise ValueError("Trailing binary difference after the last shard")
                    if process.wait() != 0:
                        raise ValueError("Failed to decompress the model package")
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"Output appeared during restore: {output}")
            os.rename(staged, output)
    print(f"Restored lambda={strength}, checkpoint {arm['step']} to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Original merged 500-point ICoT directory")
    parser.add_argument("--strength", required=True, choices=("0", "0.1", "0.25", "0.5"))
    parser.add_argument("--output", type=Path, required=True, help="New directory for the full inference model")
    args = parser.parse_args()
    restore(args.base, args.output, args.strength)


if __name__ == "__main__":
    main()
