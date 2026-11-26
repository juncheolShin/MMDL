"""Download pinned test snapshots and verify the upstream scoring files."""
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import snapshot_download

ROOT = Path("data/hf")
REVISIONS = {
    "MMMU": "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68",
    "MMMU_Pro": "563f3e84bb3b90893083a1f039cfa13077f2302b",
}
OFFICIAL = "https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/b26bdefc7e5e007962fde8f93b2bb53c99c8d13d/"
SCORING = {
    "mmmu/answer_dict_test.json": "45b639a937017d3d71f81f610df3a1471debe1887c5c1d24644fec1b33059d5c",
    "mmmu-pro/evaluate.py": "4c825ed353774658a1a8bc9966e4c52c0c59227e4c5fe5f480b7549817027c2c",
}


def fetch_scoring():
    dest = ROOT / "official"
    dest.mkdir(parents=True, exist_ok=True)
    for source, digest in SCORING.items():
        path = dest / Path(source).name
        if path.exists() and sha256(path.read_bytes()).hexdigest() == digest:
            continue
        data = urlopen(OFFICIAL + source, timeout=60).read()
        if sha256(data).hexdigest() != digest:
            raise ValueError(f"Upstream SHA-256 mismatch for {source}")
        path.write_bytes(data)


if __name__ == "__main__":
    fetch_scoring()
    for name, revision in REVISIONS.items():
        target = ROOT / name
        snapshot_download(repo_id="MMMU/" + name, repo_type="dataset", revision=revision,
                          local_dir=target, allow_patterns=["*/test-*.parquet"])
        if not list(target.rglob("test-*.parquet")):
            raise RuntimeError(f"No test shards found for {name}")
        (target / "MMDL_SNAPSHOT.json").write_text(json.dumps({"revision": revision, "evaluation_only": True}))
