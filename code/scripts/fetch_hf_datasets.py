"""Warm the pinned MMMU validation cache and fetch the existing MMMU-Pro snapshot.

MMMU validation is loaded through the same per-subject Hub loader used at
evaluation time. The resulting manifest records its pinned provenance. MMMU-Pro
continues to be stored as a local snapshot for its separate experiments.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / 'code' / 'eval'
sys.path.insert(0, str(EVAL_DIR))

from mmmu_data import DATA_DIR, load_mmmu  # noqa: E402

DATA_PATH = DATA_DIR.expanduser().resolve()

MMMU_PRO = ('MMMU/MMMU_Pro', '563f3e84bb3b90893083a1f039cfa13077f2302b')


def warm_mmmu_validation_cache():
    """Load and validate all pinned validation configs through the eval loader."""
    data = load_mmmu(DATA_PATH)
    provenance = data.attrs['provenance']

    manifest = {
        **provenance,
        'manifest': 'dataset_manifest.json',
        'prepared_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    manifest_path = DATA_PATH / 'dataset_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(f"{provenance['repo_id']}@{provenance['revision']}: cached {len(data)} validation rows across 30 configs")


def fetch_mmmu_pro_snapshot():
    """Keep the pre-existing pinned MMMU-Pro snapshot download unchanged."""
    from huggingface_hub import snapshot_download

    repo_id, revision = MMMU_PRO
    target = DATA_PATH / 'hf' / 'MMMU_Pro'
    snapshot_download(
        repo_id=repo_id, repo_type='dataset', revision=revision,
        local_dir=target, max_workers=4,
    )
    parquet = list(target.rglob('*.parquet'))
    if not parquet:
        raise RuntimeError(f'{repo_id} download has no Parquet data: {target}')
    expected = {'standard (4 options)', 'standard (10 options)', 'vision'}
    actual = {p.parent.name for p in parquet}
    if not expected <= actual:
        raise RuntimeError(f'MMMU-Pro snapshot missing configurations: {expected - actual}')
    (target / 'MMDL_SNAPSHOT.json').write_text(json.dumps({
        'repo_id': repo_id, 'revision': revision,
        'parquet_files': len(parquet), 'evaluation_only': True,
    }, indent=2) + '\n')
    print(f'{repo_id}@{revision}: {len(parquet)} Parquet files in {target}')


def main():
    warm_mmmu_validation_cache()
    fetch_mmmu_pro_snapshot()


if __name__ == '__main__':
    main()
