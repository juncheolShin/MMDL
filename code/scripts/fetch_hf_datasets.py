"""Bake complete, pinned MMMU and MMMU-Pro snapshots into the Docker image.

These are evaluation assets only. Do not use any split for training.
"""
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    'MMMU': ('MMMU/MMMU', '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'),
    'MMMU_Pro': ('MMMU/MMMU_Pro', '563f3e84bb3b90893083a1f039cfa13077f2302b'),
}


def main():
    for name, (repo_id, revision) in DATASETS.items():
        target = ROOT / 'data' / 'hf' / name
        snapshot_download(
            repo_id=repo_id, repo_type='dataset', revision=revision,
            local_dir=target, max_workers=4,
        )
        parquet = list(target.rglob('*.parquet'))
        if not parquet:
            raise RuntimeError(f'{repo_id} download has no Parquet data: {target}')
        if name == 'MMMU' and len({p.parent for p in parquet}) != 30:
            raise RuntimeError('MMMU snapshot must contain all 30 subject directories')
        if name == 'MMMU_Pro':
            expected = {'standard (4 options)', 'standard (10 options)', 'vision'}
            actual = {p.parent.name for p in parquet}
            if not expected <= actual:
                raise RuntimeError(f'MMMU-Pro snapshot missing configurations: {expected - actual}')
        (target / 'MMDL_SNAPSHOT.json').write_text(json.dumps({
            'repo_id': repo_id, 'revision': revision,
            'parquet_files': len(parquet), 'evaluation_only': True,
        }, indent=2) + '\n')
        print(f'{repo_id}@{revision}: {len(parquet)} Parquet files in {target}')


if __name__ == '__main__':
    main()
