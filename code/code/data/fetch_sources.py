"""Pin, download and inspect the non-benchmark training datasets.

Revisions are locked in configs/data_pins.json (tracked by git) so a later run on
another machine fetches the same bytes. Downloads land in training_data/raw/<name>
as a plain directory tree (local_dir, never the symlinked hub cache) so the whole
folder can be copied to the GPU host as-is.

    python code/data/fetch_sources.py --list
    python code/data/fetch_sources.py --pin              # resolve revisions, write the lock
    python code/data/fetch_sources.py                    # download everything pinned
    python code/data/fetch_sources.py --only ai2d --inspect
"""
import argparse
import hashlib
import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = REPO_ROOT / 'configs' / 'data_sources.toml'
DEFAULT_PINS = REPO_ROOT / 'configs' / 'data_pins.json'
DEFAULT_DEST = REPO_ROOT / 'training_data' / 'raw'
MANIFEST_NAME = 'sources_manifest.json'
ROLES = ('core', 'optional')


def fail(message):
    sys.exit(f'fetch_sources: {message}')


def load_sources(path):
    """Read the source table, rejecting anything that looks like evaluation data."""
    if not path.is_file():
        fail(f'missing source table: {path}')
    with path.open('rb') as stream:
        table = tomllib.load(stream)
    if not table:
        fail(f'{path}: no datasets defined')
    sources = {}
    for name, entry in table.items():
        if not isinstance(entry, dict):
            fail(f'{path}: [{name}] must be a table')
        repo_id = entry.get('repo_id')
        role = entry.get('role', 'core')
        if not isinstance(repo_id, str) or '/' not in repo_id:
            fail(f'{path}: [{name}].repo_id must be "owner/name"')
        if role not in ROLES:
            fail(f'{path}: [{name}].role must be one of {ROLES}')
        # Same separation the trainer enforces: benchmarks are evaluation only.
        if 'mmmu' in repo_id.lower() or 'mmmu' in name.lower():
            fail(f'{path}: [{name}] names MMMU; evaluation data is fetched by '
                 'code/scripts/fetch_hf_datasets.py and is never training data')
        sources[name] = {'repo_id': repo_id, 'role': role, 'note': entry.get('note', '')}
    return sources


def load_pins(path):
    if not path.is_file():
        return {}
    pins = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(pins, dict):
        fail(f'{path}: expected an object keyed by dataset name')
    return pins


def resolve_revisions(sources, pins, pin_path, repin):
    """Look up each repo's current commit and write any missing pin to disk."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    changed = False
    for name, source in sources.items():
        existing = pins.get(name)
        if existing and not repin:
            if existing.get('repo_id') != source['repo_id']:
                fail(f'{pin_path}: [{name}] is pinned to {existing.get("repo_id")} but '
                     f'configs/data_sources.toml now says {source["repo_id"]}. '
                     'Resolve this by hand: re-pinning silently would change the dataset.')
            continue
        try:
            info = api.dataset_info(source['repo_id'])
        except HfHubHTTPError as error:
            fail(f'{source["repo_id"]}: cannot read repository info ({error}). '
                 'A gated dataset needs `huggingface-cli login`.')
        if not info.sha:
            fail(f'{source["repo_id"]}: hub returned no commit sha')
        previous = existing.get('revision') if existing else None
        pins[name] = {
            'repo_id': source['repo_id'],
            'revision': info.sha,
            'pinned_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        }
        if previous and previous != info.sha:
            print(f'{name}: re-pinned {previous[:12]} -> {info.sha[:12]} (--repin)')
        else:
            print(f'{name}: pinned {source["repo_id"]} @ {info.sha}')
        changed = True
    if changed:
        pin_path.write_text(json.dumps(pins, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print(f'wrote {pin_path.relative_to(REPO_ROOT)} (commit this file)')
    return pins


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(name, repo_id, revision, dest, workers):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    target = dest / name
    target.mkdir(parents=True, exist_ok=True)
    print(f'{name}: downloading {repo_id} @ {revision[:12]} -> {target.relative_to(REPO_ROOT)}')
    try:
        snapshot_download(
            repo_id=repo_id, repo_type='dataset', revision=revision,
            local_dir=str(target), max_workers=workers,
        )
    except HfHubHTTPError as error:
        fail(f'{repo_id}: download failed ({error})')
    return target


def describe(target):
    """Hash every downloaded file and summarise what kinds of files arrived."""
    files, kinds, total = [], {}, 0
    for path in sorted(target.rglob('*')):
        if not path.is_file() or '.cache' in path.parts:
            continue
        size = path.stat().st_size
        total += size
        suffix = path.suffix.lower() or '<none>'
        kinds[suffix] = kinds.get(suffix, 0) + 1
        files.append({
            'path': path.relative_to(target).as_posix(),
            'bytes': size,
            'sha256': file_sha256(path),
        })
    return files, kinds, total


def inspect_parquet(target):
    """Report split row counts and columns straight from Parquet footers."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {'error': 'pyarrow not installed; install it to see row counts'}
    report = {}
    for path in sorted(target.rglob('*.parquet')):
        try:
            handle = pq.ParquetFile(path)
        except Exception as error:  # a stray or truncated file should not stop the run
            report.setdefault('_unreadable', []).append(f'{path.name}: {error}')
            continue
        # Shards are named like "train-00000-of-00003.parquet" inside a config dir.
        stem = re.sub(r'-\d+-of-\d+$', '', path.stem)
        folder = path.parent.relative_to(target).as_posix()
        split = stem if folder in ('.', '') else f'{folder}/{stem}'
        entry = report.setdefault(split, {'rows': 0, 'shards': 0, 'columns': [], 'nulls': {}})
        entry['rows'] += handle.metadata.num_rows
        entry['shards'] += 1
        entry['columns'] = [field.name for field in handle.schema_arrow]
        # Null counts come from the footer statistics, so no row data is read.
        # Keyed by the full leaf path: an image column is a struct, and collapsing
        # "image.bytes" and "image.path" into "image" would double-count.
        for group in range(handle.metadata.num_row_groups):
            meta = handle.metadata.row_group(group)
            for column in range(meta.num_columns):
                stats = meta.column(column).statistics
                if stats is None or stats.null_count is None:
                    continue
                leaf = meta.column(column).path_in_schema
                entry['nulls'][leaf] = entry['nulls'].get(leaf, 0) + stats.null_count
    return report


def human(size):
    value = float(size)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}'
        value /= 1024
    return f'{value:.1f} TiB'


def print_inspection(inspection):
    if 'error' in inspection:
        print(f'  inspect: {inspection["error"]}')
        return
    for note in inspection.pop('_unreadable', []):
        print(f'  ! {note}')
    if not inspection:
        print('  inspect: no Parquet files found (loading-script dataset?)')
        return
    for split, entry in sorted(inspection.items()):
        print(f'  {split}: {entry["rows"]:,} rows, {entry["shards"]} shard(s)')
        print(f'    columns: {", ".join(entry["columns"])}')
        nulls = {key: count for key, count in entry['nulls'].items() if count}
        if nulls:
            summary = ', '.join(f'{key}={count:,}' for key, count in sorted(nulls.items()))
            print(f'    null values: {summary}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sources', type=Path, default=DEFAULT_SOURCES)
    parser.add_argument('--pins', type=Path, default=DEFAULT_PINS)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--only', nargs='+', metavar='NAME',
                        help='download a subset of the configured datasets')
    parser.add_argument('--include-optional', action='store_true',
                        help='also fetch datasets whose role is "optional"')
    parser.add_argument('--pin', action='store_true',
                        help='resolve missing revisions into the lock file and exit')
    parser.add_argument('--repin', action='store_true',
                        help='overwrite existing pins with the current revisions')
    parser.add_argument('--inspect', action='store_true',
                        help='after downloading, report split sizes and columns')
    parser.add_argument('--list', action='store_true', help='show configured datasets and exit')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    sources = load_sources(args.sources)
    if args.only:
        unknown = sorted(set(args.only) - set(sources))
        if unknown:
            fail(f'unknown dataset(s) {unknown}; configured: {sorted(sources)}')
        sources = {name: sources[name] for name in args.only}
    elif not args.include_optional:
        sources = {name: source for name, source in sources.items() if source['role'] == 'core'}
    if not sources:
        fail('no datasets selected')
    if args.workers < 1:
        fail('--workers must be at least 1')

    pins = load_pins(args.pins)
    if args.list:
        for name, source in sorted(sources.items()):
            state = pins.get(name, {}).get('revision', '(unpinned)')
            print(f'{name:12} {source["repo_id"]:32} {source["role"]:8} {state}')
            if source['note']:
                print(f'{"":12} {source["note"]}')
        return

    pins = resolve_revisions(sources, pins, args.pins, args.repin)
    if args.pin:
        return

    args.dest.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dest / MANIFEST_NAME
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    grand_total = 0
    for name, source in sorted(sources.items()):
        revision = pins[name]['revision']
        target = download(name, source['repo_id'], revision, args.dest, args.workers)
        files, kinds, total = describe(target)
        grand_total += total
        scripts = [entry['path'] for entry in files if entry['path'].endswith('.py')]
        manifest[name] = {
            'repo_id': source['repo_id'],
            'revision': revision,
            'role': source['role'],
            'note': source['note'],
            'local_dir': target.relative_to(REPO_ROOT).as_posix(),
            'downloaded_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'file_count': len(files),
            'total_bytes': total,
            'file_kinds': kinds,
            'files': files,
        }
        kind_summary = ', '.join(f'{key} x{count}' for key, count in sorted(kinds.items()))
        print(f'  {len(files)} files, {human(total)}, kinds: {kind_summary}')
        if scripts:
            print(f'  ! loading script present ({", ".join(scripts)}); read it before any '
                  'load_dataset call, and prefer reading the Parquet files directly')
        if args.inspect:
            inspection = inspect_parquet(target)
            manifest[name]['parquet'] = inspection
            print_inspection(inspection)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n',
                             encoding='utf-8')
    print(f'\n{len(sources)} dataset(s), {human(grand_total)} total')
    print(f'manifest: {manifest_path.relative_to(REPO_ROOT)}')
    print(f'lock file: {args.pins.relative_to(REPO_ROOT)} (commit it; the data stays untracked)')


if __name__ == '__main__':
    main()
