"""Fetch the original ChartQA release (the one that carries the data tables).

The HuggingFaceM4/ChartQA parquet mirror we train from has only image/query/
label/human_or_machine. It does NOT carry each chart's underlying data table,
so a visual cue in a ChartQA sample cannot be checked against anything. The
original release adds, per split:

    png/          chart images
    tables/       <chart>.csv  <- the chart's underlying data table (what we need)
    annotations/  <chart>.json bounding boxes (noisy by the authors' own note)
    train_augmented.json / train_human.json   machine- vs human-authored Q/A

Provenance: https://github.com/vis-nlp/ChartQA (Masry et al., Findings of ACL
2022, GPL-3.0) points at the full release hosted on the Hugging Face dataset
repo below. The GitHub copy is the first version and lacks annotations/.

Authors' warning, quoted from the GitHub README: "some of the SVG files were
corrupt/noisy/missing, so the provided annotations in this dataset are a bit
noisy." That concerns annotations/, not tables/.

The archive is untrusted input: it lands in its own directory, every member path
is checked before extraction, and nothing is executed from it.

    python code/data/fetch_chartqa_original.py --check   # HEAD only, no download
    python code/data/fetch_chartqa_original.py
"""
import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = (
    'https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/'
    'ChartQA%20Dataset.zip?download=true'
)
DEFAULT_DEST = REPO_ROOT / 'training_data' / 'raw_external' / 'chartqa_original'
USER_AGENT = 'python-urllib/3 (dataset fetch)'


def fail(message):
    sys.exit(f'fetch_chartqa_original: {message}')


def human(size):
    value = float(size)
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if value < 1024 or unit == 'GiB':
            return f'{value:.1f} {unit}'
        value /= 1024
    return f'{value:.1f} GiB'


def head(url):
    request = urllib.request.Request(url, method='HEAD', headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return {
            'status': response.status,
            'final_url': response.url,
            'content_length': int(response.headers.get('Content-Length') or 0),
            'content_type': response.headers.get('Content-Type'),
            'etag': response.headers.get('ETag'),
            'last_modified': response.headers.get('Last-Modified'),
        }


def download(url, target, expected_bytes):
    partial = target.with_suffix(target.suffix + '.part')
    digest = hashlib.sha256()
    received = 0
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as sink:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            sink.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            if received % (32 << 20) < (1 << 20):
                print(f'  {human(received)} / {human(expected_bytes)}', flush=True)
    if expected_bytes and received != expected_bytes:
        partial.unlink(missing_ok=True)
        fail(f'size mismatch: got {received} bytes, server announced {expected_bytes}')
    partial.replace(target)
    return received, digest.hexdigest()


def safe_members(archive, root):
    """Reject absolute paths, parent escapes, symlinks and device entries."""
    root = root.resolve()
    members = []
    for info in archive.infolist():
        name = info.filename
        if name.endswith('/'):
            continue
        if name.startswith('/') or '\\' in name or '..' in Path(name).parts:
            fail(f'archive member escapes the extraction root: {name!r}')
        mode = info.external_attr >> 16
        if mode and not (mode & 0o170000) in (0, 0o100000):
            fail(f'archive member is not a regular file: {name!r}')
        destination = (root / name).resolve()
        if root not in destination.parents:
            fail(f'archive member resolves outside the extraction root: {name!r}')
        members.append(info)
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', default=DEFAULT_URL)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--check', action='store_true', help='HEAD the object and exit')
    args = parser.parse_args()

    info = head(args.url)
    print(f'HEAD {args.url}')
    for key, value in info.items():
        print(f'  {key}: {value}')
    print(f'  size: {human(info["content_length"])}')
    if args.check:
        return
    if info['status'] != 200 or not info['content_length']:
        fail(f'unexpected HEAD response: {info}')

    args.dest.mkdir(parents=True, exist_ok=True)
    archive_path = args.dest / 'ChartQA-Dataset.zip'
    if archive_path.is_file() and archive_path.stat().st_size == info['content_length']:
        print(f'{archive_path.name} already present; skipping download')
        digest = hashlib.sha256()
        with archive_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        size, sha = archive_path.stat().st_size, digest.hexdigest()
    else:
        print(f'downloading -> {archive_path}')
        size, sha = download(args.url, archive_path, info['content_length'])
    print(f'  {human(size)}  sha256={sha}')

    extract_root = args.dest / 'extracted'
    if extract_root.exists() and any(extract_root.iterdir()):
        fail(f'{extract_root} is not empty; remove it to re-extract')
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_members(archive, extract_root)
        print(f'extracting {len(members)} files (all member paths checked)')
        for member in members:
            archive.extract(member, extract_root)

    tree = {}
    for path in extract_root.rglob('*'):
        if path.is_file():
            key = '/'.join(path.relative_to(extract_root).parts[:2])
            tree[key] = tree.get(key, 0) + 1
    print('contents:')
    for key, count in sorted(tree.items()):
        print(f'  {key}: {count:,} files')

    tables = sorted(extract_root.rglob('*.csv'))
    print(f'csv tables found: {len(tables):,}')

    manifest = {
        'source_repo': 'https://github.com/vis-nlp/ChartQA',
        'source_full_release': 'https://huggingface.co/datasets/ahmed-masry/ChartQA',
        'url': args.url,
        'final_url': info['final_url'],
        'license': 'GPL-3.0 (per the GitHub repo and the HF dataset card)',
        'citation': 'Masry et al., ChartQA, Findings of ACL 2022, '
                    'https://aclanthology.org/2022.findings-acl.177',
        'bytes': size,
        'sha256': sha,
        'etag': info['etag'],
        'last_modified': info['last_modified'],
        'downloaded_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'extracted_to': extract_root.relative_to(REPO_ROOT).as_posix(),
        'file_counts': tree,
        'csv_table_count': len(tables),
        'evaluation_only': False,
        'why': 'supplies tables/<chart>.csv, the underlying data table that makes a '
               'ChartQA visual cue programmatically checkable; the HuggingFaceM4 '
               'parquet mirror we train from does not carry it',
        'authors_warning': 'per the GitHub README some source SVGs were corrupt, noisy '
                           'or missing, so annotations/ (bounding boxes) are noisy; the '
                           'README states no such caveat for tables/',
    }
    manifest_path = args.dest / 'chartqa_original_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'manifest: {manifest_path.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
