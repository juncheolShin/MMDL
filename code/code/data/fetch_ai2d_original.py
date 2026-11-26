"""Fetch the original AI2D release from AllenAI's public S3 bucket.

The HuggingFace mirror we use elsewhere carries only AI2D's 3,088-question test
split. The original release adds the train questions and, more importantly, the
per-diagram annotation JSONs (text boxes, arrows, arrowheads, blobs and the
relationship graph) that make a visual cue programmatically checkable.

Provenance: https://allenai.org/data/diagrams redirects (307) to the S3 object
below, in the bucket that https://registry.opendata.aws/allenai-diagrams/ lists
as `arn:aws:s3:::ai2-public-datasets` (us-west-2, managed by the Allen Institute
for AI, CC BY-SA 4.0).

The archive is untrusted input: it lands in its own directory, every member path
is checked before extraction, and nothing is executed from it.

    python code/data/fetch_ai2d_original.py --check      # HEAD only, no download
    python code/data/fetch_ai2d_original.py
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
DEFAULT_URL = 'https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip'
DEFAULT_DEST = REPO_ROOT / 'training_data' / 'raw_external' / 'ai2d_original'
USER_AGENT = 'python-urllib/3 (dataset fetch)'


def fail(message):
    sys.exit(f'fetch_ai2d_original: {message}')


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
    parser.add_argument('--keep-archive', action='store_true',
                        help='keep the .zip after extraction (default: keep)')
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
    archive_path = args.dest / 'ai2d-all.zip'
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
            key = path.relative_to(extract_root).parts[:2]
            tree['/'.join(key)] = tree.get('/'.join(key), 0) + 1
    print('contents:')
    for key, count in sorted(tree.items()):
        print(f'  {key}: {count:,} files')

    manifest = {
        'source_page': 'https://allenai.org/data/diagrams',
        'aws_registry': 'https://registry.opendata.aws/allenai-diagrams/',
        'url': args.url,
        'final_url': info['final_url'],
        'license': 'CC BY-SA 4.0 (per AllenAI / AWS Open Data registry)',
        'bytes': size,
        'sha256': sha,
        'etag': info['etag'],
        'last_modified': info['last_modified'],
        'downloaded_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'extracted_to': extract_root.relative_to(REPO_ROOT).as_posix(),
        'file_counts': tree,
        'evaluation_only': False,
        'note': 'training use; the lmms-lab-encoder/ai2d test question ids are excluded '
                'from training by image id in the split step',
    }
    (args.dest / 'ai2d_original_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'manifest: {(args.dest / "ai2d_original_manifest.json").relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
