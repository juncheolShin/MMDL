"""Rebuild the three published ICoT retention splits from their pinned upstream releases."""
import argparse
from collections import Counter, defaultdict
import csv
import errno
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import string
import sys
import tempfile
from zipfile import ZipFile

import pyarrow.parquet as pq
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code' / 'eval'))
from prompting import build_mmmu_prompt

RECIPE_DIR = HERE / 'data_recipe'
EMMA_IMAGE = re.compile(r'<image[ _]?(\d+)>', re.IGNORECASE)
LETTERS = string.ascii_uppercase
PARQUET_COLUMNS = {
    'm3cot': ['id', 'question', 'choices', 'answer', 'context', 'image'],
    'aokvqa': ['question_id', 'question', 'choices', 'correct_choice_idx', 'image'],
    'emma': ['pid', 'question', 'options', 'answer', 'context', 'image_1'],
}


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source_paths(recipe, roots, download):
    """Resolve the experiment's local archive layout, or use huggingface_hub's normal cache."""
    result = {}
    for source in recipe['sources'].values():
        for key, file in source['files'].items():
            candidates = (root / relative for root in roots for relative in (key, file['path']))
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None and download:
                from huggingface_hub import hf_hub_download
                path = Path(hf_hub_download(repo_id=source['repo_id'], repo_type='dataset',
                                            filename=file['path'], revision=source['revision']))
            if path is None:
                raise FileNotFoundError(f'Missing {key}. Supply --source-root or use --download.')
            checked(digest(path) == file['sha256'], f'Upstream SHA-256 mismatch: {key}')
            result[key] = path
    return result


def load_rows(spec):
    path = RECIPE_DIR / spec['recipe']
    checked(digest(path) == spec['recipe_sha256'], f'Modified split selection: {path.name}')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def rgb_digest(rgb):
    h = hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode())
    h.update(rgb.tobytes())
    return h.hexdigest()


def store_image(folder, row, blob):
    key = row['id']
    checked(hashlib.sha256(blob).hexdigest() == row['source_provenance']['encoded_sha256'],
            f'Wrong upstream image for {key}')
    # The saved image must already have the orientation used in the experiment.
    with Image.open(io.BytesIO(blob)) as image:
        with ImageOps.exif_transpose(image).convert('RGB') as rgb:
            pixel_hash = rgb_digest(rgb)
            checked(pixel_hash == row['image_sha256'], f'Decoded pixels differ for {key}')
            fmt = image.format
            convert = row['source'] == 'emma' or image.getexif().get(274, 1) != 1 or fmt not in ('JPEG', 'PNG', 'WEBP')
            extension = 'png' if convert else {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}[fmt]
            name = f'images/{pixel_hash}.{extension}'
            checked(name == row['image'], f'Unexpected image path for {key}')
            path = folder / name
            if not path.exists():
                if convert:
                    rgb.save(path, format='PNG', compress_level=1)
                else:
                    path.write_bytes(blob)


def restore_question(row, raw):
    """Use the original choices and labels; retain source context included in the training prompt."""
    source = row['source']
    context = ''
    if source == 'm3cot':
        uid, question, choices, answer = raw['id'], raw['question'], raw['choices'], raw['answer']
        context = (raw['context'] or '').strip()
    elif source == 'aokvqa':
        uid, question, choices = raw['question_id'], raw['question'], raw['choices']
        answer = LETTERS[raw['correct_choice_idx']]
    elif source == 'scivqa':
        uid, question, answer = raw['instance_id'], raw['question'], raw['answer']
        choices = [option[letter] for letter, option in zip('ABCD', raw['answer_options'], strict=True)]
    elif source == 'pmc_vqa':
        uid, question, answer = raw['index'], raw['Question'], raw['Answer']
        choices = [raw[f'Choice {letter}'].strip().removeprefix(letter + ':') for letter in 'ABCD']
    elif source == 'emma':
        uid, choices, answer = raw['pid'], raw['options'], raw['answer']
        question = EMMA_IMAGE.sub('', raw['question'])
        context = EMMA_IMAGE.sub('', raw['context'] or '').strip()
    else:
        raise ValueError(f'Unknown source {source}')
    checked(uid == row['upstream_id'], f'Unexpected upstream question for {row["id"]}')
    question = question.strip() + ('\nContext: ' + context if context else '')
    choices, answer = [s.strip() for s in choices], answer.strip()
    checked(answer in tuple(LETTERS[:len(choices)]), f'Invalid answer for {row["id"]}')
    question_line = {'id': row['id'], 'question_type': 'multiple-choice', 'question': question,
                     **dict(zip(LETTERS, choices))}
    messages = build_mmmu_prompt(question_line, [], 65536, 524288)
    prompt = messages[0]['content'][-1]['text']
    row.update(question=question, choices=choices, answer=answer,
               conversations=[{'from': 'human', 'value': '<image>\n' + prompt}, {'from': 'gpt', 'value': answer}])


def restore_parquet(file, source, requests):
    by_index = {row['source_provenance']['parquet_row']: (folder, row) for folder, row in requests}
    checked(len(by_index) == len(requests), f'Duplicate upstream rows for {file}')
    image_field = 'image_1' if source == 'emma' else 'image'
    offset = 0
    for batch in pq.ParquetFile(file).iter_batches(batch_size=64, columns=PARQUET_COLUMNS[source]):
        for i in range(batch.num_rows):
            entry = by_index.pop(offset + i, None)
            if entry is None:
                continue
            folder, row = entry
            raw = batch.slice(i, 1).to_pylist()[0]
            restore_question(row, raw)
            store_image(folder, row, raw[image_field]['bytes'])
        offset += batch.num_rows
    checked(not by_index, f'Missing upstream rows in {file}')


def restore_archive(paths, source, requests):
    where = requests[0][1]['source_provenance']
    id_field = 'instance_id' if source == 'scivqa' else 'index'
    by_id = {row['upstream_id']: (folder, row) for folder, row in requests}
    checked(len(by_id) == len(requests), f'Duplicate upstream IDs for {source}')
    with paths[where['file']].open(encoding='utf-8-sig', newline='') as handle, ZipFile(paths[where['archive']]) as images:
        records = json.load(handle) if source == 'scivqa' else csv.DictReader(handle)
        for raw in records:
            entry = by_id.pop(raw[id_field], None)
            if entry is None:
                continue
            folder, row = entry
            restore_question(row, raw)
            member = 'images_train/' + raw['image_file'] if source == 'scivqa' else 'figures/' + raw['Figure_path']
            checked(member == row['source_provenance']['image_member'], f'Wrong archive member for {row["id"]}')
            store_image(folder, row, images.read(member))
    checked(not by_id, f'Missing upstream IDs for {source}')


def manifests(recipe):
    public = {name: recipe['splits']['public/' + name]['annotation'] for name in ('train', 'dev')}
    expert = recipe['splits']['emma/dev']['annotation']
    return {
        'public': {'schema': 'icot_specialist_mc_v1', 'artifacts': public,
                   'annotation_sha256': public['train']['sha256'], 'validation_sha256': public['dev']['sha256'],
                   'sources': {k: v for k, v in recipe['sources'].items() if k != 'emma'},
                   'benchmark_audit': recipe['benchmark_audit']},
        'emma': {'schema': 'emma_retention_dev', 'artifacts': {'dev': expert},
                 'source': recipe['sources']['emma'], 'upstream_split': 'test',
                 'purpose': 'external model selection', 'subjects': dict.fromkeys(('Math', 'Physics', 'Chemistry', 'Coding'), 80)},
    }


def verify_annotation(path, artifact, *, verify_images):
    checked(digest(path) == artifact['sha256'], f'Annotation SHA-256 mismatch: {path}')
    with path.open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    counts = Counter(row['source'] for row in rows)
    checked(len(rows) == artifact['count'] and counts == artifact['by_source'], f'Wrong counts: {path}')
    checked(len({row['id'] for row in rows}) == len(rows), f'Duplicate IDs: {path}')
    images = {row['image']: row['image_sha256'] for row in rows}
    checked(len(images) == artifact['images'], f'Unexpected image count: {path}')
    if verify_images:
        for name, expected in images.items():
            with Image.open(path.parent / name) as im, im.convert('RGB') as rgb:
                checked(rgb_digest(rgb) == expected, f'Image SHA-256 mismatch: {name}')


def prepare(recipe, paths, output):
    assignments = defaultdict(list)
    splits = {}
    for group in manifests(recipe):
        (output / group / 'images').mkdir(parents=True)
    for split, spec in recipe['splits'].items():
        folder = output / split.split('/')[0]
        splits[split] = rows = load_rows(spec)
        for row in rows:
            source, prov = row['source'], row['source_provenance']
            if source == 'emma':
                key = f"emma/{recipe['sources']['emma']['revision']}/{prov['subject']}.parquet"
            else:
                key = prov['file']
            assignments[(source, key)].append((folder, row))
    for (source, key), requests in sorted(assignments.items()):
        print(f'{key}: {len(requests)} rows', flush=True)
        if source in PARQUET_COLUMNS:
            restore_parquet(paths[key], source, requests)
        else:
            restore_archive(paths, source, requests)
    for split, rows in splits.items():
        path = output / (split + '.jsonl')
        with path.open('w', encoding='utf-8', newline='\n') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        verify_annotation(path, recipe['splits'][split]['annotation'], verify_images=False)
    for group, manifest in manifests(recipe).items():
        (output / group / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def verify(recipe, output):
    for group, manifest in manifests(recipe).items():
        checked(read_json(output / group / 'data_manifest.json') == manifest, f'Incorrect {group} manifest')
        for split, artifact in manifest['artifacts'].items():
            verify_annotation(output / group / (split + '.jsonl'), artifact, verify_images=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Destination containing public/ and emma/')
    parser.add_argument('--source-root', type=Path, action='append', default=[], help='Local archive root; repeat for multiple roots')
    parser.add_argument('--download', action='store_true', help='Fetch missing source files through huggingface_hub')
    parser.add_argument('--verify-only', action='store_true', help='Verify annotations, manifests and every decoded image offline')
    args = parser.parse_args()
    recipe = read_json(RECIPE_DIR / 'recipe.json')
    output = args.output.resolve()
    if args.verify_only or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        verify(recipe, output)
    else:
        paths = source_paths(recipe, args.source_root, args.download)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix='.prepare-icot-') as tmp:
            staged = Path(tmp) / 'dataset'
            prepare(recipe, paths, staged)
            try:
                staged.rename(output)
            except OSError as err:
                # Another process may have published the complete dataset during preparation.
                if err.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                verify(recipe, output)
    print(f'Verified {output}: 18,956 train / 1,044 public dev / 320 EMMA dev.')


if __name__ == '__main__':
    main()
