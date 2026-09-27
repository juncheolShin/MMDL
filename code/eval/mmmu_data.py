"""Load all 30 MMMU validation configurations at the assignment revision."""
import ast
import io
import math
import os
import re
import string
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get('DATA_PATH', REPO_ROOT / 'data'))
RESULTS_DIR = REPO_ROOT / 'results'

HF_REPO_ID = 'MMMU/MMMU'
HF_REVISION = '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'
SUBJECTS = (
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine',
    'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics',
    'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
    'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music',
    'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology',
)
N_PER_SUBJECT = 30
IMAGE_COLUMNS = tuple(f'image_{n}' for n in range(1, 8))
OPTION_COLUMNS = tuple(string.ascii_uppercase)
DATA_COLUMNS = (
    'id', 'index', 'split', 'question_type', 'question', 'answer',
    *OPTION_COLUMNS, *IMAGE_COLUMNS,
)


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _parse_options(raw, sample_id):
    """Return an ordered letter-to-text map without evaluating arbitrary code."""
    if _is_missing(raw):
        return {}
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f'cannot parse HF options for {sample_id}: {raw!r}') from exc

    parsed = {}
    if isinstance(raw, dict):
        entries = list(raw.items())
        # Preserve letter keys; otherwise use the dataset's mapping iteration order.
        explicit = all(str(key).strip().upper() in OPTION_COLUMNS for key, _ in entries)
        if explicit:
            entries.sort(key=lambda item: OPTION_COLUMNS.index(str(item[0]).strip().upper()))
        for position, (key, value) in enumerate(entries):
            if position >= len(OPTION_COLUMNS):
                raise ValueError(f'too many HF options for {sample_id}: {len(entries)}')
            letter = str(key).strip().upper() if explicit else OPTION_COLUMNS[position]
            if letter in parsed:
                raise ValueError(f'duplicate HF option key {letter} for {sample_id}')
            parsed[letter] = value
    elif isinstance(raw, (list, tuple)):
        for position, value in enumerate(raw):
            if position >= len(OPTION_COLUMNS):
                raise ValueError(f'too many HF options for {sample_id}: {len(raw)}')
            letter = OPTION_COLUMNS[position]
            parsed[letter] = value
    else:
        raise ValueError(f'unsupported HF options type for {sample_id}: {type(raw).__name__}')

    return {letter: parsed[letter] for letter in OPTION_COLUMNS if letter in parsed}


def _normalize_question_type(raw, options, sample_id):
    if isinstance(raw, str):
        value = raw.strip().lower().replace('_', '-').replace(' ', '-')
        if value in {'multiple-choice', 'multiplechoice'}:
            return 'multiple-choice'
        if value in {'open', 'open-ended', 'free-response'}:
            return 'open'
    if _is_missing(raw):
        return 'multiple-choice' if options else 'open'
    raise ValueError(f'unknown question_type for {sample_id}: {raw!r}')


def _dataset_rows(dataset, subject, start_index=0):
    if len(dataset) != N_PER_SUBJECT:
        raise ValueError(f'{subject} validation has {len(dataset)} rows; expected {N_PER_SUBJECT}')
    normalized = []
    for source in dataset:
        sample_id = str(source['id'])
        expected_prefix = f'validation_{subject}_'
        if not sample_id.startswith(expected_prefix):
            raise ValueError(f'{subject} returned unexpected sample id {sample_id!r}')
        options = _parse_options(source.get('options'), sample_id)
        question_type = _normalize_question_type(source.get('question_type'), options, sample_id)
        if question_type == 'multiple-choice' and not options:
            raise ValueError(f'multiple-choice question has no options: {sample_id}')

        row = {
            'id': sample_id,
            # A stable, zero-based evaluation ordinal.
            'index': start_index + len(normalized),
            'split': 'validation',
            'question_type': question_type,
            'question': source['question'],
            # Keep open answers as supplied, including list-valued answers.
            'answer': source['answer'],
        }
        row.update({letter: options.get(letter) for letter in OPTION_COLUMNS})
        row.update({column: source.get(column) for column in IMAGE_COLUMNS})
        normalized.append(row)
    return normalized


def load_mmmu(data_path):
    """Load 900 rows; store Hub data, Arrow files, and images under data_path."""
    data_path = Path(data_path).expanduser().resolve()
    os.environ['HF_HOME'] = str(data_path)
    from datasets import load_dataset

    dataset_path = data_path / 'datasets'
    rows = []
    rows_by_subject = {}
    for subject in SUBJECTS:
        dataset = load_dataset(
            HF_REPO_ID, subject, split='validation', revision=HF_REVISION,
            cache_dir=str(dataset_path),
            data_files={'validation': f'{subject}/validation-*.parquet'},
            # Hub metadata lists dev/test too. Validate the requested split below.
            verification_mode='no_checks',
        )
        subject_rows = _dataset_rows(dataset, subject, start_index=len(rows))
        rows.extend(subject_rows)
        rows_by_subject[subject] = len(subject_rows)

    ids = [row['id'] for row in rows]
    seen = set()
    duplicates = set()
    for sample_id in ids:
        if sample_id in seen:
            duplicates.add(sample_id)
        seen.add(sample_id)
    duplicates = sorted(duplicates)
    if duplicates:
        raise ValueError(f'duplicate MMMU validation ids: {duplicates[:5]}')
    if len(rows) != len(SUBJECTS) * N_PER_SUBJECT:
        raise ValueError(f'MMMU validation has {len(rows)} rows; expected 900')

    data = pd.DataFrame(rows, columns=DATA_COLUMNS)
    data.attrs['provenance'] = {
        'repo_id': HF_REPO_ID,
        'revision': HF_REVISION,
        'split': 'validation',
        'subjects': list(SUBJECTS),
        'rows_by_subject': rows_by_subject,
        'n_samples': len(data),
        'source': 'huggingface_hub',
        'data_path': str(data_path),
        'image_order': list(IMAGE_COLUMNS),
        'image_dump_format': 'PNG',
    }
    return data


def _save_png(image, destination):
    """Save a decoded HF image to PNG without JPEG re-encoding."""
    from PIL import Image

    if isinstance(image, dict):
        image_bytes = image.get('bytes')
        image_path = image.get('path')
        if image_bytes is not None:
            image = Image.open(io.BytesIO(image_bytes))
        elif image_path:
            image = Image.open(image_path)
        else:
            raise ValueError('HF image object has neither bytes nor path')
    elif isinstance(image, (bytes, bytearray, memoryview)):
        image = Image.open(io.BytesIO(bytes(image)))
    elif isinstance(image, (str, os.PathLike)):
        image = Image.open(image)

    if not hasattr(image, 'save'):
        raise TypeError(f'unsupported HF image value: {type(image).__name__}')

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
    try:
        image.save(temporary, format='PNG')
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def dump_image(row, data_path):
    """Write a row's decoded images as lossless PNGs in image_1..image_7 order."""
    image_root = Path(data_path).expanduser().resolve() / 'images' / HF_REVISION
    sample_id = str(row['id'])
    safe_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', sample_id).strip('._') or 'sample'
    paths = []
    for image_index, column in enumerate(IMAGE_COLUMNS, start=1):
        image = row.get(column)
        if _is_missing(image):
            continue
        path = image_root / safe_id / f'image_{image_index}.png'
        if not path.is_file() or path.stat().st_size == 0:
            _save_png(image, path)
        paths.append(str(path))
    return paths
