"""Load MMMU_DEV_VAL.tsv and the vendored upstream modules.

The TSV is loaded with Qwen3-VL's own `dataset_utils.load_dataset`, so every row matches what
their evaluation code feeds to `build_mmmu_prompt`.
"""
import hashlib
import importlib.util
import os
import string
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
QWEN_DIR = ROOT / 'third_party' / 'qwen3vl_mmmu'
MMMU_OFFICIAL_DIR = ROOT / 'third_party' / 'mmmu_official'
TSV_PATH = DATA_DIR / 'MMMU_DEV_VAL.tsv'
ANSWER_DICT_PATH = DATA_DIR / 'mmmu_answer_dict_val.json'
IMAGE_ROOT = DATA_DIR / 'images' / 'MMMU'

# See third_party/SOURCES.md: Qwen pins an older MD5; this is the file VLMEvalKit serves today.
VLMEVALKIT_TSV_MD5 = '585e8ad75e73f75dcad265dfd0417d64'

# Qwen's modules use flat imports (`from common_utils import ...`), so their directory goes on sys.path.
if str(QWEN_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_DIR))


def file_digest(path, algo='sha256'):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_mmmu(split='validation'):
    """Return the MMMU_DEV_VAL rows of `split` as a DataFrame, processed exactly like Qwen's loader."""
    md5 = file_digest(TSV_PATH, 'md5')
    if md5 != VLMEVALKIT_TSV_MD5:
        raise RuntimeError(f'{TSV_PATH} has MD5 {md5}, expected {VLMEVALKIT_TSV_MD5}')

    import dataset_utils
    os.environ['LMUData'] = str(DATA_DIR)
    # With the MD5 matching, load_dataset reads the local file instead of re-downloading it.
    dataset_utils.MMMU_DATASET_MD5 = VLMEVALKIT_TSV_MD5
    data = dataset_utils.load_dataset('MMMU_DEV_VAL')
    RESTORED_OPTION_CELLS[:] = _restore_na_like_options(data)
    if split != 'all':
        data = data[data['split'] == split].reset_index(drop=True)
    return data


# (id, option letter, text) cells restored by the last load_mmmu call; recorded in run_config.json.
RESTORED_OPTION_CELLS = []


def _restore_na_like_options(data):
    """Put back option cells that pandas' default NA parsing turned into NaN.

    The option text "None" (validation_Geography_15, whose answer is D) is read as missing, which
    drops the correct option from Qwen's prompt and from the parsers.
    """
    letters = [c for c in string.ascii_uppercase if c in data.columns]
    raw = pd.read_csv(TSV_PATH, sep='\t', usecols=['id', *letters], dtype=str,
                      keep_default_na=False, na_values=['']).set_index('id')
    restored = []
    for c in letters:
        values = data['id'].map(raw[c])
        mask = data[c].isna() & values.notna()
        if mask.any():
            restored += [(i, c, v) for i, v in zip(data.loc[mask, 'id'], values[mask])]
            data.loc[mask, c] = values[mask]
    return restored


def dump_image(line):
    import dataset_utils
    return dataset_utils.dump_image(line, str(IMAGE_ROOT))


def load_mmmu_official(name):
    """Import a vendored MMMU module under a unique name (Qwen also ships an `eval_utils`)."""
    spec = importlib.util.spec_from_file_location(f'mmmu_official_{name}', MMMU_OFFICIAL_DIR / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vendored_digests():
    files = sorted((ROOT / 'third_party').glob('*/*.py')) + [TSV_PATH, ANSWER_DICT_PATH]
    return {str(p.relative_to(ROOT)): file_digest(p) for p in files}
