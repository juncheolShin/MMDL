import hashlib
import json
from pathlib import Path
import sys

OUT=Path('results/tab640_retrain')
DATA=Path('training_data/tab640')
PARENT=Path('icot/adapters/498')
SAVE=Path('checkpoints/tab640_retrained')
BASE=Path('models/Qwen3-VL-4B-Instruct')
sys.path.insert(0,'code')
from third_party.qwen3vl_mmmu.input_utils import prepare_inputs_for_vllm


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read_rows(split):
    path=DATA/(split+'.json')
    return json.loads(path.read_text())


def request(row,processor):
    image=Path(row['image'])
    assert sha(image)==row['image_sha256']
    text=row['question']
    if row['mode']=='number':
        text+='\nAnswer with the number only.'
    else:
        assert row['mode'] in ('mc','bridge')
        text+='\n'+'\n'.join(f'{letter}. {value}' for letter,value in zip('ABCDEFGHIJ',row['options'],strict=True))
        if row['mode']=='bridge':
            text+='\nGive the numeric result, then its matching option letter.'
        else:
            text+='\nAnswer with the option letter only.'
    content=[{'type':'image','image':str(image.resolve()),'min_pixels':262144,'max_pixels':2097152},
             {'type':'text','text':text}]
    return prepare_inputs_for_vllm([{'role':'user','content':content}],processor)
