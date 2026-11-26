"""Encode the scored single-image QA rows exactly as the training and validation inputs."""
import hashlib
import json
import math
import re
from pathlib import Path
from string import ascii_uppercase

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from training.common import require



def verify_prepared_images(annotations, image_root):
    """Bind the audited JSONL to the exact decoded images used for training."""
    root = Path(image_root).resolve()
    verified, split_hashes = {}, []
    for annotation in annotations:
        hashes = set()
        with Path(annotation).open(encoding='utf-8') as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                names = row.get('image', [])
                names = [names] if isinstance(names, str) else names
                expected = row.get('image_sha256', [])
                expected = [expected] if isinstance(expected, str) else expected
                if not names or len(names) != len(expected):
                    raise ValueError('prepared rows must bind every image with image_sha256')
                for name, digest in zip(names, expected):
                    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                        raise ValueError('invalid prepared image_sha256')
                    path = (root / name).resolve()
                    if not path.is_relative_to(root) or not path.is_file():
                        raise ValueError('prepared image missing or outside image root')
                    if path not in verified:
                        with Image.open(path) as opened:
                            rgb = ImageOps.exif_transpose(opened).convert('RGB')
                            actual = hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode())
                            actual.update(rgb.tobytes())
                            verified[path] = actual.hexdigest()
                            rgb.close()
                    if verified[path] != digest:
                        raise ValueError(f'prepared image hash mismatch: {path}')
                    hashes.add(digest)
        split_hashes.append(hashes)
    if any(hashes & prev for i, hashes in enumerate(split_hashes) for prev in split_hashes[:i]):
        raise ValueError('training and validation must have disjoint image groups')
    return {'verified_images': len(verified), 'image_counts': [len(group) for group in split_hashes]}


def conversation_messages(row, image_root):
    """Preserve image/text order rather than moving all image tokens to the front."""
    paths = row.get('image', [])
    paths = [paths] if isinstance(paths, str) else paths
    if not paths:
        raise ValueError('vision training requires at least one image per example')
    image_root = Path(image_root).resolve()
    resolved = [(image_root / path).resolve() for path in paths]
    if any(not path.is_relative_to(image_root) or not path.is_file() for path in resolved):
        raise ValueError('missing image or path outside image root')
    turns = row['conversations']
    if len(turns) < 2 or len(turns) % 2:
        raise ValueError('expected alternating human/gpt conversations')
    messages, cursor = [], 0
    for i, turn in enumerate(turns):
        expected = 'human' if i % 2 == 0 else 'gpt'
        if turn.get('from') != expected or not isinstance(turn.get('value'), str):
            raise ValueError('expected alternating human/gpt text turns')
        value = turn['value']
        if expected == 'gpt':
            if '<image>' in value or not value.strip():
                raise ValueError('assistant answer must be nonempty and contain no image tags')
            content = [{'type': 'text', 'text': value}]
        else:
            content = []
            for part in re.split('(<image>)', value):
                if part == '<image>':
                    if cursor >= len(resolved):
                        raise ValueError('more image tags than images')
                    content.append({'type': 'image', 'image': str(resolved[cursor])})
                    cursor += 1
                elif part:
                    content.append({'type': 'text', 'text': part})
        messages.append({'role': 'user' if expected == 'human' else 'assistant', 'content': content})
    if cursor != len(resolved):
        raise ValueError('image count does not match human image tags')
    return messages, resolved


def expand_image_placeholders(text, processor, grid):
    """Use the same image-token expansion as the pinned Qwen3VLProcessor."""
    marker = '<|mmdl_image_expansion|>'
    if marker in text:
        raise ValueError('reserved image expansion marker in text')
    token = processor.image_token
    grids = list(grid)
    if text.count(token) > len(grids):
        raise ValueError('image token/grid mismatch')
    for item in grids[:text.count(token)]:
        count = int(item.prod().item()) // processor.image_processor.merge_size**2
        text = text.replace(token, marker * count, 1)
    return text.replace(marker, token)


def encode_example(row, processor, image_root, min_pixels, max_pixels, model_max_length):
    messages, paths = conversation_messages(row, image_root)
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    images = []
    try:
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        encoded = processor(
            text=[full_text], images=images, return_tensors='pt',
            add_special_tokens=False, padding=False, truncation=False,
            images_kwargs={'min_pixels': min_pixels, 'max_pixels': max_pixels},
        )
    finally:
        for image in images:
            image.close()
    ids = encoded['input_ids'][0]
    if len(ids) > model_max_length:
        raise ValueError(
            f"example {row.get('id', '<unknown>')} has {len(ids)} tokens > {model_max_length}; "
            'refusing to silently truncate images or answer tokens'
        )
    grid = encoded['image_grid_thw']

    def token_ids(text):
        expanded = expand_image_placeholders(text, processor, grid)
        return processor.tokenizer(expanded, add_special_tokens=False)['input_ids']

    full_ids = ids.tolist()
    if token_ids(full_text) != full_ids:
        raise RuntimeError('processor/template token alignment changed')
    labels = torch.full_like(ids, -100)
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        before = processor.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        through = processor.apply_chat_template(messages[:i + 1], tokenize=False, add_generation_prompt=False)
        prefix, suffix = token_ids(before), token_ids(through)
        if full_ids[:len(prefix)] != prefix or full_ids[:len(suffix)] != suffix:
            raise RuntimeError('assistant span is not an exact token prefix of the full conversation')
        if len(suffix) <= len(prefix):
            raise ValueError('assistant answer contains no supervised tokens')
        labels[len(prefix):len(suffix)] = ids[len(prefix):len(suffix)]
    if not (labels[1:] != -100).any():
        raise ValueError('example contains no causal assistant targets')
    # Batch size is deliberately one: visual patches are not sequence padding.
    encoded['labels'] = labels.unsqueeze(0)
    return dict(encoded)


class VisionConversationDataset(Dataset):
    def __init__(self, annotation, processor, image_root, settings):
        self.rows = [json.loads(line) for line in Path(annotation).read_text(encoding='utf-8').splitlines() if line.strip()]
        self.processor, self.image_root, self.settings = processor, image_root, settings

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return encode_example(
            self.rows[index], self.processor, self.image_root,
            self.settings['min_pixels'], self.settings['max_pixels'], self.settings['model_max_length'],
        )


class MultimodalCollator:
    """Right-pad text while preserving each example's ordered visual patch rows."""
    FIELDS = frozenset(('input_ids', 'attention_mask', 'labels', 'pixel_values', 'image_grid_thw'))

    def __init__(self, pad_token_id, image_token_id, merge_size=2, max_batch_size=2):
        if type(pad_token_id) is not int or pad_token_id < 0 or pad_token_id == image_token_id:
            raise ValueError('a distinct nonnegative text padding token is required')
        if type(image_token_id) is not int or image_token_id < 0:
            raise ValueError('invalid image token ID')
        if merge_size < 1 or max_batch_size not in (1, 2):
            raise ValueError('invalid merge size or maximum batch size')
        self.pad_token_id, self.image_token_id = pad_token_id, image_token_id
        self.merge_size, self.max_batch_size = merge_size, max_batch_size

    def __call__(self, examples):
        if not 1 <= len(examples) <= self.max_batch_size:
            raise ValueError('multimodal batch must contain one or two examples within the configured limit')
        reference = examples[0]
        lengths = []
        for example in examples:
            if set(example) != self.FIELDS:
                raise ValueError(f'unsupported multimodal fields: missing={self.FIELDS - set(example)}, extra={set(example) - self.FIELDS}')
            if any(not isinstance(value, torch.Tensor) for value in example.values()):
                raise TypeError('every encoded field must be a tensor')
            ids, mask, labels = (example[key] for key in ('input_ids', 'attention_mask', 'labels'))
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
                raise ValueError('each encoded example must have input_ids shape [1, length>=2]')
            if ids.dtype != torch.long or labels.dtype != torch.long or mask.dtype not in (torch.long, torch.int, torch.bool):
                raise TypeError('input IDs/labels must be int64 and attention mask integral')
            if mask.shape != ids.shape or labels.shape != ids.shape or not (mask == 1).all():
                raise ValueError('collator expects aligned unpadded single-example text tensors')
            supervised = labels != -100
            if not supervised[:, 1:].any() or not torch.equal(labels[supervised], ids[supervised]):
                raise ValueError('labels must retain at least one causal target and match input token IDs')
            if (labels[ids == self.image_token_id] != -100).any():
                raise ValueError('image tokens must be excluded from assistant loss')
            grid, pixels = example['image_grid_thw'], example['pixel_values']
            if grid.dtype != torch.long or grid.ndim != 2 or grid.shape[1] != 3 or not len(grid) or not (grid > 0).all():
                raise ValueError('image grid must be a nonempty positive int64 [images,3] tensor')
            if (grid[:, 1:] % self.merge_size != 0).any():
                raise ValueError('spatial grid dimensions must be divisible by merger size')
            patches = grid.prod(dim=-1)
            if pixels.ndim != 2 or not pixels.is_floating_point() or pixels.shape[0] != int(patches.sum()):
                raise ValueError('visual patch rows do not match image grids')
            expected_tokens = int((patches // self.merge_size**2).sum())
            if int((ids == self.image_token_id).sum()) != expected_tokens:
                raise ValueError('image token count does not match image grids')
            for key in self.FIELDS:
                if example[key].device != reference[key].device or example[key].dtype != reference[key].dtype:
                    raise ValueError(f'inconsistent device or dtype for {key}')
            if pixels.shape[1] != reference['pixel_values'].shape[1]:
                raise ValueError('inconsistent visual patch feature dimension')
            lengths.append(ids.shape[1])
        length = max(lengths)
        batch = {}
        for key, fill in (('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)):
            output = reference[key].new_full((len(examples), length), fill)
            for i, (example, count) in enumerate(zip(examples, lengths)):
                output[i, :count] = example[key][0]
            batch[key] = output
        for key in ('pixel_values', 'image_grid_thw'):
            batch[key] = torch.cat([example[key] for example in examples], dim=0)
        return batch


class RetentionDataset(torch.utils.data.Dataset):
    """Bind the teacher decision to the exact original row and option alphabet."""
    def __init__(self, dataset, teacher_path):
        self.dataset = dataset
        self.rows = dataset.rows
        with Path(teacher_path).open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        by_id = {entry['id']: entry for entry in records}
        require(len(records) == len(by_id) == len(self.rows), 'teacher/annotation count or duplicate ID mismatch')
        max_options = max(len(row['choices']) for row in self.rows)
        require(2 <= max_options <= len(ascii_uppercase), 'unsupported choice count')
        token_ids = [dataset.processor.tokenizer.encode(c, add_special_tokens=False)
                     for c in ascii_uppercase[:max_options]]
        require(all(len(ids) == 1 for ids in token_ids), 'single-token option letters required')
        self.letter_ids = [ids[0] for ids in token_ids]
        require(len(set(self.letter_ids)) == max_options, 'option token IDs must be distinct')
        self.teacher = []
        for row in self.rows:
            key = row['id']
            entry = by_id.pop(key, None)
            require(entry is not None, f'missing teacher ID {key}')
            count = len(row['choices'])
            choices = ascii_uppercase[:count]
            gold = row['answer']
            require(count >= 2 and len(gold) == 1 and gold in choices and entry['gold'] == gold
                    and entry['source'] == row['source'] and entry['option_token_ids'] == self.letter_ids[:count],
                    f'teacher option/source/answer mismatch at {key}')
            probs, prediction = entry['probabilities'], entry['prediction']
            require(len(probs) == count and all(type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1
                                                for p in probs) and abs(sum(probs) - 1) <= 1e-5
                    and prediction == choices[max(range(count), key=probs.__getitem__)]
                    and type(entry['correct']) is bool and entry['correct'] == (prediction == gold),
                    f'invalid teacher distribution or correctness at {key}')
            self.teacher.append((probs, entry['correct'], choices.index(gold)))
        require(not by_id, 'teacher labels contain unknown IDs')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        example = self.dataset[index]
        probs, correct, gold_index = self.teacher[index]
        ids = self.letter_ids[:len(probs)]
        first_label = example['labels'][example['labels'] != -100][0].item()
        require(first_label == ids[gold_index], f"gold first assistant token mismatch at {self.rows[index]['id']}")
        return {**example, 'retention': (index, ids, probs, correct)}


class RetentionCollator:
    def __init__(self, base):
        self.base = base

    def __call__(self, examples):
        metadata = [e['retention'] for e in examples]
        batch = self.base([{k: v for k, v in e.items() if k != 'retention'} for e in examples])
        width = max(len(ids) for _, ids, _, _ in metadata)
        batch['option_ids'] = torch.tensor([ids + [-1] * (width-len(ids)) for _, ids, _, _ in metadata])
        batch['teacher_probs'] = torch.tensor([p + [0.] * (width-len(p)) for _, _, p, _ in metadata],
                                              dtype=torch.float32)
        batch['teacher_correct'] = torch.tensor([correct for _, _, _, correct in metadata], dtype=torch.bool)
        batch['row_index'] = torch.tensor([idx for idx, _, _, _ in metadata])
        return batch
