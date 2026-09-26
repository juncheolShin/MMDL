"""Validate a non-benchmark JSONL and run Qwen3-VL's pinned multimodal LoRA trainer."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

EVAL_DATA_ROOT = Path(__file__).resolve().parents[2] / 'data'
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / 'models' / 'Qwen3-VL-4B-Instruct'
UPSTREAM_COMMIT = '96588727e44c78b25ba03ea03b8e12f7e64fd0da'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration import DEFAULT_CONFIG, TRAINING_TYPES, file_sha256, load_section  # noqa: E402


def within(path, root):
    return path == root or root in path.parents


def validate_data(annotation, image_root):
    annotation = annotation.resolve()
    image_root = image_root.resolve()
    eval_root = EVAL_DATA_ROOT.resolve()
    for path in (annotation, image_root):
        if within(path, eval_root) or any('mmmu' in p.lower() for p in path.parts):
            raise ValueError(f'MMMU/MMMU-Pro evaluation data cannot be used for training: {path}')
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)
    count = 0
    with annotation.open(encoding='utf-8') as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_name = str(row.get('source', '')).lower()
            if 'mmmu' in source_name:
                raise ValueError(f'line {line_no}: benchmark source is evaluation only')
            turns = row.get('conversations')
            if not isinstance(turns, list) or len(turns) < 2:
                raise ValueError(f'line {line_no}: expected human/gpt conversations')
            if any(t.get('from') not in ('human', 'gpt') or not isinstance(t.get('value'), str)
                   for t in turns):
                raise ValueError(f'line {line_no}: invalid conversation turn')
            if turns[0]['from'] != 'human' or turns[-1]['from'] != 'gpt':
                raise ValueError(f'line {line_no}: conversations must start with human and end with gpt')
            images = row.get('image') or []
            images = [images] if isinstance(images, str) else images
            if not isinstance(images, list) or any(not isinstance(p, str) for p in images):
                raise ValueError(f'line {line_no}: image must be a path or a list of paths')
            image_tags = sum(t['value'].count('<image>') for t in turns if t['from'] == 'human')
            if image_tags != len(images):
                raise ValueError(f'line {line_no}: {image_tags} <image> tags for {len(images)} images')
            if any('<image>' in t['value'] for t in turns if t['from'] == 'gpt'):
                raise ValueError(f'line {line_no}: assistant answer must not contain <image>')
            for name in images:
                image = (image_root / name).resolve()
                if not within(image, image_root) or not image.is_file():
                    raise ValueError(f'line {line_no}: missing or out-of-root image {name}')
                if within(image, eval_root):
                    raise ValueError(f'line {line_no}: evaluation image cannot be used for training')
            count += 1
    if count == 0:
        raise ValueError('annotation contains no examples')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', type=Path, required=True, help='Qwen conversation JSONL')
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    training = load_section(args.config, 'training', TRAINING_TYPES)
    for key, value in training.items():
        setattr(args, key, value)
    count = validate_data(args.annotation, args.image_root)
    print(f'validated {count} non-benchmark training examples')
    if args.validate_only:
        return
    if not args.model.is_dir():
        parser.error(f'model directory does not exist: {args.model}')
    if not 0 < args.min_pixels <= args.max_pixels:
        parser.error('expected 0 < min-pixels <= max-pixels')
    if (args.epochs <= 0 or args.learning_rate <= 0 or args.gradient_accumulation_steps < 1
            or args.per_device_train_batch_size < 1 or args.lora_r < 1 or args.lora_alpha < 1
            or not 0 <= args.lora_dropout < 1 or args.model_max_length < 1
            or args.max_steps == 0):
        parser.error('invalid training values in config')
    output = args.output_dir.resolve()
    if within(output, EVAL_DATA_ROOT.resolve()):
        parser.error('output directory must be outside evaluation data')
    output.mkdir(parents=True, exist_ok=True)

    train_root = Path(os.environ.get('QWEN_TRAIN_ROOT', '/opt/qwen3-vl/qwen-vl-finetune'))
    if not train_root.is_dir():
        parser.error(f'pinned Qwen training code not found: {train_root}')
    sys.path.insert(0, str(train_root))
    sys.path.insert(0, str(train_root / 'qwenvl' / 'train'))
    import qwenvl.data as qwen_data
    qwen_data.data_dict['mmdl_custom'] = {
        'annotation_path': str(args.annotation.resolve()),
        'data_path': str(args.image_root.resolve()),
    }
    from qwenvl.train.train_qwen import train

    digest = hashlib.sha256(args.annotation.read_bytes()).hexdigest()
    (output / 'mmdl_training_config.json').write_text(json.dumps({
        'annotation_sha256': digest, 'training_examples': count,
        'model': str(args.model), 'qwen_training_commit': UPSTREAM_COMMIT,
        'epochs': args.epochs, 'learning_rate': args.learning_rate,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'min_pixels': args.min_pixels, 'max_pixels': args.max_pixels,
        'model_max_length': args.model_max_length, 'max_steps': args.max_steps,
        'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'config_sha256': file_sha256(args.config),
        'config_values': training,
    }, indent=2) + '\n')
    sys.argv = [sys.argv[0],
        '--model_name_or_path', str(args.model.resolve()),
        '--dataset_use', 'mmdl_custom', '--output_dir', str(output),
        '--bf16', '--lora_enable', 'True', '--lora_r', str(args.lora_r),
        '--lora_alpha', str(args.lora_alpha), '--lora_dropout', str(args.lora_dropout),
        '--per_device_train_batch_size', str(args.per_device_train_batch_size),
        '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
        '--gradient_checkpointing', 'True', '--learning_rate', str(args.learning_rate),
        '--num_train_epochs', str(args.epochs), '--max_steps', str(args.max_steps),
        '--min_pixels', str(args.min_pixels), '--max_pixels', str(args.max_pixels),
        '--model_max_length', str(args.model_max_length),
        '--save_strategy', 'epoch', '--save_total_limit', '1',
        '--dataloader_num_workers', '0', '--remove_unused_columns', 'False',
        '--report_to', 'none', '--logging_steps', '1',
    ]
    train(attn_implementation='sdpa')


if __name__ == '__main__':
    main()
