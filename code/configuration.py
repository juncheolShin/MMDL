"""Read versioned experiment settings with strict key and type checks."""
import hashlib
import tomllib
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'rtx4090.toml'

EVALUATION_TYPES = {
    'temperature': float, 'top_p': float, 'top_k': int,
    'repetition_penalty': float, 'presence_penalty': float, 'seed': int,
    'max_new_tokens': int, 'max_model_len': int,
    'min_pixels': int, 'max_pixels': int,
    'gpu_memory_utilization': float, 'max_num_seqs': int,
    'tensor_parallel_size': int, 'max_images_per_prompt': int,
}
TRAINING_TYPES = {
    'epochs': float, 'learning_rate': float,
    'gradient_accumulation_steps': int, 'per_device_train_batch_size': int,
    'lora_r': int, 'lora_alpha': int, 'lora_dropout': float,
    'max_steps': int, 'min_pixels': int, 'max_pixels': int,
    'model_max_length': int,
}


def load_section(path, section, types):
    path = Path(path)
    with path.open('rb') as stream:
        profile = tomllib.load(stream)
    if section not in profile or not isinstance(profile[section], dict):
        raise ValueError(f'{path}: missing [{section}] section')
    values = profile[section]
    extra = set(values) - set(types)
    missing = set(types) - set(values)
    if extra or missing:
        raise ValueError(f'{path} [{section}]: extra={sorted(extra)}, missing={sorted(missing)}')
    for key, expected in types.items():
        if type(values[key]) is not expected:
            raise TypeError(f'{path} [{section}].{key} must be {expected.__name__}')
    return values


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
