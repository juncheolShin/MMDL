"""Guard the shared experiment profile against accidental parameter drift."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configuration import DEFAULT_CONFIG, EVALUATION_TYPES, TRAINING_TYPES, load_section  # noqa: E402


evaluation = load_section(DEFAULT_CONFIG, 'evaluation', EVALUATION_TYPES)
training = load_section(DEFAULT_CONFIG, 'training', TRAINING_TYPES)
assert evaluation['temperature'] == 0.0
assert evaluation['seed'] == 0
assert evaluation['max_new_tokens'] == 512
assert evaluation['min_pixels'] == 256 * 32 * 32
assert evaluation['max_pixels'] == 2048 * 32 * 32
assert training['lora_r'] > 0

with tempfile.TemporaryDirectory() as directory:
    invalid = Path(directory) / 'invalid.toml'
    invalid.write_text(DEFAULT_CONFIG.read_text().replace('seed = 0', 'sead = 0'))
    try:
        load_section(invalid, 'evaluation', EVALUATION_TYPES)
    except ValueError as exc:
        assert 'sead' in str(exc) and 'seed' in str(exc)
    else:
        raise AssertionError('a misspelled parameter was accepted')

print('PASS shared experiment configuration')
