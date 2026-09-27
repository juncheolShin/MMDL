"""Fixed MC/open prompts documented in reports/mmmu_baseline.md."""
import string


def _has_option(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    try:
        return bool(value == value)  # NaN is not a real option.
    except (TypeError, ValueError):
        return False


def build_mmmu_prompt(line, image_paths, min_pixels, max_pixels):
    """Preserve image order; use distinct MC and open-answer instructions."""
    question = str(line['question']).strip()
    if line['question_type'] == 'multiple-choice':
        options = [(letter, line[letter]) for letter in string.ascii_uppercase
                   if letter in line and _has_option(line[letter])]
        if not options:
            raise ValueError(f"multiple-choice question has no options: {line['id']}")
        option_lines = '\n'.join(f'{letter}. {value}' for letter, value in options)
        instruction = 'Answer with the option letter only.'
        prompt = f'Question: {question}\nOptions:\n{option_lines}\n{instruction}'
    else:
        prompt = (f'{question}\n\n'
                  'Solve the question using the image(s) when relevant.'
                  'Answer with exactly one line in this format: Answer: <your short answer>.'
                  'You must not show reasoning.')

    content = [
        {'type': 'image', 'image': path,
         'min_pixels': min_pixels, 'max_pixels': max_pixels}
        for path in image_paths
    ]
    content.append({'type': 'text', 'text': prompt})
    return [{'role': 'user', 'content': content}]
