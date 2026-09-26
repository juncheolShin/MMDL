"""Team-agreed MMMU prompts from the 2026-09-25 meeting notes."""
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


def build_mmmu_prompt(line, dump_image, min_pixels, max_pixels):
    """Preserve image order; use distinct MC and open-answer instructions."""
    question = str(line['question']).strip()
    if line['question_type'] == 'multiple-choice':
        options = [(letter, line[letter]) for letter in string.ascii_uppercase
                   if letter in line and _has_option(line[letter])]
        if not options:
            raise ValueError(f"multiple-choice question has no options: {line['id']}")
        option_lines = '\n'.join(f'{letter}. {value}' for letter, value in options)
        prompt = (f'Question: {question}\nOptions:\n{option_lines}\n'
                  'Answer with the option letter only.')
    else:
        prompt = (f'{question}\n\n'
                  'Solve the question using the image(s) when relevant. You may reason briefly. '
                  'End with exactly one line in this format: Final answer: <your short answer>.')

    paths = dump_image(line)
    paths = paths if isinstance(paths, list) else [paths]
    content = [
        {'type': 'image', 'image': path,
         'min_pixels': min_pixels, 'max_pixels': max_pixels}
        for path in paths
    ]
    content.append({'type': 'text', 'text': prompt})
    return [{'role': 'user', 'content': content}]
