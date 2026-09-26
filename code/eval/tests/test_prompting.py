"""Exact team-agreed prompts and image ordering. Run directly with Python."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompting import build_mmmu_prompt  # noqa: E402


def test_multiple_choice():
    row = {'id': 'validation_Test_1', 'question_type': 'multiple-choice',
           'question': 'Which figure?', 'A': 'first', 'B': 'second', 'C': float('nan')}
    messages = build_mmmu_prompt(row, lambda _: ['one.png', 'two.png'], 262144, 2097152)
    items = messages[0]['content']
    assert [x['image'] for x in items[:-1]] == ['one.png', 'two.png']
    assert all(x['min_pixels'] == 262144 and x['max_pixels'] == 2097152 for x in items[:-1])
    assert items[-1]['text'] == (
        'Question: Which figure?\nOptions:\nA. first\nB. second\n'
        'Answer with the option letter only.')


def test_open():
    row = {'id': 'validation_Test_2', 'question_type': 'open', 'question': 'Compute the value.'}
    text = build_mmmu_prompt(row, lambda _: 'image.png', 262144, 2097152)[0]['content'][-1]['text']
    assert text == ('Compute the value.\n\nSolve the question using the image(s) when relevant. '
                    'You may reason briefly. End with exactly one line in this format: '
                    'Final answer: <your short answer>.')


if __name__ == '__main__':
    test_multiple_choice()
    test_open()
    print('PASS team MMMU prompts')
