"""Response-format cases for answer_extraction. Run: python code/eval/tests/test_answer_extraction.py (or pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from answer_extraction import answer_type_of, extract_choice, extract_open, open_is_correct  # noqa: E402

ART8 = {'A': 'Herne Bay, Kent', 'B': 'St Ives, Cornwall', 'C': 'Aldeburgh, Suffolk', 'D': 'Sandown, Isle of Wight'}
MONEY = {'A': '$63,020', 'B': '$58,410', 'C': '$71,320', 'D': '$77,490'}

# (name, response, options, expected letter or None)
MC_CASES = [
    ('slide p.10', 'The correct answer is: **C. Aldeburgh, Suffolk**\n\n**Explanation:**\n"A Conversation with the Sea" is a '
     'large-scale sculpture by artist Antony Gormley, located on the pebble beach in **Aldeburgh, Suffolk**, England. '
     'A giant scallop.\n\nThus, the correct and only correct option is **C. Aldeburgh, Suffolk**.', ART8, 'C'),
    ('slide p.11', 'The correct answer is C. Aldeburgh, Suffolk...', ART8, 'C'),
    ('bold answer + sentence-initial A', '**Answer: C**\n\nThe sculpture stands on the beach in Aldeburgh. A scallop-shaped '
     'steel work, it honours Britten.', ART8, 'C'),
    ('mentions a rejected option', 'The answer is C. It is not in St Ives, Cornwall, which is on the west coast.', ART8, 'C'),
    ('final line Answer:', 'The sculpture stands on the beach. A scallop-shaped steel work.\n\nAnswer: C', ART8, 'C'),
    ('letter alone', 'C', ART8, 'C'),
    ('letter with option text', 'C. Aldeburgh, Suffolk', ART8, 'C'),
    ('parenthesised', 'The answer is (B).', ART8, 'B'),
    ('answer is A because', 'The answer is A because Herne Bay hosts it.', ART8, 'A'),
    ('article after statement is not a letter', 'The answer is A coastal town: Aldeburgh, Suffolk.', ART8, 'C'),
    ('statement on next line', 'To determine the location, recall the sculpture.\n\n### Final Answer:\n\n**D. Sandown, Isle of Wight**', ART8, 'D'),
    ('boxed', 'Computing the total gives 77,490.\n\n$\\boxed{D}$', MONEY, 'D'),
    ('boxed with text macro', 'Final answer: $\\boxed{\\text{B}}$', MONEY, 'B'),
    ('answer stated as option text', 'After adding the rows, the correct answer is $77,490.', MONEY, 'D'),
    ('revision, last statement wins', 'The answer is B. Wait, re-checking the total: the correct answer is D.', MONEY, 'D'),
    ('option X is correct', 'Option A is incorrect because ... Option C is correct since the sums match.', MONEY, 'C'),
    ('option X is not correct', 'Option B is not correct. Option C is correct.', MONEY, 'C'),
    ('X is the correct answer', 'Checking each value, D is the correct answer.', MONEY, 'D'),
    ('marked analysis lines', '- A. Herne Bay, Kent: Incorrect\n- B. St Ives, Cornwall: Incorrect\n- C. Aldeburgh, Suffolk: Correct ✅\n'
     '- D. Sandown, Isle of Wight: Incorrect', ART8, 'C'),
    ('only option text', 'The sculpture is on the shingle beach at Aldeburgh, Suffolk, north of the town centre.', ART8, 'C'),
    ('echoed prompt is not a statement', 'Please select the correct answer from the options above. I cannot see the image.', ART8, None),
    ('no answer', 'I am not able to determine this from the image.', ART8, None),
    ('invalid letter', 'The answer is E.', ART8, None),
    ('echoed option list (LLaVA, Pharmacy_27)', '(A) 20.0%,0.50 mol\n(B) 30.0%,0.50 mol\n(C) 40.0%,0.50 mol',
     {'A': '20.0%,0.50 mol', 'B': '30.0%,0.50 mol', 'C': '40.0%,0.50 mol'}, None),
    ('I would choose', 'Given the coastline shape, I would choose B.', ART8, 'B'),
    ('answer is not X', 'The answer is not A; the sculpture is in Aldeburgh, Suffolk.', ART8, 'C'),
    ('answer line then option walk-through', 'C. Aldeburgh, Suffolk\n\nA. Herne Bay, Kent is in the south-east.\nB. St Ives is in Cornwall.', ART8, 'C'),
]

# (name, response, ground truth, expected correct)
OPEN_CASES = [
    ('number, statement', 'Using the summation, the deflection is found.\n\nThe answer is 1.06 in.', '1.06', True),
    ('number, boxed', 'Therefore $\\boxed{12.97}$', '12.97', True),
    ('number, 2-decimal rounding', 'The final answer is 0.99648.', '0.9965', True),
    ('number, wrong', 'The answer is 0.98 in.', '1.06', False),
    ('intermediate equals GT, final differs', 'At t = 2, x = 0.9965 initially.\nAfter four terms, the sum is 0.9812.', '0.9965', False),
    ('condition before result', 'Final answer: at t = 2, the sum is 0.9965', '0.9965', True),
    ('fraction latex', 'The probability that individual #7 is affected is $\\frac{1}{64}$.', '1/64', True),
    ('fraction as decimal', 'So the probability is 0.015625.', '1/64', True),
    ('alternatives', 'The bottom slides at 24/7 ≈ 3.43 ft/s.', ['24/7', '3.429'], True),
    ('thousands separators', 'The company will raise $7,243,000 in the auction.', '7243000', True),
    ('million word', 'Answer: about $2 million', '2000000', True),
    ('percent vs ratio', 'The probability is 10%.', '0.10', True),
    ('negative degrees', 'The phase of $I_{cC}$ is $-120^\\circ$.', '-120', True),
    ('unit exponent ignored', 'Answer: 12.97 m/s^2', '12.97', True),
    ('scientific notation', 'The answer is 2 \\times 10^{6}.', '2000000', True),
    ('parenthetical ignored', 'The answer is 1.06 in (about 27 mm).', '1.06', True),
    ('label, labeled arrow', 'The arrow labeled **C** points to the hydrogen bond between the strands.', 'C', True),
    ('label, point', 'Point A lies where the force is negative. Thus, point A shows the repulsive nature, unlike B, C and D.', 'A', True),
    ('label, not the article', 'A hydrogen bond forms between strands. The answer is B.', 'C', False),
    ('label, statement', 'The regioselective step is reaction B.\n\nAnswer: B', 'B', True),
    ('label, lower case after noun (LLaVA, Pharmacy_19)', 'Step b', 'B', True),
    ('label, bare letter', 'a', 'A', True),
    ('label, a word is not a label', 'Right', 'C', False),
    ('label, nothing to read', 'I cannot determine this.', 'A', False),
    ('text, statement', 'The answer is: **Transformation**', 'Transformation', True),
    ('text, answer first', 'The place described in the image is Tampa, Florida. It is a coastal city.', ['Tampa', 'Florida'], True),
    ('text, latex formula', 'The formula of the compound is therefore $MgS$.', ['$MgS$', 'MgS'], True),
    ('text, wrong', 'The answer is conjugation.', 'Transformation', False),
]


def test_multiple_choice():
    failures = []
    for name, response, options, expected in MC_CASES:
        got = extract_choice(response, options)
        if got.pred != expected:
            failures.append(f'{name}: expected {expected}, got {got.pred} via {got.method}')
    assert not failures, '\n'.join(failures)


def test_open():
    failures = []
    for name, response, gt, expected in OPEN_CASES:
        atype = answer_type_of(gt)
        got = extract_open(response, atype)
        if open_is_correct(got, gt, atype) != expected:
            failures.append(f'{name}: expected {expected}, got pred={got.pred} via {got.method} ({atype})')
    assert not failures, '\n'.join(failures)


def test_answer_types():
    assert answer_type_of('C') == 'label'
    assert answer_type_of('1/64') == 'number'
    assert answer_type_of(['24/7', '3.429']) == 'number'
    assert answer_type_of('-120') == 'number'
    assert answer_type_of(['$MgS$', 'MgS']) == 'text'
    assert answer_type_of('trans-1-Chloro-4-methylcyclohexane') == 'text'


if __name__ == '__main__':
    ok = True
    for fn in (test_answer_types, test_multiple_choice, test_open):
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except AssertionError as e:
            ok = False
            print(f'FAIL {fn.__name__}\n{e}')
    sys.exit(0 if ok else 1)
