"""Check score_mmmu.py against MMMU's own main_parse_and_eval.py, on the LLaVA-1.5-13B val outputs that
ship with the MMMU repo (needs network for a shallow fetch of that repo; CPU only).

  1. Our wrapper of the official parser must give the official script's verdict on every item the
     parser does not answer with a random guess (exit code 1 otherwise).
  2. On these short answers ("C", "A) Josef Albers"), the structured extractor should read the same
     letter as the official parser.

    python code/eval/tests/check_official_parity.py --out results/pipeline_validation/official_parity.txt
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
MMMU_REPO = 'https://github.com/MMMU-Benchmark/MMMU.git'
MMMU_COMMIT = '268471d0d488258990025331c7528359c324aa25'  # same commit as code/third_party/mmmu_official


def fetch_mmmu(dest):
    subprocess.run(['git', 'init', '-q', str(dest)], check=True)
    subprocess.run(['git', '-C', str(dest), 'fetch', '-q', '--depth', '1', MMMU_REPO, MMMU_COMMIT], check=True)
    subprocess.run(['git', '-C', str(dest), 'checkout', '-q', 'FETCH_HEAD'], check=True)
    return dest / 'mmmu'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', help='also write the report to this file')
    ap.add_argument('--keep', action='store_true', help='keep the temporary work directory')
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix='mmmu_parity_'))
    mmmu = fetch_mmmu(work / 'MMMU')
    outputs = mmmu / 'example_outputs' / 'llava1.5_13b'

    # Official script, on a copy: it writes parsed_output.json next to each output.json.
    shutil.copytree(outputs, work / 'official')
    subprocess.run([sys.executable, 'main_parse_and_eval.py', '--path', str(work / 'official'), '--subject', 'ALL'],
                   cwd=mmmu, check=True, stdout=subprocess.DEVNULL)
    official = {s['id']: s for f in (work / 'official').glob('*/parsed_output.json') for s in json.load(open(f))}

    # The same responses through score_mmmu.py.
    llava = {s['id']: s for f in sorted(outputs.glob('*/output.json')) for s in json.load(open(f))}
    run = work / 'llava_run'
    run.mkdir()
    with open(run / 'predictions.jsonl', 'w') as f:
        for s in llava.values():
            f.write(json.dumps({'id': s['id'], 'response': s['response'], 'finish_reason': 'stop', 'n_output_tokens': 0}) + '\n')
    subprocess.run([sys.executable, str(EVAL_DIR / 'score_mmmu.py'), str(run)], check=True, stdout=subprocess.DEVNULL)
    ours = {r['id']: r for r in csv.DictReader(open(run / 'per_sample.csv'))}
    scores = json.load(open(run / 'scores.json'))

    guessed = {i for i, r in ours.items() if r['official_random_fallback'] == '1'}
    mismatch = [i for i in official if i not in guessed and (official[i]['judge'] == 'Correct') != (ours[i]['official_correct'] == '1')]
    mc = [i for i in ours if ours[i]['question_type'] == 'multiple-choice' and i not in guessed]
    differ = [i for i in mc if ours[i]['pred'] != ours[i]['official_parsed']]
    open_ids = [i for i in ours if ours[i]['question_type'] != 'multiple-choice']

    lines = [
        f'MMMU-Benchmark/MMMU@{MMMU_COMMIT[:7]} example_outputs/llava1.5_13b: {len(llava)} val responses',
        f'official script accuracy        {sum(s["judge"] == "Correct" for s in official.values()) / len(official):.4f}',
        f'score_mmmu.py official accuracy {scores["official"]["overall"]["acc"]:.4f}',
        f'verdict mismatches (excluding {len(guessed)} random guesses): {len(mismatch)} {mismatch}',
        f'random guesses with a different verdict (RNG order): '
        f'{sum((official[i]["judge"] == "Correct") != (ours[i]["official_correct"] == "1") for i in guessed)}',
        f'MC letters, structured vs official (official not guessing): {len(mc) - len(differ)}/{len(mc)} identical',
    ]
    for i in differ:
        lines.append(f'  {i}: structured={ours[i]["pred"] or None} ({ours[i]["method"]}), official={ours[i]["official_parsed"]}, '
                     f'response={llava[i]["response"][:70]!r}')
    lines.append(f'open questions correct: structured {sum(ours[i]["correct"] == "1" for i in open_ids)}/{len(open_ids)}, '
                 f'official {sum(ours[i]["official_correct"] == "1" for i in open_ids)}/{len(open_ids)}')
    lines.append(f'overall accuracy: structured {scores["structured"]["overall"]["acc"]:.4f}, '
                 f'official {scores["official"]["overall"]["acc"]:.4f}')

    report = '\n'.join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + '\n')
    if not args.keep:
        shutil.rmtree(work)
    sys.exit(1 if mismatch else 0)


if __name__ == '__main__':
    main()
