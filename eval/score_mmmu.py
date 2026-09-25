"""Score runs/<run>/predictions.jsonl: s = (1/N) sum_n g(r_gt,n, r̂_n), g = exact match after normalization.

Two extractors turn each response r_n into r̂_n:
  structured  answer_extraction.py (primary). Reads the response's final answer statement, so long
              explanatory answers are handled; never guesses (unparsed = wrong, and counted).
  official    MMMU's parse_multi_choice_response / parse_open_response + evaluate, kept as a reference
              point. It targets short "letter only" answers and guesses randomly when it finds nothing.

Writes scores.json, per_sample.csv and mc_disagreements.csv (MC items where the two extractors differ,
for manual auditing).

    python eval/score_mmmu.py runs/base_greedy
"""
import argparse
import collections
import csv
import json
import random
import string
from pathlib import Path

import pandas as pd

import mmmu_data
from answer_extraction import answer_type_of, extract_choice, extract_open, open_is_correct


class RecordingRandom:
    """Replaces `random` inside the MMMU parser so its random-guess fallbacks can be counted.

    The parser seeds the global RNG with 42 at import and only ever calls random.choice, so
    Random(42) yields the same guesses.
    """

    def __init__(self, seed=42):
        self._rng = random.Random(seed)
        self.calls = 0

    def choice(self, seq):
        self.calls += 1
        return self._rng.choice(seq)


def subject_of(sample_id):
    return '_'.join(sample_id.split('_')[1:-1])


def summarize(correct_by_id, answers, domains):
    def acc(ids):
        hits = [correct_by_id[i] for i in ids]
        return {'n': len(hits), 'correct': sum(hits), 'acc': sum(hits) / len(hits) if hits else None}

    ids = sorted(correct_by_id)
    by_subject = collections.defaultdict(list)
    for i in ids:
        by_subject[subject_of(i)].append(i)
    return {
        'overall': acc(ids),
        'multiple_choice': acc([i for i in ids if answers[i]['question_type'] == 'multiple-choice']),
        'open': acc([i for i in ids if answers[i]['question_type'] != 'multiple-choice']),
        'by_domain': {d: acc([i for s in subs for i in by_subject.get(s, [])])
                      for d, subs in domains.items() if any(s in by_subject for s in subs)},
        'by_subject': {s: acc(v) for s, v in sorted(by_subject.items())},
    }


def score_structured(preds, answers, options_by_id):
    rows = {}
    for p in preds:
        gt = answers[p['id']]
        if gt['question_type'] == 'multiple-choice':
            atype = 'choice'
            ex = extract_choice(p['response'], options_by_id[p['id']])
            correct = ex.pred == gt['ground_truth']
            pred = ex.pred
        else:
            atype = answer_type_of(gt['ground_truth'])
            ex = extract_open(p['response'], atype)
            correct = open_is_correct(ex, gt['ground_truth'], atype)
            pred = f'{ex.pred[0]:g}{"%" if ex.pred[1] else ""}' if atype == 'number' and ex.pred else ex.pred
        rows[p['id']] = {'answer_type': atype, 'pred': pred, 'method': ex.method, 'conflict': ex.conflict,
                         'correct': bool(correct), 'span': ex.span}
    return rows


def score_official(preds, answers, options_by_id):
    eu = mmmu_data.load_mmmu_official('eval_utils')
    rng = RecordingRandom()
    eu.random = rng
    rows = {}
    for p in sorted(preds, key=lambda x: x['id']):
        gt = answers[p['id']]
        before = rng.calls
        if gt['question_type'] == 'multiple-choice':
            index2ans = options_by_id[p['id']]
            parsed = eu.parse_multi_choice_response(p['response'], list(index2ans), index2ans)
        else:
            parsed = eu.parse_open_response(p['response'])
        sample = {'id': p['id'], 'question_type': gt['question_type'], 'answer': gt['ground_truth'], 'parsed_pred': parsed}
        judge, _ = eu.evaluate([sample])
        rows[p['id']] = {'parsed': parsed, 'random_fallback': rng.calls > before, 'correct': judge[p['id']] == 'Correct'}
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    preds = [json.loads(l) for l in open(run_dir / 'predictions.jsonl')]
    answers = json.load(open(mmmu_data.ANSWER_DICT_PATH))
    unknown = [p['id'] for p in preds if p['id'] not in answers]
    if unknown:
        raise SystemExit(f'{len(unknown)} ids are not in the official val answer dict, e.g. {unknown[:3]}')
    meta = mmmu_data.load_mmmu('all')
    options_by_id = {
        r['id']: {c: r[c] for c in string.ascii_uppercase if c in meta.columns and not pd.isna(r[c])}
        for _, r in meta.iterrows()
    }
    domains = mmmu_data.load_mmmu_official('data_utils').DOMAIN_CAT2SUB_CAT

    st = score_structured(preds, answers, options_by_id)
    off = score_official(preds, answers, options_by_id)
    ids = sorted(st)
    mc_ids = [i for i in ids if st[i]['answer_type'] == 'choice']
    disagree = [i for i in mc_ids if st[i]['pred'] is not None and st[i]['pred'] != off[i]['parsed']]
    n_tokens = [p['n_output_tokens'] for p in preds]

    scores = {
        'run': run_dir.name,
        'structured': {
            **summarize({i: st[i]['correct'] for i in ids}, answers, domains),
            'unparsed': {'multiple_choice': sum(st[i]['pred'] is None for i in mc_ids),
                         'open': sum(st[i]['pred'] is None for i in ids) - sum(st[i]['pred'] is None for i in mc_ids)},
            'conflicting_statements': sum(st[i]['conflict'] for i in ids),
            'method_counts': dict(collections.Counter(f"{st[i]['answer_type']}:{st[i]['method']}" for i in ids)),
        },
        'official': {
            **summarize({i: off[i]['correct'] for i in ids}, answers, domains),
            'random_fallbacks': sum(off[i]['random_fallback'] for i in ids),
        },
        'mc_extractor_disagreements': len(disagree),
        'generation': {
            'finish_reason': dict(collections.Counter(p['finish_reason'] for p in preds)),
            'output_tokens_mean': sum(n_tokens) / len(n_tokens),
            'output_tokens_max': max(n_tokens),
        },
    }
    (run_dir / 'scores.json').write_text(json.dumps(scores, indent=2, ensure_ascii=False))

    by_id = {p['id']: p for p in preds}
    with open(run_dir / 'per_sample.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'subject', 'question_type', 'answer_type', 'gt', 'pred', 'method', 'conflict', 'correct',
                    'official_parsed', 'official_correct', 'official_random_fallback', 'finish_reason',
                    'n_output_tokens', 'span'])
        for i in ids:
            s, o, p = st[i], off[i], by_id[i]
            w.writerow([i, subject_of(i), answers[i]['question_type'], s['answer_type'], answers[i]['ground_truth'],
                        s['pred'], s['method'], int(s['conflict']), int(s['correct']), o['parsed'], int(o['correct']),
                        int(o['random_fallback']), p['finish_reason'], p['n_output_tokens'], s['span'][:200]])
    with open(run_dir / 'mc_disagreements.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'gt', 'structured', 'method', 'official', 'official_random_fallback', 'response'])
        for i in disagree:
            w.writerow([i, answers[i]['ground_truth'], st[i]['pred'], st[i]['method'], off[i]['parsed'],
                        int(off[i]['random_fallback']), by_id[i]['response']])

    s, o = scores['structured'], scores['official']
    print(f"[{run_dir.name}] structured: overall {s['overall']['acc']:.4f} ({s['overall']['correct']}/{s['overall']['n']}), "
          f"MC {s['multiple_choice']['acc']:.4f} ({s['multiple_choice']['correct']}/{s['multiple_choice']['n']}), "
          f"open {s['open']['correct']}/{s['open']['n']}; unparsed MC {s['unparsed']['multiple_choice']}, "
          f"open {s['unparsed']['open']}; conflicting statements {s['conflicting_statements']}")
    print(f"[{run_dir.name}] official:   overall {o['overall']['acc']:.4f}, MC {o['multiple_choice']['acc']:.4f}, "
          f"open {o['open']['correct']}/{o['open']['n']}; random fallbacks {o['random_fallbacks']}; "
          f"MC disagreements with structured {len(disagree)} (see mc_disagreements.csv)")
    for d, r in s['by_domain'].items():
        print(f'  {d:32s} n={r["n"]:3d} acc={r["acc"]:.3f}')
    g = scores['generation']
    print(f"  finish_reason {g['finish_reason']}, output tokens mean {g['output_tokens_mean']:.0f} max {g['output_tokens_max']}")


if __name__ == '__main__':
    main()
