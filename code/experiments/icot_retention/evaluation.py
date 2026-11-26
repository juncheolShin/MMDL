"""Select using external validation only, then replay the team's MMMU scorer on paired responses."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import string
import sys
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'icot/scoring'))
from score_mmmu import score_structured

STRENGTHS = ('0', '0.1', '0.25', '0.5')
MODELS = ('teacher', *STRENGTHS)
STEPS = (500, 1000, 1500, 2000, 2370)
REVISION = '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_rows(path, rows):
    with path.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode())


def read_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def by_id(rows, count):
    result = {row['id']: row for row in rows}
    check(len(rows) == len(result) == count, f'Expected {count} unique questions')
    return result


def paired(before, after):
    check(before.keys() == after.keys(), 'Cannot pair different question IDs')
    return {
        'correct': sum(after.values()),
        'gained': sum(after[key] and not before[key] for key in before),
        'lost': sum(before[key] and not after[key] for key in before),
    }


def external_rows(labels, teacher, predictions):
    """Join by question ID, never by the incidental order of a worker output."""
    ids = labels.keys()
    check(teacher.keys() == ids and all(run.keys() == ids for run in predictions.values()), 'External ID mismatch')
    rows = []
    for key, row in labels.items():
        ref = teacher[key]
        letters = string.ascii_uppercase[:len(row['choices'])]
        check(ref['gold'] == row['answer'] and ref['source'] == row['source'], f'Mismatched teacher label: {key}')
        answers = {'teacher': ref['prediction']}
        answers.update({name: run[key] for name, run in predictions.items()})
        check(row['answer'] in tuple(letters) and all(v in tuple(letters) for v in answers.values()),
              f'Invalid answer letter: {key}')
        rows.append(dict(id=key, gold=row['answer'], choices=letters, predictions=answers))
    return rows


def external_scores(rows, columns, n):
    by_id(rows, n)
    results = {name: {} for name in ('teacher', *columns)}
    for row in rows:
        pred = row['predictions']
        letters = list(row['choices'])
        check(set(pred) == results.keys() and row['gold'] in letters and all(p in letters for p in pred.values()),
              f"Invalid predictions at {row['id']}")
        for name in results:
            results[name][row['id']] = pred[name] == row['gold']
    return {name: paired(results['teacher'], correct) for name, correct in results.items()}


def select(public, emma):
    columns = [f'{w}@{s}' for w in STRENGTHS for s in STEPS]
    pub = external_scores(public, columns, 1044)
    expert = external_scores(emma, STRENGTHS, 320)
    arms = {}
    for w in STRENGTHS:
        step = max(STEPS, key=lambda s: (pub[f'{w}@{s}']['correct'], -pub[f'{w}@{s}']['lost'], -s))
        arms[w] = {'step': step, 'public': pub[f'{w}@{step}'], 'emma': expert[w]}

    def ranking(w):
        # Comparing integer cross-products implements the average of two accuracies exactly.
        p, e = arms[w]['public'], arms[w]['emma']
        return (p['correct'] * 320 + e['correct'] * 1044,
                min((p['gained'] - p['lost']) * 320, (e['gained'] - e['lost']) * 1044),
                -(p['lost'] * 320 + e['lost'] * 1044), -float(w))
    return {'teacher': {'public': pub['teacher'], 'emma': expert['teacher']}, 'arms': arms,
            'selected_strength': max(STRENGTHS[1:], key=ranking), 'best_including_control': max(STRENGTHS, key=ranking)}


def teacher_decisions(rows):
    by_id(rows, 18956)
    counts = {}
    for row in rows:
        probs = row['probabilities']
        check(2 <= len(probs) <= 26 and all(math.isfinite(v) and 0 <= v <= 1 for v in probs)
              and math.isclose(sum(probs), 1, abs_tol=1e-5), f"Invalid teacher probabilities: {row['id']}")
        prediction = string.ascii_uppercase[max(range(len(probs)), key=probs.__getitem__)]
        check(row['prediction'] == prediction and row['gold'] in list(string.ascii_uppercase[:len(probs)])
              and row['correct'] is (prediction == row['gold']), f"Invalid teacher decision: {row['id']}")
        group = counts.setdefault(row['source'], {'count': 0, 'correct': 0})
        group['count'] += 1
        group['correct'] += int(row['correct'])
    return {'count': len(rows), 'correct': sum(g['correct'] for g in counts.values()), 'by_source': counts}


def archive_teacher(teacher_scores, train_labels, dest):
    rows = read_rows(teacher_scores / 'train.jsonl')
    labels = by_id(read_rows(train_labels), 18956)
    result = teacher_decisions(rows)
    check(all(r['id'] in labels and (r['gold'], r['source']) == (labels[r['id']]['answer'], labels[r['id']]['source'])
              and len(r['probabilities']) == len(labels[r['id']]['choices']) for r in rows), 'Training teacher label mismatch')
    write_rows(dest / 'teacher_train.jsonl.gz', rows)
    write_json(dest / 'teacher_train.json', {**result, 'sha256': digest(dest / 'teacher_train.jsonl.gz')})


def lock_selection(data, teacher_scores, models, expert_files, dest):
    """Write the external predictions and lock the selected checkpoint before opening MMMU."""
    data = Path(data)
    public = by_id(read_rows(data / 'public/dev.jsonl'), 1044)
    emma = by_id(read_rows(data / 'emma/dev.jsonl'), 320)
    pub_pred, expert_pred, best_steps, weights = {}, {}, {}, {}
    expert_data_sha = digest(data / 'emma/dev.jsonl')
    for name in MODELS:
        model = models[name]
        weights[name] = {p.name: digest(p) for p in sorted(model.glob('*.safetensors'))}
        check(bool(weights[name]), f'Missing exported model: {model}')
        score = read(expert_files[name])
        # The expert runner binds its predictions to the actual model shards.
        check(score['bindings']['shards_sha256'] == weights[name], f'EMMA model mismatch: {name}')
        bindings = score['bindings']
        check(bindings.get('data_sha256', bindings.get('dev_sha256')) == expert_data_sha, f'EMMA data mismatch: {name}')
        items = by_id(score['items'], 320)
        check(items.keys() == emma.keys() and all(r['gold'] == emma[k]['answer'] for k, r in items.items()),
              f'Incorrect EMMA labels for {name}')
        if name == 'teacher':
            expert_teacher = items
            continue
        expert_pred[name] = {k: r['prediction'] for k, r in items.items()}
        best_steps[name] = read(model / 'trainer_state.json')['best_global_step']
        for step in STEPS:
            checkpoint = read(model / 'retention_dev' / f'step-{step}.json')
            check(checkpoint['step'] == step, 'Mismatched public checkpoint')
            pub_pred[f'{name}@{step}'] = {k: v['pred'] for k, v in by_id(checkpoint['items'], 1044).items()}
    teacher_cache = read(teacher_scores / 'manifest.json')
    bindings = teacher_cache['bindings']
    check(all(bindings['model_sha256'].get(k) == v for k, v in weights['teacher'].items()), 'Training teacher model mismatch')
    for split in ('train', 'dev'):
        check(bindings['input_sha256'][split] == digest(data / 'public' / f'{split}.jsonl')
              and teacher_cache['outputs'][split]['sha256'] == digest(teacher_scores / f'{split}.jsonl'),
              f'Training teacher data mismatch: {split}')
    pub_rows = external_rows(public, by_id(read_rows(teacher_scores / 'dev.jsonl'), 1044), pub_pred)
    expert_rows = external_rows(emma, expert_teacher, expert_pred)
    selection = select(pub_rows, expert_rows)
    check(all(selection['arms'][w]['step'] == best_steps[w] for w in STRENGTHS), 'Exported checkpoint differs from selection')
    dest.mkdir(parents=True, exist_ok=True)
    archive_teacher(teacher_scores, data / 'public/train.jsonl', dest)
    write_rows(dest / 'public.jsonl.gz', pub_rows)
    write_rows(dest / 'emma.jsonl.gz', expert_rows)
    write_json(dest / 'selection.json', {**selection, 'models_sha256': weights,
               'inputs_sha256': {n: digest(dest / n) for n in ('public.jsonl.gz', 'emma.jsonl.gz')}})


def mmmu_scores(rows):
    items = by_id(rows, 900)
    answers = {key: r['label'] for key, r in items.items()}
    check(sum(v['question_type'] == 'multiple-choice' for v in answers.values()) == 847
          and sum(v['question_type'] == 'open' for v in answers.values()) == 53, 'Unexpected MMMU split')
    correct = {}
    for name in MODELS:
        preds = [dict(id=r['id'], response=r['responses'][name]) for r in rows]
        scored = score_structured(preds, answers)
        correct[name] = {key: scored[key]['correct'] for key in items}
    return {name: paired(correct['teacher'], values) for name, values in correct.items()}


def verify_external(dest):
    teacher_file = dest / 'teacher_train.jsonl.gz'
    teacher = teacher_decisions(read_rows(teacher_file))
    check(read(dest / 'teacher_train.json') == {**teacher, 'sha256': digest(teacher_file)}, 'Training teacher record differs')
    locked = read(dest / 'selection.json')
    public, emma = (dest / n for n in ('public.jsonl.gz', 'emma.jsonl.gz'))
    for p in (public, emma):
        check(digest(p) == locked['inputs_sha256'][p.name], f'Selection evidence hash mismatch: {p.name}')
    for k, v in select(read_rows(public), read_rows(emma)).items():
        check(locked[k] == v, f'Cannot reproduce external selection: {k}')
    return locked


def normalized_protocol(cfg):
    check(cfg['dataset']['repo_id'] == 'MMMU/MMMU' and cfg['dataset']['revision'] == REVISION
          and cfg['dataset']['split'] == 'validation' and len(cfg['dataset']['subjects']) == 30 and cfg['n_samples'] == 900,
          'Unexpected MMMU protocol or coverage')
    per_subject = cfg['dataset']['rows_by_subject']
    check(len(per_subject) == 30 and set(per_subject) == set(cfg['dataset']['subjects'])
          and all(n == 30 for n in per_subject.values()), 'Incomplete MMMU subject coverage')
    for rel, sha in read(ROOT / 'icot/evidence/evaluation_498.json')['data_and_vendored_sha256'].items():
        if rel != 'evaluation_data.json':
            check(sha == cfg['data_and_vendored_sha256'][rel] == digest(ROOT / rel), f'Changed evaluation helper: {rel}')
    protocol = {'revision': REVISION, 'split': 'validation', 'config_sha256': cfg['config_sha256'],
            'pipeline_sha256': cfg['pipeline_sha256'], 'sampling': cfg['sampling_params'],
            'engine': {k: v for k, v in cfg['llm_kwargs'].items() if k != 'model'},
            'pixels': [cfg['args'][k] for k in ('min_pixels', 'max_pixels')]}

    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol):
    config_path = ROOT / 'configs/evaluation.toml'
    config = tomllib.loads(config_path.read_text())['evaluation']
    pipeline = {}
    for name in ('answer_extraction', 'check_prompts', 'infer_mmmu', 'mmmu_data', 'prompting', 'score_mmmu'):
        folder = 'icot/scoring' if name in ('answer_extraction', 'score_mmmu') else 'code/eval'
        pipeline[f'code/eval/{name}.py'] = digest(ROOT / folder / f'{name}.py')
    sampling = {k: config[k] for k in ('temperature', 'top_p', 'top_k', 'repetition_penalty', 'presence_penalty')}
    sampling['max_tokens'] = config['max_new_tokens']
    engine = {k: config[k] for k in ('tensor_parallel_size', 'gpu_memory_utilization', 'max_model_len', 'max_num_seqs', 'seed')}
    engine.update(trust_remote_code=True, limit_mm_per_prompt={'image': config['max_images_per_prompt']})
    check(protocol == dict(revision=REVISION, split='validation', config_sha256=digest(config_path),
                           pipeline_sha256=pipeline, sampling=sampling, engine=engine,
                           pixels=[config['min_pixels'], config['max_pixels']]), 'Pinned ICoT evaluation protocol differs')


def verify_packaged_weights(locked):
    weights = read(HERE / 'weights/manifest.json')
    base = {name: entry['sha256'] for name, entry in weights['shards'].items()}
    check(base == locked['models_sha256']['teacher'], 'Packaged teacher differs from measured teacher')
    check(weights['selected_strength'] == locked['selected_strength'], 'Packaged selection differs')
    for name in STRENGTHS:
        arm = weights['candidates'][name]
        check(arm['target_sha256'] == locked['models_sha256'][name] and arm['step'] == locked['arms'][name]['step'],
              f'Packaged model differs from evaluated checkpoint: {name}')


def collect_mmmu(dest, directories):
    locked = verify_external(dest)
    rows, protocol, expected = {}, None, {}
    for name in MODELS:
        folder = directories[name]
        cfg = read(folder / 'run_config.json')
        current = normalized_protocol(cfg)
        check(protocol is None or protocol == current, f'Evaluation protocol mismatch: {name}')
        protocol = current
        fingerprint = {k: h for k, h in cfg['model_fingerprint_sha256'].items() if k.endswith('.safetensors')}
        check(fingerprint == locked['models_sha256'][name], f'MMMU evaluated a different model: {name}')
        labels = folder / 'evaluation_data.json'
        check(digest(labels) == cfg['data_and_vendored_sha256']['evaluation_data.json'], 'MMMU labels have changed')
        answers = read(labels)['samples']
        predictions = by_id(read_rows(folder / 'predictions.jsonl'), 900)
        check(answers.keys() == predictions.keys() and (not rows or rows.keys() == answers.keys()), 'Unmatched MMMU IDs')
        for key, label in answers.items():
            item = rows.setdefault(key, dict(id=key, label=label, responses={}))
            check(item['label'] == label, f'Mismatched MMMU label: {key}')
            item['responses'][name] = predictions[key]['response']
        expected[name] = read(folder / 'scores.json')['structured']['overall']['correct']
    examples = list(rows.values())
    scores = mmmu_scores(examples)
    check(all(scores[name]['correct'] == expected[name] for name in MODELS), 'Mismatch with original structured scores')
    write_rows(dest / 'mmmu.jsonl.gz', examples)
    write_json(dest / 'summary.json', make_summary(dest, locked, scores, protocol))


def make_summary(dest, locked, scores, protocol):
    winner = locked['selected_strength']
    return {'selection_sha256': digest(dest / 'selection.json'), 'mmmu_sha256': digest(dest / 'mmmu.jsonl.gz'),
            'protocol': protocol, 'mmmu': scores, 'selected_strength': winner, 'target_correct': 501,
            'target_reached': scores[winner]['correct'] >= 501}


def verify(dest):
    locked = verify_external(dest)
    report = read(dest / 'summary.json')
    validate_protocol(report['protocol'])
    if dest.resolve() == (HERE / 'evidence').resolve():
        verify_packaged_weights(locked)
    scores = mmmu_scores(read_rows(dest / 'mmmu.jsonl.gz'))
    check(report == make_summary(dest, locked, scores, report['protocol']), 'MMMU evidence or summary mismatch')
    print('strength  step  public/1044  EMMA/320  MMMU/900  gained  lost')
    for name in MODELS:
        arm = locked['teacher'] if name == 'teacher' else locked['arms'][name]
        result = scores[name]
        print(f"{name:>8} {arm.get('step', '—'):>5} {arm['public']['correct']:>12} {arm['emma']['correct']:>9}"
              f" {result['correct']:>9} {result['gained']:>7} {result['lost']:>5}")
    print(f"External selection: λ={locked['selected_strength']}  |  ≥501: {report['target_reached']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=HERE / 'evidence')
    verify(parser.parse_args().evidence)
