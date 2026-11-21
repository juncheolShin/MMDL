"""Reproduce the four-GPU joint-training sweep and publish its verified evidence."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.headerregistry import Address
from email.message import EmailMessage
from pathlib import Path

import evaluation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECIPE = HERE / 'recipe.toml'
ASSESS = ROOT / 'icot/evaluate.sh'
LOCKED = ('selection.json', 'public.jsonl.gz', 'emma.jsonl.gz', 'teacher_train.json', 'teacher_train.jsonl.gz')
MMMU = tuple('mmmu/' + n for n in ('scores.json', 'evaluation_data.json', 'run_config.json', 'predictions.jsonl'))
DEFAULT_ENV = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'NCCL_P2P_DISABLE': '1',
               'NCCL_IB_DISABLE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_record(path, value):
    # Same-directory rename keeps a signal or crash from publishing a partial receipt.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=f'.writing-{path.name}-', delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def input_record(args, devices):
    import prepare_data

    recipe = HERE / 'data_recipe/recipe.json'
    prepare_data.verify(read(recipe), args.data)
    data_files = ('public/train.jsonl', 'public/dev.jsonl', 'emma/dev.jsonl',
                  'public/data_manifest.json', 'emma/data_manifest.json')
    manifest = read(HERE / 'weights/manifest.json')
    expected = {**manifest['assets'], **{n: v['sha256'] for n, v in manifest['shards'].items()}}
    if {p.name for p in args.teacher.glob('*.safetensors')} != set(manifest['shards']):
        raise ValueError('The supplied teacher does not have the released ICoT shards')
    for name, digest in expected.items():
        if sha(args.teacher / name) != digest:
            raise ValueError(f'Teacher differs from the released ICoT: {name}')
    sources = [
        Path(__file__), RECIPE, recipe, HERE / 'prepare_data.py', HERE / 'evaluation.py', ASSESS,
        ROOT / 'icot/evidence/evaluation_498.json',
        ROOT / 'configs/evaluation.toml', ROOT / 'code/configuration.py', *HERE.glob('training/*.py'),
        *(ROOT / 'code/eval' / f'{n}.py' for n in ('infer_mmmu', 'check_prompts', 'mmmu_data', 'prompting')),
        *(ROOT / 'icot/scoring').glob('*.py'), *(ROOT / 'code/third_party').rglob('*.py'),
    ]
    return {
        'format': 1, 'teacher': expected, 'external_data': {n: sha(args.data / n) for n in data_files},
        'code': {str(p.relative_to(ROOT)): sha(p) for p in sources},
        'packages': {p: importlib.metadata.version(p) for p in
                     ('torch', 'torchvision', 'transformers', 'accelerate', 'safetensors',
                      'vllm', 'datasets', 'pillow', 'pyarrow')},
        'environment': {k: os.environ.get(k, v) for k, v in DEFAULT_ENV.items()},
        'python': list(sys.version_info[:3]), 'cuda_visible_devices': devices,
        'mmmu_cache': str(args.mmmu_data), 'mmmu_revision': evaluation.REVISION,
    }


class Stages:
    def __init__(self, root):
        self.root = root
        self.attempts = root / 'attempts'
        self.receipts = root / 'receipts'
        self.attempts.mkdir(exist_ok=True)
        self.receipts.mkdir(exist_ok=True)
        self.cancel = threading.Event()

    def run(self, name, dependencies, required, action):
        receipt = self.receipts / (name + '.json')
        prior = {'inputs': sha(self.root / 'run.json'),
                 **{n: sha(self.receipts / (n + '.json')) for n in dependencies}}
        if receipt.exists():
            saved = read(receipt)
            attempt = saved['attempt']
            if Path(attempt).name != attempt or not attempt.startswith(name + '-'):
                raise ValueError(f'{name}: invalid receipt path')
            out = self.attempts / attempt / 'result'
            if (saved['dependencies'] != prior or set(saved['files']) != set(required)
                    or any(not (out / n).is_file() or sha(out / n) != digest for n, digest in saved['files'].items())):
                raise ValueError(f'{name}: recorded outputs or dependencies changed; choose a fresh --run')
            print(f'{name}: verified completed stage', flush=True)
            return out
        if self.cancel.is_set():
            raise RuntimeError(f'{name}: cancelled')
        attempt = Path(tempfile.mkdtemp(prefix=name + '-', dir=self.attempts))
        out = attempt / 'result'
        print(f'{name}: {attempt}', flush=True)
        try:
            action(out)
            hashes = {n: sha(out / n) for n in required}
            save_record(receipt, {'attempt': attempt.name, 'dependencies': prior, 'files': hashes})
            return out
        except BaseException as exc:
            exc.add_note(f'Stage {name}, files at {attempt}')
            raise


def command(argv, env, log, cancel):
    """Catch interrupts at the orchestrator and stop the entire child process group."""
    with log.open('x') as stream:
        proc = subprocess.Popen([str(s) for s in argv], cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                if cancel.is_set():
                    raise RuntimeError(f'Cancelled subprocess: {log}')
                try:
                    status = proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if status:
                raise subprocess.CalledProcessError(status, argv)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            except ProcessLookupError:
                pass
            raise


def sweep(args, stages, devices):
    teacher = args.teacher
    env = os.environ.copy()
    for name, value in DEFAULT_ENV.items():
        env.setdefault(name, value)
    env['PATH'] = os.pathsep.join((str(Path(sys.executable).parent), env.get('PATH', '')))
    env['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
    weights = read(HERE / 'weights/manifest.json')
    model_files = (*weights['shards'], *weights['assets'], 'export.json', 'training_config.json', 'trainer_state.json',
                   *(f'retention_dev/step-{step}.json' for step in evaluation.STEPS))

    def distributed(script, out, *flags):
        argv = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                HERE / 'training' / script, '--config', RECIPE, *flags]
        command(argv, env, out.parent / 'process.log', stages.cancel)

    scores = stages.run('teacher-cache', (), ('train.jsonl', 'dev.jsonl', 'manifest.json'),
                        lambda out: distributed('teacher.py', out, '--model', teacher, '--data-dir',
                                                args.data / 'public', '--output-dir', out))
    models = {'teacher': teacher}
    expert = {}
    for w in evaluation.MODELS:
        deps = ()
        if w != 'teacher':
            models[w] = stages.run(
                f'train-{w}', ('teacher-cache',), model_files,
                lambda out: distributed('train.py', out, '--model', teacher, '--data-dir', args.data / 'public',
                                        '--teacher-dir', scores, '--output-dir', out, '--distill-weight', w))
            deps = (f'train-{w}',)
        output = stages.run(
            f'emma-{w}', deps, ('scores.json',),
            lambda out: distributed('teacher.py', out, '--model', models[w], '--data-dir', args.data / 'emma',
                                    '--expert-output', out / 'scores.json'))
        expert[w] = output / 'scores.json'

    def lock(out):
        evaluation.lock_selection(args.data, scores, models, expert, out)
        evaluation.verify_external(out)

    lock_deps = ('teacher-cache', *(f'emma-{w}' for w in evaluation.MODELS))
    locked = stages.run('selection', lock_deps, LOCKED, lock)

    def mmmu(w, gpu, deps):
        child_env = {**env, 'CUDA_VISIBLE_DEVICES': gpu,
                     'DATA_PATH': str(args.mmmu_data), 'HF_HOME': str(args.mmmu_data)}
        out = stages.run(f'mmmu-{w}', deps, MMMU, lambda dest: command(
            ['bash', ASSESS, models[w], dest], child_env, dest.parent / 'process.log', stages.cancel))
        return out / 'mmmu'

    # Populate the shared dataset/image cache once before the four independent evaluations.
    directories = {'teacher': mmmu('teacher', devices[0], ('selection',))}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {}
        try:
            for w, gpu in zip(evaluation.STRENGTHS, devices, strict=True):
                jobs[pool.submit(mmmu, w, gpu, ('selection', 'mmmu-teacher'))] = w
            for job in as_completed(jobs):
                directories[jobs[job]] = job.result()
        except BaseException:
            stages.cancel.set()
            raise

    def collect(out):
        shutil.copytree(locked, out)
        evaluation.collect_mmmu(out, directories)
        evaluation.verify(out)

    all_evidence = (*LOCKED, 'summary.json', 'mmmu.jsonl.gz')
    result = stages.run('final', tuple(f'mmmu-{w}' for w in evaluation.MODELS), all_evidence, collect)
    published = stages.root / 'evidence'
    if not published.exists():
        stage = Path(tempfile.mkdtemp(prefix='publish-', dir=stages.attempts)) / 'evidence'
        shutil.copytree(result, stage)
        stage.rename(published)
    if any(sha(published / n) != sha(result / n) for n in all_evidence):
        raise ValueError(f'Published evidence differs from the final stage: {published}')
    evaluation.verify(published)
    return published


def execute(args):
    devices = [x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')]
    if len(devices) != 4 or len(set(devices)) != 4 or not all(devices):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to four distinct GPU IDs')
    args.run.mkdir(exist_ok=True, parents=True)
    record = args.run / 'run.json'
    if not record.exists() and any(p.name != '.lock' and not p.name.startswith('.writing-run.json-')
                                   for p in args.run.iterdir()):
        raise ValueError('Choose an empty --run directory for a new experiment')
    with (args.run / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = input_record(args, devices)
        if record.exists():
            if read(record) != identity:
                raise ValueError('Inputs, runtime or recipe changed; choose a fresh --run')
        else:
            save_record(record, identity)
        return sweep(args, Stages(args.run), devices)


def mail_transport(recipient):
    addr = Address(addr_spec=recipient)
    if not addr.username or not addr.domain or str(addr) != recipient:
        raise ValueError('--notify-email requires one email address')
    path = shutil.which('sendmail') or '/usr/sbin/sendmail'
    if not os.access(path, os.X_OK):
        raise ValueError('--notify-email requires a configured local sendmail executable')
    return path


def notify(path, recipient, run, exc):
    message = EmailMessage()
    message['To'] = recipient
    message['Subject'] = 'MMDL ICoT experiment interrupted'
    message.set_content('\n'.join((f'Run: {run}', f'{type(exc).__name__}: {exc}', *getattr(exc, '__notes__', ()))))
    subprocess.run([path, '-oi', '-t'], input=message.as_bytes(), check=True, timeout=30)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher', 'data', 'mmmu-data', 'run'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--notify-email', metavar='ADDRESS', help='Alert on failures via the local sendmail service')
    args = parser.parse_args(argv)
    for name in ('teacher', 'data', 'mmmu_data', 'run'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    transport = mail_transport(args.notify_email) if args.notify_email else None

    def terminate(number, frame):
        raise KeyboardInterrupt(f'Received {signal.Signals(number).name}')

    previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        return execute(args)
    except BaseException as exc:
        if transport:
            try:
                notify(transport, args.notify_email, args.run, exc)
            except Exception as mail_error:
                print(f'Failure email could not be submitted: {mail_error}', file=sys.stderr)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
