"""Exercise orchestration, restart integrity and notification without GPUs or mail."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
import reproduce as run


def emit(directory, names):
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)


class SweepTest(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        folder = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.args = argparse.Namespace(teacher=folder / 'teacher', data=folder / 'data',
                                       mmmu_data=folder / 'cache', run=folder / 'sweep')
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,2,1,0'}))
        self.events = []
        self.input_check = self.stack.enter_context(mock.patch.object(run, 'input_record', return_value={'inputs': 1}))
        self.stack.enter_context(mock.patch.object(run, 'command', side_effect=self.command))
        for name, effect in [('lock_selection', self.lock), ('collect_mmmu', self.collect),
                             ('verify_external', self.verify_locked), ('verify', self.verify)]:
            self.stack.enter_context(mock.patch.object(run.evaluation, name, side_effect=effect))
        self.fail = None

    def lock(self, data, scores, models, expert, dest):
        self.assertEqual(set(models), set(run.evaluation.MODELS))
        self.assertTrue(all(p.is_file() for p in expert.values()))
        self.events.append('lock')
        emit(dest, run.LOCKED)

    def verify_locked(self, folder):
        self.assertTrue(all((folder / n).is_file() for n in run.LOCKED))

    def collect(self, dest, directories):
        self.verify_locked(dest)
        self.assertEqual(set(directories), set(run.evaluation.MODELS))
        self.assertTrue(all((p / 'scores.json').is_file() for p in directories.values()))
        self.events.append('collect')
        emit(dest, ('summary.json', 'mmmu.jsonl.gz'))

    def verify(self, dest):
        self.verify_locked(dest)
        self.assertTrue((dest / 'summary.json').is_file())

    def command(self, argv, env, log, cancel):
        gpu = env['CUDA_VISIBLE_DEVICES']
        if argv[0] == 'bash':
            self.assertIn('lock', self.events)
            self.assertEqual(argv[1], run.ASSESS)
            self.assertEqual(env['HF_HOME'], str(self.args.mmmu_data))
            self.assertEqual(env['DATA_PATH'], env['HF_HOME'])
            w = 'teacher' if argv[2] == self.args.teacher else argv[2].parent.name.split('-')[1]
            if w != 'teacher':
                self.assertIn('mmmu-teacher@3', self.events)
            self.events.append(f'mmmu-{w}@{gpu}')
            emit(argv[-1], run.MMMU)
            return
        self.assertIn('--standalone', argv)
        self.assertIn('--nproc_per_node=4', argv)
        self.assertEqual(gpu, '3,2,1,0')
        self.assertEqual(argv[argv.index('--config') + 1], run.RECIPE)
        if '--expert-output' in argv:
            target = argv[argv.index('--expert-output') + 1]
            self.events.append('emma')
            emit(target.parent, ('scores.json',))
        elif '--distill-weight' in argv:
            w = argv[argv.index('--distill-weight') + 1]
            self.events.append(f'train-{w}')
            out = argv[argv.index('--output-dir') + 1]
            if self.fail == w:
                emit(out, ('partial-training',))
                raise RuntimeError('worker failed')
            weights = run.read(EXPERIMENT / 'weights/manifest.json')
            emit(out, (*weights['shards'], *weights['assets'], 'export.json', 'trainer_state.json', 'training_config.json',
                       *(f'retention_dev/step-{s}.json' for s in run.evaluation.STEPS)))
        else:
            self.events.append('teacher')
            emit(argv[argv.index('--output-dir') + 1], ('manifest.json', 'train.jsonl', 'dev.jsonl'))

    def test_order_and_gpu_assignment(self):
        dest = run.execute(self.args)
        self.assertEqual(dest, self.args.run / 'evidence')
        self.assertEqual(self.events.count('emma'), 5)
        self.assertEqual(self.events.count('lock'), 1)
        self.assertGreater(self.events.index('lock'), self.events.index('train-0.5'))
        self.assertEqual(set(self.events[self.events.index('mmmu-teacher@3') + 1:-1]),
                         {'mmmu-0@3', 'mmmu-0.1@2', 'mmmu-0.25@1', 'mmmu-0.5@0'})
        events = self.events[:]
        self.assertEqual(run.execute(self.args), dest)
        self.assertEqual(self.events, events)

    def test_partial_attempt_is_preserved_and_completed_stages_are_verified(self):
        self.fail = '0.1'
        with self.assertRaisesRegex(RuntimeError, 'worker failed') as raised:
            run.execute(self.args)
        self.assertIn('train-0.1', str(raised.exception.__notes__))
        self.assertNotIn('lock', self.events)
        self.fail = None
        run.execute(self.args)
        self.assertEqual(self.events.count('train-0.1'), 2)
        self.assertEqual(self.events.count('train-0'), 1)
        self.assertEqual(len(list(self.args.run.glob('attempts/train-0.1-*/result/partial-training'))), 1)
        completed = next(self.args.run.glob('attempts/train-0-*/result/export.json'))
        completed.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'recorded outputs'):
            run.execute(self.args)
        self.input_check.return_value = {'inputs': 2}
        with self.assertRaisesRegex(ValueError, 'Inputs, runtime or recipe'):
            run.execute(self.args)

    def test_alert_identifies_stage_without_masking_original_failure(self):
        args = ['--teacher', 't', '--data', 'd', '--mmmu-data', 'm', '--run', 'r', '--notify-email', 'a@example.org']
        cause = KeyboardInterrupt('Received SIGTERM')
        cause.add_note('Stage emma-0.1')
        with (mock.patch.object(run, 'execute', side_effect=cause), mock.patch.object(run, 'mail_transport', return_value='sendmail'),
              mock.patch.object(run.subprocess, 'run', side_effect=RuntimeError('mail error')) as submit,
              contextlib.redirect_stderr(io.StringIO()), self.assertRaises(KeyboardInterrupt) as caught):
            run.main(args)
        self.assertIs(caught.exception, cause)
        self.assertIn(b'Stage emma-0.1', submit.call_args.kwargs['input'])
        self.assertIn(b'Received SIGTERM', submit.call_args.kwargs['input'])

    def test_terminal_disconnect_submits_alert_and_restores_handler(self):
        before = signal.getsignal(signal.SIGHUP)
        args = ['--teacher', 't', '--data', 'd', '--mmmu-data', 'm', '--run', 'r', '--notify-email', 'a@example.org']
        with (mock.patch.object(run, 'execute', side_effect=lambda _: os.kill(os.getpid(), signal.SIGHUP)),
              mock.patch.object(run, 'mail_transport', return_value='sendmail'),
              mock.patch.object(run.subprocess, 'run') as submit, self.assertRaisesRegex(KeyboardInterrupt, 'SIGHUP')):
            run.main(args)
        self.assertIn(b'SIGHUP', submit.call_args.kwargs['input'])
        self.assertEqual(signal.getsignal(signal.SIGHUP), before)


if __name__ == '__main__':
    unittest.main()
