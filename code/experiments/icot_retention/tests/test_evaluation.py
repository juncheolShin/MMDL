from pathlib import Path
import sys
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
import evaluation


class EvaluationTest(unittest.TestCase):
    def test_external_selection_and_weights_match_recorded_run(self):
        locked = evaluation.verify_external(EXPERIMENT / 'evidence')
        self.assertEqual(locked['selected_strength'], '0.5')
        self.assertEqual([locked['arms'][w]['step'] for w in evaluation.STRENGTHS], [2370, 2000, 1500, 2370])
        evaluation.verify_packaged_weights(locked)

    def test_raw_responses_reproduce_gains_and_losses(self):
        result = evaluation.mmmu_scores(evaluation.read_rows(EXPERIMENT / 'evidence/mmmu.jsonl.gz'))
        self.assertEqual(result['teacher']['correct'], 500)
        self.assertEqual(result['0.5'], {'correct': 488, 'gained': 31, 'lost': 43})
        for values in result.values():
            self.assertEqual(values['correct'], 500 + values['gained'] - values['lost'])

    def test_mismatched_questions_and_protocol_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different question IDs'):
            evaluation.paired({'a': True}, {'b': True})
        protocol = evaluation.read(EXPERIMENT / 'evidence/summary.json')['protocol']
        protocol['sampling']['temperature'] = 1
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            evaluation.validate_protocol(protocol)


if __name__ == '__main__':
    unittest.main()
