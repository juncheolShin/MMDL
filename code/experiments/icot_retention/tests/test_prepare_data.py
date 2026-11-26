import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_data as data


class DataRecipeTests(unittest.TestCase):
    def test_splits_have_disjoint_ids_and_images_and_carry_no_upstream_answers(self):
        catalog = data.read_json(data.RECIPE_DIR / 'recipe.json')
        all_ids = set()
        all_images = set()
        for name, count in [('public/train', 18956), ('public/dev', 1044), ('emma/dev', 320)]:
            rows = data.load_rows(catalog['splits'][name])
            self.assertEqual(len(rows), count)
            self.assertFalse(any('answer' in r or 'conversations' in r for r in rows))
            ids = {r['id'] for r in rows}
            self.assertEqual(len(ids), count)
            self.assertTrue(ids.isdisjoint(all_ids))
            all_ids.update(ids)
            images = {r['image_sha256'] for r in rows}
            self.assertTrue(images.isdisjoint(all_images))
            all_images.update(images)

    def test_exact_parquet_lookup_by_row_number_and_original_answer_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'images').mkdir()
            stream = io.BytesIO()
            with Image.new('RGB', (3, 2), 'red') as rgb:
                rgb.save(stream, format='PNG')
                pixel_digest = data.rgb_digest(rgb)
            blob = stream.getvalue()
            upstream = {'question_id': 'target', 'question': 'Which color?', 'choices': ['Blue', 'Red'],
                        'correct_choice_idx': 1, 'image': {'bytes': blob}}
            path = folder / 'source.parquet'
            pq.write_table(pa.Table.from_pylist([dict(upstream, question_id='unselected'), upstream]), path)
            row = {'id': 'aokvqa:target', 'upstream_id': 'target', 'source': 'aokvqa',
                   'image': f'images/{pixel_digest}.png', 'image_sha256': pixel_digest,
                   'source_provenance': {'parquet_row': 1, 'encoded_sha256': hashlib.sha256(blob).hexdigest()}}
            data.restore_parquet(path, 'aokvqa', [(folder, row)])
            self.assertEqual(row['answer'], 'B')
            self.assertEqual(row['choices'], ['Blue', 'Red'])
            self.assertEqual((folder / row['image']).read_bytes(), blob)
            self.assertIn('A. Blue\nB. Red', row['conversations'][0]['value'])
            row['source_provenance']['parquet_row'] = 0
            with self.assertRaisesRegex(ValueError, 'Unexpected upstream question'):
                data.restore_parquet(path, 'aokvqa', [(folder, row)])

    def test_verify_detects_replaced_pixels_and_annotation_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with Image.new('RGB', (4, 2), 'red') as rgb:
                rgb.save(folder / 'test.png')
                row = {'image': 'test.png', 'image_sha256': data.rgb_digest(rgb), 'id': 'row', 'source': 'm3cot'}
            path = folder / 'train.jsonl'
            path.write_text(json.dumps(row) + '\n')
            artifact = {'sha256': data.digest(path), 'count': 1, 'images': 1, 'by_source': {'m3cot': 1}}
            data.verify_annotation(path, artifact, verify_images=True)
            with Image.new('RGB', (4, 2), 'blue') as rgb:
                rgb.save(folder / 'test.png')
            with self.assertRaisesRegex(ValueError, 'Image SHA-256 mismatch'):
                data.verify_annotation(path, artifact, verify_images=True)
            path.write_text(json.dumps(row, separators=(',', ':')) + '\n')
            with self.assertRaisesRegex(ValueError, 'Annotation SHA-256 mismatch'):
                data.verify_annotation(path, artifact, verify_images=False)

    def test_failed_build_never_publishes_partial_output(self):
        def fail_during_build(recipe, sources, stage):
            stage.mkdir()
            (stage / 'incomplete').touch()
            raise OSError('interrupted write')

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'dataset'
            with (
                patch.object(sys, 'argv', ['prepare_data.py', '--output', str(output)]),
                patch.object(data, 'source_paths', return_value={}),
                patch.object(data, 'prepare', side_effect=fail_during_build),
                self.assertRaisesRegex(OSError, 'interrupted write'),
            ):
                data.main()
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
