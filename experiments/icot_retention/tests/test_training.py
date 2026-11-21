"""Exercise the loss, exact epoch coverage, public-dev selection, and full BF16 model export."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.data import MultimodalCollator, RetentionCollator, RetentionDataset
from training.model import CHECKPOINT, configure_attention, standalone, trainables
from training.teacher import prompt_prefix, scored
from training.trainer import EvaluateFinalStep, RetentionTrainer, TailBalancedBatchSampler, forward_choices, score_predictions


def four_merger_model():
    config = Qwen3VLConfig(
        text_config={
            'vocab_size': 128, 'hidden_size': 32, 'intermediate_size': 64,
            'num_hidden_layers': 4, 'num_attention_heads': 4,
            'num_key_value_heads': 2, 'head_dim': 8, 'max_position_embeddings': 128,
            'rope_scaling': {'rope_type': 'default', 'mrope_section': [1, 1, 2], 'mrope_interleaved': True},
        },
        vision_config={
            'depth': 4, 'hidden_size': 32, 'intermediate_size': 64, 'num_heads': 4,
            'patch_size': 16, 'spatial_merge_size': 2, 'temporal_patch_size': 2,
            'out_hidden_size': 32, 'num_position_embeddings': 16,
            'deepstack_visual_indexes': [0, 1, 2],
        },
        image_token_id=120, video_token_id=119,
        vision_start_token_id=121, vision_end_token_id=122,
    )
    config._attn_implementation = 'sdpa'
    model = Qwen3VLForConditionalGeneration(config)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model


def example(width=4, answer_length=2, offset=0):
    image_tokens = 4 * width // 4
    ids = torch.tensor([[1, 121] + [120] * image_tokens + [122, 6, 7, 8, 9, 10]])
    labels = torch.full_like(ids, -100)
    labels[:, -answer_length:] = ids[:, -answer_length:]
    return {
        'input_ids': ids, 'labels': labels, 'attention_mask': torch.ones_like(ids),
        'pixel_values': torch.randn(4 * width, 3 * 2 * 16 * 16) + offset,
        'image_grid_thw': torch.tensor([[1, 4, width]]),
    }


class Rows:
    def __init__(self):
        self.rows, self.encoded, self.labels = [], [], []
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda c, **kwargs: [8 + ord(c)-ord('A')]))
        for i in range(4):
            count = 2 + i % 2
            encoded = example(2 + 2 * (i % 2), 2 + i % 2)
            first_label = encoded['labels'][encoded['labels'] != -100][0].item()
            gold = chr(ord('A') + first_label - 8)
            pred = gold if i < 2 else ('B' if gold == 'A' else 'A')
            probabilities = [.15 / (count - 1)] * count
            probabilities[ord(pred)-ord('A')] = .85
            self.rows.append({'id': str(i), 'choices': list(range(count)), 'answer': gold, 'source': 'fixed'})
            self.encoded.append(encoded)
            self.labels.append({'id': str(i), 'gold': gold, 'source': 'fixed', 'prediction': pred,
                                'correct': pred == gold, 'option_token_ids': list(range(8, 8+count)),
                                'probabilities': probabilities})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.encoded[index]


def dataset(tmp):
    base = Rows()
    path = Path(tmp) / 'teacher.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in reversed(base.labels)))
    return RetentionDataset(base, path)


class RetentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_masked_kl_and_eos_ce_at_distinct_positions(self):
        torch.manual_seed(313)
        model = four_merger_model().eval()
        base = MultimodalCollator(0, 120)
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp)
            for indices in ([0, 1], [1, 2], [2, 3]):
                subset = [data[i] for i in indices]
                batch = RetentionCollator(base)(subset)
                ce, _, _ = forward_choices(model, batch, 0)
                plain = base([data.dataset[i] for i in indices])
                ids = plain.pop('labels')
                logits = model(**plain).logits
                # Independent full-vocabulary CE at every unmasked next-token position.
                gold = torch.stack([
                    F.cross_entropy(values[:-1][target[1:] != -100], target[1:][target[1:] != -100])
                    for values, target in zip(logits, ids, strict=True)
                ]).mean()
                torch.testing.assert_close(ce, gold)
                kl = torch.tensor(0.)
                for row, sample in enumerate(subset):
                    _, tokens, probs, keep = sample['retention']
                    if keep:
                        pos = int((ids[row] != -100).nonzero()[0])-1
                        kl += F.kl_div(logits[row, pos, tokens].float().log_softmax(0),
                                       torch.tensor(probs), reduction='sum')
                total, _, _ = forward_choices(model, batch, .5)
                torch.testing.assert_close(total, gold + .5 * kl / len(indices))
                (gradient,) = torch.autograd.grad(total, model.lm_head.weight)
                self.assertTrue(torch.isfinite(gradient).all())
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                score_predictions(data, [0, 1, 2, 2], [0, 0, 0, 0])
            metrics, items = score_predictions(data, [3, 1, 0, 2], [0, 1, 1, 0])
            self.assertEqual(len(items), 4)
            self.assertEqual(metrics['correct'], 2)
            self.assertEqual(metrics['net_gain'], 0)
            self.assertEqual(metrics['lost'], 1)

    def test_teacher_receives_only_prompt_and_available_options(self):
        item = example(2, 2)
        prefix = prompt_prefix(item, 9)
        self.assertEqual(prefix.tolist(), item['input_ids'][0, :-2].tolist())
        row = {'id': 'test', 'gold': 'B', 'option_token_ids': [8, 9]}
        result = scored(row, torch.tensor([1., 3., 100.]))
        self.assertTrue(result['correct'])
        torch.testing.assert_close(torch.tensor(result['probabilities']), torch.tensor([1., 3.]).softmax(0))

    def test_each_of_18956_examples_appears_once_with_equal_ddp_tail(self):
        batches = list(TailBalancedBatchSampler(range(18956), batch_size=2, num_processes=4))
        self.assertEqual(len(batches), 2370 * 4)
        self.assertEqual([len(batch) for batch in batches[-4:]], [1] * 4)
        self.assertEqual([i for batch in batches for i in batch], list(range(18956)))

    def test_optimizer_checkpoint_and_standalone_export(self):
        torch.manual_seed(314)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'source', root / 'output'
            data = dataset(root)
            model = four_merger_model().bfloat16()
            model.config.tie_word_embeddings = model.config.text_config.tie_word_embeddings = True
            model.tie_weights()
            model.save_pretrained(source, max_shard_size='100KB')
            (source / 'merges.txt').write_text('test\n')
            scope = configure_attention(model)
            args = TrainingArguments(
                output_dir=str(output), use_cpu=True, bf16=True, bf16_full_eval=True, report_to='none',
                per_device_train_batch_size=2, per_device_eval_batch_size=2, max_steps=3,
                learning_rate=.001, eval_strategy='steps', save_strategy='steps', eval_steps=2, save_steps=2,
                load_best_model_at_end=True, metric_for_best_model='eval_selection_score', greater_is_better=True,
                save_only_model=True, remove_unused_columns=False, dataloader_pin_memory=False,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                logging_strategy='no', disable_tqdm=True)
            trainer = RetentionTrainer(
                model=model, args=args, train_dataset=data, eval_dataset=data, lora_learning_rate=.002,
                distill_weight=.5, callbacks=[EvaluateFinalStep()],
                data_collator=RetentionCollator(MultimodalCollator(0, 120)))
            trainer.train()
            metrics = [row for row in trainer.state.log_history if 'eval_loss' in row]
            self.assertEqual([row['step'] for row in metrics], [2, 3])
            best = max(metrics, key=lambda row: (row['eval_correct'], -row['eval_lost'], -row['step']))
            self.assertEqual(trainer.state.best_global_step, best['step'])
            for metric in metrics:
                items = json.loads((output / 'retention_dev' / f"step-{metric['step']}.json").read_text())['items']
                self.assertEqual([item['id'] for item in items], [r['id'] for r in data.rows])
                self.assertEqual(metric['eval_correct'], sum(i['pred'] == r['answer'] for i, r in zip(items, data.rows)))
            checkpoint = Path(trainer.state.best_model_checkpoint)
            with safe_open(checkpoint / CHECKPOINT, framework='pt') as delta:
                for name, tensor in trainables(model).items():
                    self.assertTrue(torch.equal(tensor, delta.get_tensor(name)), name)
            export_args = (source, checkpoint, output, scope['targets'], {'config.json', 'generation_config.json'})
            report = standalone(*export_args)
            self.assertEqual(report, standalone(*export_args, verify=True))
            self.assertEqual(report['changed_projections'], len(scope['targets']))
            restored = Qwen3VLForConditionalGeneration.from_pretrained(output, dtype=torch.bfloat16)
            state = restored.state_dict()
            visual = 'model.visual.merger.linear_fc2.weight'
            self.assertTrue(torch.equal(state[visual], model.state_dict()[visual].bfloat16()))
            stem = scope['targets'][0]
            attention = model.state_dict()
            expected = (attention[stem + '.weight'].float() +
                        2 * attention[stem + '.lora_B'] @ attention[stem + '.lora_A']).bfloat16()
            self.assertTrue(torch.equal(state[stem + '.weight'], expected))
            self.assertTrue(torch.equal(restored.lm_head.weight, model.lm_head.weight))


if __name__ == '__main__':
    unittest.main()
