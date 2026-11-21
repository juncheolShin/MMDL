"""Selective option distillation, exact DDP epoch sampling and checkpoint selection."""
from functools import partial
import os
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader
from safetensors.torch import save_file
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import seed_worker

from training.common import require, write_json
from training.model import CHECKPOINT, PREFIX, restore_delta, trainables


class TailBalancedBatchSampler(BatchSampler):
    """Give every DDP rank an equal, unpadded tail from a single global shuffle.

    For N=18956, batch=2, ranks=4 this yields 2369 groups of four size-2
    batches followed by four size-1 batches. Accelerate can then shard these
    batches without duplicating four examples or leaving two ranks behind.
    """
    def __init__(self, sampler, batch_size=2, num_processes=4):
        if type(num_processes) is not int or num_processes < 1:
            raise ValueError('number of sampler processes must be positive')
        if len(sampler) % num_processes:
            raise ValueError('exact one-pass sampling requires examples divisible by the number of processes')
        super().__init__(sampler, batch_size=batch_size, drop_last=False)
        self.num_processes = num_processes

    def __len__(self):
        complete, remainder = divmod(len(self.sampler), self.batch_size * self.num_processes)
        return (complete + bool(remainder)) * self.num_processes

    def __iter__(self):
        global_size = self.batch_size * self.num_processes
        pending = []
        for index in self.sampler:
            pending.append(index)
            if len(pending) == global_size:
                for start in range(0, global_size, self.batch_size):
                    yield pending[start:start + self.batch_size]
                pending = []
        if pending:
            if len(pending) % self.num_processes:
                raise RuntimeError('sampler length changed; cannot split the final batch evenly across ranks')
            local_size = len(pending) // self.num_processes
            for start in range(0, len(pending), local_size):
                yield pending[start:start + local_size]


def forward_choices(model, inputs, weight):
    """Baseline CE on every assistant token; conditional KL at the first answer predictor only."""
    inputs = dict(inputs)
    ids = inputs.pop('option_ids')
    probs = inputs.pop('teacher_probs')
    correct = inputs.pop('teacher_correct')
    inputs.pop('row_index')
    targets = inputs.pop('labels')[:, 1:]
    positions = (targets != -100).any(dim=0).nonzero(as_tuple=True)[0]
    labels = targets.index_select(1, positions)
    supervised = labels != -100
    counts = supervised.sum(1)
    require(bool((counts > 0).all()), 'each example must have an assistant target')
    outputs = model(**inputs, logits_to_keep=positions)
    logits = outputs.logits.float()
    losses = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction='none')
    ce = (losses.sum(1) / counts).mean()
    first = supervised.long().argmax(1)
    first_logits = logits[torch.arange(len(first), device=first.device), first]
    valid = ids >= 0
    choices = first_logits.gather(1, ids.clamp_min(0)).masked_fill(~valid, float('-inf'))
    loss = ce
    if weight:
        # Zero-probability padded slots have zero value and zero gradient.
        log_q = F.log_softmax(choices, dim=1).masked_fill(~valid, 0.)
        kl = F.kl_div(log_q, probs, reduction='none').sum(1)
        # DDP ranks see equal local batch sizes, including the balanced final batch.
        # Averaging over all examples preserves the exact global CE + weight*I(correct)*KL.
        loss = ce + weight * (kl * correct).mean()
    return loss, outputs, choices.argmax(1)


def score_predictions(dataset, row_indices, predictions):
    row_indices = np.asarray(row_indices).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    count = len(dataset)
    require(len(predictions) == len(row_indices) == count and
            np.array_equal(np.sort(row_indices), np.arange(count)), 'incomplete or duplicate dev predictions')
    ordered = np.empty(count, dtype=np.int64)
    ordered[row_indices] = predictions
    items, correct, recovered, lost = [], 0, 0, 0
    for ix, choice in enumerate(ordered):
        probabilities, was_correct, gold = dataset.teacher[ix]
        require(0 <= choice < len(probabilities), 'predicted unavailable option')
        is_correct = int(choice) == gold
        correct += is_correct
        recovered += is_correct and not was_correct
        lost += was_correct and not is_correct
        items.append({'id': dataset.rows[ix]['id'], 'pred': ascii_uppercase[choice]})
    return {
        'correct': correct, 'recovered': recovered, 'lost': lost, 'net_gain': recovered - lost,
        'accuracy': correct / count, 'selection_score': correct * (count + 1) - lost,
    }, items


class EvaluateFinalStep(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


class RetentionTrainer(Trainer):
    def __init__(self, *args, lora_learning_rate, distill_weight, **kwargs):
        self.lora_learning_rate = lora_learning_rate
        self.distill_weight = distill_weight
        super().__init__(*args, compute_metrics=self._dev_metrics, **kwargs)
        self.model_accepts_loss_kwargs = False

    def get_train_dataloader(self):
        if self._train_batch_size != 2:
            return super().get_train_dataloader()
        if self.train_dataset is None or not hasattr(self.train_dataset, '__len__'):
            raise ValueError('exact one-pass batching requires a finite training dataset')
        if self.args.dataloader_drop_last:
            raise ValueError('exact one-pass batching must not drop training examples')
        if self.accelerator.split_batches or self.accelerator.dispatch_batches:
            raise ValueError('exact one-pass batching requires split_batches=False and dispatch_batches=False')
        sampler = self._get_train_sampler(self.train_dataset)
        batches = TailBalancedBatchSampler(sampler, batch_size=2, num_processes=self.accelerator.num_processes)
        loader = DataLoader(
            self.train_dataset, batch_sampler=batches, collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            worker_init_fn=partial(seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index),
        )
        previous_even_batches = self.accelerator.even_batches
        self.accelerator.even_batches = False
        try:
            # Preparation replaces RandomSampler with SeedableRandomSampler and
            # DataLoaderShard.set_epoch() propagates epochs through our sampler.
            # Only training uses uneven local batch sizes; evaluation is unchanged.
            return self.accelerator.prepare_data_loader(loader)
        finally:
            self.accelerator.even_batches = previous_even_batches

    def create_optimizer(self):
        if self.optimizer is None:
            decay = set(self.get_decay_parameter_names(self.model))
            groups = {}
            for name, parameter in trainables(self.model).items():
                lr = self.args.learning_rate if name.startswith(PREFIX) else self.lora_learning_rate
                key = (lr, self.args.weight_decay if name in decay else 0.0)
                groups.setdefault(key, []).append(parameter)
            self.optimizer = torch.optim.AdamW(
                [dict(params=params, lr=lr, weight_decay=wd) for (lr, wd), params in groups.items()],
                betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon,
            )
        return self.optimizer

    def _save(self, output_dir=None, state_dict=None):
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        tmp = destination / f'{CHECKPOINT}.{os.getpid()}.tmp'
        save_file({n: p.detach().cpu().contiguous() for n, p in trainables(self.model).items()}, tmp,
                  metadata={'format': 'pt'})
        os.replace(tmp, destination / CHECKPOINT)

    def _load_best_model(self):
        restore_delta(self.model, self.state.best_model_checkpoint)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs, _ = forward_choices(model, inputs, self.distill_weight)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, _, predictions = forward_choices(model, inputs, 0.)
        return loss.detach(), predictions.unsqueeze(1), inputs['row_index'].unsqueeze(1)

    def _dev_metrics(self, evaluation):
        metrics, items = score_predictions(self.eval_dataset, evaluation.label_ids, evaluation.predictions)
        if self.is_world_process_zero():
            destination = Path(self.args.output_dir) / 'retention_dev'
            destination.mkdir(parents=True, exist_ok=True)
            step = self.state.global_step
            write_json(destination / f'step-{step}.json', {'step': step, 'items': items})
        return metrics
