"""ICoT-SI token deletion primitives. Schedule transitions occur at update boundaries."""
from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class Schedule:
    updates_per_epoch: int
    remove_per_epoch: float = 8.0
    warmup_updates: int = 0
    smoothing_lambda: float = 4.0

    def __post_init__(self):
        if self.updates_per_epoch < 1 or self.warmup_updates < 0:
            raise ValueError('Invalid update counts')
        if not math.isfinite(self.remove_per_epoch) or self.remove_per_epoch <= 0:
            raise ValueError('Removal rate must be finite and positive')
        if math.isnan(self.smoothing_lambda) or self.smoothing_lambda <= 0:
            raise ValueError('Smoothing lambda must be positive')

    def removed(self, completed_updates):
        if completed_updates < 0:
            raise ValueError('Negative update count')
        return math.floor(self.remove_per_epoch * max(0, completed_updates - self.warmup_updates)
                          / self.updates_per_epoch)

    def offset(self, rng, completed_updates):
        if completed_updates < self.warmup_updates or math.isinf(self.smoothing_lambda):
            return 0
        # Geometric distribution: P(o=k)=(1-exp(-lambda))*exp(-lambda*k).
        return math.floor(-math.log1p(-rng.random()) / self.smoothing_lambda)

    def to_dict(self):
        return asdict(self)


def rationale_span(item, target, tokenizer):
    """Locate a token-aligned E/K/R prefix, retaining Answer and end tokens."""
    if '\nAnswer:' not in target:
        raise ValueError('Explicit Answer boundary required')
    rationale, answer = target.rsplit('\nAnswer:', 1)
    if not rationale.startswith('Evidence:') or not answer.strip():
        raise ValueError('Expected nonempty E/K/R/A target')
    ids, labels = item['input_ids'].flatten(), item['labels'].flatten()
    active = (labels != -100).nonzero().flatten()
    if not len(active) or not torch.equal(active, torch.arange(active[0], len(ids), device=ids.device)):
        raise ValueError('Expected one contiguous assistant target at sequence end')
    expected = tokenizer.encode(target + '<|im_end|>\n', add_special_tokens=False)
    if labels[active].tolist() != expected or not torch.equal(ids[active], labels[active]):
        raise ValueError('Cache does not match the full target')
    prefix = tokenizer.encode(rationale + '\n', add_special_tokens=False)
    if expected[:len(prefix)] != prefix:
        raise ValueError('Answer boundary crosses a tokenizer token')
    start = int(active[0])
    return start, start + len(prefix)


def remove_prefix(item, span, count):
    """Delete actual causal inputs AND labels; text-only suffix shifts in all mRoPE axes."""
    start, stop = span
    ids, labels, positions = item['input_ids'], item['labels'], item['position_ids']
    if not (0 <= start < stop < ids.shape[-1]) or count < 0:
        raise ValueError('Invalid rationale span/count')
    if ids.shape != labels.shape or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError('Expected an unpadded single-example cache')
    if positions.shape != (3, 1, ids.shape[-1]):
        raise ValueError('Expected Qwen3-VL mRoPE positions')
    if not torch.all(labels[..., :start] == -100) or torch.any(labels[..., start:] == -100):
        raise ValueError('Rationale must be in the supervised suffix')
    # This shortcut is valid only after the last image/video: all remaining positions
    # must be ordinary contiguous text on all axes. Compare with model.get_rope_index
    # in the preflight before using it on real training data.
    text_positions = positions[..., start:]
    if not torch.equal(text_positions, text_positions[0:1].expand_as(text_positions)):
        raise ValueError('Non-text multimodal positions inside target')
    if not torch.all(torch.diff(text_positions, dim=-1) == 1):
        raise ValueError('Noncontiguous target positions')
    removed = min(count, stop - start)
    end = start + removed
    output = dict(item)
    for key in ('input_ids', 'labels'):
        output[key] = torch.cat((item[key][..., :start], item[key][..., end:]), dim=-1)
    output['position_ids'] = torch.cat((positions[..., :start], positions[..., end:] - removed), dim=-1)
    # Upstream uncollated cache uses a list of segment lengths, not a binary mask.
    if item.get('attention_mask') != [ids.shape[-1]]:
        raise ValueError('Unexpected uncollated attention mask')
    output['attention_mask'] = [output['input_ids'].shape[-1]]
    return output, removed


def reset_optimizer(optimizer):
    """Keep parameter identities and LR; discard moments only at a clean update boundary."""
    if any(p.grad is not None for group in optimizer.param_groups for p in group['params']):
        raise ValueError('Cannot reset with accumulated gradients')
    optimizer.state.clear()


def natural_span(item, target, tokenizer):
    rationale, marker, answer = target.rpartition('\nAnswer: ')
    if (not marker or not rationale.strip() or not answer.strip() or '\n' in answer or
            any(line.lstrip().startswith('Answer:') for line in rationale.splitlines())):
        raise ValueError('Expected nonempty natural reasoning and one final Answer: line')

    ids, labels = item['input_ids'].flatten(), item['labels'].flatten()
    active = (labels != -100).nonzero().flatten()
    if not len(active) or not torch.equal(active, torch.arange(active[0], len(ids), device=ids.device)):
        raise ValueError('Expected a contiguous supervised assistant suffix')
    expected = tokenizer.encode(target + '<|im_end|>\n', add_special_tokens=False)
    if labels[active].tolist() != expected or not torch.equal(ids[active], labels[active]):
        raise ValueError('Cached supervision differs from the target text')
    prefix = tokenizer.encode(rationale + '\n', add_special_tokens=False)
    suffix = tokenizer.encode('Answer: ' + answer + '<|im_end|>\n', add_special_tokens=False)
    if not prefix or not suffix or prefix + suffix != expected:
        raise ValueError('The final Answer: boundary is not token aligned')
    if set(prefix) & set(tokenizer.all_special_ids):
        raise ValueError('Reasoning contains a model special token')
    if set(suffix[:-len(tokenizer.encode('<|im_end|>\n', add_special_tokens=False))]) & set(tokenizer.all_special_ids):
        raise ValueError('Answer contains a model special token')
    start = int(active[0])
    return start, start + len(prefix)
