"""Train the visual stack plus text attention LoRA and export the restored checkpoint."""
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from training.common import file_sha256, read_json, require

PREFIX = 'model.visual.'
PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
RANK = 8
ALPHA = 16
CHECKPOINT = 'trainable.safetensors'


class PackedPatchEmbedding(torch.nn.Module):
    """One input row is exactly one disjoint C,T,H,W patch, as in vLLM's torch 2.9 path."""
    def __init__(self, patch):
        super().__init__()
        self.proj = patch.proj
        conv = self.proj
        require(conv.kernel_size == conv.stride and conv.groups == 1 and conv.padding == (0, 0, 0)
                and conv.dilation == (1, 1, 1), 'unsupported Qwen patch convolution')

    def forward(self, hidden_states):
        return F.linear(hidden_states.to(self.proj.weight.dtype), self.proj.weight.flatten(1), self.proj.bias)


def parameter_group(name):
    if not name.startswith(PREFIX):
        return 'frozen'
    parts = name.removeprefix(PREFIX).split('.')
    if parts[0] in ('blocks', 'deepstack_merger_list') and len(parts) >= 3 and parts[1].isdigit():
        return '.'.join(parts[:2])
    if parts[0] in ('patch_embed', 'pos_embed', 'merger') and len(parts) >= 2:
        return parts[0]
    raise ValueError(f'unrecognized visual tensor: {name}')


def configure_joint(model):
    visual = model.model.visual
    require(len(visual.deepstack_merger_list) == 3, 'three DeepStack mergers required')
    model.requires_grad_(False)
    visual.patch_embed = PackedPatchEmbedding(visual.patch_embed)
    visual.requires_grad_(True)
    names, grouped = [], set()
    for name, tensor in model.named_parameters():
        group = parameter_group(name)
        require((group != 'frozen') == tensor.requires_grad, f'inconsistent trainable set: {name}')
        if tensor.requires_grad:
            tensor.data = tensor.data.float()
            names.append(name)
            grouped.add(group)
    expected = {'patch_embed', 'pos_embed', 'merger', *(f'blocks.{i}' for i in range(len(visual.blocks))),
                *(f'deepstack_merger_list.{i}' for i in range(3))}
    require(grouped == expected, 'incomplete visual module coverage')
    return {'visual_groups': sorted(grouped), 'trainable_names': names,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'frozen_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad)}


class AttentionLoRA(torch.nn.Module):
    """Keep each published Qwen projection, adding B(A(x)) * alpha/r in parallel."""
    def __init__(self, linear):
        super().__init__()
        require(type(linear) is torch.nn.Linear, 'unsupported Qwen attention projection')
        self.weight, self.bias = linear.weight, linear.bias
        self.lora_A = torch.nn.Parameter(torch.empty(RANK, linear.in_features, device=self.weight.device))
        self.lora_B = torch.nn.Parameter(torch.zeros(linear.out_features, RANK, device=self.weight.device))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        activation = x if torch.is_autocast_enabled(x.device.type) else x.to(self.lora_A.dtype)
        delta = F.linear(F.linear(activation, self.lora_A), self.lora_B)
        return base + (delta * (ALPHA / RANK)).to(base.dtype)


def projection_names(layers):
    return {f'model.language_model.layers.{i}.self_attn.{p}' for i in range(layers) for p in PROJECTIONS}


def adapter_names(targets):
    return {f'{name}.{factor}' for name in targets for factor in ('lora_A', 'lora_B')}


def configure_attention(model):
    visual = configure_joint(model)
    layers = model.model.language_model.layers
    targets = projection_names(len(layers))
    for layer in layers:
        for name in PROJECTIONS:
            linear = AttentionLoRA(getattr(layer.self_attn, name))
            require(torch.count_nonzero(linear.lora_B).item() == 0, 'nonzero initial attention delta')
            setattr(layer.self_attn, name, linear)
    trainables = {n: p for n, p in model.named_parameters() if p.requires_grad}
    expect = adapter_names(targets)
    expect.update(visual['trainable_names'])
    require(set(trainables) == expect and all(p.dtype == torch.float32 for p in trainables.values()),
            'incorrect trainable parameter scope or optimizer precision')
    return {'visual_groups': visual['visual_groups'], 'visual_names': visual['trainable_names'],
            'trainable_names': sorted(trainables), 'targets': sorted(targets), 'zero_delta_at_start': True,
            'trainable_parameters': sum(p.numel() for p in trainables.values())}


def trainables(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def restore_delta(model, checkpoint):
    named = trainables(model)
    with safe_open(Path(checkpoint) / CHECKPOINT, framework='pt', device='cpu') as stored, torch.no_grad():
        require(set(stored.keys()) == set(named), 'invalid selected trainable checkpoint keys')
        for name, parameter in named.items():
            value = stored.get_tensor(name)
            require(value.shape == parameter.shape and value.dtype == parameter.dtype, f'invalid checkpoint {name}')
            parameter.copy_(value)


def merged_tensor(name, source, delta, targets):
    if name.removesuffix('.weight') in targets and name.endswith('.weight'):
        stem = name.removesuffix('.weight')
        a, b = (delta.get_tensor(f'{stem}.{factor}') for factor in ('lora_A', 'lora_B'))
        require(a.dtype == b.dtype == torch.float32 and a.shape == (RANK, source.shape[1]) and
                b.shape == (source.shape[0], RANK), f'invalid LoRA factors: {stem}')
        return (source.float() + (b @ a) * (ALPHA / RANK)).to(source.dtype)
    if name.startswith(PREFIX):
        direct = delta.get_tensor(name)
        require(direct.dtype == torch.float32 and direct.shape == source.shape, f'invalid visual tensor: {name}')
        return direct.to(source.dtype)
    return source


def standalone(source, checkpoint, out, targets, assets, verify=False):
    """Stream one published shard at a time, preserving its keys (including omitted tied aliases)."""
    mapping = read_json(source / 'model.safetensors.index.json')['weight_map']
    targets = set(targets)
    visual = {n for n in mapping if n.startswith(PREFIX)}
    expected = adapter_names(targets) | visual
    require(all(f'{name}.weight' in mapping for name in targets), 'adapter targets absent from published source')
    shard_hashes = {}
    changed_projections = 0
    with safe_open(checkpoint / CHECKPOINT, framework='pt', device='cpu') as delta:
        require(set(delta.keys()) == expected, 'selected checkpoint scope differs from planned arm')
        if verify:
            require(read_json(out / 'model.safetensors.index.json') == read_json(source / 'model.safetensors.index.json'),
                    'export index differs from the published architecture')
        for file in sorted(set(mapping.values())):
            require(Path(file).name == file, 'unexpected shard path')
            with safe_open(source / file, framework='pt', device='cpu') as original:
                names = sorted(n for n, f in mapping.items() if f == file)
                require(set(names) == set(original.keys()), 'published shard/index mismatch')
                export = {}
                for name in names:
                    value = original.get_tensor(name)
                    require(value.dtype == torch.bfloat16, f'unexpected source dtype {name}')
                    result = merged_tensor(name, value, delta, targets)
                    if name.endswith('.weight') and name.removesuffix('.weight') in targets:
                        require(bool(result.isfinite().all()), f'nonfinite merged matrix {name}')
                        changed_projections += int(not torch.equal(value, result))
                    export[name] = result.contiguous()
                if verify:
                    with safe_open(out / file, framework='pt', device='cpu') as stored:
                        require(set(stored.keys()) == set(names), 'export tensor set differs from source')
                        for name in names:
                            saved = stored.get_tensor(name)
                            require(saved.dtype == export[name].dtype and torch.equal(saved, export[name]),
                                    f'export differs from selected model: {name}')
                else:
                    tmp = out / (file + '.tmp')
                    save_file(export, tmp, metadata={'format': 'pt'})
                    os.replace(tmp, out / file)
                del export
            shard_hashes[file] = file_sha256(out / file)
    for file in set(assets) | {'model.safetensors.index.json', 'merges.txt'}:
        if file.endswith('.safetensors'):
            continue
        if verify:
            require(file_sha256(out / file) == file_sha256(source / file), f'inference asset mismatch {file}')
        else:
            shutil.copyfile(source / file, out / file)
    return {'merged_projections': len(targets), 'changed_projections': changed_projections, 'num_visual_tensors': len(visual), 'dtype': 'bfloat16',
            'shards_sha256': shard_hashes}
