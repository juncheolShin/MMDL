"""Make Qwen's FlashAttention patch import lazy for the SDPA LoRA path.

Only the pinned upstream commit is supported. This leaves its trainer logic
unchanged when data_flatten/data_packing explicitly request FlashAttention.
"""
import os
from pathlib import Path

root = Path(os.environ.get('QWEN_TRAIN_ROOT', '/opt/qwen3-vl/qwen-vl-finetune'))
target = root / 'qwenvl' / 'train' / 'train_qwen.py'
source = target.read_text(encoding='utf-8')
eager = 'from trainer import replace_qwen2_vl_attention_class\n'
call = ('    if data_args.data_flatten or data_args.data_packing:\n'
        '        replace_qwen2_vl_attention_class()')
replacement = ('    if data_args.data_flatten or data_args.data_packing:\n'
               '        from trainer import replace_qwen2_vl_attention_class\n'
               '        replace_qwen2_vl_attention_class()')
if source.count(eager) != 1 or source.count(call) != 1:
    raise RuntimeError('Qwen training source differs from the pinned, reviewed commit')
target.write_text(source.replace(eager, '').replace(call, replacement), encoding='utf-8')
print(f'lazy FlashAttention import applied to {target}')
