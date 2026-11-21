"""Export a verified mixed480 adapter, then check every merged tensor on CPU."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--base', type=Path, default=Path('models/Qwen3-VL-4B-Instruct'))
    p.add_argument('--preserve-inference-assets', action='store_true')
    args = p.parse_args()
    if args.output.exists() or args.verification.exists():
        raise FileExistsError('Refusing to overwrite an export or verification')
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    saved = json.loads((args.checkpoint/'checkpoint.json').read_text())
    base_hashes = {p.name:file_sha256(p) for p in args.base.iterdir() if p.suffix in ('.json','.safetensors')}
    if saved['manifest']['base_files_sha256'] != base_hashes:
        raise ValueError('Base fingerprint differs from the trained checkpoint')
    for name, digest in saved['files_sha256'].items():
        if file_sha256(args.checkpoint/name) != digest:
            raise ValueError(f'Checkpoint changed: {name}')
    command = [sys.executable, str(Path(__file__).with_name('merge_lora.py')),
               '--base-model', str(args.base), '--adapter', str(args.checkpoint/'adapter'),
               '--output', str(args.output)]
    if args.preserve_inference_assets:
        command.append('--preserve-inference-assets')
    subprocess.run(command, check=True)
    config = json.loads((args.checkpoint/'adapter/adapter_config.json').read_text())
    if any(config.get(k) for k in ('use_rslora','use_dora','rank_pattern','alpha_pattern')):
        raise ValueError('Expected uniform ordinary LoRA')
    adapter = load_file(args.checkpoint/'adapter/adapter_model.safetensors')
    prefixes = {k.removesuffix('.lora_A.weight') for k in adapter if k.endswith('.lora_A.weight')}
    if set(adapter) != {f'{prefix}.lora_{matrix}.weight' for prefix in prefixes for matrix in ('A','B')}:
        raise ValueError('Unexpected adapter tensor keys')
    targets = {p.removeprefix('base_model.model.')+'.weight':p for p in prefixes}
    indices = [json.loads((root/'model.safetensors.index.json').read_text())['weight_map'] for root in (args.base,args.output)]
    if set(indices[0]) != set(indices[1]) or len(targets) != 144 or not set(targets) <= set(indices[0]):
        raise ValueError('Unexpected base/merged/LoRA tensor mapping')
    changed = 0
    with ExitStack() as stack:
        readers = [{name:stack.enter_context(safe_open(root/name,framework='pt',device='cpu')) for name in set(index.values())}
                   for root,index in zip((args.base,args.output),indices)]
        for name in indices[0]:
            original, merged = [reader[index[name]].get_tensor(name) for reader,index in zip(readers,indices)]
            expected = original
            if name in targets:
                prefix = targets[name]
                delta = (adapter[prefix+'.lora_B.weight'] @ adapter[prefix+'.lora_A.weight']) * (config['lora_alpha']/config['r'])
                expected = (original.float()+delta).to(original.dtype)
                changed += int(not torch.equal(original,merged))
            if not torch.equal(expected,merged):
                raise ValueError(f'Export tensor mismatch: {name}')
    if not changed:
        raise ValueError('Adapter export changed no target weights')
    result = {'passed':True,'checkpoint':str(args.checkpoint),'checkpoint_sha256':file_sha256(args.checkpoint/'checkpoint.json'),
              'output':str(args.output),'all_tensors_verified':len(indices[0]),'lora_targets':len(targets),'changed_targets':changed,
              'export_sha256':{p.name:file_sha256(p) for p in args.output.iterdir() if p.suffix in ('.json','.safetensors')},
              'note':'Exact base+FP32 delta rounded to BF16; native unmerged generation can differ.'}
    args.verification.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: every exported tensor verified',flush=True)


if __name__ == '__main__':
    main()
