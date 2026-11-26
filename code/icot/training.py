import hashlib
import json
import math
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed

sys.path.insert(0, str(Path('code').resolve()))
sys.path.insert(0, str(Path('code/eval').resolve()))
from prompting import build_mmmu_prompt
from third_party.qwen3vl_mmmu.input_utils import prepare_inputs_for_vllm

BASE = Path('models/Qwen3-VL-4B-Instruct')
ACCUM = 4
TEMP = 2.0


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inputs(processor, row, cue=None):
    instruction = f'Checked evidence: {cue}\nAnswer with the option letter only.' if cue else None
    request = prepare_inputs_for_vllm(messages(row, instruction), processor)
    return processor(text=[request['prompt']], images=request['multi_modal_data']['image'],
                     return_tensors='pt').to('cuda')


def logits(model, processor, row, cue=None):
    return model(**inputs(processor, row, cue), use_cache=False, logits_to_keep=1).logits[0, -1].float()


def option_ids(processor, row):
    tokens = [processor.tokenizer.encode(c, add_special_tokens=False) for c in row['options']]
    assert all(len(ids) == 1 for ids in tokens)
    return [ids[0] for ids in tokens]


def messages(row, instruction=None):
    line = {'id': row['id'], 'question_type': 'multiple-choice', 'question': row['question'], **row['options']}
    result = build_mmmu_prompt(line, [str(Path(row['image']).resolve())], 262144, 2097152)
    if instruction:
        content = result[0]['content'][-1]
        content['text'] = content['text'].removesuffix('Answer with the option letter only.') + instruction
    return result

def fit_adapter(model, processor, prepared, save, parent_path, learning_rate, epochs, *, epoch_examples=None):
    params = [p for n,p in model.named_parameters() if p.requires_grad]
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0)
    model.train()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache = False
    steps=[]
    for epoch in range(epochs):
        if epoch_examples is not None:
            prepared = epoch_examples[epoch]
        indices=list(range(len(prepared)))
        random.Random(42+epoch).shuffle(indices)
        for start in range(0,len(indices),ACCUM):
            batch=indices[start:start+ACCUM]
            opt.zero_grad(set_to_none=True)
            losses=[]
            for i in batch:
                item=prepared[i]
                row=item['row']
                ids=option_ids(processor,row)
                gold=ids[list(row['options']).index(row['gold'])]
                pred=logits(model,processor,row)
                hard=-F.log_softmax(pred,dim=0)[gold]
                kl=F.kl_div(F.log_softmax(pred[ids]/TEMP,dim=0),
                            pred.new_tensor(item['target']),reduction='sum') * TEMP**2
                loss=(hard+kl)/2
                assert torch.isfinite(loss)
                (loss/len(batch)).backward()
                losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            opt.step()
            steps.append(dict(epoch=epoch+1,ids=[prepared[i]['row']['id'] for i in batch],
                              mean_loss=sum(losses)/len(losses),gradient_norm=float(norm)))
            print('update',len(steps),'/',epochs*math.ceil(len(prepared)/ACCUM),steps[-1]['mean_loss'],flush=True)
    save.mkdir()
    model.save_pretrained(save/'adapter')
    parent=json.loads((parent_path/'checkpoint.json').read_text())
    files={str(p.relative_to(save)):digest(p) for p in sorted((save/'adapter').glob('*')) if p.is_file()}
    assert files['adapter/adapter_model.safetensors']!=digest(parent_path/'adapter/adapter_model.safetensors')
    proof={'manifest':{'base_files_sha256':parent['manifest']['base_files_sha256'],
                        'parent_checkpoint_sha256':digest(parent_path/'checkpoint.json')},
           'files_sha256':files,'completed_updates':len(steps)}
    (save/'checkpoint.json').write_text(json.dumps(proof,indent=2))
    return steps, sum(p.numel() for p in params)
