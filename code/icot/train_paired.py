import json
from pathlib import Path
import sys
from training import (BASE, AutoProcessor, PeftModel, Qwen3VLForConditionalGeneration, digest, messages, prepare_inputs_for_vllm, set_seed, torch)
import torch.nn.functional as F

out=Path('results/icot_paired')
out.mkdir(parents=True,exist_ok=True)
name=sys.argv[1]
assert name in ('answer','short')
design=json.loads(Path('icot/recipes/paired.json').read_text())
pool=json.loads(Path('training_data/icot/rows.json').read_text())
design['epochs']=[[pool[row['id']] | row for row in epoch] for epoch in design['epochs']]
parent=Path('icot/adapters/498')
save=Path('checkpoints')/(out.name+'_'+name)
assert digest(parent/'checkpoint.json')==design['parent_sha256'] and not save.exists()
torch.set_num_threads(2)
set_seed(design['seed'])
processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
tok=processor.tokenizer
eos=tok.convert_tokens_to_ids('<|im_end|>')
assert eos==tok.eos_token_id
base=Qwen3VLForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
model=PeftModel.from_pretrained(base,parent/'adapter',is_trainable=True)
params=[p for n,p in model.named_parameters() if p.requires_grad]
assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
assert sum(p.numel() for p in params)==5898240
model.train()
model.enable_input_require_grads()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
model.config.use_cache=False
opt=torch.optim.AdamW(params,lr=design['lr'],weight_decay=0)
steps=[]
set_seed(design['seed'])
for epoch,items in enumerate(design['epochs'],start=1):
    for start in range(0,len(items),4):
        batch=items[start:start+4]
        opt.zero_grad(set_to_none=True)
        losses=[]; counts=[]; prompt_digests=[]
        for row in batch:
            request=prepare_inputs_for_vllm(messages(row),processor)
            prompt=request['prompt']
            target=row['target_'+name]
            suffix=tok.encode(target,add_special_tokens=False)+[eos]
            assert 2<=len(suffix)<=129
            assert tok.encode(prompt+target+'<|im_end|>',add_special_tokens=False)==tok.encode(prompt,add_special_tokens=False)+suffix
            inp=processor(text=[prompt+target+'<|im_end|>'],images=request['multi_modal_data']['image'],return_tensors='pt').to('cuda')
            labels=inp['input_ids'][0,-len(suffix):]
            assert labels.tolist()==suffix
            pred=model(**inp,logits_to_keep=len(suffix)+1,use_cache=False).logits[0,:-1].float()
            assert pred.shape[0]==len(suffix)
            loss=F.cross_entropy(pred,labels)
            assert torch.isfinite(loss)
            (loss/len(batch)).backward()
            losses.append(loss.detach().item());counts.append(len(suffix))
            import hashlib
            prompt_digests.append(hashlib.sha256(prompt.encode()).hexdigest())
        norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        opt.step()
        step={'epoch':epoch,'ids':[v['id'] for v in batch],'loss':sum(losses)/len(batch),'target_lengths':counts,'prompt_hashes':prompt_digests,'gradient_norm':norm.item()}
        steps.append(step);print(name,len(steps),'/ 6',step['loss'],flush=True)
assert len(steps)==6
save.mkdir(parents=True)
model.save_pretrained(save/'adapter')
files={str(p.relative_to(save)):digest(p) for p in sorted((save/'adapter').glob('*')) if p.is_file()}
assert files['adapter/adapter_model.safetensors']!=digest(parent/'adapter/adapter_model.safetensors')
ck={'manifest':{'base_files_sha256':json.loads((parent/'checkpoint.json').read_text())['manifest']['base_files_sha256'],'parent_checkpoint_sha256':design['parent_sha256']},'files_sha256':files,'completed_updates':len(steps)}
(save/'checkpoint.json').write_text(json.dumps(ck,indent=2))
(out/f'train_{name}.json').write_text(json.dumps({'design_sha256':digest(Path('icot/recipes/paired.json')),'steps':steps,'checkpoint_sha256':digest(save/'checkpoint.json')},indent=2))
