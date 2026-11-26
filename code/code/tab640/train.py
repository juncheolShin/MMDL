import json
import os
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from peft import PeftModel
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration,set_seed
from common import OUT,DATA,PARENT,SAVE,BASE,request,read_rows,sha

rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
assert dist.get_world_size()==3 and dist.get_rank()==rank
torch.set_num_threads(2)
set_seed(42)
assert not SAVE.exists()
assert sha(PARENT/'checkpoint.json')=='89609b0bd6629983e34f9e6b2dae0ca3e0564cc8696dd67e047996c8f50b44bb'
parent=json.loads((PARENT/'checkpoint.json').read_text())
assert all(sha(PARENT/k)==v for k,v in parent['files_sha256'].items())
proc=AutoProcessor.from_pretrained(BASE,local_files_only=True)
base=Qwen3VLForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,
                                                      attn_implementation='sdpa').to(rank)
model=PeftModel.from_pretrained(base,PARENT/'adapter',is_trainable=True)
model.train()
model.enable_input_require_grads()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
model.config.use_cache=False
params=[p for n,p in model.named_parameters() if p.requires_grad]
assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
assert sum(p.numel() for p in params)==5898240
optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0)
set_seed(42+rank)
rows=read_rows('train')
assert len(rows)==1280
eos=proc.tokenizer.eos_token_id
steps=[]
start=time.monotonic()
for step in range(320):
    batch=rows[step*4:step*4+4]
    optimizer.zero_grad(set_to_none=True)
    local_ids=[]
    total=torch.zeros(1,device=rank)
    for j,row in enumerate(batch):
        if (step+j)%3!=rank:continue
        req=request(row,proc)
        suffix=proc.tokenizer.encode(row['target'],add_special_tokens=False)+[eos]
        assert len(suffix)>=2
        text=req['prompt']+row['target']+'<|im_end|>'
        inputs=proc(text=[text],images=req['multi_modal_data']['image'],return_tensors='pt').to(rank)
        labels=inputs['input_ids'][0,-len(suffix):]
        assert labels.tolist()==suffix
        logits=model(**inputs,logits_to_keep=len(suffix)+1,use_cache=False).logits[0,:-1].float()
        loss=F.cross_entropy(logits,labels)
        (loss/4).backward()
        total+=loss.detach()/4
        local_ids.append(row['id'])
    # Each sample contributes exactly 1/4 to the shared gradient, including double-work ranks.
    grads=[p.grad for p in params]
    assert all(g is not None for g in grads)
    flat=parameters_to_vector(grads)
    dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    vector_to_parameters(flat,grads)
    norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
    optimizer.step()
    dist.all_reduce(total)
    distributed_ids=[None]*3
    dist.all_gather_object(distributed_ids,local_ids)
    assert sorted(sum(distributed_ids,[]))==sorted(r['id'] for r in batch)
    if rank==0:
        steps.append(dict(update=step+1,ids_by_rank=distributed_ids,loss=total.item(),grad_norm=norm.item()))
        print('step',step+1,'/320',total.item(),round(time.monotonic()-start,1),flush=True)
if rank==0:
    temp=SAVE.with_name(SAVE.name+'.saving')
    temp.mkdir(parents=True)
    model.save_pretrained(temp/'adapter')
    files={str(p.relative_to(temp)):sha(p) for p in (temp/'adapter').iterdir() if p.is_file()}
    assert files['adapter/adapter_model.safetensors']!=parent['files_sha256']['adapter/adapter_model.safetensors']
    (temp/'checkpoint.json').write_text(json.dumps(dict(manifest=dict(base_files_sha256=parent['manifest']['base_files_sha256'],
        parent_checkpoint_sha256=sha(PARENT/'checkpoint.json')),files_sha256=files,completed_updates=320),indent=2))
    temp.rename(SAVE)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'training.json').write_text(json.dumps(dict(steps=steps,world_size=3,effective_batch=4,lr=1e-5,
        train_manifest=sha(DATA/'train.json'),seconds=time.monotonic()-start),indent=2))
dist.barrier()
dist.destroy_process_group()
