import json,sys,random
from collections import Counter
from pathlib import Path
from training import BASE,digest,fit_adapter,PeftModel,AutoProcessor,Qwen3VLForConditionalGeneration,set_seed,torch
ref=Path('icot/recipes')
p=Path('results/icot_498')
p.mkdir(parents=True,exist_ok=True)
parent=Path('icot/adapters/495')
save=Path('checkpoints/icot_498')
save.parent.mkdir(exist_ok=True)
if save.exists(): raise FileExistsError(save)
samples=json.loads((ref/'498.json').read_text())
pool=json.loads(Path('training_data/icot/rows.json').read_text())
original=[{'row':pool[v['id']],'kind':v['role'],'target':v['probabilities']} for v in samples]
letters='ABCD'
gold=[x['row']['gold'] for x in original];old=Counter(gold)
assert len(gold)==12 and all(list(x['row']['options'])==list(letters) for x in original)
targets=[c for c in letters for _ in range(6-old[c])]
rng=random.Random(496)
for _ in range(50000):
    rng.shuffle(targets)
    if all(x!=y for x,y in zip(gold,targets)): break
else: raise ValueError('No balanced assignment')
second=[]
proof=[]
for item,g in zip(original,targets,strict=True):
    r=item['row'];shift=(letters.index(g)-letters.index(r['gold']))%4
    order=[(i-shift)%4 for i in range(4)]
    opts={c:r['options'][letters[i]] for c,i in zip(letters,order)}
    y=[item['target'][i] for i in order]
    assert opts[g]==r['options'][r['gold']] and {v:p for v,p in zip(opts.values(),y)}==dict(zip(r['options'].values(),item['target']))
    second.append({**item,'row':{**r,'gold':g,'options':opts},'target':y})
    proof.append(dict(id=r['id'],old=r['gold'],new=g,shift=shift))
assert Counter(gold+targets)==dict.fromkeys(letters,6)
(p/'design.json').write_text(json.dumps(dict(parent=str(parent),reference=496,selection_digest=digest(ref/'498.json'),epochs=2,lr=1e-5,updates=6,seed=42,only_intervention='Rotate options and corresponding soft targets during second epoch. External dev and MMMU unchanged.',permutations=proof),indent=2))
torch.set_num_threads(2);set_seed(42)
processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
base=Qwen3VLForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
model=PeftModel.from_pretrained(base,parent/'adapter',is_trainable=True)
set_seed(42)
steps,n=fit_adapter(model,processor,original,save,parent,1e-5,2,epoch_examples=[original,second])
prev=json.loads(Path('icot/evidence/training_498.json').read_text())
assert [x['ids'] for x in steps]==[x['ids'] for x in prev['steps']] and len(steps)==6 and n==prev['trainable']
(p/'training.json').write_text(json.dumps(dict(steps=steps,trainable=n,first_epoch_max_loss_delta_vs_recorded_498=max(abs(a['mean_loss']-b['mean_loss']) for a,b in zip(steps[:3],prev['steps'][:3])),parent_sha256=digest(parent/'checkpoint.json')),indent=2))
