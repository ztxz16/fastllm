import argparse,json
from pathlib import Path
import torch
from safetensors import safe_open
p=argparse.ArgumentParser(description='Independent TP8 draft body fixture with BF16 intermediate rounding')
p.add_argument('--draft',required=True,help='Draft model.safetensors checkpoint')
p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False
with safe_open(a.draft,framework='pt',device='cpu') as f:
 raw={k:f.get_tensor(k) for k in f.keys() if k.startswith('layers.') or k in ('norm.weight','mask_embedding')}
tensors={}
for layer in range(5):
 p=f'layers.{layer}.'
 for n in ['input_layernorm.weight','post_attention_layernorm.weight','self_attn.q_norm.weight','self_attn.k_norm.weight']:
  tensors['dspark.'+p+n]=raw[p+n].float()
 tensors['dspark.'+p+'self_attn.mergeqkv.weight']=torch.cat([raw[p+'self_attn.'+n+'_proj.weight'] for n in ('q','k','v')])
 tensors['dspark.'+p+'self_attn.o_proj.weight']=raw[p+'self_attn.o_proj.weight']
 tensors['dspark.'+p+'mlp.gateup_proj.weight']=torch.cat([raw[p+'mlp.'+n+'_proj.weight'] for n in ('gate','up')])
 tensors['dspark.'+p+'mlp.down_proj.weight']=raw[p+'mlp.down_proj.weight']
tensors['dspark.norm.weight']=raw['norm.weight'].float()
tensors['dspark.mask_embedding']=raw['mask_embedding'].bfloat16()
tensors['model.embed_tokens.weight']=torch.randn(64,4096).bfloat16()
tensors['test.context']=torch.randn(64,17,4096).bfloat16()
def norm(x,g):return ((x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)).bfloat16().float()*g.float()).bfloat16()
def mm(x,w):return (x.float()@w.float().T).bfloat16()
def rope(x,position):
 ang=position.float()[:,None]*(10000.**(-torch.arange(64,device=0).float()/64))[None,:]
 c,s=ang.cos().bfloat16().float(),ang.sin().bfloat16().float()
 while c.ndim<x.ndim:c=c.unsqueeze(0)
 p,q=x[...,:64].float(),x[...,64:].float()
 return torch.cat([(p*c).bfloat16().float()-(q*s).bfloat16().float(),(q*c).bfloat16().float()+(p*s).bfloat16().float()],-1).bfloat16()
x=torch.cat([tensors['model.embed_tokens.weight'][:,None,:],tensors['dspark.mask_embedding'].reshape(1,1,4096).expand(64,6,4096)],1).to(0)
context=tensors['test.context'].to(0)
for layer in range(5):
 p=f'layers.{layer}.';w={n[len(p):]:v.to(0) for n,v in raw.items() if n.startswith(p)}
 nx=norm(x,w['input_layernorm.weight']);outputs=[]
 for rank in range(8):
  kv=rank//2
  q=mm(nx,w['self_attn.q_proj.weight'][rank*512:(rank+1)*512]).reshape(64,7,4,128).transpose(1,2)
  k=mm(nx,w['self_attn.k_proj.weight'][kv*128:(kv+1)*128]).unsqueeze(1)
  v=mm(nx,w['self_attn.v_proj.weight'][kv*128:(kv+1)*128]).unsqueeze(1)
  q=rope(norm(q,w['self_attn.q_norm.weight']),torch.arange(17,24,device=0))
  k=rope(norm(k,w['self_attn.k_norm.weight']),torch.arange(17,24,device=0))
  ck=mm(context,w['self_attn.k_proj.weight'][kv*128:(kv+1)*128]).unsqueeze(1)
  cv=mm(context,w['self_attn.v_proj.weight'][kv*128:(kv+1)*128]).unsqueeze(1)
  ck=rope(norm(ck,w['self_attn.k_norm.weight']),torch.arange(17,device=0))
  keys=torch.cat([ck,k],2);values=torch.cat([cv,v],2)
  logits=((q.float()@keys.float().transpose(-1,-2)).bfloat16().float()/(128**.5)).bfloat16().float()
  # DFlash draft attention is non-causal within its proposed seven-row block.
  attn=(logits.softmax(-1).bfloat16().float()@values.float()).bfloat16().transpose(1,2).reshape(64,7,512)
  outputs.append(mm(attn,w['self_attn.o_proj.weight'][:,rank*512:(rank+1)*512]))
 x=(x.float()+torch.stack([o.float() for o in outputs]).sum(0).bfloat16().float()).bfloat16()
 nx=norm(x,w['post_attention_layernorm.weight']);outputs=[]
 for rank in range(8):
  gate=mm(nx,w['mlp.gate_proj.weight'][rank*512:(rank+1)*512]);up=mm(nx,w['mlp.up_proj.weight'][rank*512:(rank+1)*512])
  activated=(torch.nn.functional.silu(gate.float()).bfloat16().float()*up.float()).bfloat16()
  outputs.append(mm(activated,w['mlp.down_proj.weight'][:,rank*512:(rank+1)*512]))
 x=(x.float()+torch.stack([o.float() for o in outputs]).sum(0).bfloat16().float()).bfloat16()
 print('REFERENCE_LAYER',layer,flush=True)
tensors['test.expected']=norm(x,raw['norm.weight'].to(0)).float().cpu()
manifest=dict(hidden_size=4096,layers=5,heads=32,kv_heads=4,head_dim=128,window=1024,block=7,eps=1e-5,theta=10000.,tensors=[])
for i,(name,value) in enumerate(tensors.items()):
 value=value.cpu().contiguous();file=f'{i}.bin';value.view(torch.uint8).numpy().tofile(out/file)
 manifest['tensors'].append(dict(name=name,shape=list(value.shape),dtype='float32' if value.dtype==torch.float32 else 'bfloat16',file=file))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('FIXTURE_PASS',flush=True)
