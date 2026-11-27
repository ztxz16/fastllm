import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import argparse, ctypes as C, json, time
from pathlib import Path
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--library', required=True, help='Built libfastllm_tools.so or a standalone fused CUDA library')
ap.add_argument('--draft', required=True, help='Draft model.safetensors checkpoint')
ap.add_argument('--target-head', help='Target safetensors shard containing lm_head.weight')
ap.add_argument('--mlp-only', action='store_true')
ap.add_argument('--layers', type=int, default=5)
ap.add_argument('--rows', type=int, default=8)
ap.add_argument('--bench', action='store_true')
ap.add_argument('--requests', help='Comma-separated request counts for batch operator tests')
ap.add_argument('--window', type=int, default=1024)
ap.add_argument('--positions', default='1,17,256,1024,8193,65529,2', help='Request live lengths for batch attention checks')
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(1749)
assert torch.cuda.device_count() >= 8, 'These operator tests require eight peer-connected SM90 or SM103 GPUs'
lib = C.CDLL(a.library)
print('library loaded',flush=True)
P = C.c_void_p
class Weight(C.Structure):
    _fields_ = [(n,P) for n in ['qkv','o','inputNorm','qNorm','kNorm','gateUp','down','postNorm']]
lib.FastllmDraftFusedCreate.argtypes = [P,C.c_int,C.c_int,P,C.c_int,C.c_int,C.c_float,C.c_float]
lib.FastllmDraftFusedCreate.restype = P
lib.FastllmDraftFusedPrepare.argtypes = [P,C.c_int]
lib.FastllmDraftFusedPrepare.restype = C.c_bool
lib.FastllmDraftFusedMlpProbe.argtypes = [P,C.c_int,C.c_int,C.c_int,P,P,P]
lib.FastllmDraftFusedMlpProbe.restype = C.c_bool
lib.FastllmDraftFusedAttentionProbe.argtypes = [P,C.c_int,C.c_int,C.c_int,P,P,P,P,C.c_int,P]
lib.FastllmDraftFusedAttentionProbe.restype = C.c_bool
lib.FastllmDraftFusedWatchdog.argtypes = [P]
lib.FastllmDraftFusedWatchdog.restype = C.c_uint
lib.FastllmDraftFusedDestroy.argtypes = [P]
lib.FastllmDraftFusedRun.argtypes = [P,C.c_int,C.c_int,P,P,P,P]
lib.FastllmDraftFusedRun.restype = C.c_bool
lib.FastllmDraftFusedError.restype = C.c_char_p
devices = (C.c_int*8)(*range(8))
streams = [torch.cuda.Stream(device=r) for r in range(8)]
for r in range(8):
    with torch.cuda.device(r):
        torch.empty(1, device=r).add_(1)
    torch.cuda.synchronize(r)

with safe_open(a.draft, framework='pt',device='cpu') as f:
    raw = [{n:f.get_tensor(f'layers.{i}.'+n) for n in [
        'self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight',
        'self_attn.q_norm.weight','self_attn.k_norm.weight','input_layernorm.weight',
        'post_attention_layernorm.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']}
        for i in range(a.layers)]
weights = [[] for _ in range(8)]
flat = []
for r in range(8):
    for w in raw:
        kv = r//2
        d = dict(qkv=torch.cat([w['self_attn.q_proj.weight'][r*512:(r+1)*512],
                w['self_attn.k_proj.weight'][kv*128:(kv+1)*128],
                w['self_attn.v_proj.weight'][kv*128:(kv+1)*128]]).to(r),
            o=w['self_attn.o_proj.weight'][:,r*512:(r+1)*512].contiguous().to(r),
            inputNorm=w['input_layernorm.weight'].float().to(r),
            qNorm=w['self_attn.q_norm.weight'].float().to(r),
            kNorm=w['self_attn.k_norm.weight'].float().to(r),
            gateUp=torch.cat([w['mlp.gate_proj.weight'][r*512:(r+1)*512],
                w['mlp.up_proj.weight'][r*512:(r+1)*512]]).to(r),
            down=w['mlp.down_proj.weight'][:,r*512:(r+1)*512].contiguous().to(r),
            postNorm=w['post_attention_layernorm.weight'].float().to(r))
        weights[r].append(d)
        flat.append(Weight(*(d[n].data_ptr() for n,_ in Weight._fields_)))
flat = (Weight*len(flat))(*flat)
state = lib.FastllmDraftFusedCreate(devices,8,a.layers,flat,a.rows,a.window,1e-5,10000.)
print('state created',flush=True)
assert not lib.FastllmDraftFusedCreate(devices,4,a.layers,flat,a.rows,a.window,1e-5,10000.)
for invalid_rows in (1,9):
    assert not lib.FastllmDraftFusedCreate(devices,8,a.layers,flat,invalid_rows,1024,1e-5,10000.)
assert not lib.FastllmDraftFusedCreate(devices,8,a.layers,flat,a.rows,1024,1e-5,float('nan'))
assert state, lib.FastllmDraftFusedError()
prepared_capacity=max([4]+[int(x) for x in a.requests.split(',')]) if a.requests else 4
assert lib.FastllmDraftFusedPrepare(state,prepared_capacity), lib.FastllmDraftFusedError()

def sync():
    for r in range(8): torch.cuda.synchronize(r)
def norm(x,g):
    return ((x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)).bfloat16().float()*g).bfloat16()
def gemm(x,w):
    return (x.float()@w.float().T).bfloat16()
def reduce(xs):
    return torch.stack([x.to(0).float() for x in xs]).sum(0).bfloat16()
def metrics(got,ref,x):
    y=got.float().cpu(); z=ref.float().cpu(); xx=x.float().cpu(); diff=y-z
    return dict(l2=float(diff.norm()/z.norm().clamp_min(1e-20)),
        max_row_l2=float((diff.norm(dim=-1)/z.norm(dim=-1).clamp_min(1e-20)).max()),
        branch_l2=float(diff.norm()/(z-xx).norm().clamp_min(1e-20)),max_abs=float(diff.abs().max()))


class ReclaimedMemoryGuard:
    """Overwrite freed cuBLAS arenas while captured graph nodes remain alive."""
    def __init__(self):
        self.cuda = C.CDLL('libcudart.so')
        self.cuda.cudaMalloc.argtypes = [C.POINTER(P), C.c_size_t]
        self.cuda.cudaMemset.argtypes = [P, C.c_int, C.c_size_t]
        self.cuda.cudaMemcpy.argtypes = [P, P, C.c_size_t, C.c_int]
        self.cuda.cudaFree.argtypes = [P]
        self.buffers = []
        for rank in range(8):
            torch.cuda.set_device(rank)
            for _ in range(2):
                ptr = P()
                assert self.cuda.cudaMalloc(C.byref(ptr), 64 << 20) == 0
                assert self.cuda.cudaMemset(ptr, 0xa5, 64 << 20) == 0
                self.buffers.append((rank, ptr))
        sync()

    def close(self):
        sync()
        host = C.create_string_buffer(64 << 20)
        for rank, ptr in self.buffers:
            torch.cuda.set_device(rank)
            assert self.cuda.cudaMemcpy(host, ptr, 64 << 20, 2) == 0
            assert host.raw.count(b'\xa5') == 64 << 20, ('graph wrote released BLAS arena', rank)
            assert self.cuda.cudaFree(ptr) == 0
        self.buffers = []
