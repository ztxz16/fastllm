"""Optional GPU development test; PyTorch is only the independent oracle.

Usage: python test/ops/test_dsv41_mega.py /path/to/mega-test.so [checkpoint]
Checks real dimensions, modulo expert placement, empty ranks, and graph replay.
"""
import ctypes as C
import json
import sys
import time
from pathlib import Path
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(1789)
lib = C.CDLL(sys.argv[1])
ptr = C.c_void_p
lib.mega_test_workspace_bytes.restype = C.c_size_t
lib.mega_test_init.argtypes = [ptr, C.POINTER(C.c_int), C.c_int]
lib.mega_test_create.argtypes = [C.POINTER(ptr), C.POINTER(ptr), C.POINTER(ptr), C.c_int]
lib.mega_test_create.restype = ptr
lib.mega_test_run.argtypes = [ptr, ptr, ptr, ptr, ptr, C.c_int, ptr]
lib.mega_test_run_rows.argtypes = lib.mega_test_run.argtypes
lib.mega_test_run_local.argtypes = lib.mega_test_run.argtypes
lib.mega_test_local_post.argtypes = [ptr]*8 + [C.c_int, ptr]
lib.mega_test_shared.argtypes = [ptr]*5
lib.mega_test_run_shared.argtypes = lib.mega_test_run.argtypes
lib.mega_test_destroy.argtypes = [ptr]
H, I, E, R, K = 5120, 2304, 384, 8, 6
devices = (C.c_int * R)(*range(R))
workspace = [torch.empty(lib.mega_test_workspace_bytes(), dtype=torch.uint8, device=r) for r in range(R)]
peers = (ptr * R)(*[t.data_ptr() for t in workspace])
layers, storage, refs = [], [], {}
checkpoint = Path(sys.argv[2]) if len(sys.argv) > 2 else None
weight_map = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map'] if checkpoint else None

def checkpoint_weight(expert, part, device):
    from safetensors import safe_open
    prefix = f'layers.0.ffn.experts.{expert}.'
    def read(suffix):
        key = prefix + suffix
        with safe_open(str(checkpoint / weight_map[key]), framework='pt') as f:
            return f.get_tensor(key).view(torch.uint8)
    names = ['w1', 'w3'] if part == 0 else ['w2']
    raw = torch.cat([read(name+'.weight').flatten() for name in names])
    sf = torch.cat([read(name+'.scale').flatten() for name in names])
    return torch.cat([raw, sf]).to(device)

def make_weight(n, k, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    raw = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device, generator=gen)
    sf = torch.randint(117, 121, (n, k // 32), dtype=torch.uint8, device=device, generator=gen)
    return torch.cat([raw.flatten(), sf.flatten()])

def dequant(w, n, k):
    raw = w[:n*k//2].reshape(n, k//2).long()
    idx = torch.stack([raw & 15, raw >> 4], -1).reshape(n, k)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=w.device)
    sf = torch.exp2(w[n*k//2:].reshape(n, k//32).float()-127).repeat_interleave(32, dim=1)
    return lut[idx] * sf

def quant(x):
    shape = x.shape
    y = x.float().reshape(-1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(y.abs().amax(-1).clamp_min(1e-4) / 448)))
    return ((y / scale[:, None]).to(torch.float8_e4m3fn).float() * scale[:, None]).reshape(shape)

# Six experts per rank are nonzero. Routes span all eight ranks, reuse experts,
# and deliberately include IDs > 8 to detect modulo-vs-contiguous ID mistakes.
chosen = [r + 8 * (j * 7 + 1) for r in range(R) for j in range(6)]
for r in range(R):
    torch.cuda.set_device(r)
    lib.mega_test_init(peers[r], devices, r)
    zero_g = torch.zeros(I*H + I*2*H//32, dtype=torch.uint8, device=r)
    zero_d = torch.zeros(H*I//2 + H*I//32, dtype=torch.uint8, device=r)
    zero_g[I*H:] = 117
    zero_d[H*I//2:] = 117
    gates, downs = [zero_g]*48, [zero_d]*48
    for e in chosen:
        if e % R != r: continue
        gates[e//R] = checkpoint_weight(e, 0, r) if checkpoint else make_weight(2*I, H, e*2+1, r)
        downs[e//R] = checkpoint_weight(e, 1, r) if checkpoint else make_weight(H, I, e*2+2, r)
        refs[e] = (gates[e//R], downs[e//R])
    storage.append((gates, downs))
    layers.append(lib.mega_test_create((ptr*48)(*[v.data_ptr() for v in gates]),
        (ptr*48)(*[v.data_ptr() for v in downs]), peers, r))
print('packed layers', flush=True)

def shared_weight(names, n, k):
    if checkpoint:
        from safetensors import safe_open
        def read(name, kind):
            key = 'layers.0.ffn.shared_experts.' + name + '.' + kind
            with safe_open(str(checkpoint / weight_map[key]), framework='pt') as f:
                return f.get_tensor(key).view(torch.uint8)
        raw = torch.cat([read(name, 'weight') for name in names])
        sf = torch.cat([read(name, 'scale') for name in names])
    else:
        raw = torch.randn(n,k).to(torch.float8_e4m3fn).view(torch.uint8)
        sf = torch.full((n//32,k//32), 120, dtype=torch.uint8)
    weight = raw.to(0).view(torch.float8_e4m3fn).float()
    factor = torch.exp2(sf.to(device=0, dtype=torch.float32)-127)
    return raw, sf, weight * factor.repeat_interleave(32,0).repeat_interleave(32,1)

shared_gate = shared_weight(['w1','w3'], 2*I, H)
shared_down = shared_weight(['w2'], H, I)
for r in range(R):
    torch.cuda.set_device(r)
    gate, down = shared_gate[0].to(r), shared_down[0].to(r)
    lib.mega_test_shared(layers[r], gate.data_ptr(), shared_gate[1].data_ptr(),
                         down.data_ptr(), shared_down[1].data_ptr())
results = []
for rows in [1, 17, 128]:
    x0 = torch.randn(rows, H, device=0, dtype=torch.bfloat16)
    ids0 = torch.tensor([[chosen[(row*5+j*9) % len(chosen)] for j in range(K)] for row in range(rows)], dtype=torch.int32, device=0)
    scores0 = torch.softmax(torch.randn(rows, K, device=0), dim=-1)
    x = [x0.to(r) for r in range(R)]
    ids = [ids0.to(r) for r in range(R)]
    scores = [scores0.to(r) for r in range(R)]
    out = [torch.empty(rows, H, device=r) for r in range(R)]
    streams = [torch.cuda.Stream(device=r) for r in range(R)]
    for r in range(R): torch.cuda.synchronize(r)
    def run():
        for r in range(R):
            torch.cuda.set_device(r)
            lib.mega_test_run(layers[r], x[r].data_ptr(), ids[r].data_ptr(), scores[r].data_ptr(), out[r].data_ptr(), rows, streams[r].cuda_stream)
    def sync():
        for stream in streams: stream.synchronize()
    run(); sync()
    actual = sum(t.to(0) for t in out)
    ref_parts = [torch.zeros_like(t) for t in out]
    # Reuse a dequantized expert across all tokens routed to it.
    for e in chosen:
        rank = e % R
        ri, slots = (ids[rank] == e).nonzero(as_tuple=True)
        if ri.numel() == 0:
            continue
        wg, wd = refs[e]
        gu = (quant(x[rank][ri]) @ dequant(wg, 2*I, H).T).to(torch.bfloat16).float()
        gate, up = gu.split(I, -1)
        gate = gate.clamp(max=10)
        act = (gate / (1 + torch.exp(-gate)) * up.clamp(-10, 10) * scores[rank][ri, slots, None]).to(torch.bfloat16).float()
        y = (quant(act) @ dequant(wd, H, I).T).to(torch.bfloat16).float()
        ref_parts[rank][ri] += y
    ref = sum(t.to(0) for t in ref_parts)
    rank_errors = [((out[r]-ref_parts[r]).norm()/ref_parts[r].norm().clamp_min(1e-20)).item() for r in range(R)]
    assert max(rank_errors) < 1e-4, rank_errors
    relative_l2 = ((actual-ref).norm()/ref.norm()).item()
    max_error = (actual-ref).abs().max().item()
    assert torch.isfinite(actual).all() and relative_l2 < 1e-4, (rows, relative_l2, max_error)
    if rows % R == 0:
        for r in range(R):
            torch.cuda.set_device(r)
            lib.mega_test_run_rows(layers[r], x[r].data_ptr(), ids[r].data_ptr(), scores[r].data_ptr(),
                                   out[r].data_ptr(), rows, streams[r].cuda_stream)
        sync()
        gathered = torch.cat([out[r].chunk(R)[r].to(0) for r in range(R)])
        assert torch.equal(actual, gathered), 'owner sum differs from rank-ordered reduction'
    local = [torch.empty(max(1, (rows+R-1-r)//R), H, device=r, dtype=torch.bfloat16) for r in range(R)]
    def run_shared(r):
        lib.mega_test_run_shared(layers[r], x[r].data_ptr(), ids[r].data_ptr(), scores[r].data_ptr(),
                                 local[r].data_ptr(), rows, streams[r].cuda_stream)
    for r in range(R):
        torch.cuda.set_device(r)
        run_shared(r)
    sync()
    def combine():
        return torch.cat([t[:(rows+R-1-r)//R].to(0) for r,t in enumerate(local)])
    shared_actual = combine()
    gate, up = (quant(x0) @ shared_gate[2].T).to(torch.bfloat16).float().split(I,-1)
    v = gate.clamp(max=10)
    act = (v/(1+torch.exp(-v)) * up.clamp(-10,10)).to(torch.bfloat16)
    shared_reference = (quant(act) @ shared_down[2].T).to(torch.bfloat16).float()
    joint_ref = (shared_reference + ref).to(torch.bfloat16).float()
    joint_error = ((shared_actual.float()-joint_ref).norm()/joint_ref.norm()).item()
    assert joint_error < 0.0005, (rows, joint_error)
    # Deliberately unbalance one real batch across ranks. Decode ownership
    # follows cache placement and can have empty ranks between active ones.
    counts = [min(32, max(0, rows - 32*i)) for i in range(R)]
    counts = [counts[i] for i in (2,0,4,1,6,3,7,5)]
    begin = 0
    pieces = []
    for r, count in enumerate(counts):
        torch.cuda.set_device(r)
        yy = torch.empty(max(1,count),H,device=r,dtype=torch.bfloat16)
        xx, ii, ss = x[r][begin:begin+count], ids[r][begin:begin+count], scores[r][begin:begin+count]
        lib.mega_test_run_local(layers[r], xx.data_ptr(), ii.data_ptr(), ss.data_ptr(), yy.data_ptr(),
                                count, streams[r].cuda_stream)
        pieces.append(yy[:count]); begin += count
    sync()
    assert torch.equal(torch.cat([y.to(0) for y in pieces]), shared_actual), 'rank-local ownership mismatch'
    residuals = [torch.randn((count,4,H), device=r, dtype=torch.bfloat16) for r,count in enumerate(counts)]
    posts = [torch.rand(count,4,device=r) for r,count in enumerate(counts)]
    combs = [torch.rand(count,4,4,device=r) for r,count in enumerate(counts)]
    hidden = [torch.empty_like(z) for z in residuals]
    begin = 0
    for r, count in enumerate(counts):
        torch.cuda.set_device(r)
        torch.cuda.current_stream().synchronize()
        lib.mega_test_local_post(layers[r], x[r][begin:begin+count].data_ptr(),
            ids[r][begin:begin+count].data_ptr(), scores[r][begin:begin+count].data_ptr(),
            residuals[r].data_ptr(), posts[r].data_ptr(), combs[r].data_ptr(),
            hidden[r].data_ptr(), count, streams[r].cuda_stream)
        begin += count
    sync()
    for r, count in enumerate(counts):
        if count == 0: continue
        total = torch.zeros_like(residuals[r], dtype=torch.float32)
        for c in range(4):
            total += residuals[r][:,c:c+1].float() * combs[r][:,c,:,None]
        expected = (total + pieces[r].float()[:,None,:] * posts[r][:,:,None]).bfloat16()
        assert torch.equal(hidden[r], expected), 'combined expert residual differs'
    joint_graphs = []
    for r in range(R):
        torch.cuda.set_device(r)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=streams[r]):
            for _ in range(16): run_shared(r)
        joint_graphs.append(g)
    for r,g in enumerate(joint_graphs):
        torch.cuda.set_device(r)
        with torch.cuda.stream(streams[r]): g.replay()
    sync()
    assert torch.equal(combine(),shared_actual), 'shared expert graph replay differs'
    graphs = []
    graph_steps = 32
    for r in range(R):
        torch.cuda.set_device(r)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=streams[r]):
            for _ in range(graph_steps):
                lib.mega_test_run(layers[r], x[r].data_ptr(), ids[r].data_ptr(), scores[r].data_ptr(), out[r].data_ptr(), rows, streams[r].cuda_stream)
        graphs.append(graph)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(R)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(R)]
    for r, graph in enumerate(graphs):
        torch.cuda.set_device(r)
        with torch.cuda.stream(streams[r]):
            start_events[r].record()
            graph.replay()
            end_events[r].record()
    sync()
    replay = sum(t.to(0) for t in out)
    assert torch.equal(actual, replay), 'graph replay differs'
    graph_ms = [start_events[r].elapsed_time(end_events[r])/graph_steps for r in range(R)]
    start = time.perf_counter()
    for _ in range(20): run()
    sync()
    results.append(dict(rows=rows, relative_l2=relative_l2, max_abs_error=max_error, shared_relative_l2=joint_error, graph_exact=True,
                        rank_relative_l2=rank_errors,
                        graph_steps=graph_steps, graph_cycle_ms_per_rank=graph_ms,
                        eager_host_cycle_ms=(time.perf_counter()-start)*1000/20))
    print(json.dumps(results[-1]), flush=True)
for r, layer in enumerate(layers):
    torch.cuda.set_device(r); lib.mega_test_destroy(layer)
print(json.dumps({'results': results}), flush=True)
