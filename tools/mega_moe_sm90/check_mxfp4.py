#!/usr/bin/env python3
"""Check native MXFP4 EP4/EP8 against canonical real-weight references.

PyTorch and safetensors are test dependencies only. The tested shared library
uses neither. Run on idle H20 GPUs; --ranks controls visible devices 0..R-1.
The reference does not call the native repacker or upstream preprocessing.
"""
import argparse, ctypes as C, json, pathlib, time
from concurrent.futures import ThreadPoolExecutor
import torch
from safetensors import safe_open
p = argparse.ArgumentParser()
p.add_argument('--lib', required=True)
p.add_argument('--model', required=True)
p.add_argument('--out', required=True)
p.add_argument('--ranks', type=int, choices=[4, 8], default=8)
a = p.parse_args()
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
lib = C.CDLL(a.lib)
ptr = C.c_void_p
ints = C.c_int
sz = C.c_size_t
specs = {'prefill_capacity': ([], ints), 'workspace_bytes': ([ints], sz), 'init': ([ptr, sz, ptr, ints, ints], ints), 'create': ([ptr, ptr, ptr, sz, ptr, ints, ints], ptr), 'run': ([ptr, ptr, ptr, ptr, ptr, ints, ints, ptr], ints), 'destroy': ([ptr], None), 'error': ([], C.c_char_p)}
for (name, (args, res)) in specs.items():
    f = getattr(lib, 'fastllm_mxfp4_ep_' + name)
    f.argtypes = args
    f.restype = res
    globals()[name] = f
R = a.ranks
chunk_capacity = prefill_capacity()
LE = 256 // R
model = pathlib.Path(a.model)
index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
files = {}

def tensor(name):
    path = str(model / index[name])
    if path not in files:
        files[path] = safe_open(path, framework='pt', device='cpu')
    return files[path].get_tensor(name)

def get(e, leg):
    prefix = f'model.layers.1.mlp.experts.{e}.{leg}_proj.'
    return (tensor(prefix + 'weight'), tensor(prefix + 'weight_scale'))

def address(t):
    return t.data_ptr()
weights = []
arenas = []
handles = []
records = []
started = time.time()
for rank in range(R):
    torch.cuda.set_device(rank)
    local = []
    for e in range(rank * LE, (rank + 1) * LE):
        (g, gs) = get(e, 'gate')
        (u, us) = get(e, 'up')
        (d, ds) = get(e, 'down')
        gate = torch.cat([torch.cat([g, u]).flatten(), torch.cat([gs, us]).flatten()]).cuda()
        down = torch.cat([d.flatten(), ds.flatten()]).cuda()
        local.append((gate, down))
    weights.append(local)
    arenas.append(torch.empty(workspace_bytes(R), device='cuda', dtype=torch.uint8))
    torch.cuda.synchronize()
peers = (ptr * R)(*[address(x) for x in arenas])
devices = (ints * R)(*range(R))
for rank in range(R):
    torch.cuda.set_device(rank)
    assert init(address(arenas[rank]), arenas[rank].numel(), devices, rank, R), error()
    gate = (ptr * LE)(*[address(x[0]) for x in weights[rank]])
    down = (ptr * LE)(*[address(x[1]) for x in weights[rank]])
    h = create(gate, down, address(arenas[rank]), arenas[rank].numel(), peers, rank, R)
    assert h, error()
    handles.append(h)
print('prepared', time.time() - started, 'workspace', workspace_bytes(R), flush=True)
cache = {}
lattice = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device='cuda:0')

# Independently decode canonical adjacent E2M1 nibbles and UE8M0 scales.
def unpack(e):
    if e in cache:
        return cache[e]
    ans = []
    for leg in ['gate', 'up', 'down']:
        (q, s) = get(e, leg)
        q = q.to('cuda:0')
        s = s.to('cuda:0')
        code = torch.stack([q & 15, q >> 4], -1).flatten(-2)
        w = lattice[(code & 7).long()] * torch.where(code < 8, 1.0, -1.0)
        w *= torch.exp2(s.float() - 127).repeat_interleave(32, -1)
        ans.append(w)
    cache[e] = (torch.cat(ans[:2]), ans[2])
    return cache[e]

# Match block-128 power-of-two E4M3 activation quantization.
def quant(x):
    v = x.reshape(x.shape[0], -1, 128)
    sf = torch.exp2(torch.ceil(torch.log2((v.abs().amax(-1) / 448).clamp_min(2 ** (-126)))))
    return ((v / sf[..., None]).to(torch.float8_e4m3fn).float() * sf[..., None]).reshape_as(x)

def reference(xs, idxs, ss, lives, fp8=True):
    torch.cuda.set_device(0)
    x = torch.cat([z.to('cuda:0').float() for z in xs])
    idx = torch.cat([z.to('cuda:0') for z in idxs])
    scores = torch.cat([z.to('cuda:0') for z in ss])
    live = torch.cat([z.to('cuda:0') for z in lives])
    idx = torch.where(live[:, None] > 0, idx, -1)
    x = quant(x) if fp8 else x
    out = torch.zeros((x.shape[0], 8, 4096), device='cuda:0')
    for expert in idx.unique().tolist():
        if expert < 0:
            continue
        (row, slot) = torch.where(idx == expert)
        (w1, w2) = unpack(expert)
        v = x[row] @ w1.T
        (g, u) = v.chunk(2, -1)
        v = torch.nn.functional.silu(g) * u * scores[row, slot, None]
        v = quant(v) if fp8 else v
        out[row, slot] = (v @ w2.T).bfloat16().float()
    return out.sum(1).bfloat16()
pool = ThreadPoolExecutor(R)
cases = [(1, 'spread'), (4, 'hot'), (16, 'spread'), (17, 'spread'),
         (32, 'uniform'), (33, 'spread'), (128, 'spread'), (129, 'spread'),
         (3, 'idle'), (192, 'spread'), (193, 'idle'), (512, 'uniform'),
         (97, 'spread'), (512, 'hot'), (513, 'idle'),
         (chunk_capacity, 'uniform'), (chunk_capacity, 'hot'),
         (chunk_capacity, 'empty'), (chunk_capacity + 1, 'ragged'),
         (2 * chunk_capacity + 1, 'idle'), (16, 'spread')]
for (ci, (n, pattern)) in enumerate(cases):
    torch.manual_seed(101 + ci)
    xs = []
    idxs = []
    ss = []
    ys = []
    lives = []
    for rank in range(R):
        torch.cuda.set_device(rank)
        rows = (0 if rank % 2 else n) if pattern == 'idle' else (
            n // (rank + 1) if pattern == 'ragged' else n)
        # Mix aligned and unaligned BF16 inputs and outputs within one collective.
        offset = int(n == 33 and rank % 2 == 1)
        x = torch.randn(rows * 4096 + offset, device='cuda')
        x = (x * (3 if ci == 1 else 0.4)).bfloat16()[offset:].view(rows, 4096)
        ids = torch.arange(8, device='cuda').repeat(rows, 1)
        if pattern == 'uniform':
            ids = ((rank * n + torch.arange(rows, device='cuda'))[:, None] * 8 + ids) % 256
        elif pattern != 'hot':
            ids = (ids * LE + rank + torch.arange(rows, device='cuda')[:, None] % 4) % 256
        s = torch.softmax(torch.randn((rows, 8), device='cuda'), -1)
        live = torch.ones(rows, device='cuda', dtype=torch.int32)
        if pattern == 'empty':
            live.zero_()
        elif rows > 2:
            live[-1] = 0
        xs.append(x)
        idxs.append(ids.int())
        ss.append(s)
        lives.append(live)
        elements = (max(1, rows) + 1) * 4096
        storage = torch.full((offset + elements,), 77.0, device='cuda', dtype=torch.bfloat16)
        ys.append(storage[offset:].view(-1, 4096))
        torch.cuda.synchronize()
    expected = reference(xs, idxs, ss, lives)
    torch.cuda.synchronize(0)

    def invoke(rank, graph=None, capture=False):
        torch.cuda.set_device(rank)
        rows = xs[rank].shape[0]

        def call():
            dummy = address(ys[rank])
            for start in range(0, n, chunk_capacity):
                count = max(0, min(chunk_capacity, rows - start))
                xptr = address(xs[rank]) + start * 4096 * 2 if count else dummy
                iptr = address(idxs[rank]) + start * 8 * 4 if count else dummy
                sptr = address(ss[rank]) + start * 8 * 4 if count else dummy
                yptr = address(ys[rank]) + start * 4096 * 2 if count else dummy
                live = address(lives[rank]) + start * 4 if count else None
                ok = run(handles[rank], xptr, yptr, iptr, sptr, count, min(chunk_capacity, n-start), live)
                if not ok:
                    raise RuntimeError(error().decode())
        if capture:
            stream = torch.cuda.ExternalStream(2, device=rank)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream, capture_error_mode='thread_local'):
                call()
            return g
        if graph is None:
            call()
        else:
            graph.replay()
        torch.cuda.synchronize(rank)

    def allcall(**kw):
        return list(pool.map(lambda rank: invoke(rank, **kw), range(R)))
    t = time.time()
    allcall()
    t = time.time() - t
    actual = torch.cat([y[:x.shape[0]].to('cuda:0') for (y, x) in zip(ys, xs)])
    diff = actual.float() - expected.float()
    l2 = (diff.norm() / expected.float().norm().clamp_min(1e-20)).item()
    mx = diff.abs().max().item()
    baseline = reference(xs, idxs, ss, lives, fp8=False)
    w4a16_l2 = ((actual.float() - baseline.float()).norm() / baseline.float().norm().clamp_min(1e-20)).item()
    assert torch.isfinite(actual).all() and l2 < 0.025, ('accuracy', n, pattern, l2, mx)
    assert all((torch.all(y[x.shape[0]:] == 77).item() for (y, x) in zip(ys, xs))), 'output guard'
    # Exact repeat/replay checks detect stale routes and workspace races.
    first = [y.clone() for y in ys]
    allcall()
    assert all((torch.equal(x, y) for (x, y) in zip(first, ys))), 'repeat'
    gs = allcall(capture=True)
    list(pool.map(lambda rank: invoke(rank, graph=gs[rank]), range(R)))
    assert all((torch.equal(x, y) for (x, y) in zip(first, ys))), 'graph'
    for rank in range(R):
        torch.cuda.set_device(rank)
        xs[rank].mul_(0.75)
        torch.cuda.synchronize()
    # Keep captured addresses but change their contents.
    expected2 = reference(xs, idxs, ss, lives)
    torch.cuda.synchronize(0)
    list(pool.map(lambda rank: invoke(rank, graph=gs[rank]), range(R)))
    actual2 = torch.cat([y[:x.shape[0]].to('cuda:0') for (y, x) in zip(ys, xs)])
    l22 = ((actual2.float() - expected2.float()).norm() / expected2.float().norm().clamp_min(1e-20)).item()
    assert l22 < 0.025, l22
    records.append({'rows_per_rank': n, 'pattern': pattern, 'relative_l2': l2, 'vs_w4a16_l2': w4a16_l2, 'max_abs': mx, 'changed_graph_l2': l22, 'repeat_exact': True, 'graph_exact': True, 'output_guard': True, 'first_call_s': t})
    print(records[-1], flush=True)
    for rank in range(R):
        torch.cuda.set_device(rank)
        gs[rank].reset()
    pathlib.Path(a.out).write_text(json.dumps({'records': records, 'seconds': time.time() - started}, indent=2))
pool.shutdown()
# Unsupported scale encodings must fail before dispatch, never silently corrupt.
torch.cuda.set_device(0)
gate = (ptr * LE)(*[address(x[0]) for x in weights[0]])
down = (ptr * LE)(*[address(x[1]) for x in weights[0]])
scale = weights[0][0][0][4096 * 4096 // 2]
original_scale = scale.clone()
for invalid_scale in (0, 255, 1):
    scale.fill_(invalid_scale)
    torch.cuda.synchronize()
    invalid = create(gate, down, address(arenas[0]), arenas[0].numel(), peers, 0, R)
    assert not invalid and b'decoder range' in error(), error()
scale.copy_(original_scale)
torch.cuda.synchronize()
pathlib.Path(a.out).write_text(json.dumps({
    'records': records, 'invalid_scale_rejected': [0, 255, 1],
    'seconds': time.time() - started,
}, indent=2))
for (rank, h) in enumerate(handles):
    torch.cuda.set_device(rank)
    destroy(h)
print('PASS', flush=True)
