"""Validate batched Indexer against the committed FastLLM CUDA implementation.

No model or third-party Indexer implementation is imported. Timing uses CUDA
Graphs, two measured repetitions, and reports microseconds per DSA layer.
Timing compares batched fallback scoring (old) with runtime selection (batch),
sharing the quantization and TopK stages. Precision uses the committed reference.
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np
import torch

P, I = C.c_void_p, C.c_int
class Row(C.Structure):
    _fields_ = [(n, P) for n in ('key', 'live', 'query', 'scale', 'scores', 'workspace', 'indices')] + [(n, I) for n in ('sequence', 'stride')]

lib = C.CDLL(str(Path(__file__).resolve().parent / 'build/indexer.so'))
lib.workspace_size.restype = C.c_size_t
lib.run.argtypes = [P, P, C.POINTER(Row), I, I, I, I, P]
torch.set_num_threads(1)
torch.manual_seed(4321)

def check(code):
    if code: raise RuntimeError(f'CUDA error {code}')

def timing(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(4): fn()
    for _ in range(3): graph.replay()
    results = []
    for _ in range(2):
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record()
        for _ in range(10): graph.replay()
        b.record(); b.synchronize()
        results.append(a.elapsed_time(b) * 1000 / 40)
    return results

class Case:
    def __init__(self, batch, n, stride=320):
        self.batch, self.capacity = batch, n
        self.q = torch.randn(batch, 16, 128, device='cuda', dtype=torch.bfloat16)
        self.w = torch.randn(batch, 16, device='cuda', dtype=torch.bfloat16)
        self.k = torch.randn(batch, n, stride, device='cuda', dtype=torch.bfloat16)
        self.live = torch.full((batch,), n, device='cuda', dtype=torch.int32)
        self.buffers, self.rows = [], []
        for old in range(2):
            data = [torch.empty(batch, 16, 128, device='cuda', dtype=torch.bfloat16),
                    torch.empty(batch, 16, device='cuda'),
                    torch.full((batch, n), -1, device='cuda', dtype=torch.int32),
                    torch.empty(batch, lib.workspace_size(), device='cuda', dtype=torch.uint8),
                    torch.full((batch, 2048), -2, device='cuda', dtype=torch.int32)]
            self.buffers.append(data)
            self.rows.append((Row * batch)(*[Row(self.k[b].data_ptr(), self.live[b].data_ptr(),
                *[x[b].data_ptr() for x in data], b, stride) for b in range(batch)]))
    def run(self, old=0, stage=0):
        check(lib.run(self.q.data_ptr(), self.w.data_ptr(), self.rows[int(bool(old))], self.batch, self.capacity, old, stage,
                      torch.cuda.current_stream().cuda_stream))
    def compare(self, old=1):
        self.run(old); self.run(); torch.cuda.synchronize()
        for name, a, b in zip(('quantized_q', 'scale', 'scores', 'workspace', 'indices'), *self.buffers):
            if name == 'workspace': continue
            if not torch.equal(a, b):
                bad = (a != b).nonzero()
                raise AssertionError(f'{name}: {len(bad)} differences; first {bad[:8].tolist()}')
    def reference_topk(self, values, lengths):
        # Ordered unsigned bits are the exact comparator contract, including NaNs.
        bits = np.asarray(values, dtype=np.uint32)
        tensor = torch.from_numpy(bits.view(np.int32)).cuda()
        self.buffers[0][2].copy_(tensor)
        self.live.copy_(torch.tensor(lengths, dtype=torch.int32, device='cuda'))
        self.run(stage=3)
        got = self.buffers[0][4].cpu().numpy()
        for b, n in enumerate(lengths):
            want = np.full(2048, -1, dtype=np.int32)
            want[:min(n, 2048)] = np.argsort(-bits[b, :n].astype(np.int64), kind='stable')[:2048]
            np.testing.assert_array_equal(got[b], want)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = {'precision': [], 'timings': []}
    for batch, n, stride in [(1,2049,320),(2,4095,321),(4,8193,320),(32,8576,320),(65,16385,320),(256,8320,320),(257,4097,320)]:
        c = Case(batch, n, stride)
        c.compare()
        c.compare(2)
        # Graph replay must read new per-row lengths, not captured host values.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): c.run()
        lengths = torch.arange(batch, device='cuda', dtype=torch.int32) % (n - 2048) + 2049
        c.live.copy_(lengths)
        for data in c.buffers: data[2].fill_(-1)
        graph.replay(); c.run(1); torch.cuda.synchronize()
        for i in (0,1,2,4): assert torch.equal(c.buffers[0][i], c.buffers[1][i]), (batch,n,i)
        # Descriptor order may differ from tensor row order; sparse subsets
        # must leave the other rows untouched.
        order = list(range(batch - 1, -1, -2))
        subset = (Row * len(order))(*[c.rows[0][b] for b in order])
        for b in order: c.buffers[0][4][b].fill_(-2)
        check(lib.run(c.q.data_ptr(), c.w.data_ptr(), subset, len(order), c.capacity, 0, 0,
                      torch.cuda.current_stream().cuda_stream))
        assert torch.equal(c.buffers[0][4], c.buffers[1][4])
        report['precision'].append({'batch':batch,'capacity':n,'stride':stride,'bitwise_equal':True,'graph_dynamic_lengths':True})
        print('PASS scoring/TopK', batch,n,stride,flush=True)
        del graph, c
    # Exercise cancellation, mixed magnitudes, ties, and scalar cache loads.
    for kind in ('scaled', 'zero_weights', 'uniform_keys', 'strided_keys'):
        c = Case(4, 4097, 321 if kind == 'strided_keys' else 320)
        if kind == 'scaled':
            c.q.mul_(torch.logspace(-6, 4, 16, device='cuda').view(1, 16, 1))
            c.k.mul_(torch.logspace(-3, 3, 320, device='cuda'))
            c.w.mul_(torch.logspace(-3, 3, 16, device='cuda'))
        elif kind == 'zero_weights': c.w.zero_()
        elif kind == 'uniform_keys': c.k.copy_(c.k[:, :1].clone().expand_as(c.k))
        c.compare()
        report['precision'].append({'scoring_edge':kind,'bitwise_equal':True})
        print('PASS scoring edge',kind,flush=True)
        del c
    rng = np.random.default_rng(4321)
    for n in (2049,8192,16385,65536,262144):
        c = Case(8, n)
        bits = rng.integers(0, 2**32, size=(8,n), dtype=np.uint32)
        bits[1].fill(0x80000000) # uniform, signed zero canonical form
        bits[2] = (np.arange(n) % 3) * 0x40000000 # large threshold tie groups
        bits[3] = 0xffffffff # max NaN bit pattern
        bits[3,::7] = 0xfffffffe
        bits[4] = np.resize(np.array([0,0x007fffff,0x7fffffff,0x80000000,0xff800000,0xffffffff],np.uint32),n)
        bits[5] = np.arange(n, dtype=np.uint32)
        bits[6] = bits[5,::-1]
        lengths = [n,n,n,n,n,max(1,n-31),0,2048]
        c.reference_topk(bits,lengths)
        report['precision'].append({'topk_edge_capacity':n,'stable_reference_equal':True})
        print('PASS edge TopK',n,flush=True)
        del c
    if args.bench:
        for n in (8320,16448):
            for batch in (1,2,4,8,16,32,64,128,256):
                c = Case(batch,n); c.run(2); c.run(); torch.cuda.synchronize()
                row = {'batch':batch,'keys':n}
                for old,name in ((2,'old'),(0,'batch')):
                    for stage,part in ((1,'quantize'),(2,'scores'),(3,'topk'),(0,'total')):
                        row[f'{name}_{part}_us'] = timing(lambda: c.run(old,stage))
                report['timings'].append(row)
                print(json.dumps(row),flush=True)
                del c
    if args.output: Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
