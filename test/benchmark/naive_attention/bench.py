import ctypes as C
import json
import sys
from pathlib import Path

import torch

torch.manual_seed(123)
torch.set_num_threads(1)
here = Path(__file__).resolve().parent / "build"
lib = C.CDLL(str(here / "bulk.so"))
P = C.c_void_p
I = C.c_int


class Row(C.Structure):
    _fields_ = [
        ("key", P),
        ("value", P),
        ("indices", P),
        ("live", P),
        ("length", I),
        ("capacity", I),
        ("stride", I),
        ("selected", I),
    ]


lib.run.argtypes = [P, C.POINTER(Row), P, P, I, I, I, I, I, P]
lib.run_tma.argtypes = [P, C.POINTER(Row), P, P, I, I, I, I, P]
lib.old.argtypes = [P, C.POINTER(Row), P, P, P, I, I, I, I, I, P]


def ptr(t):
    return t.data_ptr() if t is not None else None


def check(code):
    if code:
        raise RuntimeError("CUDA error " + str(code))


def timing(f):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            f()
    for _ in range(3):
        graph.replay()
    results = []
    for _ in range(2):
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record()
        for _ in range(10):
            graph.replay()
        b.record()
        b.synchronize()
        results.append(a.elapsed_time(b) * 1000 / 200)
    return sum(results) / len(results)


def case(batch, ring, n=768, capacity=896, sparse=False):
    assert batch >= 32, "The historical baseline below is the large-batch dispatch."
    if ring:
        n = 128
        capacity = 128
    heads = 8
    stride = 192 if ring else 320
    q = torch.randn(batch, heads, 192, device="cuda", dtype=torch.bfloat16) * 0.6
    k = torch.randn(batch, capacity, 1, stride, device="cuda", dtype=torch.bfloat16) * 0.6
    v = torch.randn(batch, capacity, 1, 128, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda", dtype=torch.bfloat16).float()
    live = torch.full((batch,), n, device="cuda", dtype=torch.int32)
    out = torch.empty(batch, heads, 128, device="cuda", dtype=torch.bfloat16)
    indices = None
    selected = 2048 if sparse else capacity
    if sparse:
        indices = torch.stack(
            [torch.randperm(capacity, device="cuda")[:2048] for _ in range(batch)]
        ).int()
        indices[:, ::17] = -1
    rows = (Row * batch)(
        *[
            Row(
                ptr(k[b]),
                ptr(v[b]),
                ptr(indices[b]) if sparse else None,
                ptr(live[b]),
                n,
                capacity,
                stride,
                2048 if sparse else 0,
            )
            for b in range(batch)
        ]
    )
    workspace = torch.empty(32 * heads * selected, device="cuda", dtype=torch.float32)

    def run():
        check(
            lib.run(
                ptr(q),
                rows,
                ptr(sink),
                ptr(out),
                batch,
                heads,
                1,
                selected,
                ring,
                torch.cuda.current_stream().cuda_stream,
            )
        )

    def old():
        check(
            lib.old(
                ptr(q),
                rows,
                ptr(sink),
                ptr(out),
                ptr(workspace),
                batch,
                heads,
                1,
                selected,
                ring,
                torch.cuda.current_stream().cuda_stream,
            )
        )

    old()
    ref = out.clone()
    result = {
        "batch": batch,
        "kind": "swa" if ring else "dsa",
        "length": n,
        "capacity": capacity,
        "sparse": sparse,
        "us": {},
        "error": {},
    }
    result["us"]["old"] = timing(old)
    run()
    got = out.clone().float()
    gold = ref.float()
    result["error"]["tensor"] = {
        "nrmse": ((got - gold).square().mean().sqrt() / gold.square().mean().sqrt()).item(),
        "max": (got - gold).abs().max().item(),
        "different": (got != gold).sum().item(),
        "finite": bool(got.isfinite().all()),
    }
    result["us"]["tensor"] = timing(run)
    if not ring and not sparse and lib.supports_tma():

        def tma():
            check(
                lib.run_tma(
                    ptr(q),
                    rows,
                    ptr(sink),
                    ptr(out),
                    batch,
                    heads,
                    1,
                    capacity,
                    torch.cuda.current_stream().cuda_stream,
                )
            )

        tma()
        got = out.clone().float()
        gold = ref.float()
        result["error"]["tma"] = {
            "nrmse": ((got - gold).square().mean().sqrt() / gold.square().mean().sqrt()).item(),
            "max": (got - gold).abs().max().item(),
            "finite": bool(got.isfinite().all()),
        }
        result["us"]["tma"] = timing(tma)
    assert all(v["finite"] and v["nrmse"] < 0.003 for v in result["error"].values()), result
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    batches = [32, 64, 128, 256] if "--full" in sys.argv else [128]
    results = [case(b, r) for r in [True, False] for b in batches]
    (here / "bench.json").write_text(json.dumps(results, indent=2))
