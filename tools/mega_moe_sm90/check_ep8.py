"""Check full-expert EP8 against installed single-rank DeepGEMM on identical W8A8 data."""

import argparse, ctypes, json, types
from pathlib import Path
import torch, deep_gemm as dg

p = argparse.ArgumentParser()
p.add_argument("--lib", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
torch.manual_seed(1026)
R, E, H, I, K, C = 8, 256, 4096, 2048, 8, 128
ptr = ctypes.c_void_p
P = ctypes.POINTER(ptr)
lib = ctypes.CDLL(a.lib)
lib.fastllm_ep8_mega_moe_workspace_bytes.restype = ctypes.c_size_t
lib.fastllm_ep8_mega_moe_error.restype = ctypes.c_char_p
lib.fastllm_ep8_mega_moe_init_workspace.argtypes = [
    ptr,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
lib.fastllm_ep8_mega_moe_create.argtypes = [P] * 4 + [
    ptr,
    ctypes.c_size_t,
    P,
    ctypes.c_int,
    ctypes.c_int,
]
lib.fastllm_ep8_mega_moe_create.restype = ptr
lib.fastllm_ep8_mega_moe_run.argtypes = [ptr, ptr, ptr, ptr, ptr, ctypes.c_int]
lib.fastllm_ep8_mega_moe_destroy.argtypes = [ptr]
size = lib.fastllm_ep8_mega_moe_workspace_bytes()
assert size <= 16 * 1024**2
print("WORKSPACE_BYTES", size, flush=True)
torch.cuda.set_device(0)
w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16).to(
    torch.float8_e4m3fn
)
w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
s1 = torch.rand(E, 2 * I // 128, H // 128, device="cuda") * 0.01 + 0.003
s2 = torch.rand(E, H // 128, I // 128, device="cuda") * 0.01 + 0.003
ranks = []
devices = (ctypes.c_int * R)(*range(R))
for r in range(R):
    torch.cuda.set_device(r)
    weights = [v[r * 32 : (r + 1) * 32].to(r) for v in [w1, w2, s1, s2]]
    scratch = torch.empty(size, device=r, dtype=torch.uint8)
    assert lib.fastllm_ep8_mega_moe_init_workspace(
        scratch.data_ptr(), size, devices, r
    ), lib.fastllm_ep8_mega_moe_error()
    ranks.append(dict(weights=weights, scratch=scratch))
peers = (ptr * R)(*[v["scratch"].data_ptr() for v in ranks])
for r, v in enumerate(ranks):
    torch.cuda.set_device(r)
    arrays = [(ptr * 32)(*[t[e].data_ptr() for e in range(32)]) for t in v["weights"]]
    v["handle"] = lib.fastllm_ep8_mega_moe_create(
        *arrays, v["scratch"].data_ptr(), size, peers, r, 0
    )
    assert v["handle"], lib.fastllm_ep8_mega_moe_error()
torch.cuda.set_device(0)
refbytes, slicer = dg._C.get_symm_buffer_size_for_sm90_mega_moe(
    1, E, C, K, H, I, True, "swiglu"
)
buffer = torch.zeros(refbytes, device="cuda", dtype=torch.uint8)
views = [torch.from_dlpack(t) for t in slicer(buffer)]
sym = types.SimpleNamespace(
    buffer=buffer,
    handle=types.SimpleNamespace(buffer_ptrs=[buffer.data_ptr()]),
    group=types.SimpleNamespace(rank=lambda: 0),
    num_max_tokens_per_rank=C,
    num_experts=E,
    num_topk=K,
)
l1, l2 = dg.transform_weights_for_mega_moe_sm90((w1, s1), (w2, s2))
result = []
for N, scenario in [
    (1, "replicated_spread"),
    (1, "one_owner"),
    (2, "different_ranks"),
    (8, "different_ranks"),
    (8, "one_owner"),
]:
    for r, v in enumerate(ranks):
        torch.cuda.set_device(r)
        x = torch.randn(N, H, device=r, dtype=torch.bfloat16) * 0.1
        idx = torch.stack(
            [torch.randperm(E, device=r, dtype=torch.int32)[:K] for _ in range(N)]
        )
        if scenario == "one_owner":
            idx = torch.arange(K, device=r, dtype=torch.int32).repeat(N, 1)
        if scenario == "replicated_spread":
            idx = (torch.arange(K, device=r, dtype=torch.int32) * 32).repeat(N, 1)
        score = torch.softmax(torch.randn(N, K, device=r), -1)
        if scenario == "replicated_spread" and r:
            x = ranks[0]["x"].to(r)
            score = ranks[0]["score"].to(r)
        v.update(x=x, idx=idx, score=score, y=torch.empty_like(x))
    for r in range(R):
        torch.cuda.synchronize(r)
    # No allocations or synchronization between launches: every peer must enter.
    for repeat in range(2):
        for r, v in enumerate(ranks):
            torch.cuda.set_device(r)
            assert lib.fastllm_ep8_mega_moe_run(
                v["handle"],
                v["x"].data_ptr(),
                v["y"].data_ptr(),
                v["idx"].data_ptr(),
                v["score"].data_ptr(),
                N,
            ), lib.fastllm_ep8_mega_moe_error()
        for r in range(R):
            torch.cuda.synchronize(r)
    torch.cuda.set_device(0)
    x = torch.cat([v["x"].to(0) for v in ranks])
    idx = torch.cat([v["idx"].to(0) for v in ranks])
    score = torch.cat([v["score"].to(0) for v in ranks])
    ref = torch.empty_like(x)
    dg.mega_moe_pre_dispatch_sm90(
        x,
        idx,
        score,
        views[0].view(torch.float8_e4m3fn),
        views[1],
        views[2],
        views[3],
        N * R,
        C,
    )
    dg.fp8_mega_moe(ref, l1, l2, sym, recipe=(128, 128, 128))
    torch.cuda.synchronize()
    y = torch.cat([v["y"].to(0) for v in ranks])
    delta = y.float() - ref.float()
    nrmse = float(delta.square().mean().sqrt() / ref.float().square().mean().sqrt())
    assert torch.isfinite(y).all() and nrmse < 0.003, (N, scenario, nrmse)
    item = dict(
        rows_per_rank=N,
        scenario=scenario,
        nrmse=nrmse,
        max_abs=float(delta.abs().max()),
        exact_equal=bool(torch.equal(y, ref)),
    )
    print(json.dumps(item), flush=True)
    result.append(item)
# Capture every peer before replaying any of them. Each graph contains the
# quantization and fused EP kernel on the per-thread stream. Descriptors are
# already resident and are not uploaded during capture or replay.
cuda = ctypes.CDLL("libcudart.so.13")
cuda.cudaStreamBeginCapture.argtypes = [ptr, ctypes.c_int]
cuda.cudaStreamEndCapture.argtypes = [ptr, ctypes.POINTER(ptr)]
cuda.cudaGraphInstantiateWithFlags.argtypes = [
    ctypes.POINTER(ptr),
    ptr,
    ctypes.c_ulonglong,
]
cuda.cudaGraphLaunch.argtypes = [ptr, ptr]
cuda.cudaGraphExecDestroy.argtypes = [ptr]
cuda.cudaGraphDestroy.argtypes = [ptr]
graphs = []
for r, v in enumerate(ranks):
    torch.cuda.set_device(r)
    g = ptr()
    ex = ptr()
    assert cuda.cudaStreamBeginCapture(ptr(2), 1) == 0
    assert lib.fastllm_ep8_mega_moe_run(
        v["handle"],
        v["x"].data_ptr(),
        v["y"].data_ptr(),
        v["idx"].data_ptr(),
        v["score"].data_ptr(),
        N,
    )
    assert cuda.cudaStreamEndCapture(ptr(2), ctypes.byref(g)) == 0
    assert cuda.cudaGraphInstantiateWithFlags(ctypes.byref(ex), g, 0) == 0
    graphs.append((g, ex))
for repeat in range(3):
    for r, (g, ex) in enumerate(graphs):
        torch.cuda.set_device(r)
        assert cuda.cudaGraphLaunch(ex, ptr(2)) == 0
    for r in range(R):
        torch.cuda.synchronize(r)
torch.cuda.set_device(0)
graph_y = torch.cat([v["y"].to(0) for v in ranks])
assert torch.equal(graph_y, ref), "EP8 Graph output differs from reference"
print("EP8_CUDA_GRAPH_PASS", flush=True)
for r, (g, ex) in enumerate(graphs):
    torch.cuda.set_device(r)
    assert cuda.cudaGraphExecDestroy(ex) == 0
    assert cuda.cudaGraphDestroy(g) == 0
for r, v in enumerate(ranks):
    torch.cuda.set_device(r)
    lib.fastllm_ep8_mega_moe_destroy(v["handle"])
Path(a.output).write_text(
    json.dumps(
        {
            "workspace_bytes_per_gpu": size,
            "cases": result,
            "cuda_graph_exact_equal": True,
        },
        indent=2,
    )
    + "\n"
)
print("EP8_KERNEL_PASS", flush=True)
