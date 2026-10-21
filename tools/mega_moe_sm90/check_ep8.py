"""Check full-expert EP4/EP8 against installed single-rank DeepGEMM on identical W8A8 data."""

import argparse, ctypes, json, types
from pathlib import Path
import torch, deep_gemm as dg

p = argparse.ArgumentParser()
p.add_argument("--lib", required=True)
p.add_argument("--output", required=True)
p.add_argument("--ranks", type=int, choices=(4, 8), default=8)
p.add_argument("--prefill", action="store_true")
a = p.parse_args()
torch.manual_seed(1026)
R, E, H, I, K, C = a.ranks, 256, 4096, 2048, 8, (8192 if a.prefill else 512)
ptr = ctypes.c_void_p
P = ctypes.POINTER(ptr)
lib = ctypes.CDLL(a.lib)
lib.fastllm_ep8_mega_moe_workspace_bytes.argtypes = [ctypes.c_int]
lib.fastllm_ep8_mega_moe_workspace_bytes.restype = ctypes.c_size_t
lib.fastllm_ep8_mega_moe_error.restype = ctypes.c_char_p
lib.fastllm_ep8_mega_moe_init_workspace.argtypes = [
    ptr,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]
lib.fastllm_ep8_mega_moe_create.argtypes = [P] * 4 + [
    ptr,
    ctypes.c_size_t,
    P,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.fastllm_ep8_mega_moe_create.restype = ptr
lib.fastllm_ep8_mega_moe_run.argtypes = [ptr, ptr, ptr, ptr, ptr, ctypes.c_int]
lib.fastllm_ep8_mega_moe_destroy.argtypes = [ptr]
size = lib.fastllm_ep8_mega_moe_workspace_bytes(R)
assert size <= 48 * 1024**2
lib.fastllm_ep8_mega_moe_max_rows.restype = ctypes.c_int
assert lib.fastllm_ep8_mega_moe_max_rows() == 64
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
    weights = [v[r * (E // R) : (r + 1) * (E // R)].to(r) for v in [w1, w2, s1, s2]]
    # Keep a guard immediately after the arena, especially for the routing
    # pattern that fills its final aligned expert tile.
    allocation = torch.full((size + 4096,), 165, device=r, dtype=torch.uint8)
    scratch = allocation[:size]
    assert lib.fastllm_ep8_mega_moe_init_workspace(
        scratch.data_ptr(), size, devices, r, R
    ), lib.fastllm_ep8_mega_moe_error()
    ranks.append(dict(weights=weights, scratch=scratch, guard=allocation[size:]))
peers = (ptr * R)(*[v["scratch"].data_ptr() for v in ranks])
for r, v in enumerate(ranks):
    torch.cuda.set_device(r)
    arrays = [(ptr * (E // R))(*[t[e].data_ptr() for e in range(E // R)]) for t in v["weights"]]
    v["handle"] = lib.fastllm_ep8_mega_moe_create(
        *arrays, v["scratch"].data_ptr(), size, peers, r, 0, R
    )
    assert v["handle"], lib.fastllm_ep8_mega_moe_error()
    # Idle ranks still need valid pointers. Test row bounds separately so
    # null-pointer rejection cannot hide a missing capacity check.
    assert not lib.fastllm_ep8_mega_moe_run(v["handle"], None, None, None, None, 0)
    scratch = v["scratch"].data_ptr()
    for invalid_rows in (-1, 65):
        assert not lib.fastllm_ep8_mega_moe_run(
            v["handle"], scratch, scratch, scratch, scratch, invalid_rows
        )
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
cases = [
    (1, "replicated_spread"),
    (1, "one_owner"),
    (2, "different_ranks"),
    (8, "different_ranks"),
    (8, "one_owner"),
    (9, "different_ranks"),
    (16, "different_ranks"),
    (32, "different_ranks"),
    (63, "different_ranks"),
    (64, "different_ranks"),
    (64, "one_owner"),
    # A small call after full-capacity calls checks reused control state.
    (1, "replicated_spread"),
    (1, "idle_ranks"),
    (7, "idle_ranks"),
    (64, "idle_ranks"),
    (64, "uneven_rows"),
    (64, "pool_bound"),
]
if a.prefill:
    cases += [(n, "prefill") for n in (128, 256, 512, 1024)]
    # End on decode so the graph test uses the persistent 64-row arena.
    cases += [(64, "different_ranks")]
    lib.fastllm_ep8_mega_moe_prefill_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.fastllm_ep8_mega_moe_prefill_workspace_bytes.restype = ctypes.c_size_t
    lib.fastllm_ep8_mega_moe_prepare_prefill.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_int]
    lib.fastllm_ep8_mega_moe_run_prefill.argtypes = [ptr, P, ptr, ptr, ptr, ptr, ctypes.c_int]
for N, scenario in cases:
    if scenario == "pool_bound":
        # 31 experts receive 129 rows and one receives 97. Their BM=64
        # aligned spans occupy exactly 6080 rows: the compiled pool bound.
        local_e = E // R
        # Put one route in every local expert, then distribute the remaining
        # full 64-row groups to saturate the compact aligned pool.
        remaining = [1] * local_e
        for i in range((N * R * K - local_e) // 64):
            remaining[i % local_e] += 64
        remaining[-1] += (N * R * K - local_e) % 64
        routes = []
        for _ in range(N * R):
            chosen = sorted(range(local_e), key=lambda e: (-remaining[e], e))[:K]
            assert all(remaining[e] > 0 for e in chosen)
            for e in chosen:
                remaining[e] -= 1
            routes.append(chosen)
        assert not any(remaining)
        pool_routes = torch.tensor(routes, dtype=torch.int32)
    for r, v in enumerate(ranks):
        torch.cuda.set_device(r)
        x = torch.randn(N, H, device=r, dtype=torch.bfloat16) * 0.1
        idx = torch.stack(
            [torch.randperm(E, device=r, dtype=torch.int32)[:K] for _ in range(N)]
        )
        if scenario == "pool_bound":
            idx = pool_routes[r * N : (r + 1) * N].to(r)
        if scenario == "one_owner":
            idx = torch.arange(K, device=r, dtype=torch.int32).repeat(N, 1)
        if scenario == "replicated_spread":
            idx = (torch.arange(K, device=r, dtype=torch.int32) * 32).repeat(N, 1)
        score = torch.softmax(torch.randn(N, K, device=r), -1)
        if scenario == "replicated_spread" and r:
            x = ranks[0]["x"].to(r)
            score = ranks[0]["score"].to(r)
        active = (N if r == 0 else 0) if scenario == "idle_ranks" else (
            [64, 63, 1, 0, 7, 0, 31, 2][r] if scenario == "uneven_rows" else N)
        v.update(x=x, idx=idx, score=score, y=torch.full_like(x, 777), active=active)
    if scenario == "prefill":
        prefill_size = lib.fastllm_ep8_mega_moe_prefill_workspace_bytes(N * R, R)
        for r, v in enumerate(ranks):
            torch.cuda.set_device(r)
            v["prefill"] = torch.full((prefill_size + 4096,), 165, device=r, dtype=torch.uint8)
            assert lib.fastllm_ep8_mega_moe_prepare_prefill(
                v["handle"], v["prefill"].data_ptr(), prefill_size, N * R) == N, lib.fastllm_ep8_mega_moe_error()
        prefill_peers = (ptr * R)(*[v["prefill"].data_ptr() for v in ranks])
    for r in range(R):
        torch.cuda.synchronize(r)
    # No allocations or synchronization between launches: every peer must enter.
    for repeat in range(2):
        for r, v in enumerate(ranks):
            torch.cuda.set_device(r)
            runner = lib.fastllm_ep8_mega_moe_run_prefill if scenario == "prefill" else lib.fastllm_ep8_mega_moe_run
            extra = [prefill_peers] if scenario == "prefill" else []
            assert runner(
                v["handle"], *extra,
                v["x"].data_ptr(),
                v["y"].data_ptr(),
                v["idx"].data_ptr(),
                v["score"].data_ptr(),
                v["active"],
            ), lib.fastllm_ep8_mega_moe_error()
        for r in range(R):
            torch.cuda.synchronize(r)
    torch.cuda.set_device(0)
    x = torch.cat([v["x"][:v["active"]].to(0) for v in ranks])
    idx = torch.cat([v["idx"][:v["active"]].to(0) for v in ranks])
    score = torch.cat([v["score"][:v["active"]].to(0) for v in ranks])
    ref = torch.empty_like(x)
    # MoE is token-independent. Bound reference launches so DeepGEMM uses
    # the same numerical implementation for decode and large prefill tests.
    # Its large-row selector can choose a different FP8 arithmetic path.
    for begin in range(0, x.shape[0], 512):
        end = min(begin + 512, x.shape[0])
        dg.mega_moe_pre_dispatch_sm90(
            x[begin:end], idx[begin:end], score[begin:end],
            views[0].view(torch.float8_e4m3fn), views[1], views[2], views[3],
            end - begin, 128,
        )
        dg.fp8_mega_moe(ref[begin:end], l1, l2, sym, recipe=(128, 128, 128))
    torch.cuda.synchronize()
    y = torch.cat([v["y"][:v["active"]].to(0) for v in ranks])
    assert all(bool((v["y"][v["active"]:] == 777).all()) for v in ranks), "Idle output overwritten"
    assert all(bool((v["guard"] == 165).all()) for v in ranks), "Workspace overflow"
    if scenario == "prefill":
        assert all(bool((v["prefill"][prefill_size:] == 165).all()) for v in ranks), "Prefill workspace overflow"
    delta = y.float() - ref.float()
    nrmse = float(delta.square().mean().sqrt() / ref.float().square().mean().sqrt())
    if not (torch.isfinite(y).all() and nrmse < 0.003):
        row_error = delta.square().mean(-1).sqrt() / ref.float().square().mean(-1).sqrt().clamp_min(1e-12)
        print("NUMERICAL_FAILURE", json.dumps(dict(rows=N, scenario=scenario,
            nrmse=nrmse, max_abs=float(delta.abs().max()), max_row=float(row_error.max()),
            different_rows=int((row_error > .003).sum()), total_rows=y.shape[0],
            top_rows=torch.topk(row_error, min(20, row_error.numel())).indices.cpu().tolist())), flush=True)
        chunked = [torch.empty_like(v["y"]) for v in ranks]
        for start in range(0, N, 64):
            for rank, v in enumerate(ranks):
                torch.cuda.set_device(rank)
                assert lib.fastllm_ep8_mega_moe_run(v["handle"],
                    v["x"][start:].data_ptr(), chunked[rank][start:].data_ptr(),
                    v["idx"][start:].data_ptr(), v["score"][start:].data_ptr(), min(64, N-start))
        for rank in range(R): torch.cuda.synchronize(rank)
        torch.cuda.set_device(0)
        chunked_y = torch.cat([t.to(0) for t in chunked])
        def nrmse_against(u, v): return float((u.float()-v.float()).square().mean().sqrt()/v.float().square().mean().sqrt())
        print("CHUNKED_REFERENCE", json.dumps(dict(prefill_vs_decode=nrmse_against(y,chunked_y),
            decode_vs_deepgemm=nrmse_against(chunked_y,ref))),flush=True)
        torch.save(dict(x=x.cpu(), idx=idx.cpu(), score=score.cpu(), y=y.cpu(), reference=ref.cpu(), chunked=chunked_y.cpu()),
                   str(Path(a.output).with_suffix(".failure.pt")))
        raise AssertionError((N, scenario, nrmse))
    item = dict(
        rows_per_rank=N,
        active_rows_per_rank=[v["active"] for v in ranks],
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
assert all(bool((v["guard"] == 165).all()) for v in ranks), "Graph workspace overflow"
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
            "ranks": R,
            "workspace_bytes_per_gpu": size,
            "cases": result,
            "cuda_graph_exact_equal": True,
        },
        indent=2,
    )
    + "\n"
)
print("EP8_KERNEL_PASS", flush=True)
