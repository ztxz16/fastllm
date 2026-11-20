"""Blackwell EP4/EP8 tile switching, masks, idle ranks and CUDA Graph regression.

Requires PyTorch and the built FastLLM library; no model checkpoint is required.
Run: python test_blackwell_mega_moe.py --lib /path/libfastllm_tools.so --out result.json
"""

import argparse
import ctypes
import json
from pathlib import Path


def main():
    import torch
    import torch.nn.functional as F

    p = argparse.ArgumentParser()
    p.add_argument("--lib", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ranks", type=int, default=8, choices=(4, 8))
    p.add_argument("--mxfp4", action="store_true", help="Use native block-32 E2M1 experts")
    p.add_argument("--prefill", action="store_true", help="Compare large prefill kernels with the 128-row reference")
    p.add_argument("--checkpoint")
    p.add_argument("--layer", type=int, default=3)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1207)
    R, E, H, I, K, C = a.ranks, 256, 4096, 2048, 8, 128
    if not torch.cuda.is_available() or torch.cuda.device_count() < R:
        print("SKIP: requires", R, "Blackwell GPUs")
        raise SystemExit(77)
    ptr = ctypes.c_void_p
    P = ctypes.POINTER(ptr)
    lib = ctypes.CDLL(a.lib)
    prefix = "fastllm_ep8_mega_moe_"

    def bind(n, args, rest=ctypes.c_int):
        f = getattr(lib, prefix + n)
        f.argtypes = args
        f.restype = rest
        return f

    err = bind("error", [], ctypes.c_char_p)
    supported = bind("supported", [])
    backend = bind("backend_name", [], ctypes.c_char_p)
    for r in range(R):
        torch.cuda.set_device(r)
        if not backend().startswith(b"sm10x-") or not supported():
            print("SKIP: unsupported Blackwell MegaMoE device", r)
            raise SystemExit(77)
    size = bind("workspace_bytes", [ctypes.c_int], ctypes.c_size_t)(R)
    init = bind(
        "init_workspace",
        [
            ptr,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ],
    )
    create = bind(
        "create_mxfp4" if a.mxfp4 else "create",
        [P] * (2 if a.mxfp4 else 4) + [ptr, ctypes.c_size_t, P, ctypes.c_int, ctypes.c_int, ctypes.c_int],
        ptr,
    )
    run = bind("run_masked", [ptr] * 5 + [ctypes.c_int, ptr])
    destroy = bind("destroy", [ptr], None)
    weights = bind("weights", [ptr, P])
    capacity = bind("decode_capacity", [ctypes.c_int, ctypes.c_int])
    batched = bind("run_batched", [ptr] * 5 + [ctypes.c_int, ctypes.c_int, ptr])
    assert capacity(32 * R, R) == 32 and capacity(32 * R + 1, R) == 128
    assert capacity(128 * R - 1, R) == 128
    assert capacity(128 * R, R) == (256 if R == 8 else 128)
    assert capacity(128 * R + 1, R) == 256
    assert capacity(256 * R + 1, R) == 256
    assert capacity(0, 3) == 0
    for runtime in ("libcudart.so.12", "libcudart.so.13"):
        try:
            cuda = ctypes.CDLL(runtime)
            break
        except OSError:
            pass
    else:
        raise RuntimeError("CUDA runtime library not found")
    cuda.cudaMemcpy.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_int]
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

    def sync():
        for r in range(R):
            torch.cuda.synchronize(r)

    def quant(x):
        v = x.float().reshape(*x.shape[:-1], -1, 32)
        scale = torch.pow(
            2.0, torch.ceil(torch.log2(v.abs().amax(-1).clamp_min(1e-4) / 448.0))
        )
        q = (v / scale[..., None]).to(torch.float8_e4m3fn)
        return q.reshape(x.shape), scale

    def deq(q, s):
        return (q.float().reshape(*q.shape[:-1], -1, 32) * s[..., None]).reshape(
            q.shape
        )

    def unpack_mxfp4(packed, scales):
        magnitudes = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
        lut = torch.cat((magnitudes, -magnitudes))
        b = packed.view(torch.uint8)
        i = torch.stack((b & 15, b >> 4), dim=-1).long()
        n = b.shape[0]
        exponent = scales.view(torch.uint8).int() - 127
        return torch.ldexp(lut[i].reshape(n, -1, 32), exponent[..., None]).reshape(n, -1)

    def original(w, s):
        n, k = w.shape
        return (
            w.float().reshape(n // 128, 128, k // 128, 128) * s[:, None, :, None]
        ).reshape(n, k)

    def sf_layout(scale, n, k, gate):
        sf = (torch.log2(scale).to(torch.int32) + 127).to(torch.uint8)
        if gate:
            sf = torch.stack(
                (
                    sf[: n // 2].reshape(-1, 8, k // 32),
                    sf[n // 2 :].reshape(-1, 8, k // 32),
                ),
                dim=1,
            ).reshape(n, k // 32)
        sf = sf.reshape(n // 128, 4, 32, k // 32).transpose(1, 2).reshape(n, k // 32)
        return sf.reshape(n, k // 128, 4).permute(1, 0, 2).contiguous().view(-1)

    def weight_layout(q, gate):
        n, k = q.shape
        if gate:
            q = torch.stack(
                (q[: n // 2].reshape(-1, 8, k), q[n // 2 :].reshape(-1, 8, k)), dim=1
            ).reshape(n, k)
        return q.contiguous().view(torch.uint8)

    assert 0 < size <= {4: 45220224, 8: 45174144}[R], ("decode workspace grew", size)
    print("WORKSPACE", size, flush=True)
    # Three distinct native experts per owner, aliased across its expert IDs.
    # Reference copies stay on GPU0 and the tested packed representation stays local.
    reference = []
    ranks = []
    devices = (ctypes.c_int * R)(*range(R))
    if a.checkpoint:
        from safetensors import safe_open

        checkpoint = Path(a.checkpoint)
        weight_map = json.loads(
            (checkpoint / "model.safetensors.index.json").read_text()
        )["weight_map"]

        def load_weight(expert, part, suffix):
            key = f"model.layers.{a.layer}.mlp.experts.{expert}.{part}.{suffix}"
            with safe_open(
                str(checkpoint / weight_map[key]), framework="pt", device="cpu"
            ) as f:
                return f.get_tensor(key).cuda(0)

    for r in range(R):
        torch.cuda.set_device(0)
        expected = ()
        refs = []
        src = []
        for j in range(3):
            if a.checkpoint:
                expert = r * (E // R) + j
                w1 = torch.cat(
                    [load_weight(expert, k, "weight") for k in ("gate_proj", "up_proj")]
                )
                w2 = load_weight(expert, "down_proj", "weight")
                scale_key = "weight_scale" if a.mxfp4 else "weight_scale_inv"
                s1 = torch.cat(
                    [
                        load_weight(expert, k, scale_key)
                        for k in ("gate_proj", "up_proj")
                    ]
                )
                s2 = load_weight(expert, "down_proj", scale_key)
            elif a.mxfp4:
                w1 = torch.randint(256, (2 * I, H // 2), dtype=torch.uint8, device=0)
                w2 = torch.randint(256, (H, I // 2), dtype=torch.uint8, device=0)
                s1 = torch.randint(116, 122, (2 * I, H // 32), dtype=torch.uint8, device=0)
                s2 = torch.randint(116, 122, (H, I // 32), dtype=torch.uint8, device=0)
            else:
                w1 = (
                    torch.randn(2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.7
                ).to(torch.float8_e4m3fn)
                w2 = (torch.randn(H, I, device="cuda", dtype=torch.bfloat16) * 0.7).to(
                    torch.float8_e4m3fn
                )
                s1 = torch.rand(2 * I // 128, H // 128, device="cuda") * 0.012 + 0.005
                s2 = torch.rand(H // 128, I // 128, device="cuda") * 0.012 + 0.005
            if a.mxfp4:
                dw1, dw2 = unpack_mxfp4(w1, s1), unpack_mxfp4(w2, s2)
                refs.append((dw1, dw2, dw1, dw2) if a.checkpoint else (dw1, dw2))
                src.append(tuple(torch.cat((w.view(torch.uint8).flatten(), scale.view(torch.uint8).flatten())).to(r)
                                 for w, scale in ((w1, s1), (w2, s2))))
                continue
            orig1 = original(w1, s1)
            orig2 = original(w2, s2)
            q1, qs1 = quant(orig1)
            q2, qs2 = quant(orig2)
            dw1, dw2 = deq(q1, qs1), deq(q2, qs2)
            refs.append((dw1, dw2, orig1, orig2) if a.checkpoint else (dw1, dw2))
            src.append(tuple(v.to(r) for v in (w1, w2, s1, s2)))
            if j == 0:
                expected = (
                    weight_layout(q1, True),
                    weight_layout(q2, False),
                    sf_layout(qs1, 2 * I, H, True),
                    sf_layout(qs2, H, I, False),
                )
        reference.append(refs)
        torch.cuda.set_device(r)
        assert supported()
        allocation = torch.full((size + 4096,), 165, dtype=torch.uint8, device=r)
        assert init(allocation.data_ptr(), size, devices, r, R), err()
        ranks.append(
            dict(
                src=src,
                allocation=allocation,
                expected=tuple(v.to(r) for v in expected),
            )
        )
    peers = (ptr * R)(*[v["allocation"].data_ptr() for v in ranks])
    for r, v in enumerate(ranks):
        torch.cuda.set_device(r)
        arrays = [
            (ptr * (E // R))(*[v["src"][e % 3][part].data_ptr() for e in range(E // R)])
            for part in range(len(v["src"][0]))
        ]
        v["handle"] = create(*arrays, v["allocation"].data_ptr(), size, peers, r, 0, R)
        assert v["handle"], err()
        if a.mxfp4:
            assert not weights(v["handle"], (ptr * 4)()), "FP4 exposed as borrowed FP8 weights"
        elif R == 8:
            addresses = (ptr * 4)()
            assert weights(v["handle"], addresses)
            for i, expected in enumerate(v["expected"]):
                actual = torch.empty_like(expected)
                assert (
                    cuda.cudaMemcpy(actual.data_ptr(), addresses[i], actual.numel(), 3)
                    == 0
                )
                assert torch.equal(actual, expected), (
                    "packed representation mismatch",
                    r,
                    i,
                    int((actual != expected).sum()),
                )
        v["x"] = torch.empty(2 * C + 2, H, dtype=torch.bfloat16, device=r)
        v["idx"] = torch.empty(2 * C + 2, K, dtype=torch.int32, device=r)
        v["score"] = torch.empty(2 * C + 2, K, dtype=torch.float32, device=r)
        v["live"] = torch.ones(2 * C + 2, dtype=torch.int32, device=r)
        v["y"] = torch.full((2 * C + 2, H), -123, dtype=torch.bfloat16, device=r)
        print("LAYER_CREATE_PASS", r, flush=True)
    sync()

    def fill(n, scenario, seed_offset=0):
        for r, v in enumerate(ranks):
            torch.cuda.set_device(r)
            torch.manual_seed(1500 + n * 17 + r + seed_offset)
            v["x"].normal_(0, 0.25)
            v["y"].fill_(-123)
            v["live"].fill_(1)
            ids = torch.argsort(torch.rand(2 * C + 2, E, device=r), dim=-1)[:, :K]
            if scenario == "one_owner":
                ids = torch.arange(K, device=r).expand(2 * C + 2, K)
            v["idx"].copy_(ids.int())
            v["score"].copy_(torch.softmax(torch.randn(2 * C + 2, K, device=r), -1))
            if scenario == "masked":
                v["live"][::3] = 0
            if scenario == "all_masked":
                v["live"].zero_()
            v["rows"] = 0 if scenario == "idle" and r % 2 else n
            if scenario == "uneven":
                v["rows"] = [n, 0, n - 1, 1, n // 2, 7, 3, 0][r]
            if scenario == "all_idle":
                v["rows"] = 0
        sync()

    def reference_chunk(v, output, start):
        count = max(0, min(C, v["rows"] - start))
        assert run(
            v["handle"], v["x"][start].data_ptr(), output[start].data_ptr(),
            v["idx"][start].data_ptr(), v["score"][start].data_ptr(),
            count, v["live"][start].data_ptr(),
        ), err()

    def launch(n):
        for start in range(0, n, C):
            for r, v in enumerate(ranks):
                torch.cuda.set_device(r)
                reference_chunk(v, v["y"], start)
        sync()

    quality = []

    def verify(n, scenario):
        errors = []
        torch.cuda.set_device(0)
        for r, v in enumerate(ranks):
            rows = v["rows"]
            y = v["y"][:rows].to(0).float()
            assert torch.isfinite(y).all()
            assert (v["y"][rows:] == -123).all(), ("output guard", r)
            assert (v["allocation"][size:] == 165).all(), ("workspace guard", r)
            if not rows:
                continue
            x = v["x"][:rows].to(0)
            ids = v["idx"][:rows].to(0)
            score = v["score"][:rows].to(0)
            live = v["live"][:rows].to(0)
            xq, xs = quant(x)
            xd = deq(xq, xs)
            ref = torch.zeros(rows, H, device=0)
            # Group by expert while preserving ordered top-k accumulation in the final sum.
            parts = torch.zeros(rows, K, H, device=0)
            original_parts = torch.zeros_like(parts) if a.checkpoint else None
            for expert in ids.unique().tolist():
                row, slot = torch.where((ids == expert) & (live[:, None] > 0))
                if row.numel() == 0:
                    continue
                weights = reference[expert // (E // R)][(expert % (E // R)) % 3]
                w1, w2 = weights[:2]
                gu = (xd[row] @ w1.T).to(torch.bfloat16).float()
                mid = F.silu(gu[:, :I]) * gu[:, I:] * score[row, slot, None]
                q, s = quant(mid)
                out = (deq(q, s) @ w2.T).to(torch.bfloat16).float()
                parts[row, slot] = out
                if a.checkpoint:
                    gu0 = (x[row].float() @ weights[2].T).to(torch.bfloat16).float()
                    mid0 = (
                        (F.silu(gu0[:, :I]) * gu0[:, I:] * score[row, slot, None])
                        .to(torch.bfloat16)
                        .float()
                    )
                    original_parts[row, slot] = (
                        (mid0 @ weights[3].T).to(torch.bfloat16).float()
                    )
            for k in range(K):
                ref += parts[:, k]
            ref = ref.to(torch.bfloat16).float()
            rel = float((y - ref).norm() / ref.norm().clamp_min(1e-20))
            errors.append(rel)
            if rel >= 0.015:
                print(
                    "ROW_ERRORS",
                    r,
                    ((y - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-20)).tolist(),
                    flush=True,
                )
                torch.save(
                    dict(y=y.cpu(), ref=ref.cpu(), idx=ids.cpu(), live=live.cpu()),
                    a.out + ".failure.pt",
                )
            assert rel < 0.015, (
                r,
                n,
                scenario,
                rel,
                float(y.abs().max()),
                float(ref.abs().max()),
            )
            assert (y[live == 0] == 0).all(), ("masked output", r)
            if a.checkpoint:
                ref0 = original_parts.sum(dim=1).to(torch.bfloat16).float()
                quality.append(
                    dict(
                        rows=n,
                        scenario=scenario,
                        rank=r,
                        relative_l2_vs_unrequantized_bf16=float(
                            (y - ref0).norm() / ref0.norm().clamp_min(1e-20)
                        ),
                    )
                )
        return max(errors or [0.0])

    reports = []
    scenarios = [
        (1, "spread"),
        (32, "spread"),
        (33, "spread"),
        (64, "spread"),
        (128, "spread"),
        (64, "one_owner"),
        (64, "idle"),
        (128, "uneven"),
        (129, "uneven"),
        (256, "spread"),
        (256, "one_owner"),
        (256, "masked"),
        (64, "masked"),
        (64, "all_masked"),
        (64, "all_idle"),
    ]
    for n, scenario in scenarios:
        fill(n, scenario)
        launch(n)
        rel = (
            verify(n, scenario)
            if (n, scenario) in [(1, "spread"), (64, "spread"), (64, "masked"), (256, "masked")]
            else None
        )
        expected = [v["y"].clone() for v in ranks]
        sync()
        for r, v in enumerate(ranks):
            torch.cuda.set_device(r)
            assert batched(
                v["handle"],
                v["x"].data_ptr(),
                v["y"].data_ptr(),
                v["idx"].data_ptr(),
                v["score"].data_ptr(),
                v["rows"],
                capacity(n * R, R),
                v["live"].data_ptr(),
            ), err()
        sync()
        for r, v in enumerate(ranks):
            assert torch.equal(v["y"], expected[r]), (
                "eager tile mismatch",
                R,
                n,
                scenario,
                r,
            )
            assert (v["allocation"][size:] == 165).all(), ("workspace guard", r)
        # Capture old/new pairs in the same arena, retaining both outputs for checking.
        graphs = []
        for r, v in enumerate(ranks):
            torch.cuda.set_device(r)
            v["old_y"] = torch.full_like(v["y"], -123)
            g = ptr()
            ex = ptr()
            assert cuda.cudaStreamBeginCapture(ptr(2), 2) == 0
            for repeat in range(3):
                for start in range(0, n, C):
                    reference_chunk(v, v["old_y"], start)
                assert batched(
                    v["handle"],
                    v["x"].data_ptr(),
                    v["y"].data_ptr(),
                    v["idx"].data_ptr(),
                    v["score"].data_ptr(),
                    v["rows"],
                    capacity(n * R, R),
                    v["live"].data_ptr(),
                ), err()
            assert cuda.cudaStreamEndCapture(ptr(2), ctypes.byref(g)) == 0
            assert cuda.cudaGraphInstantiateWithFlags(ctypes.byref(ex), g, 0) == 0
            graphs.append((g, ex))
        for iteration in range(4):
            fill(n, scenario, seed_offset=iteration * 203)
            # Live mask contents change without recapturing or changing common capacity.
            if iteration % 2:
                for r, v in enumerate(ranks):
                    torch.cuda.set_device(r)
                    v["live"][1::4] = 0
                sync()
            for r, (_, ex) in enumerate(graphs):
                torch.cuda.set_device(r)
                assert cuda.cudaGraphLaunch(ex, ptr(2)) == 0
            sync()
            for r, v in enumerate(ranks):
                assert torch.equal(v["y"], v["old_y"]), (
                    "graph tile mismatch",
                    R,
                    n,
                    scenario,
                    iteration,
                    r,
                )
                assert torch.isfinite(v["y"]).all()
                assert (v["y"][v["rows"] :] == -123).all(), ("output guard", r)
                assert (v["allocation"][size:] == 165).all(), ("workspace guard", r)
        for r, (g, ex) in enumerate(graphs):
            torch.cuda.set_device(r)
            assert cuda.cudaGraphExecDestroy(ex) == 0
            assert cuda.cudaGraphDestroy(g) == 0
        report = dict(
            rows=n,
            scenario=scenario,
            capacity=capacity(n * R, R),
            exact=True,
            relative_l2=rel,
            graph_replays=4,
        )
        reports.append(report)
        print("PASS", json.dumps(report), flush=True)
        Path(a.out).write_text(
            json.dumps(
                dict(
                    ranks=R,
                    weight_format="mxfp4" if a.mxfp4 else "fp8",
                    workspace_bytes=size,
                    reports=reports,
                    checkpoint_quality=quality,
                ),
                indent=2,
            )
            + "\n"
        )
    def check_prefill(max_rows, uneven):
        total = max_rows * R
        span = scratch_bytes(total, R)
        if not span:
            # Small router buffers can use the bounded fallback, especially at EP4.
            assert max_rows <= 512
            return
        assert span <= total * H * 4
        buffers = []
        for rank in range(R):
            torch.cuda.set_device(rank)
            rows = max_rows
            if uneven:
                rows = [max_rows, 0, 13, max_rows - 1, max_rows // 2, 1, 0, 73][rank]
            x = torch.randn(max_rows + 1, H, dtype=torch.bfloat16, device=rank)
            first = torch.randint(E, (max_rows, 1), dtype=torch.int32, device=rank)
            idx = (first + 17 * torch.arange(K, dtype=torch.int32, device=rank)) % E
            scores = torch.rand(max_rows, K, device=rank).softmax(-1)
            y = torch.full_like(x, -123.)
            reference = torch.full_like(x, -123.)
            arena = torch.full((span + 4096,), 165, dtype=torch.uint8, device=rank)
            buffers.append((rows, x, idx, scores, y, reference, arena))
        sync()

        # Execute the original 128-row kernel over exactly the same routed tokens.
        for start in range(0, max_rows, C):
            for rank, (rows, x, idx, scores, _, reference, _) in enumerate(buffers):
                torch.cuda.set_device(rank)
                count = min(C, max(0, rows - start))
                assert batched(
                    ranks[rank]["handle"], x[start].data_ptr(), reference[start].data_ptr(),
                    idx[start].data_ptr(), scores[start].data_ptr(), count, C, None
                ), err()
        sync()

        peers = (ptr * R)(*[record[-1].data_ptr() for record in buffers])
        capacities = []
        for rank, record in enumerate(buffers):
            torch.cuda.set_device(rank)
            handle, arena = ranks[rank]["handle"], record[-1]
            assert prepare_prefill(handle, arena.data_ptr(), span - 1, total) == 0
            capacities.append(prepare_prefill(handle, arena.data_ptr(), span, total))
        capacity = capacities[0]
        assert len(set(capacities)) == 1 and C < capacity <= 32768, capacities
        sync()
        for start in range(0, max_rows, capacity):
            for rank, (rows, x, idx, scores, y, _, _) in enumerate(buffers):
                torch.cuda.set_device(rank)
                count = min(capacity, max(0, rows - start))
                assert run_prefill(
                    ranks[rank]["handle"], peers, x[start].data_ptr(), y[start].data_ptr(),
                    idx[start].data_ptr(), scores[start].data_ptr(), count
                ), err()
        sync()
        errors = []
        for rows, _, _, _, y, reference, arena in buffers:
            assert (arena[span:] == 165).all() and (y[rows:] == -123).all()
            if rows:
                actual, expected = y[:rows].float(), reference[:rows].float()
                rel = ((actual - expected).norm() / expected.norm().clamp_min(1e-20)).item()
                assert rel < 0.015 and actual.isfinite().all(), (max_rows, uneven, rel)
                errors.append(rel)
        reports.append(dict(prefill_rows=max_rows, capacity=capacity, uneven=uneven,
                            workspace_bytes=span, max_relative_l2=max(errors)))
        print("PREFILL_PASS", reports[-1], flush=True)

    if a.prefill:
        scratch_bytes = bind("prefill_workspace_bytes", [ctypes.c_int, ctypes.c_int], ctypes.c_size_t)
        prepare_prefill = bind("prepare_prefill", [ptr, ptr, ctypes.c_size_t, ctypes.c_int])
        run_prefill = bind("run_prefill", [ptr, P, ptr, ptr, ptr, ptr, ctypes.c_int])
        cases = [(512, False), (2048, True), (4096, False), (4096, True)]
        if R == 8:
            cases.extend([(8192, True), (16384, True), (32768, False), (32768, True)])
        for max_rows, uneven in cases:
            check_prefill(max_rows, uneven)
        Path(a.out).write_text(json.dumps(dict(
            ranks=R, weight_format="mxfp4" if a.mxfp4 else "fp8", workspace_bytes=size,
            reports=reports, checkpoint_quality=quality
        ), indent=2) + "\n")

    for r, v in enumerate(ranks):
        torch.cuda.set_device(r)
        destroy(v["handle"])
    print("ALL_PASS", flush=True)


if __name__ == "__main__":
    main()
