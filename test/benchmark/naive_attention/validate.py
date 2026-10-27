import json
import torch
from bench import Row, lib, ptr, check, here

torch.manual_seed(712)
checks = []


def error(x, y):
    x = x.double()
    y = y.double()
    d = x - y
    return {
        "max": d.abs().max().item(),
        "nrmse": (d.square().sum() / y.square().sum().clamp_min(1e-30)).sqrt().item(),
        "finite": bool(x.isfinite().all()),
    }


def validate(
    batch,
    capacity,
    ring=False,
    sparse=False,
    sink_on=True,
    heads=8,
    kvheads=1,
    scale=0.6,
    poison=False,
    all_invalid=False,
):
    stride = kvheads * 192 + (0 if ring else 128)
    count = 128 if ring else 2048 if sparse else capacity
    q = torch.randn(batch, heads, 192, device="cuda", dtype=torch.bfloat16) * scale
    k = torch.randn(batch, capacity, stride, device="cuda", dtype=torch.bfloat16) * scale
    v = torch.randn(batch, capacity, kvheads, 128, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda") * 3 if sink_on else None
    live = torch.full((batch,), capacity, device="cuda", dtype=torch.int32)
    indices = None
    if sparse:
        indices = torch.stack(
            [torch.randperm(capacity, device="cuda")[:2048] for _ in range(batch)]
        ).int()
        indices[:, ::17] = -1
        indices[:, ::23] = capacity + 2
        if all_invalid:
            indices.fill_(-1)
    out = torch.empty(batch, heads, 128, device="cuda", dtype=torch.bfloat16)
    rows = (Row * batch)(
        *[
            Row(
                ptr(k[b]),
                ptr(v[b]),
                ptr(indices[b]) if sparse else None,
                ptr(live[b]),
                capacity,
                capacity,
                stride,
                count if sparse else 0,
            )
            for b in range(batch)
        ]
    )
    funcs = {
        "cuda": lambda: check(
            lib.run(
                ptr(q),
                rows,
                ptr(sink),
                ptr(out),
                batch,
                heads,
                kvheads,
                count,
                ring,
                torch.cuda.current_stream().cuda_stream,
            )
        )
    }
    if not ring and not sparse and kvheads == 1 and heads % 8 == 0 and lib.supports_tma():
        funcs["hopper"] = lambda: check(
            lib.run_tma(
                ptr(q),
                rows,
                ptr(sink),
                ptr(out),
                batch,
                heads,
                kvheads,
                count,
                torch.cuda.current_stream().cuda_stream,
            )
        )
    graphs = {}
    for name, fn in funcs.items():
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        graphs[name] = g
    bounds = [
        0,
        1,
        31,
        32,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        511,
        512,
        739,
        768,
        802,
        896,
        1023,
        1024,
        1536,
        2047,
        2048,
    ]
    original_k = k.clone() if poison else None
    original_v = v.clone() if poison else None
    for trial in range(3):
        if ring:
            lengths = [bounds[(b + trial * 7) % len(bounds)] for b in range(batch)]
        elif trial == 0:
            lengths = [capacity] * batch
        else:
            lengths = [min(capacity, bounds[(b + trial * 7) % len(bounds)]) for b in range(batch)]
        live.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        if poison:
            k.copy_(original_k)
            v.copy_(original_v)
            for b, length in enumerate(lengths):
                if length < capacity:
                    k[b, length:].fill_(float("nan"))
                    v[b, length:].fill_(float("nan"))
        # Independent float64 reference, including BF16 materialization points.
        want = []
        truth = []
        for start in range(0, batch, 8):
            size = min(8, batch - start)
            lens = live[start : start + size].long()
            if ring:
                logical = torch.arange(count, device="cuda")[None, :].expand(size, -1)
                idx = ((lens - 128).clamp_min(0)[:, None] + logical) & 127
                valid = logical < lens.clamp_max(128)[:, None]
            elif sparse:
                idx = indices[start : start + size].long()
                valid = (idx >= 0) & (idx < lens[:, None])
                idx = idx.clamp(0, capacity - 1)
            else:
                idx = torch.arange(count, device="cuda")[None, :].expand(size, -1)
                valid = idx < lens[:, None]
            kr = (
                k[start : start + size]
                .gather(1, idx[:, :, None].expand(-1, -1, stride))[:, :, : kvheads * 192]
                .view(size, count, kvheads, 192)
                .double()
            )
            vr = (
                v[start : start + size]
                .gather(1, idx[:, :, None, None].expand(-1, -1, kvheads, 128))
                .double()
            )
            kr = kr.repeat_interleave(heads // kvheads, 2).permute(0, 2, 1, 3)
            vr = vr.masked_fill(~valid[:, :, None, None], 0)
            vr = vr.repeat_interleave(heads // kvheads, 2).permute(0, 2, 1, 3)
            raw = (q[start : start + size].double()[:, :, None, :] * kr).sum(-1)

            def ref(rounding):
                score = (
                    (raw.bfloat16().double() * (192**-0.5)).bfloat16().double()
                    if rounding
                    else raw * (192**-0.5)
                )
                score = score.masked_fill(~valid[:, None, :], float("-inf"))
                if sink is not None:
                    score = torch.cat(
                        (score, sink[None, :, None].double().expand(size, -1, 1)), dim=-1
                    )
                p = score.softmax(-1)[..., :count]
                p = torch.nan_to_num(p)
                if rounding:
                    p = p.bfloat16().double()
                return (p[:, :, :, None] * vr).sum(-2).bfloat16()

            want.append(ref(True))
            truth.append(ref(False))
        want = torch.cat(want)
        truth = torch.cat(truth)
        for name, g in graphs.items():
            g.replay()
            got = out.clone()
            er = error(got, want)
            et = error(got, truth)
            assert er["finite"] and er["nrmse"] < 0.003 and er["max"] < 0.032, (
                batch,
                capacity,
                trial,
                name,
                er,
            )
            assert et["finite"] and et["nrmse"] < 0.03, (batch, capacity, trial, name, et)
            checks.append(
                {
                    "batch": batch,
                    "capacity": capacity,
                    "ring": ring,
                    "sparse": sparse,
                    "sink": sink_on,
                    "heads": heads,
                    "kvheads": kvheads,
                    "trial": trial,
                    "poison": poison,
                    "all_invalid": all_invalid,
                    "path": name,
                    "staged_reference": er,
                    "fp64_reference": et,
                }
            )
    print("PASS", batch, capacity, ring, sparse, heads, kvheads, flush=True)


if __name__ == "__main__":
    for b in [17, 32, 65, 128, 129, 256]:
        validate(b, 128, ring=True, sink_on=(b != 65))
    for b, n in [(17, 31), (32, 128), (65, 896), (128, 1024), (129, 2047), (256, 2048)]:
        validate(b, n, sink_on=(b != 32))
    for b in [32, 128]:
        validate(b, 4097, sparse=True, sink_on=(b == 128))
    validate(17, 896, heads=16, kvheads=2)
    validate(17, 896, heads=16, kvheads=1)
    validate(32, 128, ring=True, heads=16, kvheads=2)
    validate(32, 896, scale=2.0)
    validate(32, 896, poison=True)
    validate(32, 128, ring=True, poison=True)
    validate(32, 4097, sparse=True, sink_on=False, all_invalid=True)
    (here / "validation.json").write_text(json.dumps({"passed": True, "checks": checks}, indent=2))
    print("PASS_CHECKS", len(checks), flush=True)
