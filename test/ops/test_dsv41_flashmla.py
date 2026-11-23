"""Compare the native SM103 adapter with independent gathers, FP64 attention and real projection weights."""

import argparse
import ctypes as ct
import json
import math
import os
from pathlib import Path


class Request(ct.Structure):
    _fields_ = [(n, ct.c_void_p) for n in ("ring", "compressed", "index", "raw", "score")] + [
        (n, ct.c_int) for n in ("position", "begin", "length")
    ]


def relative(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-24)).item()


def main():
    import torch
    from safetensors import safe_open

    ap = argparse.ArgumentParser()
    ap.add_argument("bridge")
    ap.add_argument("checkpoint")
    opt = ap.parse_args()
    lib = ct.CDLL(str(Path(opt.bridge).resolve()))
    lib.test_v41_mla.argtypes = [ct.c_int] * 3 + [ct.c_float] + [ct.c_void_p] * 10
    lib.test_v41_output.argtypes = [ct.c_int] + [ct.c_void_p] * 7
    dev = "cuda"
    bf = torch.bfloat16
    f8 = torch.float8_e4m3fn
    torch.manual_seed(8021)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(opt.checkpoint)
    index = json.loads(next(root.glob("*index.json")).read_text())["weight_map"]

    def weight(name):
        k = "layers.0.attn." + name
        with safe_open(root / index[k], framework="pt") as f:
            return f.get_tensor(k)

    aw, bw = (weight(n + ".weight").view(torch.uint8).contiguous() for n in ("wo_a", "wo_b"))
    af, bf_scale = (
        torch.pow(2.0, weight(n + ".scale").view(torch.uint8).to(torch.int) - 127).contiguous()
        for n in ("wo_a", "wo_b")
    )

    def dequant(w, s):
        n, k = w.shape
        return (
            w.cuda().view(f8).float().view(n // 32, 32, k // 32, 32) * s.cuda()[:, None, :, None]
        ).reshape(n, k)

    a, b = dequant(aw, af).view(8, 1024, 4096), dequant(bw, bf_scale)
    shapes = ((1, 0, 1), (3, 1, 17), (16, 2, 128), (17, 1, 129), (128, 2, 32), (3, 2, 1024), (128, 1, 1024))
    for m, ratio, width in shapes:
        pos = (torch.arange(m) * 91 + 129) % 384
        if m > 1:
            pos[:3] = torch.tensor([0, 127, 128])
        if m == 3 and width == 1024:
            pos[:] = torch.tensor([8191, 2049, 2048])
        pitch = (m + 3) // 4 * 4
        maxc = int(pos.max()) + 1
        rr = torch.randint(0, 120, (m, 128, 512), device=dev, dtype=torch.uint8)
        rr ^= torch.randint(0, 2, rr.shape, device=dev, dtype=torch.uint8) * 128
        rs = torch.randint(119, 125, (m, 128, 16), device=dev, dtype=torch.uint8)
        ring = torch.cat((rr, rs), dim=-1).contiguous()
        rv = (
            (rr.view(f8).float().view(m, 128, 16, 32) * torch.exp2(rs.float() - 127)[..., None])
            .reshape(m, 128, 512)
            .bfloat16()
        )
        cb = torch.randint(0, 256, (m, maxc, 256), device=dev, dtype=torch.uint8)
        cs = torch.randint(32, 48, (m, maxc, 32), device=dev, dtype=torch.uint8)
        cmp = torch.cat((cb, cs), dim=-1).contiguous()
        grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=dev)
        nibbles = torch.stack((cb & 15, cb >> 4), dim=-1).long()
        cv = (
            (grid[nibbles].reshape(m, maxc, 32, 16) * cs.view(f8).float()[..., None])
            .reshape(m, maxc, 512)
            .bfloat16()
        )
        select = torch.randint(-2, maxc, (m, width), dtype=torch.int32, device=dev)
        rawq = torch.randn(m, 64, 512, device=dev, dtype=bf) * 0.25
        cur = torch.randn(m, 512, device=dev, dtype=bf) * 1.5
        currentPacked = None
        if os.environ.get("FASTLLM_DSV41_MLA_PAGED") == "1":
            # Production passes the output of the exact block-32 ring encoder.
            # Distinct values here catch accidental reads of an old ring slot.
            ex = torch.exp2(
                torch.ceil(torch.log2(cur.float().reshape(m, 16, 32).abs().amax(-1).clamp_min(1e-4) / 448))
            )
            code = (cur.float().view(m, 16, 32) / ex[..., None]).to(f8)
            cur = (code.float() * ex[..., None]).reshape(m, 512).bfloat16()
            currentPacked = torch.cat(
                (code.view(torch.uint8).reshape(m, 512), (ex.log2() + 127).to(torch.uint8)), dim=-1
            ).contiguous()
        sink = torch.randn(64, device=dev)
        phase = pos.cuda().float()[:, None] * torch.linspace(0.0001, 0.4, 32, device=dev)
        angles = torch.cat((phase.cos(), phase.sin()), dim=-1).contiguous()
        packed = rawq.view(m, 64, 32, 16).transpose(1, 2).contiguous()
        order = list(reversed(range(m)))
        desc = (Request * m)(
            *[
                Request(ring[i].data_ptr(), cmp[i].data_ptr(), 0, 0, 0, int(pos[t]), 0, 0)
                for t, i in enumerate(order)
            ]
        )
        req = torch.frombuffer(bytearray(bytes(desc)), dtype=torch.uint8).cuda()
        actual = torch.empty_like(rawq)
        codes = torch.empty((m, 32768), device=dev, dtype=torch.uint8)
        sf = torch.empty((8, 32, pitch), device=dev, dtype=torch.int32)
        ptr = lambda x: x.data_ptr()
        torch.cuda.synchronize()
        rc = lib.test_v41_mla(
            m,
            width,
            ratio,
            1 / math.sqrt(512),
            *[ptr(x) for x in (packed, cur, angles, req, select, sink, actual, codes, sf)],
            ptr(currentPacked) if currentPacked is not None else None
        )
        assert rc == 0, rc

        def rope(x, t, inverse):
            y = x.clone()
            xy = x[..., 448:].reshape(-1, 32, 2)
            c = angles[t, :32].double()
            s = angles[t, 32:].double() * (-1 if inverse else 1)
            y[..., 448:] = torch.stack(
                (xy[..., 0] * c - xy[..., 1] * s, xy[..., 0] * s + xy[..., 1] * c), -1
            ).flatten(-2)
            return y

        ref = []
        for t, i in enumerate(order):
            end = int(pos[t])
            size = (end + 1) // ratio if ratio else 0
            ids = select[t, : min(width, size)].long()
            ids = ids[(ids >= 0) & (ids < size)]
            hist = torch.arange(max(0, end - 127), end, device=dev) % 128
            keys = torch.cat((rv[i, hist], cur[t : t + 1], cv[i, ids])).double()
            q = rope(rawq[t].double(), t, False).bfloat16().double()
            prob = torch.cat((q @ keys.T / math.sqrt(512), sink.double()[:, None]), dim=-1).softmax(-1)
            value = prob[:, :-1] @ keys
            round_bf16 = os.environ.get("FASTLLM_DSV41_MLA_BF16_ROUND") == "1"
            if round_bf16:
                value = value.bfloat16().double()
            rotated = rope(value, t, True)
            ref.append(rotated.bfloat16().double() if round_bf16 else rotated)
        ref = torch.stack(ref)
        err = relative(actual, ref)
        assert err < 0.055, (m, "attention", err)
        # Independent inverse of the FlashMLA output layout and packed word bytes.
        exp = sf[:, :, :m].permute(2, 0, 1).contiguous().view(torch.uint8).reshape(m, 8, 128)
        values = codes.view(f8).float().reshape(m, 8, 128, 32) * torch.exp2(exp.float() - 127)[..., None]
        if os.environ.get("FASTLLM_DSV41_MLA_HEAD_MAJOR") == "1":
            decoded = values.reshape(m, 64, 512)
        else:
            decoded = values.reshape(m, 8, 16, 8, 32).transpose(2, 3).reshape(m, 64, 512)
        assert torch.equal(decoded.bfloat16(), actual)
        # Compare the quantized grid as well as the unquantized FP64 answer.
        # A large error in one request must not disappear in the whole-batch norm.
        blocks = ref.reshape(m, 64, 16, 32)
        oracle_scale = (blocks.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil().exp2()
        oracle_codes = (blocks / oracle_scale[..., None]).float().clamp(-448, 448).to(f8)
        expected = (oracle_codes.float() * oracle_scale[..., None]).reshape(m, -1)
        row_error = (actual.float().reshape(m, -1) - expected).norm(dim=1) / expected.norm(dim=1)
        max_row_error = row_error.max().item()
        assert max_row_error < 0.055, (m, "quantized attention row", max_row_error)
        y = torch.empty(m, 5120, device=dev, dtype=bf)
        rc = lib.test_v41_output(m, *[ptr(x) for x in (aw, af, bw, bf_scale, codes, sf, y)])
        assert rc == 0, (m, rc)
        oa = (
            torch.einsum("mgk,gnk->mgn", decoded.reshape(m, 8, 4096), a).bfloat16().float().reshape(m, -1, 32)
        )
        scales = torch.exp2(torch.ceil(torch.log2(oa.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
        rounded = (oa / scales).clamp(-448, 448).to(f8).float() * scales
        refy = (rounded.reshape(m, -1) @ b.T).bfloat16()
        ep = relative(y, refy)
        assert ep < 0.015, (m, "output", ep)
        print(
            json.dumps(
                dict(
                    rows=m,
                    ratio=ratio,
                    width=width,
                    attention_relative_l2=err,
                    max_quantized_attention_row_l2=max_row_error,
                    output_relative_l2=ep,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
