"""Compare fused FP8 verification against the same paged KV expanded exactly to FP16."""
from pathlib import Path
import sys

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_triton_server import fastllm_inkling_window_verify


def check(batch, chain, page, mapped):
    heads, group, dim, window, extent = 6, 3, 128, 129, 64
    kv_heads = heads // group
    past = 141 + torch.arange(batch) % 19
    first = (past + 1 - window) // page
    live = torch.arange(batch) % (chain + 1)
    pages = int(((past + chain - 1) // page - first + 1).max())
    ids = torch.stack([torch.randperm(pages, device="cuda") for _ in range(batch * 2)])
    ids = ids.view(batch, 2, pages).transpose(1, 2).int().contiguous()
    shape = (pages, page, kv_heads, dim)
    kv8 = [torch.randn(shape, device="cuda").to(torch.float8_e4m3fn) for _ in range(2)]
    query, bias, sink, gate = [torch.randn(batch, chain, heads, width, device="cuda", dtype=torch.bfloat16)
                                for width in (dim, extent, 1, dim)]

    def run(kv, fp8):
        key, value = kv
        desc = torch.zeros(batch, chain, 80, dtype=torch.uint8, device="cuda")
        desc64 = desc.view(torch.int64)[:, 0]
        desc64[:, 0], desc64[:, 1] = key.data_ptr(), value.data_ptr()
        desc64[:, 2] = ids.data_ptr() + torch.arange(batch, device="cuda") * ids.stride(0) * ids.element_size()
        if mapped:
            # Graph padding and host-backed requests can coexist. Give every other
            # request a direct page table and lay V immediately after each K page.
            byte_pages = [t.view(torch.uint8)[ids[:, :, i].long()] for i, t in enumerate(kv)]
            pairs = torch.stack(byte_pages, dim=2)
            offsets = torch.arange(batch * pages, device="cuda", dtype=torch.int64).view(batch, pages)
            addresses = pairs.data_ptr() + offsets * pairs.stride(1)
            desc64[::2, :2] = 0
            desc64[::2, 2] = (addresses.data_ptr() + torch.arange(batch, device="cuda") * pages * 8)[::2]
        info = desc.view(torch.int32)[:, 0]
        info[:, 6], info[:, 7] = first.to("cuda"), page
        info[:, 18], info[:, 19] = past.to("cuda"), live.to("cuda")
        out = torch.full_like(query, float("nan"))
        fastllm_inkling_window_verify[(triton.cdiv(chain, 32 // group), kv_heads, batch)](
            desc, query, bias, sink, gate, out, 80, 72, chain, heads, heads * dim,
            heads, group, page, window, extent, MAPPED=mapped, FP8_KV=fp8, num_warps=4, num_stages=2)
        return out

    expected = run([t.half() for t in kv8], False)
    actual = run(kv8, True)
    # The two generated MMA kernels can differ at BF16 rounding boundaries.
    delta = actual.float() - expected.float()
    rel_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float())
    print(f"batch={batch} chain={chain} mapped={mapped} relL2={rel_l2.item():.4g} maxAbs={delta.abs().max().item():.4g}")
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.008)
    assert rel_l2 < 1e-4
    padding = torch.arange(chain)[None, :] >= live[:, None]
    assert torch.count_nonzero(actual[padding.cuda()]) == 0


if __name__ == "__main__":
    torch.manual_seed(257)
    for case in ((65, 8, 16, False), (129, 13, 37, True)):
        check(*case)
    print("FP8 compact verification matches expanded KV")
