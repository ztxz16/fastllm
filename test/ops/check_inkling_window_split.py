"""Check the singleton Triton scan's split-major output against explicit paged attention."""
from pathlib import Path
import sys

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_triton_server import fastllm_inkling_window_split


def check_scan(past, page_len, heads, group, window, extent, splits):
    dim = 128
    length = min(past + 1, window)
    start = past + 1 - length
    first_page = start // page_len
    pages = past // page_len - first_page + 1
    kv_heads = heads // group
    # K and V have independent physical page numbers after eviction.
    ids = torch.stack([torch.randperm(pages + 2, device="cuda")[:pages] for _ in range(2)], dim=1).int()
    key, value = [torch.randn(pages + 2, page_len, kv_heads, dim, device="cuda", dtype=torch.float16)
                  for _ in range(2)]
    query = torch.randn(heads, dim, device="cuda")
    bias = torch.randn(heads, extent, device="cuda")
    visit = torch.zeros(80, dtype=torch.uint8)
    visit.view(torch.int64)[:3] = torch.tensor([key.data_ptr(), value.data_ptr(), ids.data_ptr()])
    visit.view(torch.int32)[6:8] = torch.tensor([first_page, page_len])
    visit.view(torch.int32)[18] = past
    visit = visit.cuda()
    partial = torch.full((splits * heads * (dim + 1),), float("nan"), device="cuda")
    unused = query.bfloat16()
    fastllm_inkling_window_split[(splits, kv_heads, 1)](
        visit, query, bias, unused, unused, partial, 80, 72, 1, heads, heads * dim,
        heads, group, 128, window, extent, max(16, triton.next_power_of_2(triton.cdiv(window, splits))),
        num_warps=4, num_stages=2)

    pos = torch.arange(start, past + 1, device="cuda")
    logical = pos // page_len - first_page
    k = key[ids[logical, 0].long(), pos % page_len].float().repeat_interleave(group, dim=1)
    v = value[ids[logical, 1].long(), pos % page_len].float().repeat_interleave(group, dim=1)
    dist = past - pos
    scores = (k * query).sum(dim=-1) * dim**-0.5
    scores += torch.where(dist < extent, bias[:, dist.clamp_max(extent - 1)], 0.).T
    values = partial[:splits * heads * dim].view(splits, heads, dim)
    lse = partial[splits * heads * dim:].view(splits, heads)
    chunk = max(16, triton.cdiv(length, splits))
    for i, part in enumerate(scores.split(chunk)):
        expected = (v[i * chunk:i * chunk + len(part)] * part.softmax(dim=0).unsqueeze(-1)).sum(dim=0)
        torch.testing.assert_close(values[i], expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(lse[i], part.logsumexp(dim=0) * 1.4426950408889634, atol=2e-5, rtol=2e-5)
    live = triton.cdiv(length, chunk)
    assert values[live:].isnan().all() and lse[live:].isnan().all()


if __name__ == "__main__":
    torch.manual_seed(257)
    for splits, heads in ((16, 24), (32, 12)):
        for past, page_len in ((0, 1), (16, 16), (127, 128), (513, 37), (1023, 128), (8317, 37)):
            check_scan(past, page_len, heads, 3, 1024, 512, splits)
    print("Inkling split window scan passed")
