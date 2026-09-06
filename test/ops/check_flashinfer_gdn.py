"""Check the SM90 C export, layout adapters, tails, and fallback state safety.

Requires the optional FlashInfer/CuTe compiler environment and a CUDA build.
Set FASTLLM_CUDA_TRITON_PYTHON and FASTLLM_CUDA_TRITON_SERVER_SCRIPT to select
the compiler. Each run checks lazy compilation in an isolated, corrupt cache.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import tempfile

import torch
from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm90 import cp_delta_rule_dsl_sm90


def error(actual, reference):
    diff = actual.float() - reference.float()
    rms = (diff.square().mean() / reference.float().square().mean().clamp_min(1e-20)).sqrt()
    return dict(nrmse=rms.item(), max_abs=diff.abs().max().item())


def check(lib, tokens, empty=False, recurrent_reference=False):
    qkv = torch.randn(1, tokens, 10240, device='cuda', dtype=torch.float16) * .3
    weight = torch.full((128,), 128 ** -.5, device='cuda')
    g = (-torch.rand(1, tokens, 48, device='cuda') * .05).half()
    beta = torch.rand_like(g)
    state = (torch.randn(1, 48, 128, 128, device='cuda') * .03).half()
    initial = torch.zeros_like(state) if empty else state.clone()
    out = torch.full((tokens, 48, 128), float('nan'), device='cuda', dtype=torch.float16)
    pointers = [x.data_ptr() for x in (qkv, weight, g, beta, state, out)]
    assert lib.FlashInferGdnTestRun(*pointers, tokens, empty, 0)
    torch.cuda.synchronize()
    rawq, rawk, v = qkv.reshape(tokens, -1).split((2048, 2048, 6144), dim=-1)
    def norm(x):
        x = x.reshape(tokens, 16, 128).float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight).half()
    q, k, v = norm(rawq), norm(rawk), v.reshape(tokens, 48, 128).contiguous()
    alpha, b = g.float().reshape(tokens, 48).exp(), beta.float().reshape(tokens, 48)
    seq = torch.tensor([0, tokens], device='cuda', dtype=torch.int64)
    ref, refstate = torch.empty_like(out), torch.empty_like(state, dtype=torch.float32)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    target = max(1, sms // 48)
    cp_len = ((tokens + target - 1) // target + 511) // 512 * 512
    cp_delta_rule_dsl_sm90(ref, refstate, q, k, v, alpha, b, seq, 128 ** -.5,
        initial_state=initial.float().transpose(-1, -2).contiguous(), cp_chunk_len=cp_len)
    so = refstate.transpose(-1, -2).contiguous().half()
    metrics = dict(tokens=tokens, empty=empty, output=error(out, ref), state=error(state, so))
    assert torch.isfinite(out).all() and torch.isfinite(state).all()
    assert metrics['output']['nrmse'] < .003, metrics
    assert metrics['state']['nrmse'] < .003, metrics
    if recurrent_reference:
        # Independent FP32 token recurrence, state in [K,V] throughout.
        h = initial[0].float()
        qr, kr = q.float().repeat_interleave(3, 1), k.float().repeat_interleave(3, 1)
        rows = []
        for i in range(tokens):
            h = h * alpha[i, :, None, None]
            delta = (v[i].float() - torch.einsum('hk,hkv->hv', kr[i], h)) * b[i, :, None]
            h = h + kr[i, :, :, None] * delta[:, None, :]
            rows.append(torch.einsum('hk,hkv->hv', qr[i], h) * 128 ** -.5)
        metrics['fp32_recurrence_output'] = error(out, torch.stack(rows))
        metrics['fp32_recurrence_state'] = error(state[0], h)
        assert metrics['fp32_recurrence_output']['nrmse'] < .006, metrics
        assert metrics['fp32_recurrence_state']['nrmse'] < .006, metrics
    # Both a continued invocation and a different tail reuse the same export.
    if not empty:
        initial.copy_(state)
        assert lib.FlashInferGdnTestRun(*pointers, tokens, False, 0)
        cp_delta_rule_dsl_sm90(ref, refstate, q, k, v, alpha, b, seq, 128 ** -.5,
            initial_state=initial.float().transpose(-1, -2).contiguous(), cp_chunk_len=cp_len)
        torch.cuda.synchronize()
        assert error(out, ref)['nrmse'] < .003
        assert error(state, refstate.transpose(-1, -2))['nrmse'] < .003
    state.fill_(.25)
    out.fill_(17)
    for invalid in range(1, 11):
        assert not lib.FlashInferGdnTestRun(*pointers, tokens, False, invalid), invalid
    assert not lib.FlashInferGdnTestRun(*pointers, 1023, False, 0)
    if tokens == 1024:
        previous_cache = os.environ.get('FASTLLM_CUDA_TRITON_CACHE_DIR')
        with tempfile.TemporaryDirectory(prefix='fastllm-gdn-bad-cache-') as cache:
            meta = dict(ok=True, op='flashinfer_gdn', abi=1, arch=90, dtype='fp16', key_heads=16,
                        value_heads=48, library=str(Path(cache) / 'missing.so'))
            (Path(cache) / 'flashinfer_gdn_v1_fp16_sm90_k16_v48.json').write_text(json.dumps(meta))
            os.environ['FASTLLM_CUDA_TRITON_CACHE_DIR'] = cache
            assert not lib.FlashInferGdnTestRun(*pointers, tokens, False, 0)
        if previous_cache is None:
            os.environ.pop('FASTLLM_CUDA_TRITON_CACHE_DIR')
        else:
            os.environ['FASTLLM_CUDA_TRITON_CACHE_DIR'] = previous_cache
    for env in ('FASTLLM_CUDA_TRITON', 'FASTLLM_CUDA_TRITON_FLASHINFER_GDN'):
        os.environ.pop(env)
        assert not lib.FlashInferGdnTestRun(*pointers, tokens, False, 0)
        os.environ[env] = '0'
        assert not lib.FlashInferGdnTestRun(*pointers, tokens, False, 0)
        os.environ[env] = '1'
    torch.cuda.synchronize()
    assert torch.all(state == .25) and torch.all(out == 17)
    print(json.dumps(metrics), flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lib', type=Path, default=root / 'build-fastllm/tools/ftllm/libfastllm_tools.so')
    parser.add_argument('--cuda-root', type=Path, default=Path('/usr/local/cuda'))
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        print('SKIP: requires SM90')
        return 77
    os.environ['FASTLLM_CUDA_TRITON'] = '1'
    os.environ['FASTLLM_CUDA_TRITON_FLASHINFER_GDN'] = '1'
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    with tempfile.TemporaryDirectory(prefix='fastllm-gdn-test-') as tmp:
        cache = Path(tmp) / 'cache'
        cache.mkdir()
        (cache / 'flashinfer_gdn_v1_fp16_sm90_k16_v48.json').write_text('{broken')
        os.environ['FASTLLM_CUDA_TRITON_CACHE_DIR'] = str(cache)
        bridge = Path(tmp) / 'bridge.so'
        subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-O2',
            f'-I{root / "include"}', f'-I{args.cuda_root / "include"}',
            f'-I{root / "third_party/json11"}',
            str(Path(__file__).with_name('flashinferGdnTestBridge.cpp')),
            str(args.lib.resolve()), f'-Wl,-rpath,{args.lib.resolve().parent}',
            '-o', str(bridge)], check=True)
        lib = ctypes.CDLL(str(bridge))
        lib.FlashInferGdnTestStream.restype = ctypes.c_void_p
        lib.FlashInferGdnTestRun.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_bool, ctypes.c_int]
        lib.FlashInferGdnTestRun.restype = ctypes.c_bool
        stream = torch.cuda.ExternalStream(lib.FlashInferGdnTestStream())
        with torch.cuda.stream(stream):
            for tokens, empty in ((1024, True), (1025, False), (4096, False), (4103, False), (8192, False)):
                check(lib, tokens, empty, recurrent_reference=tokens == 1024)
        meta = json.loads((cache / 'flashinfer_gdn_v1_fp16_sm90_k16_v48.json').read_text())
        assert {p.name for p in Path(meta['library']).parent.iterdir()} == {'gdn.so', 'gdn.cubin'}
    print('PASS: exports, adapters, FP32 recurrence, tails, continuation, fallback, cache recovery')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
