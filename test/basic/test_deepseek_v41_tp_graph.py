#!/usr/bin/env python3
"""Expand the CPU fixture to CUDA TP dimensions and compare eager/graph execution.

Requires numpy, torch, safetensors and two CUDA GPUs. The fixed input tokens
exercise Engram, shared KV, both indexer stages and changing request lengths.
"""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile

import torch
from safetensors.torch import load_file, save_file
from test_deepseek_v41_cpu_fixture import unpack


def prepare(directory, hc_mult=4, expert_cache=False):
    unpack(str(Path(__file__).with_name('deepseek_v41_fixture.npz')), str(directory))
    config = json.loads((directory / 'config.json').read_text())
    config['text_config'].update(num_attention_heads=64, head_dim=512,
                                 index_head_dim=128, moe_intermediate_size=256,
                                 vocab_size=256, hc_mult=hc_mult)
    if expert_cache:
        config['text_config']['n_routed_experts'] = 4
    (directory / 'config.json').write_text(json.dumps(config))
    weights = load_file(str(directory / 'model.safetensors'))
    for name, tensor in list(weights.items()):
        value = tensor
        if hc_mult == 4 and name.endswith(('hc_attn_fn', 'hc_ffn_fn')):
            value = (tensor.float().repeat(3, 2) / 2).to(tensor.dtype)
        elif hc_mult == 4 and name.endswith(('hc_attn_base', 'hc_ffn_base')):
            value = tensor.repeat(3)
        elif hc_mult == 4 and name.endswith(('engram.q_weight', 'engram.k_weight')):
            value = tensor.repeat(2, 1)
        elif hc_mult == 4 and name.endswith('engram.wkv.weight'):
            value = tensor.reshape(3, 128, -1)[[0, 1, 0, 1, 2]].reshape(640, -1)
        elif name in ('embed.weight', 'head.weight'):
            value = tensor.repeat(2, 1)
        elif '.attn.indexer.' in name:
            if name.endswith('k_norm.weight'):
                value = tensor.repeat(2)
            elif name.endswith('wk.weight'):
                value = (tensor.float().repeat(2, 4) / 4).to(tensor.dtype)
            elif name.endswith('wq_b.weight'):
                value = tensor.reshape(2, 64, -1).repeat(1, 2, 1).reshape(256, -1)
        elif '.attn.' in name:
            if name.endswith('attn_sink'):
                value = tensor.repeat(16)
            elif name.endswith('wq_b.weight'):
                value = tensor.reshape(4, 128, -1).repeat(16, 4, 1).reshape(32768, -1)
            elif name.endswith(('wkv.weight', 'wgate.weight')):
                value = tensor.repeat(4, 1)
            elif name.endswith(('kv_norm.weight', 'compressor.norm.weight')):
                value = tensor.repeat(4)
            elif name.endswith('wo_a.weight'):
                value = (tensor.float().repeat(1, 64) / 64).to(tensor.dtype)
        elif '.ffn.' in name and '.gate.' not in name:
            if name.endswith(('.w1.weight', '.w3.weight')):
                value = tensor.repeat(2, 1)
            elif name.endswith('.w2.weight'):
                value = (tensor.float().repeat(1, 2) / 2).to(tensor.dtype)
        weights[name] = value.contiguous()
    if expert_cache:
        # At least 16 records are required by the cache. Duplicate the two
        # fixture experts, then store block-32 E2M1 weights and UE8M0 scales.
        values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
        for name, tensor in list(weights.items()):
            if '.ffn.gate.' in name:
                weights[name] = tensor.repeat(2, *([1] * (tensor.ndim - 1)))
            if '.ffn.experts.' not in name:
                continue
            expert = int(name.split('.experts.')[1].split('.')[0])
            blocks = tensor.float().reshape(tensor.shape[0], -1, 32)
            exponent = (blocks.abs().amax(-1).clamp_min(1e-12) / 6).log2().ceil()
            scaled = blocks / exponent.exp2().unsqueeze(-1)
            codes = (scaled.abs().unsqueeze(-1) - values).abs().argmin(-1).to(torch.uint8)
            codes |= (scaled < 0).to(torch.uint8) * 8
            codes = codes.reshape(tensor.shape)
            packed = codes[:, ::2] | (codes[:, 1::2] << 4)
            scale = (exponent + 127).to(torch.uint8)
            for target in (name, name.replace(f'.experts.{expert}.', f'.experts.{expert + 2}.')):
                weights[target] = packed.clone()
                weights[target.removesuffix('weight') + 'scale'] = scale.clone()
    save_file(weights, str(directory / 'model.safetensors'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--hc-mult', type=int, choices=(2, 4), default=4)
    parser.add_argument('--expert-cache', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='v41-tp-graph-') as path:
        directory = Path(path)
        prepare(directory, args.hc_mult, args.expert_cache)
        env = dict(os.environ, FASTLLM_SKIP_WARMUP='1', FASTLLM_DSV41_REFERENCE_MATH='0',
                   FASTLLM_DSV41_DISABLE_FAKE_QUANT='1', FASTLLM_DSV41_CUDA_GRAPH_DEBUG='1')
        command = [args.binary, str(directory)] + (['--expert-cache'] if args.expert_cache else [])
        if args.expert_cache:
            env.update(FT_NUMAS='1', FT_THREADS='2', FASTLLM_DSV41_MOE_CACHE_TRACE='1')
        result = subprocess.run(command, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        print(result.stdout)
        if result.returncode == 77:
            raise SystemExit(77)
        result.check_returncode()
        assert 'PASS: DSpark graph shapes 1..6, main features and rollback match across requests' in result.stdout
        assert 'decode CUDA graph captured:' in result.stdout, 'graph was not exercised'
        for tokens in range(1, 7):
            assert re.search(r'graph captured:.*tokens=%d\b' % tokens, result.stdout), 'missing graph shape %d' % tokens
            assert 'graph replay: tokens=%d' % tokens in result.stdout, 'shape %d never replayed' % tokens
        assert 'giving up' not in result.stdout and 'graph disabled' not in result.stdout
        if args.expert_cache:
            assert re.search(r'V4.1 verify cache:.* [1-9]\d* GPU routes', result.stdout), 'verify never used GPU experts'


if __name__ == '__main__':
    main()
