#!/usr/bin/env python3
"""Expand the CPU fixture to CUDA TP dimensions and compare eager/graph execution.

Requires numpy, torch, safetensors and two CUDA GPUs. The fixed input tokens
exercise Engram, shared KV, both indexer stages and changing request lengths.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from safetensors.torch import load_file, save_file
from test_deepseek_v41_cpu_fixture import unpack


def prepare(directory, hc_mult=4):
    unpack(str(Path(__file__).with_name('deepseek_v41_fixture.npz')), str(directory))
    config = json.loads((directory / 'config.json').read_text())
    config['text_config'].update(num_attention_heads=64, head_dim=512,
                                 index_head_dim=128, moe_intermediate_size=256,
                                 vocab_size=256, hc_mult=hc_mult)
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
    save_file(weights, str(directory / 'model.safetensors'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--hc-mult', type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='v41-tp-graph-') as path:
        directory = Path(path)
        prepare(directory, args.hc_mult)
        env = dict(os.environ, FASTLLM_SKIP_WARMUP='1', FASTLLM_DSV41_REFERENCE_MATH='0',
                   FASTLLM_DSV41_DISABLE_FAKE_QUANT='1', FASTLLM_DSV41_CUDA_GRAPH_DEBUG='1')
        result = subprocess.run([args.binary, str(directory)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        print(result.stdout)
        if result.returncode == 77:
            raise SystemExit(77)
        result.check_returncode()
        assert 'decode CUDA graph captured:' in result.stdout, 'graph was not exercised'
        assert 'giving up' not in result.stdout and 'graph disabled' not in result.stdout


if __name__ == '__main__':
    main()
