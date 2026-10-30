"""Compare Wan VAE primitives; audit FP32 convolutions against FP64 as well.

cuDNN may choose Winograd, whose FP32 error can exceed the primitive gate.
Convolution accuracy therefore uses an FP64 oracle with the same inputs; both
native/cuDNN agreement and cuDNN's own FP64 error remain in the report. Full
pipeline comparisons continue to use the official FP32 VAE implementation.
"""
import argparse
import ctypes
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compare_vae import metrics, thwc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bridge', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1407)
    lib = ctypes.CDLL(str(a.bridge))
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.lingbot_test_error.restype = ctypes.c_char_p
    lib.lingbot_test_vae_norm.argtypes = [ptr, ptr, ptr, ctypes.c_int, ctypes.c_int]
    lib.lingbot_test_vae_conv.argtypes = [ptr]*4 + [ctypes.c_int]*9
    lib.lingbot_test_vae_attention.argtypes = [ptr, ptr, ctypes.c_int]
    lib.lingbot_test_vae_conv2d.argtypes = [ptr]*4 + [ctypes.c_int]*8
    def pointer(x):
        return None if x is None else x.ctypes.data_as(ptr)
    def array(x):
        return x.float().cpu().contiguous().numpy()
    results = []
    def record(name, actual, expected, fp64=None):
        oracle = expected if fp64 is None else fp64
        item = dict(name=name, reference='official_fp32' if fp64 is None else 'fp64',
                    **metrics(actual, oracle))
        item['different_values'] = int(np.count_nonzero(actual != expected))
        if fp64 is not None:
            item['official_fp32_comparison'] = metrics(actual, expected)
            item['official_fp32_accuracy'] = metrics(expected, fp64)
        item['passed'] = item['finite'] and item['relative_l2'] < 2e-6
        results.append(item)
        print(name, item['relative_l2'], item['different_values'], flush=True)
    for c in [96, 192, 384]:
        for h, w in [(8, 8), (12, 20), (48, 80)]:
            x = torch.randn(1, c, 1, h, w, device='cuda')
            gamma = torch.randn(c, device='cuda')
            expected = thwc(F.normalize(x, dim=1)*math.sqrt(c)*gamma[None, :, None, None, None])
            values, weight = thwc(x), array(gamma)
            actual = np.empty_like(values)
            status = lib.lingbot_test_vae_norm(pointer(values), pointer(weight), pointer(actual), h*w, c)
            if status:
                raise RuntimeError(lib.lingbot_test_error().decode())
            record(f'norm_{c}_{h}_{w}', actual, expected)
    for t, h, w, ic, oc, kt, kernel, stride, pad in [
        (1, 16, 24, 3, 96, 3, 3, 1, 2),
        (6, 8, 12, 96, 96, 3, 3, 1, 0),
        (3, 8, 12, 192, 192, 3, 1, 2, 0),
        (3, 12, 20, 32, 32, 1, 1, 1, 0),
    ]:
        x = torch.randn(1, ic, t, h, w, device='cuda')
        weight = torch.randn(oc, ic, kt, kernel, kernel, device='cuda')/math.sqrt(ic*kt*kernel*kernel)
        bias = torch.randn(oc, device='cuda')
        expected = thwc(F.conv3d(F.pad(x, (kernel//2,)*4+(pad, 0)), weight, bias, stride=(stride, 1, 1)))
        values, weights, biases = thwc(x), array(weight), array(bias)
        actual = np.empty_like(expected)
        status = lib.lingbot_test_vae_conv(pointer(values), pointer(weights), pointer(biases), pointer(actual),
            t, h, w, ic, oc, kt, kernel, stride, pad)
        if status:
            raise RuntimeError(lib.lingbot_test_error().decode())
        gold = F.conv3d(F.pad(x.double(), (kernel//2,)*4+(pad, 0)),
                        weight.double(), bias.double(), stride=(stride, 1, 1))
        gold = gold[0].permute(1, 2, 3, 0).cpu().numpy()
        record(f'conv_{t}_{h}_{w}_{ic}_{oc}_{kt}_{kernel}', actual, expected, gold)
    for h, w, ic, oc, kernel, stride, pad, pad_end in [
        (48, 80, 96, 96, 3, 2, 0, 1),
        (24, 40, 192, 192, 3, 2, 0, 1),
        (24, 40, 384, 192, 3, 1, 1, 0),
        (12, 20, 384, 1152, 1, 1, 0, 0),
        (12, 20, 384, 384, 1, 1, 0, 0),
    ]:
        x = torch.randn(1, ic, h, w, device='cuda')
        weight = torch.randn(oc, ic, kernel, kernel, device='cuda')/math.sqrt(ic*kernel*kernel)
        bias = torch.randn(oc, device='cuda')
        expected = array(F.conv2d(F.pad(x, (pad, pad+pad_end, pad, pad+pad_end)),
                                  weight, bias, stride=stride).permute(0, 2, 3, 1))
        values, weights, biases = array(x.permute(0, 2, 3, 1)), array(weight), array(bias)
        actual = np.empty_like(expected)
        status = lib.lingbot_test_vae_conv2d(pointer(values), pointer(weights), pointer(biases), pointer(actual),
            h, w, ic, oc, kernel, stride, pad, pad_end)
        if status:
            raise RuntimeError(lib.lingbot_test_error().decode())
        gold = F.conv2d(F.pad(x.double(), (pad, pad+pad_end, pad, pad+pad_end)),
                        weight.double(), bias.double(), stride=stride)
        gold = gold.permute(0, 2, 3, 1).cpu().numpy()
        record(f'conv2d_{h}_{w}_{ic}_{oc}_{kernel}_{stride}', actual, expected, gold)
    for rows in [64, 127, 128, 129, 240, 512, 1080]:
        qkv = torch.randn(1, 1, rows, 1152, device='cuda')
        q, k, v = qkv.chunk(3, dim=-1)
        expected = array(F.scaled_dot_product_attention(q, k, v))[0, 0]
        values = array(qkv)[0, 0]
        actual = np.empty_like(expected)
        if lib.lingbot_test_vae_attention(pointer(values), pointer(actual), rows):
            raise RuntimeError(lib.lingbot_test_error().decode())
        record(f'attention_{rows}', actual, expected)
    passed = all(item['passed'] for item in results)
    a.output.write_text(json.dumps(dict(results=results, passed=passed), indent=2))
    if not passed:
        raise SystemExit('Wan VAE primitive L2 gate failed')


if __name__ == '__main__':
    main()
