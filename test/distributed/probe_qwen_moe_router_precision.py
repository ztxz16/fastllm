#!/usr/bin/env python3
"""独立数值诊断：分离RMSNorm与router GEMM差异；不编译，不作为正式推理验收。"""

import argparse
import ctypes
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import traceback
import uuid

from run_process_tp import ROOT, capture, parse_report, save_report


def norm_variants(torch, x, weight, epsilon):
    """/**
     * 比较HF公开运算与原生256线程归约顺序的张量模拟，不调用或替代正式Norm kernel。
     * 两条路径均保留归一化后转FP16、再乘scale转FP16的边界。模拟只覆盖当前
     * hidden为256倍数的模型；不能用此结果宣称已验证原生kernel中间量。
     * @param x 同一设备上FP16 [tokens,hidden]，非空且有限。
     * @param weight 同设备FP16 [hidden] scale。
     * @param epsilon RMSNorm正数epsilon。
     * @return HF顺序和原生归约顺序模拟的两个FP16输出。
     */"""
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or x.shape[1] % 256:
        raise ValueError('probe requires nonempty [tokens,hidden], hidden divisible by 256')
    if x.dtype != torch.float16 or weight.dtype != x.dtype or weight.shape != (x.shape[1],):
        raise ValueError('probe requires FP16 activation and [hidden] scale')
    if x.device != weight.device or not math.isfinite(epsilon) or epsilon <= 0 or not torch.isfinite(x).all() or not torch.isfinite(weight).all():
        raise ValueError('invalid norm device/epsilon/nonfinite input')

    values = x.float()
    hf_scale = torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
    hf = (values * hf_scale).half() * weight

    # 每线程连续处理相隔256列的元素，再执行128/64/.../1的共享内存树归约。
    sums = torch.zeros((x.shape[0], 256), dtype=torch.float32, device=x.device)
    for begin in range(0, x.shape[1], 256):
        sums = sums + values[:, begin:begin+256].square()
    for step in (128, 64, 32, 16, 8, 4, 2, 1):
        sums = sums[:, :step] + sums[:, step:2*step]
    native_scale = torch.rsqrt(sums / x.shape[1] + epsilon)
    native_order = (values * native_scale).half() * weight
    return hf, native_order


def linked_cublas(worker):
    """/** 从已有worker的动态链接清单查找cuBLAS，拒绝悄悄换用其他版本。 */"""
    text = subprocess.run(['ldd', str(worker.resolve())], check=True, text=True,
                          capture_output=True, timeout=15).stdout
    match = re.search(r'^\s*libcublas\.so(?:\.\d+)*\s+=>\s+(\S+)', text, re.M)
    if not match or not Path(match[1]).is_file():
        raise RuntimeError('cannot resolve the existing worker cuBLAS; inspect ldd output: ' + text)
    return Path(match[1]).resolve()


class PedanticGemm:
    """/** 独立handle重放修复前的PEDANTIC配置，供历史日志对照，不代表当前MoE策略。 */"""

    def __init__(self, library, stream):
        """/** 绑定已有CUDA库、独立handle及调用方当前stream；配置失败立即释放handle。 */"""
        self.lib = ctypes.CDLL(str(library))
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        signatures = {
            'cublasCreate_v2': [ctypes.POINTER(pointer)],
            'cublasDestroy_v2': [pointer],
            'cublasSetStream_v2': [pointer, pointer],
            'cublasSetMathMode': [pointer, integer],
            'cublasGetVersion_v2': [pointer, ctypes.POINTER(integer)],
            'cublasGemmEx': [pointer, integer, integer, integer, integer, integer,
                            pointer, pointer, integer, integer, pointer, integer, integer,
                            pointer, pointer, integer, integer, integer, integer],
        }
        for name, signature in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = signature, integer
        self.handle = pointer()
        self.check(self.lib.cublasCreate_v2(ctypes.byref(self.handle)))
        try:
            self.check(self.lib.cublasSetStream_v2(self.handle, pointer(stream)))
            self.check(self.lib.cublasSetMathMode(self.handle, 0))
            version = integer()
            self.check(self.lib.cublasGetVersion_v2(self.handle, ctypes.byref(version)))
            self.version = version.value
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(status):
        """/** cuBLAS错误立即失败，不使用其他算法或CPU fallback。 */"""
        if status:
            raise RuntimeError(f'cuBLAS probe failed: status={status}')

    def close(self):
        """/** 释放诊断私有handle，可重复调用；不改变PyTorch自己的handle。 */"""
        if self.handle:
            self.check(self.lib.cublasDestroy_v2(self.handle))
            self.handle = ctypes.c_void_p()

    def linear(self, torch, x, weight):
        """/**
         * 执行FP16 Y[T,N]=X[T,K]*W[N,K]^T，保留修复前Linear的FP16输出分支参数。
         * 使用已分配的连续CUDA张量和同一stream，FP32_PEDANTIC累加，禁止编译。
         * @param torch 已安装的PyTorch，仅提供张量、设备内存与stream。
         * @param x 连续FP16 CUDA [T,K]。
         * @param weight 同设备连续FP16 CUDA [N,K]。
         * @return 同设备FP16 [T,N]；异步结果由调用方同步后读取。
         */"""
        if (not self.handle or x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
                or not x.is_cuda or x.device != weight.device or x.dtype != torch.float16
                or weight.dtype != x.dtype or not x.is_contiguous() or not weight.is_contiguous()
                or min(*x.shape, *weight.shape) <= 0):
            raise ValueError('invalid FP16 CUDA GEMM inputs/handle')
        t, k = x.shape
        n = weight.shape[0]
        output = torch.empty((t, n), dtype=x.dtype, device=x.device)
        alpha, beta = ctypes.c_float(1), ctypes.c_float(0)
        pointer = ctypes.c_void_p
        self.check(self.lib.cublasGemmEx(self.handle, 1, 0, n, t, k, ctypes.byref(alpha),
            pointer(weight.data_ptr()), 2, k, pointer(x.data_ptr()), 2, k, ctypes.byref(beta),
            pointer(output.data_ptr()), 2, n, 69, -1))
        return output


def exact_metrics(torch, actual, expected):
    """/** 统计恢复FP16后的逐元素精确差异；零容差仅用于定位，不能替代原验收阈值。 */"""
    if actual.shape != expected.shape or not actual.numel():
        raise ValueError('invalid probe comparison shape')
    a, b = actual.float(), expected.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    errors = torch.where(finite, (a-b).abs(), float('inf'))
    bad = int(((a != b) | ~finite).sum().item())
    worst = int(errors.reshape(-1).argmax().item())
    observed, target = a.reshape(-1)[worst].item(), b.reshape(-1)[worst].item()
    return dict(status='FAIL' if bad else 'PASS', max_abs_error=errors.max().item(),
                mean_abs_error=errors.mean().item(), failed_elements=bad,
                shape=list(a.shape), worst_index=worst, actual=observed, expected=target,
                reason=f'shape={list(a.shape)} worst_index={worst} actual={observed:.3f} '
                       f'expected={target:.3f}; exact FP16 comparison' if bad else '')


def main():
    """/** 从现有诊断读取同一输入及已保存router输出，对照四种组合并保存四格式报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_router_probe_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    print(f'独立诊断目录：{output}', flush=True)
    rows, details, gemm = [], [], None
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), warmup=0, iterations=1, repeats=1,
        expected_case_count=54,
        report_title='Router归一化/GEMM独立数值诊断',
        report_description='独立张量模拟与cuBLAS对照，不是正式后端或性能验收。FAIL表示精确值不同，不改变完整模型验收阈值。',
        diagnostic=str(args.diagnostic.resolve()), model=str(args.model.resolve()))
    try:
        import torch
        from safetensors import safe_open
        if not torch.cuda.is_available():
            raise RuntimeError('run this probe in the existing CUDA reference environment on the server')
        torch.cuda.set_device(args.device)
        device = torch.device('cuda', args.device)
        metadata.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                        device=args.device, compute_capability=torch.cuda.get_device_capability(device),
                        allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        library = linked_cublas(args.worker)
        gemm = PedanticGemm(library, torch.cuda.current_stream().cuda_stream)
        metadata.update(cublas_path=str(library), cublas_version=gemm.version)
        cache = args.worker.parent / 'CMakeCache.txt'
        metadata['build_cache'] = cache.read_text() if cache.is_file() else 'unavailable'

        original = json.loads((args.diagnostic / 'summary.json').read_text())
        rank = original['metadata']['rank']
        metadata['source_rank'] = rank
        metadata['source_reference'] = original['metadata'].get('reference')
        metadata['source_git'] = original['metadata']['git']
        references = {row['name']: row for row in json.loads((args.diagnostic / 'reference.json').read_text())['layers']}
        targets = {1: 0, 9: 11, 14: 2, 17: 0, 22: 19}
        needed = {f'real_layer_{layer}_moe' for layer in targets} | {'real_layer_22_full', 'real_layer_22_attention'}
        native = {}
        for path in sorted(args.diagnostic.glob('native-*/stdout.log')):
            for row in parse_report(path.read_text()).get('results', []):
                if row['task'] in needed:
                    native[row['task']] = row['ranks'][rank]
        if set(native) != needed:
            raise ValueError(f'missing saved native cases: {needed-set(native)}')
        config = json.loads((args.model / 'config.json').read_text())
        mapping = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']

        def load_weight(name):
            """/** 只读取索引中当前所需的norm/gate张量，不加载全模型或修改checkpoint。 */"""
            path = (args.model / mapping[name]).resolve()
            if not path.is_relative_to(args.model.resolve()):
                raise ValueError('weight shard escapes model directory')
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                return handle.get_tensor(name).to(device=device, dtype=torch.float16).contiguous()

        def compare(name, a, b):
            """/** 在同一stream同步读回后记录完整精度，表格由公共报告器保留三位小数。 */"""
            metrics = exact_metrics(torch, a, b)
            rows.append(dict(case=name, backend='independent_router_probe', **metrics))
            return metrics

        with torch.inference_mode():
            for layer, token in targets.items():
                task = references[f'real_layer_{layer}_moe']
                prefix = f'model.layers.{layer}.'
                scale = load_weight(prefix + 'post_attention_layernorm.weight')
                weight = load_weight(prefix + 'mlp.gate.weight')
                x = torch.tensor(task['inputs'][rank]['hidden'], dtype=torch.float16, device=device).reshape(-1, task['hidden'])
                workloads = [(f'layer{layer}_same_hf_input', x, f'real_layer_{layer}_moe')]
                if layer == 22:
                    attention = torch.tensor(native['real_layer_22_attention']['output'], dtype=x.dtype, device=device).reshape_as(x)
                    workloads.append(('layer22_native_attention_input', attention, 'real_layer_22_full'))
                for label, hidden, saved_case in workloads:
                    hf, simulated = norm_variants(torch, hidden, scale, config['rms_norm_eps'])
                    norm_error = compare(label+'/norm_order', simulated, hf)
                    print(f'{label}: norm_different={norm_error["failed_elements"]}, max_abs={norm_error["max_abs_error"]:.9g}', flush=True)
                    expected = torch.tensor(references[saved_case]['reference'][rank]['router_logits'], dtype=x.dtype, device=device).reshape(x.shape[0], -1)
                    observed = torch.tensor(native[saved_case]['router_logits'], dtype=x.dtype, device=device).reshape_as(expected)
                    variants = dict(hf_norm_torch_gemm=torch.nn.functional.linear(hf, weight),
                                    hf_norm_pedantic=gemm.linear(torch, hf, weight),
                                    simulated_norm_torch_gemm=torch.nn.functional.linear(simulated, weight),
                                    simulated_norm_pedantic=gemm.linear(torch, simulated, weight))
                    evidence = dict(case=label, token=token, shape=list(hidden.shape),
                        hf_order_norm=hf.cpu().tolist(), simulated_native_order_norm=simulated.cpu().tolist(),
                        saved_hf_logits=expected.cpu().tolist(), saved_native_logits=observed.cpu().tolist(), variants={})
                    for name, values in variants.items():
                        against_hf = compare(label+'/'+name+'/vs_saved_hf', values, expected)
                        against_native = compare(label+'/'+name+'/vs_saved_native', values, observed)
                        evidence['variants'][name] = values.cpu().tolist()
                        print(f'  {name}: HF_diff={against_hf["failed_elements"]}, '
                              f'native_diff={against_native["failed_elements"]}, '
                              f'HF_max={against_hf["max_abs_error"]:.9g}, native_max={against_native["max_abs_error"]:.9g}', flush=True)
                    details.append(evidence)
                    metadata['loaded_cublas_libraries'] = sorted({line.split()[-1]
                        for line in Path('/proc/self/maps').read_text().splitlines()
                        if '/libcublas' in line})
                    (output / 'raw.json').write_text(json.dumps(details, allow_nan=False))
                    save_report(output, rows, metadata)
    except Exception as error:
        (output / 'error.log').write_text(traceback.format_exc())
        rows.append(dict(case='probe_setup_or_execution', backend='independent_router_probe', status='FAIL',
                         failed_elements=1, reason=str(error)))
        print(f'FAIL: {error}', flush=True)
    finally:
        if gemm is not None:
            try:
                gemm.close()
            except Exception as error:
                rows.append(dict(case='probe_cleanup', backend='independent_router_probe', status='FAIL',
                                 failed_elements=1, reason=str(error)))
        status = save_report(output, rows, metadata)
        print(f'{status}: {output}（精确差异诊断，非完整模型验收）', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
