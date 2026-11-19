#!/usr/bin/env python3
"""SM70 专家 GEMM 独立验证/微基准；使用正式算子及真实专家快照，不编译任何产物。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import struct
import subprocess
import sys
import uuid

import numpy as np

from run_process_tp import ROOT, capture, save_report
from run_qwen_moe_full import request
from probe_qwen_moe_block_batches import load_blocks
from probe_qwen_moe_prefill_batches import file_hash, prefill
from fastllm_pytools.moe_batch_invariant import BACKEND


def metrics(actual, expected, *, bitwise=False, atol=0.05, rtol=0.01):
    """/** 独立数值检查或逐位不变性检查；拒绝非有限值，保留完整误差统计。 */"""
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid shape or nonfinite GEMM output')
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if bitwise:
        if a.dtype != b.dtype:
            raise ValueError('bitwise comparison requires identical dtype')
        dtype = np.dtype(f'u{a.dtype.itemsize}')
        failed = np.ascontiguousarray(a).view(dtype) != np.ascontiguousarray(b).view(dtype)
    else:
        failed = error > atol + rtol * np.abs(b.astype(np.float64))
    return dict(status='FAIL' if failed.any() else 'PASS', max_abs_error=float(error.max()) if error.size else 0.0,
                mean_abs_error=float(error.mean()) if error.size else 0.0,
                failed_elements=int(failed.sum()), compared_elements=int(error.size))


def load_weight(model, mapping, name):
    """/** 读取 safetensors 原始 FP16/BF16/FP32 权重并转换为正式 FP16，不需要 PyTorch。 */"""
    path = (model / mapping[name]).resolve()
    if not path.is_relative_to(model) or path.suffix != '.safetensors':
        raise ValueError('weight shard escapes model directory')
    with path.open('rb') as handle:
        header_size = struct.unpack('<Q', handle.read(8))[0]
        if header_size > 64 * 1024 * 1024:
            raise ValueError('invalid safetensors header size')
        info = json.loads(handle.read(header_size))[name]
        start, end = info['data_offsets']
        dtype = {'F16': '<f2', 'BF16': '<u2', 'F32': '<f4'}.get(info['dtype'])
        shape = info['shape']
        if dtype is None or len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError('unsupported expert weight dtype/shape')
        size = math.prod(shape) * np.dtype(dtype).itemsize
        if start < 0 or end - start != size or 8 + header_size + end > path.stat().st_size:
            raise ValueError('invalid safetensors offsets')
        handle.seek(8 + header_size + start)
        data = np.frombuffer(handle.read(size), dtype=dtype).reshape(shape)
    if info['dtype'] == 'BF16':
        data = (data.astype(np.uint32) << 16).view(np.float32)
    data = np.ascontiguousarray(data, dtype=np.float16)
    if not np.isfinite(data).all():
        raise ValueError('nonfinite expert weights')
    return data


class Suite:
    """/** 调用正式算子微基准，逐例保存原始输入、输出、命令、延迟及四种报告。 */"""

    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                             '_sm70_expert_gemm_' + uuid.uuid4().hex[:8])
        self.output.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.metadata = dict(scope='isolated_expert_gemm_not_full_inference', execution_complete=False,
            command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
            git_status=capture(['git', 'status', '--short']),
            gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
            cuda=capture(['nvcc', '--version']), build_cache=(args.build / 'CMakeCache.txt').read_text(),
            warmup=args.warmup, iterations=args.iterations, repeats=args.repeats, seed=20261119,
            real_replay_required=not args.synthetic_only, real_targets=0,
            report_title='SM70 专家 GEMM 独立验证与微基准',
            report_description='调用与正式推理相同的 GEMM；FP64 独立参考与跨批次逐位检查分别报告。'
                               '微基准不能替代完整生成或端到端性能验收。')

    def run(self, label, x, w, bias=None, *, fp32=False, graph=False, backend='sm70', device=0):
        """/** 使用相同 stream/同步与计时边界运行双方；实际返回 EXECUTED 后仍需数值验收。 */"""
        directory = self.output / label
        directory.mkdir(parents=True, exist_ok=False)
        x = np.ascontiguousarray(x, dtype=np.float16)
        w = np.ascontiguousarray(w, dtype=np.float16)
        x.tofile(directory / 'input.bin')
        w.tofile(directory / 'weight.bin')
        command = [str(self.args.build / 'fastllm_sm70_expert_gemm_check'), '--m', str(len(x)), '--n', str(len(w)),
                   '--k', str(w.shape[1]), '--input', str(directory / 'input.bin'), '--weight', str(directory / 'weight.bin'),
                   '--output', str(directory / 'output.bin'), '--backend', backend, '--device', str(device),
                   '--fp32', str(int(fp32)), '--graph', str(int(graph)), '--warmup', str(self.args.warmup),
                   '--iterations', str(self.args.iterations), '--repeats', str(self.args.repeats)]
        if bias is not None:
            np.asarray(bias, dtype=np.float16).tofile(directory / 'bias.bin')
            command += ['--bias', str(directory / 'bias.bin')]
        (directory / 'command.txt').write_text(shlex.join(command) + '\n')
        result = subprocess.run(command, text=True, capture_output=True, timeout=self.args.timeout)
        (directory / 'stdout.log').write_text(result.stdout)
        (directory / 'stderr.log').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f'{label}: native exit={result.returncode}: {result.stderr.strip()}')
        report = json.loads(result.stdout)
        if report.get('status') != 'EXECUTED' or report.get('backend') != (BACKEND if backend == 'sm70' else 'cublasGemmEx'):
            raise AssertionError('native benchmark did not execute requested backend')
        samples = report['samples_ms']
        if len(samples) != self.args.repeats or not all(math.isfinite(v) and (v > 0 or (not len(x) and v == 0)) for v in samples):
            raise AssertionError('invalid timing samples')
        median = statistics.median(samples)
        report.update(median_ms=median, p95_ms=sorted(samples)[math.ceil(.95 * len(samples)) - 1],
                      cv=statistics.pstdev(samples) / statistics.mean(samples) if statistics.mean(samples) else 0.0,
                      effective_tops=2 * len(x) * len(w) * w.shape[1] / (median * 1e9) if median else 0.0,
                      input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                      weight_sha256=hashlib.sha256(w.tobytes()).hexdigest(), command=command)
        (directory / 'result.json').write_text(json.dumps(report, indent=2))
        if report.get('repeat_outputs_equal') is not True:
            raise AssertionError('identical GEMM inputs changed across timing repeats')
        y = np.fromfile(directory / 'output.bin', dtype=np.float32 if fp32 else np.float16).reshape(len(x), len(w))
        return y, report

    def record(self, name, comparison, native=None, **fields):
        """/** 每个数值断言均写入部分报告；任何 FAIL 保留到最终返回码。 */"""
        row = dict(case=name, backend=BACKEND, reason='', **comparison)
        row.update(fields)
        if native:
            row.update({k: native[k] for k in ('median_ms', 'p95_ms', 'cv', 'effective_tops')})
        self.rows.append(row)
        save_report(self.output, self.rows, self.metadata)
        print(f'{row["status"]}: {name}', flush=True)

    def synthetic(self):
        """/** 覆盖块边界、行位置、排列、N=1、K尾部、bias、输出精度、空专家及非法输入。 */"""
        rng = np.random.default_rng(20261119)
        for n, k in ((1, 8), (7, 40), (127, 32), (128, 64), (129, 512)):
            x = rng.normal(0, .1, (129, k)).astype(np.float16)
            w = rng.normal(0, .1, (n, k)).astype(np.float16)
            for fp32, with_bias in ((False, False), (False, True), (True, False), (True, True)):
                bias = rng.normal(0, .1, n).astype(np.float16) if with_bias else None
                key = f'n{n}_k{k}_fp32{int(fp32)}_bias{int(with_bias)}'
                full, native = self.run(key + '_full', x, w, bias, fp32=fp32)
                reference = x.astype(np.float64) @ w.astype(np.float64).T
                if bias is not None:
                    reference += bias.astype(np.float64)
                self.record(key + '_fp64', metrics(full, reference, atol=.001, rtol=.001), native)
                for count in (1, 4, 31, 32, 33, 63, 64, 65, 128):
                    # 同一行跨 tile、warp 和尾部位置，包含反向排列。
                    indices = np.arange(129)[::-1][:count].copy()
                    y, evidence = self.run(key + f'_m{count}', x[indices], w, bias, fp32=fp32, graph=True)
                    self.record(key + f'_m{count}_bitwise', metrics(y, full[indices], bitwise=True), evidence)
        empty, native = self.run('empty_expert', np.empty((0, 8), np.float16), np.ones((1, 8), np.float16))
        self.record('empty_expert', metrics(empty, empty, bitwise=True), native)
        try:
            self.run('reject_k7', np.ones((1, 7), np.float16), np.ones((1, 7), np.float16))
        except RuntimeError as error:
            if 'invalid shape/stream' not in str(error):
                raise
            self.record('reject_k7', dict(status='PASS', max_abs_error=None, mean_abs_error=None, failed_elements=0))
        else:
            raise AssertionError('unaligned K was not rejected')

    def real(self):
        """/** 重放已定位专家的完全相同输入；快照缺失时通过正式旧模式诊断重采。 */"""
        args = self.args
        model = args.model.resolve()
        config = json.loads((model / 'config.json').read_text())
        mapping = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
        if args.baseline:
            baseline = args.baseline.resolve()
            saved = json.loads((baseline / 'summary.json').read_text())
            meta = saved['metadata']
            if not saved.get('complete') or meta.get('scope') != 'formal_moe_block_batch_trace_diagnostic':
                raise ValueError('baseline must be a complete formal MoE block trace')
            for filename, key in (('config.json', 'model_config_sha256'), ('model.safetensors.index.json', 'model_index_sha256')):
                if file_hash(model / filename) != meta.get(key):
                    raise ValueError('baseline model provenance mismatch')
            job = meta['request']
        else:
            baseline = self.output / 'collected_baseline'
            job = request([1, 2, 3, 4] * 256, 128)
            for wave in (2, 4):
                prefill(args, baseline / f'first{wave}', job, list(range(wave)), config['vocab_size'], block_trace=True)
        blocks = {wave: load_blocks(baseline / f'first{wave}/trace', wave, config, job) for wave in (2, 4)}
        self.metadata.update(real_baseline=str(baseline), real_request=job,
                             trace_manifests={str(w): {str(r): m for r, (m, _) in ranks.items()} for w, ranks in blocks.items()})
        for expert in range(config['num_experts']):
            owner = expert % 4
            a, b = blocks[2][owner][1], blocks[4][owner][1]
            prefix = f'expert_{expert}_'
            if prefix + 'indices' not in a or prefix + 'indices' not in b:
                continue
            positions = np.searchsorted(b[prefix + 'indices'], a[prefix + 'indices'])
            if (positions >= len(b[prefix + 'indices'])).any() or not np.array_equal(b[prefix + 'indices'][positions], a[prefix + 'indices']):
                raise ValueError('expert route slots cannot be paired')
            for phase, source, parts in (('gateup', 'packed', ('gate_proj', 'up_proj')), ('down', 'activated', ('down_proj',))):
                x2, x4 = a[prefix + source], b[prefix + source]
                if not np.array_equal(x2, x4[positions]):
                    continue  # 上游输入已不同，不能归因于本次 GEMM。
                old = metrics(a[prefix + phase], b[prefix + phase][positions], bitwise=True)
                if old['status'] != 'FAIL':
                    continue
                name = prefix + phase
                weights = [load_weight(model, mapping, f'model.layers.0.mlp.experts.{expert}.{part}.weight') for part in parts]
                w = np.concatenate(weights, axis=0)
                if w.shape != (a[prefix + phase].shape[1], x2.shape[1]):
                    raise ValueError('expert weight shape mismatch')
                y2, native = self.run(name + '_two', x2, w, device=owner)
                y4, _ = self.run(name + '_four', x4, w, device=owner)
                self.record(name + '_common_rows', metrics(y2, y4[positions], bitwise=True), native, baseline_difference=old)
                # 高精度参考独立于两后端；真实大矩阵固定抽查 8 行全部输出列，范围明确记录。
                indices = np.unique(np.linspace(0, len(x2) - 1, min(8, len(x2)), dtype=int))
                reference = x2[indices].astype(np.float64) @ w.astype(np.float64).T
                self.record(name + '_fp64', metrics(y2[indices], reference), reference_row_indices=indices.tolist())
                original, baseline_perf = self.run(name + '_cublas', x2, w, backend='cublas', device=owner)
                check = metrics(original[indices], reference)
                candidate_check = metrics(y2[indices], reference)
                self.record(name + '_cublas_fp64', check)
                self.record(name + '_microbenchmark', candidate_check, native,
                    speedup=baseline_perf['median_ms'] / native['median_ms'] if
                    check['status'] == candidate_check['status'] == 'PASS' else None)
                self.metadata['real_targets'] += 1
        if not self.metadata['real_targets']:
            raise AssertionError('no saved expert difference with identical inputs; original issue has not been revalidated')


def main():
    """/** 单独运行算子检查；失败保持非零及原始证据，不自动编译或下载。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--synthetic-only', action='store_true', help='仅独立边界验证，不代表真实问题已修复')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    if args.repeats < 5 or args.warmup < 1 or args.iterations < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('require warmup >= 1, iterations >= 1, repeats >= 5 and positive timeout')
    if not args.synthetic_only and not args.model:
        parser.error('--model is required for real replay')
    args.build = args.build.resolve()
    suite = Suite(args)
    print(f'专家 GEMM 验证：{suite.output}', flush=True)
    try:
        suite.synthetic()
        if not args.synthetic_only:
            suite.real()
        suite.metadata['execution_complete'] = True
    except Exception as error:
        suite.record('execution_failure', dict(status='FAIL', max_abs_error=None, mean_abs_error=None,
                                              failed_elements=1), reason=f'{type(error).__name__}: {error}')
        print(f'{type(error).__name__}: {error}', flush=True)
    finally:
        status = save_report(suite.output, suite.rows, suite.metadata)
        print(f'{status}: {suite.output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
