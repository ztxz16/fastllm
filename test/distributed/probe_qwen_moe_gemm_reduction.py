#!/usr/bin/env python3
"""独立重放MoE专家GEMM，比较默认math mode与禁止低精度归约；不修改正式后端。"""

from contextlib import contextmanager
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import traceback
import uuid

import numpy as np

from probe_qwen_moe_attention_precision import MoeGemm
from probe_qwen_moe_block_batches import load_blocks
from probe_qwen_moe_prefill_batches import difference, file_hash
from probe_qwen_moe_router_precision import linked_cublas
from run_process_tp import ROOT, capture, save_report


def reduction_flag(cuda_home):
    """/** 从已安装cuBLAS头文件读取归约控制位并记录来源；不猜测枚举值或下载头文件。 */"""
    for suffix in ('include/cublas_api.h', 'targets/x86_64-linux/include/cublas_api.h'):
        path = (cuda_home/suffix).resolve()
        if path.is_file():
            match = re.search(r'\bCUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION\s*=\s*(0x[\da-fA-F]+|\d+)\b', path.read_text())
            if match:
                value = int(match[1], 0)
                if value <= 0 or value & (value-1):
                    raise ValueError('cuBLAS reduction flag is not a single positive bit')
                return value, dict(path=str(path), sha256=file_hash(path), value=value)
    raise ValueError('cannot find reduction flag in installed cublas_api.h; check --cuda-home')


def symbol_library(library):
    """/** 用dladdr确认实际GEMM符号所属库，避免PyTorch已加载的同名库被误当作worker版本。 */"""
    class Info(ctypes.Structure):
        _fields_ = [('name', ctypes.c_char_p), ('base', ctypes.c_void_p),
                    ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]
    query = ctypes.CDLL(None).dladdr
    query.argtypes, query.restype = [ctypes.c_void_p, ctypes.POINTER(Info)], ctypes.c_int
    info = Info()
    if not query(ctypes.cast(library.cublasGemmEx, ctypes.c_void_p), ctypes.byref(info)) or not info.name:
        raise ValueError('dladdr could not identify cublasGemmEx')
    return Path(os.fsdecode(info.name)).resolve()


def set_math_mode(gemm, mode):
    """/** 设置并读回诊断私有handle的math mode；cuBLAS拒绝或未生效时直接失败。 */"""
    getter = gemm.lib.cublasGetMathMode
    getter.argtypes, getter.restype = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_int
    gemm.check(gemm.lib.cublasSetMathMode(gemm.handle, mode))
    actual = ctypes.c_int(-1)
    gemm.check(getter(gemm.handle, ctypes.byref(actual)))
    if actual.value != mode:
        raise ValueError(f'cuBLAS math mode mismatch: requested={mode} actual={actual.value}')


def tensor_sha(array):
    """/** 对连续原始张量字节计算摘要，形状和dtype另外写入报告。 */"""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def targets(blocks, first_differences):
    """/**
     * 仅选择每个专家首次不同的GEMM，并确认两边共同输入行完全相同。
     * @param blocks 两请求/四请求的正式内部快照，已通过布局及来源校验。
     * @param first_differences 旧报告的逐专家首差映射；仅支持gateup和down。
     * @return 目标列表，保留完整矩阵及共同槽位在四请求矩阵中的行索引。
     */"""
    if not first_differences:
        raise ValueError('baseline has no differing expert GEMM')
    result = []
    for key, phase in sorted(first_differences.items(), key=lambda item: int(item[0])):
        expert = int(key)
        if str(expert) != key or not 0 <= expert < blocks[2][0][0]['experts'] or phase not in ('gateup', 'down'):
            raise ValueError('unsupported expert first-difference stage')
        owner, prefix = expert % 4, f'expert_{expert}_'
        a, b = blocks[2][owner][1], blocks[4][owner][1]
        slots, full_slots = a[prefix+'indices'], b[prefix+'indices']
        positions = np.searchsorted(full_slots, slots)
        if (positions >= len(full_slots)).any() or not np.array_equal(full_slots[positions], slots):
            raise ValueError('cannot pair GEMM rows by original routing slots')
        input_name = prefix + ('packed' if phase == 'gateup' else 'activated')
        x = {2: a[input_name], 4: b[input_name]}
        y = {2: a[prefix+phase], 4: b[prefix+phase]}
        if difference(x[2].reshape(-1), x[4][positions].reshape(-1))['status'] != 'PASS':
            raise ValueError(f'expert {expert} {phase}: inputs already differ; GEMM attribution refused')
        metrics = difference(y[2].reshape(-1), y[4][positions].reshape(-1))
        if metrics['status'] != 'FAIL':
            raise ValueError(f'expert {expert} {phase}: saved GEMM difference not reproduced')
        result.append(dict(name=prefix+phase, expert=expert, owner=owner, phase=phase,
                           x=x, y=y, positions=positions, saved_difference=metrics))
    return result


def load_weight(torch, safe_open, model, config, mapping, target, provenance):
    """/**
     * 读取当前专家必要的原始张量，按正式loader的Gate在前、Up在后布局拼接。
     * @param torch 已安装的PyTorch接口，仅使用现有算子。
     * @param safe_open 已安装的safetensors读取接口。
     * @param model 已通过来源校验的模型绝对路径。
     * @param config 原模型配置，提供隐藏维度和专家中间维度。
     * @param mapping 原模型分片索引，映射完整权重名称到分片文件。
     * @param target 当前专家ID及投影阶段，不加载完整模型。
     * @param provenance 记录原dtype/形状、分片与转换后的FP16摘要；两模式间禁止变化。
     * @return CPU连续FP16 [N,K]权重；非法形状、dtype、非有限值直接失败。
     */"""
    h, m = config['hidden_size'], config['moe_intermediate_size']
    parts = ('gate_proj', 'up_proj') if target['phase'] == 'gateup' else ('down_proj',)
    shape = (m, h) if target['phase'] == 'gateup' else (h, m)
    weights = []
    for part in parts:
        name = f'model.layers.0.mlp.experts.{target["expert"]}.{part}.weight'
        path = (model/mapping[name]).resolve()
        if not path.is_relative_to(model) or path.suffix != '.safetensors':
            raise ValueError('weight shard escapes model directory')
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            value = handle.get_tensor(name)
        if tuple(value.shape) != shape or value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f'unsupported checkpoint tensor: {name}')
        dtype = str(value.dtype)
        value = value.to(dtype=torch.float16).contiguous()
        raw = value.numpy()
        difference(raw.reshape(-1), raw.reshape(-1))
        entry = dict(shard=str(path), shard_bytes=path.stat().st_size, source_dtype=dtype,
                     shape=list(shape), fp16_sha256=tensor_sha(raw))
        if name in provenance and provenance[name] != entry:
            raise ValueError('checkpoint tensor changed between modes')
        provenance[name] = entry
        weights.append(value)
    return torch.cat(weights, dim=0).contiguous() if len(weights) > 1 else weights[0]


class GpuReplay:
    """/** 用原worker的cuBLAS私有handle做独立实验；PyTorch仅负责现成的内存与拷贝操作。 */"""

    def __init__(self, args, config, metadata):
        """/** 校验GPU数量、库路径与本机枚举；记录环境，不修改PyTorch全局数学策略。 */"""
        import torch
        from safetensors import safe_open
        if torch.cuda.device_count() != 4:
            raise ValueError('expose the original four GPUs in rank order')
        self.torch, self.safe_open, self.args, self.config = torch, safe_open, args, config
        self.library = linked_cublas(args.build/'fastllm_tp_worker')
        self.flag, header = reduction_flag(args.cuda_home)
        self.mapping = json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
        self.metadata = metadata
        metadata.update(torch=torch.__version__, torch_cuda=torch.version.cuda, flag_header=header,
            current_math_mode=0, candidate_math_mode=self.flag,
            cublas_path=str(self.library), cublas_sha256=file_hash(self.library), handles={}, weights={},
            gemm=dict(api='cublasGemmEx', compute_type='CUBLAS_COMPUTE_32F', algorithm='CUBLAS_GEMM_DEFAULT_TENSOR_OP',
                      input_dtype='float16', output_dtype='float16', alpha=1, beta=0, bias=False),
            devices=[dict(rank=r, name=torch.cuda.get_device_name(r), capability=torch.cuda.get_device_capability(r))
                     for r in range(4)])

    @contextmanager
    def operation(self, target):
        """/**
         * 在专家原owner设备准备完整输入矩阵及权重，退出时释放私有handle。
         * @param target 同一专家、两种原始行数及其完整保存输入；不能压缩成共同子矩阵运行。
         * @return 可调用run(mode,wave)，同stream同步后返回完整FP16 CPU输出，不进行计时。
         */"""
        torch = self.torch
        owner = target['owner']
        torch.cuda.set_device(owner)
        stream = torch.cuda.current_stream(owner)
        gemm = MoeGemm(self.library, stream.cuda_stream)
        try:
            actual_library = symbol_library(gemm.lib)
            if not actual_library.samefile(self.library):
                raise ValueError(f'GEMM resolved to {actual_library}, expected {self.library}')
            self.metadata['handles'][str(owner)] = dict(version=gemm.version, symbol_library=str(actual_library),
                                                       stream=int(stream.cuda_stream))
            with torch.inference_mode():
                weight = load_weight(torch, self.safe_open, self.args.model, self.config, self.mapping,
                                     target, self.metadata['weights']).to(device=f'cuda:{owner}')
                inputs = {wave: torch.from_numpy(np.array(target['x'][wave], copy=True)).to(device=f'cuda:{owner}')
                          for wave in (4, 2)}

                def run(mode, wave):
                    """/** 每次只改变私有handle的归约控制位；其他GEMM参数、输入和同步方式保持相同。 */"""
                    if mode not in ('current', 'no_reduced') or wave not in inputs:
                        raise ValueError('unsupported GEMM replay mode/wave')
                    set_math_mode(gemm, 0 if mode == 'current' else self.flag)
                    result = gemm.linear(torch, inputs[wave], weight)
                    stream.synchronize()
                    return result.cpu().numpy().copy()

                yield run
        finally:
            gemm.close()


def record(rows, name, actual, expected, **details):
    """/** 保存完整张量的精确误差；只用于独立实验，不代表正式后端验收。 */"""
    if actual.shape != expected.shape:
        raise ValueError(f'{name}: output shape mismatch')
    row = dict(case=name, backend='independent_cublas_gemm', shape=list(actual.shape),
        **difference(actual.reshape(-1), expected.reshape(-1)), **details)
    rows.append(row)
    print(f'{row["status"]}: {name}; max_abs={row["max_abs_error"]:.6g}; '
          f'mean_abs={row["mean_abs_error"]:.6g}; different={row["failed_elements"]}', flush=True)
    return row


def run_experiment(engine, cases, output, repeats, rows, metadata):
    """/**
     * 先完整重放当前配置，再运行候选配置；不允许基准失败后继续解释候选结果。
     * @param engine 独立cuBLAS执行器，测试可注入纯CPU模拟器。
     * @param cases 已验证共同输入相等的专家目标。
     * @param output 新建结果根目录；保存原始输出，完全相同的重复轮次引用同一文件。
     * @param repeats 每模式每目标独立调用次数，默认5；每矩阵另warmup一次。
     * @param rows 部分比较报告；零容差差异保持FAIL，禁止用候选结果覆盖旧失败。
     * @param metadata 环境、原始样本和门禁状态；异常时由调用方保存部分报告。
     * @return 无返回值；仅所有基准重放通过后设置baseline_replay_passed。
     */"""
    for target in cases:
        record(rows, target['name']+'_same_input', target['x'][2], target['x'][4][target['positions']])
    metadata['samples'] = []
    metadata['candidate_targets'] = {}
    for mode in ('current', 'no_reduced'):
        for target in cases:
            name = target['name']
            directory = output/name/mode
            directory.mkdir(parents=True, exist_ok=False)
            first, raw_paths, passed = {}, {}, True
            with engine.operation(target) as run:
                for wave in (4, 2):
                    warmup = run(mode, wave)
                    if warmup.shape != target['y'][wave].shape or warmup.dtype != np.float16:
                        raise ValueError('warmup output shape/dtype mismatch')
                    difference(warmup.reshape(-1), warmup.reshape(-1))
                for repeat in range(1, repeats+1):
                    observed = {}
                    for wave in (4, 2):
                        array = run(mode, wave)
                        if array.dtype != np.float16 or array.shape != target['y'][wave].shape:
                            raise ValueError('GEMM output shape/dtype mismatch')
                        difference(array.reshape(-1), array.reshape(-1))
                        observed[wave] = array
                        equal_first = wave in first and np.array_equal(array, first[wave])
                        if not equal_first:
                            path = directory/f'first{wave}_repeat{repeat:02}.npy'
                            np.save(path, array, allow_pickle=False)
                        else:
                            path = raw_paths[wave]
                        if wave not in first:
                            first[wave], raw_paths[wave] = array.copy(), path
                        metadata['samples'].append(dict(target=name, mode=mode, wave=wave, repeat=repeat,
                            path=str(path.relative_to(output)), sha256=file_hash(path), shape=list(array.shape)))
                        if mode == 'current':
                            row = record(rows, f'{name}_current_first{wave}_repeat{repeat:02}_replay',
                                         array, target['y'][wave], group='baseline_replay')
                            if row['status'] != 'PASS':
                                raise ValueError('current cuBLAS did not exactly replay native snapshot; candidate attribution refused')
                        elif repeat > 1:
                            row = record(rows, f'{name}_no_reduced_first{wave}_repeat{repeat:02}_stable',
                                         array, first[wave], group='candidate_stability')
                            passed &= row['status'] == 'PASS'
                    row = record(rows, f'{name}_{mode}_repeat{repeat:02}_batch_compare', observed[2],
                        observed[4][target['positions']], group=mode+'_batch_compare',
                        actual_rows=len(target['x'][2]), expected_rows=len(target['x'][4]))
                    if mode == 'no_reduced':
                        passed &= row['status'] == 'PASS'
            if mode == 'no_reduced':
                metadata['candidate_targets'][name] = passed
            save_report(output, rows, metadata)
        if mode == 'current':
            metadata['baseline_replay_passed'] = True
            print('当前配置全部精确重放通过；开始禁止低精度归约对照。', flush=True)
    metadata['candidate_batch_invariant'] = all(metadata['candidate_targets'].values())


def main():
    """/** 校验基准、权重和worker来源后执行独立GEMM实验；任何失败均保留部分四格式报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='已完成的moe_block_batches目录')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, default=Path('build-attention-ep-cu126'))
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda-12.6'))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT/'test/distributed/results')
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error('repeats must be at least 5')
    for field in ('baseline', 'model', 'build', 'cuda_home'):
        setattr(args, field, getattr(args, field).resolve())
    output = args.output.resolve()/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+
                                    '_moe_gemm_reduction_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f'专家GEMM独立归约诊断：{output}', flush=True)
    rows, metadata = [], dict(scope='independent_expert_gemm_reduction', execution_complete=False,
        expected_case_count=1, baseline_replay_passed=False, candidate_batch_invariant=None,
        command=shlex.join([sys.executable, *sys.argv]), baseline=str(args.baseline),
        git=capture(['git', 'rev-parse', 'HEAD']), gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture([str(args.cuda_home/'bin/nvcc'), '--version']),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        math_environment={k: os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE')},
        warmup=1, iterations=1, repeats=args.repeats, cuda_graph=False, prefix_cache='not_applicable',
        report_title='专家GEMM低精度归约独立实验',
        report_description='复用正式快照与原权重；先精确重放默认cuBLAS，再比较禁止低精度归约。'
            '默认批次差异保持FAIL；候选结论单列，不代表正式推理修复或性能验收。')
    try:
        report = json.loads((args.baseline/'summary.json').read_text())
        old = report['metadata']
        if (not report['complete'] or old['scope'] != 'formal_moe_block_batch_trace_diagnostic' or
                not old['trace_attribution_valid'] or not old['expert_pairing_valid'] or
                old['layer'] != 0 or old['cuda_graph'] or old['prefix_cache']):
            raise ValueError('baseline is not a complete valid layer-zero block attribution')
        config = json.loads((args.model/'config.json').read_text())
        for filename, key in (('config.json', 'model_config_sha256'), ('model.safetensors.index.json', 'model_index_sha256')):
            metadata[key] = file_hash(args.model/filename)
            if metadata[key] != old[key]:
                raise ValueError(f'model provenance mismatch: {filename}')
        worker_hash = file_hash(args.build/'fastllm_tp_worker')
        if worker_hash != old['binary_sha256']['fastllm_tp_worker']:
            raise ValueError('worker changed since baseline; use the original build without rebuilding')
        metadata.update(worker_sha256=worker_hash, build_cache=(args.build/'CMakeCache.txt').read_text(),
                        baseline_git=old['git'], baseline_summary_sha256=file_hash(args.baseline/'summary.json'))
        blocks = {}
        for wave in (4, 2):
            blocks[wave] = load_blocks(args.baseline/f'first{wave}/trace', wave, config, old['request'])
            for rank, (manifest, _) in blocks[wave].items():
                for key in ('manifest_sha256', 'binary_sha256'):
                    if manifest[key] != old['traces'][str(wave)][str(rank)][key]:
                        raise ValueError('native snapshot changed since baseline')
        cases = targets(blocks, old['expert_first_difference'])
        metadata['expected_case_count'] = len(cases)*(6*args.repeats-1)
        metadata['targets'] = [dict(name=t['name'], expert=t['expert'], owner=t['owner'], phase=t['phase'],
            input_shapes={str(w): list(t['x'][w].shape) for w in (4, 2)},
            output_shapes={str(w): list(t['y'][w].shape) for w in (4, 2)},
            saved_difference=t['saved_difference']) for t in cases]
        save_report(output, rows, metadata)
        engine = GpuReplay(args, config, metadata)
        run_experiment(engine, cases, output, args.repeats, rows, metadata)
        metadata['execution_complete'] = len(rows) == metadata['expected_case_count']
    except BaseException as error:
        metadata['candidate_batch_invariant'] = None
        (output/'diagnostic.traceback.log').write_text(traceback.format_exc())
        rows.append(dict(case='diagnostic_error_or_interrupt', status='FAIL', failed_elements=1,
                         reason=f'{type(error).__name__}: {error}'))
        print(f'FAIL: {type(error).__name__}: {error}', flush=True)
    finally:
        status = save_report(output, rows, metadata)
    print('当前配置精确重放通过：'+str(metadata['baseline_replay_passed']), flush=True)
    print('候选配置消除本次批次差异：'+str(metadata['candidate_batch_invariant']), flush=True)
    print('各目标候选结果：'+json.dumps(metadata.get('candidate_targets', {}), ensure_ascii=False), flush=True)
    print(f'{status}: {output}', flush=True)
    return int(status != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
