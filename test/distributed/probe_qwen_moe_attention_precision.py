#!/usr/bin/env python3
"""独立Attention投影数值实验；使用保存的输入，不编译、不替代正式后端验收。"""

import argparse
from contextlib import ExitStack
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import traceback
import uuid

from probe_qwen_moe_router_precision import PedanticGemm, exact_metrics, linked_cublas, norm_variants
from run_process_tp import ROOT, capture, parse_report, save_report


MODES = ('hf_replay', 'split_bias', 'merged_qkv', 'fp32_o', 'native_norm', 'combined_projections')


class MoeGemm(PedanticGemm):
    """/** 复用私有handle管理，以当前MoE的32F/TENSOR_OP参数进行独立对照。 */"""

    def linear(self, torch, x, weight, fp32=False):
        """/**
         * 重放Y[T,N]=X[T,K]*W[N,K]^T的cuBLAS调用，使用调用方当前stream。
         * @param x 连续FP16 CUDA [T,K]。
         * @param weight 同设备连续FP16 CUDA [N,K]。
         * @param fp32 true保留FP32输出，false输出FP16；不融合bias。
         * @return 同设备输出；参数或cuBLAS失败时抛异常，不fallback。
         */"""
        if (not self.handle or x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
                or not x.is_cuda or x.device != weight.device or x.dtype != torch.float16
                or weight.dtype != x.dtype or not x.is_contiguous() or not weight.is_contiguous()
                or min(*x.shape, *weight.shape) <= 0):
            raise ValueError('invalid FP16 CUDA GEMM inputs/handle')
        t, k = x.shape
        n = weight.shape[0]
        output = torch.empty((t, n), dtype=torch.float32 if fp32 else x.dtype, device=x.device)
        alpha, beta = ctypes.c_float(1), ctypes.c_float(0)
        pointer = ctypes.c_void_p
        self.check(self.lib.cublasGemmEx(self.handle, 1, 0, n, t, k, ctypes.byref(alpha),
            pointer(weight.data_ptr()), 2, k, pointer(x.data_ptr()), 2, k, ctypes.byref(beta),
            pointer(output.data_ptr()), 0 if fp32 else 2, n, 68, 99))
        return output


def attention_variant(torch, attention, normalized, simulated_norm, residual, positions, embeddings, mode, gemm):
    """/**
     * 在HF独立Attention中替换一个投影因素，记录FP16残差及中间量。
     * RoPE、QK、softmax、PV仍由HF执行；组合实验也不是原生Attention重放。
     * 所有hook限定在本次调用内，成功和异常均卸载，不修改权重或全局后端。
     * @param attention 已加载本层权重且eval的HF eager模块。
     * @param normalized HF归一化的FP16 [1,T,H]。
     * @param simulated_norm 原生归约顺序的张量模拟，布局同normalized。
     * @param residual 双方共同的原始FP16 [1,T,H]，无padding/cache。
     * @param positions [1,T]位置；embeddings为HF产生的cos/sin。
     * @param mode MODES中的实验名称；gemm为独立MoE cuBLAS调用器。
     * @return 残差输出和中间量；失败抛异常，不更换路径重试。
     */"""
    if mode not in MODES or residual.ndim != 3 or residual.shape[0] != 1 or not residual.shape[1]:
        raise ValueError('unsupported attention experiment mode/shape')
    if any(value.shape != residual.shape or value.dtype != torch.float16 or value.device != residual.device
           for value in (normalized, simulated_norm, residual)):
        raise ValueError('attention experiment requires matching FP16 [1,T,H] inputs')
    x = simulated_norm if mode in ('native_norm', 'combined_projections') else normalized
    qkv = [attention.q_proj, attention.k_proj, attention.v_proj]
    evidence = {'norm': x.detach().clone()}
    merged = None
    if mode in ('merged_qkv', 'combined_projections'):
        weight = torch.cat([module.weight for module in qkv], dim=0).contiguous()
        merged = gemm.linear(torch, x.reshape(-1, x.shape[-1]).contiguous(), weight)
        biases = [module.bias for module in qkv]
        if any(bias is not None for bias in biases):
            if any(bias is None for bias in biases):
                raise ValueError('mixed QKV bias is not supported by the native projection')
            merged = merged + torch.cat(biases)
        merged = merged.split([module.weight.shape[0] for module in qkv], dim=-1)

    with ExitStack() as hooks:
        for index, (name, module) in enumerate(zip(('q', 'k', 'v'), qkv)):
            def projection(module, args, value, index=index, name=name):
                if mode == 'split_bias':
                    value = torch.nn.functional.linear(args[0], module.weight)
                    if module.bias is not None:
                        value = value + module.bias
                elif merged is not None:
                    value = merged[index].reshape_as(value).contiguous()
                evidence[name] = value.detach().clone()
                return value
            hooks.callback(module.register_forward_hook(projection).remove)

        def output_projection(module, args, value):
            evidence['context'] = args[0].detach().clone()
            if mode in ('fp32_o', 'combined_projections'):
                if module.bias is not None:
                    raise ValueError('Qwen2MoE output projection must have no bias')
                value = gemm.linear(torch, args[0].reshape(-1, args[0].shape[-1]).contiguous(),
                                    module.weight, fp32=True).half().reshape_as(value)
            evidence['o'] = value.detach().clone()
            return value
        hooks.callback(attention.o_proj.register_forward_hook(output_projection).remove)

        tokens = x.shape[1]
        mask = torch.full((tokens, tokens), torch.finfo(x.dtype).min, dtype=x.dtype,
                          device=x.device).triu(1).reshape(1, 1, tokens, tokens)
        value = attention(hidden_states=x, attention_mask=mask, position_ids=positions,
                          position_embeddings=embeddings, use_cache=False)[0]
        output = residual + value
        evidence['output'] = output.detach().clone()
    return output, evidence


def require_hf_replay(metrics):
    """/** 未精确复现已保存HF边界时停止归因，避免把环境或参考差异当成投影原因。 */"""
    if metrics['status'] != 'PASS':
        raise RuntimeError('HF replay does not match saved HF attention; stop projection attribution')


def main():
    """/** 仅读取指定层的权重和保存的输入；逐层输出原始中间量及四格式报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 9, 16, 18, 22, 23])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_attention_probe_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    print(f'独立诊断目录：{output}', flush=True)
    rows, gemm = [], None
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), warmup=0, iterations=1, repeats=1,
        expected_case_count=len(args.layers)*len(MODES)*2, layers=args.layers, modes=list(MODES),
        probe_gemm='CUBLAS_COMPUTE_32F/CUBLAS_GEMM_DEFAULT_TENSOR_OP; FP16 inputs; FP16 or FP32 output',
        diagnostic=str(args.diagnostic.resolve()), model=str(args.model.resolve()),
        report_title='Attention投影独立数值诊断',
        report_description='零容差用于差异定位；HF Attention核心保留，不能替代原生后端或完整模型验收。')
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('run in the existing CUDA reference environment on the server')
        import transformers
        from safetensors import safe_open
        from transformers.models.qwen2_moe.modeling_qwen2_moe import (
            Qwen2MoeAttention, Qwen2MoeRMSNorm, Qwen2MoeRotaryEmbedding)
        if transformers.__version__ != '4.57.6':
            raise RuntimeError('this isolated probe requires the recorded Transformers 4.57.6 API')
        torch.manual_seed(20261115)
        torch.cuda.set_device(args.device)
        device = torch.device('cuda', args.device)
        metadata.update(torch=torch.__version__, transformers=transformers.__version__, torch_cuda=torch.version.cuda,
                        compute_capability=torch.cuda.get_device_capability(device), seed=20261115,
                        allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        library = linked_cublas(args.worker)
        gemm = MoeGemm(library, torch.cuda.current_stream().cuda_stream)
        metadata.update(cublas_path=str(library), cublas_version=gemm.version)
        cache = args.worker.parent / 'CMakeCache.txt'
        metadata['build_cache'] = cache.read_text() if cache.is_file() else 'unavailable'

        original = json.loads((args.diagnostic / 'summary.json').read_text())
        rank = original['metadata']['rank']
        metadata.update(source_rank=rank, source_git=original['metadata']['git'],
                        source_reference=original['metadata'].get('reference'))
        references = {row['name']: row for row in json.loads((args.diagnostic / 'reference.json').read_text())['layers']}
        native = {}
        for path in sorted(args.diagnostic.glob('native-*/stdout.log')):
            for row in parse_report(path.read_text()).get('results', []):
                if row['task'] in native:
                    raise ValueError('duplicate saved task: ' + row['task'])
                native[row['task']] = row['ranks'][rank]
        config = transformers.AutoConfig.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
        if (config.model_type != 'qwen2_moe' or len(set(args.layers)) != len(args.layers)
                or any(layer < 0 or layer >= config.num_hidden_layers for layer in args.layers)
                or getattr(config, 'rope_scaling', None) or getattr(config, 'use_sliding_window', False)):
            raise ValueError('probe requires unique valid layers of Qwen2MoE with default RoPE and full attention')
        mapping = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']

        def load_weight(name):
            """/** 仅读取索引内所需张量；拒绝模型目录外的分片，不修改原始checkpoint。 */"""
            path = (args.model / mapping[name]).resolve()
            if not path.is_relative_to(args.model.resolve()):
                raise ValueError('weight shard escapes model directory')
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                return handle.get_tensor(name).to(device=device, dtype=torch.float16).contiguous()

        def compare(label, actual, expected):
            """/** 同步读取精确差异，并附原验收容差的数值统计；不改变正式判定标准。 */"""
            metrics = exact_metrics(torch, actual, expected)
            metrics['tolerance_failed_elements'] = int((
                ~torch.isfinite(actual) | ~torch.isfinite(expected) |
                ((actual.float()-expected.float()).abs() > .05 + .01*expected.float().abs())).sum().item())
            rows.append(dict(case=label, backend='independent_hf_attention_projection_probe', **metrics))
            return metrics

        with torch.inference_mode():
            for layer in args.layers:
                name = f'real_layer_{layer}_attention'
                task = references[name]
                if task['stage'] != 1 or task['inputs'][rank]['position_begin'] != 0:
                    raise ValueError('probe requires saved Attention prefill with position_begin=0')
                x = torch.tensor(task['inputs'][rank]['hidden'], device=device, dtype=torch.float16).reshape(1, -1, task['hidden'])
                expected = torch.tensor(task['reference'][rank]['output'], device=device, dtype=x.dtype).reshape_as(x)
                observed = torch.tensor(native[name]['output'], device=device, dtype=x.dtype).reshape_as(x)
                prefix = f'model.layers.{layer}.'
                attention = Qwen2MoeAttention(config, layer_idx=layer).to(device=device, dtype=x.dtype).eval()
                attention.load_state_dict({key: load_weight(prefix+'self_attn.'+key)
                                           for key in attention.state_dict()}, strict=True)
                norm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device=device, dtype=x.dtype)
                norm.load_state_dict({'weight': load_weight(prefix+'input_layernorm.weight')})
                normalized = norm(x)
                _, simulated = norm_variants(torch, x[0], norm.weight, config.rms_norm_eps)
                positions = torch.arange(x.shape[1], device=device).reshape(1, -1)
                # 独立构造FP32 inv_freq，避免将其随Attention权重转换成FP16。
                rotary = Qwen2MoeRotaryEmbedding(config=config, device=device)
                embeddings = rotary(x, positions)
                raw = dict(input=x.cpu().tolist(), saved_hf=expected.cpu().tolist(), saved_native=observed.cpu().tolist(), variants={})

                for mode in MODES:
                    values, intermediates = attention_variant(torch, attention, normalized, simulated.unsqueeze(0),
                                                              x, positions, embeddings, mode, gemm)
                    hf = compare(f'layer{layer}/{mode}/vs_saved_hf', values, expected)
                    actual = compare(f'layer{layer}/{mode}/vs_saved_native', values, observed)
                    raw['variants'][mode] = {key: value.cpu().tolist() for key, value in intermediates.items()}
                    (output / f'layer-{layer:02d}.json').write_text(json.dumps(raw, allow_nan=False))
                    metadata['loaded_cublas_libraries'] = sorted({line.split()[-1]
                        for line in Path('/proc/self/maps').read_text().splitlines() if '/libcublas' in line})
                    save_report(output, rows, metadata)
                    print(f'layer{layer}/{mode}: HF_diff={hf["failed_elements"]} native_diff={actual["failed_elements"]} '
                          f'HF_max={hf["max_abs_error"]:.9g} native_max={actual["max_abs_error"]:.9g}', flush=True)
                    if mode == 'hf_replay':
                        require_hf_replay(hf)
                del attention, norm, rotary
    except Exception as error:
        (output / 'error.log').write_text(traceback.format_exc())
        rows.append(dict(case='probe_setup_or_execution', backend='independent_attention_probe', status='FAIL',
                         failed_elements=1, reason=str(error)))
        print(f'FAIL: {error}', flush=True)
    finally:
        if gemm is not None:
            try:
                gemm.close()
            except Exception as error:
                rows.append(dict(case='probe_cleanup', backend='independent_attention_probe', status='FAIL',
                                 failed_elements=1, reason=str(error)))
        status = save_report(output, rows, metadata)
        print(f'{status}: {output}（精确差异诊断，非完整模型验收）', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
