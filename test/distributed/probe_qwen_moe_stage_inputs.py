#!/usr/bin/env python3
"""独立双输入MoE诊断：分离传入误差与同输入实现差异，复用已编译的正式层入口。"""

import argparse
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import struct
import sys
import traceback
import uuid

from probe_qwen_moe_forward_precision import compare, digest, hf_trajectory, load_trace, TRACE_ENV
from qwen_moe_transformers_reference import moe_routing_observer, reference_device_map
from recheck_qwen_moe_forward_precision import recheck
from run_process_tp import ROOT, capture, parse_report, save_report
from run_qwen_moe_precision_diagnostic import ATOL, RTOL, compare_layer, run_logged, task_chunks


def restore_half_wire(values):
    """/**
     * 从json11小数记录恢复FP16精确值；FP16相邻值间距大于传输舍入宽度。
     * 仅用于原生FP16层输出的精确重放门禁，不能用于FP32 logits或路由累加值。
     * @param values 有限的%.12f传输数值；必须重新编码后逐值相同。
     * @return FP16精确值；任意非FP16编码、溢出或非有限值均拒绝。
     */"""
    result = []
    for value in values:
        if not math.isfinite(value):
            raise ValueError('nonfinite FP16 wire value')
        try:
            decoded, = struct.unpack('<e', struct.pack('<e', value))
        except (OverflowError, struct.error) as error:
            raise ValueError('FP16 wire overflow') from error
        if float(format(decoded, '.12f')) != value:
            raise ValueError('value does not match native FP16 JSON encoding')
        result.append(decoded)
    return result


def observe_moe(torch, block, norm, x):
    """/**
     * 调用HF原版Norm和MoE公开forward，观察实际路由并按Decoder语义加残差。
     * 整段输入为FP16 [1,tokens,hidden]；不只抽取目标Token，以保持专家GEMM形状。
     * 不改写Top-K、专家计算或累加精度；退出及异常时移除全部观察hook。
     * @return 与正式层检查器字段一致的精确Tensor值列表。
     */"""
    record = {}
    observer = moe_routing_observer(torch)

    def gate(module, args, value):
        """/** 观察实际router输出，并开始记录同一次MoE的路由算子。 */"""
        observer.begin(record, 'paired-input MoE')
        record['router_logits'] = value.detach().clone()

    def save(name):
        """/** 保存共享专家的原始输出供既有观察器核对实际合成关系。 */"""
        def hook(module, args, value):
            """/** 复制当前输出，不改变传给后续运算的张量。 */"""
            record[name] = value.detach().clone()
        return hook

    def finish(module, args, value):
        """/** 核对HF实际路由输出和共享专家能重建MoE返回值。 */"""
        observer.finish(value)

    with ExitStack() as hooks, observer, torch.inference_mode():
        hooks.callback(block.gate.register_forward_hook(gate).remove)
        hooks.callback(block.shared_expert.register_forward_hook(save('shared')).remove)
        hooks.callback(block.shared_expert_gate.register_forward_hook(save('shared_gate')).remove)
        hooks.callback(block.register_forward_hook(finish).remove)
        delta, _ = block(norm(x))
        record['output'] = x + delta
    return {name: value.detach().cpu().reshape(-1).tolist() for name, value in record.items()}


def hf_pair(model, config, layer, inputs, expected_versions, observer=None):
    """/**
     * 仅加载指定层的HF Norm/MoE权重，在原参考设备上执行两种完整输入。
     * 不构造完整模型，不下载或编译；检查参考版本并保存两次真实路由输出。
     * @param model 本地模型目录，权重分片必须位于该目录内。
     * @param config 模型配置字典，确定维度和参考层所在设备。
     * @param layer 从零开始的层索引。
     * @param inputs HF和原生的Attention后FP16激活列表，形状一致。
     * @param expected_versions 已保存的torch、transformers和dtype版本约束。
     * @param observer 可选独立诊断观察器；接收torch、MoE、Norm和完整输入，默认沿用原观察器。
     * @return 各输入参考及实际Torch/Transformers/设备信息。
     */"""
    os.environ.pop('USE_TORCH', None)
    import torch
    import transformers
    from safetensors import safe_open
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeRMSNorm, Qwen2MoeSparseMoeBlock

    versions = dict(torch=torch.__version__, transformers=transformers.__version__, dtype='float16')
    if torch.cuda.device_count() != 4 or transformers.__version__ != '4.57.6' or any(
            versions[key] != expected_versions[key] for key in versions):
        raise ValueError(f'reference environment differs from saved reference: {versions}')
    cfg = transformers.Qwen2MoeConfig.from_dict(config)
    device = torch.device('cuda', reference_device_map(cfg)[f'model.layers.{layer}'])
    torch.cuda.set_device(device)
    versions.update(device=str(device),
        allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    with torch.device('meta'):
        modules = torch.nn.ModuleDict(dict(
            post_attention_layernorm=Qwen2MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps),
            mlp=Qwen2MoeSparseMoeBlock(cfg)))
    mapping = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    state = {}
    for name in modules.state_dict():
        full = f'model.layers.{layer}.{name}'
        path = (model / mapping[full]).resolve()
        if not path.is_relative_to(model):
            raise ValueError('weight shard escapes model directory')
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            state[name] = handle.get_tensor(full).to(device=device, dtype=torch.float16).contiguous()
    modules.load_state_dict(state, strict=True, assign=True)
    modules.eval()
    result = {}
    try:
        for name, values in inputs.items():
            x = torch.tensor(values, device=device, dtype=torch.float16).reshape(1, -1, cfg.hidden_size)
            result[name] = (observer or observe_moe)(torch, modules['mlp'], modules['post_attention_layernorm'], x)
            del x
    finally:
        del modules, state
        torch.cuda.empty_cache()
    return result, versions


def attribution_rows(layer, ids, hidden, native_input, hf_input, saved_native, saved_hf, references, results):
    """/**
     * 先核对两侧重放与历史输出，再分别保存上游传播和同输入实现差异。
     * 所有数值比较沿用原阈值；重放门禁为精确比较。门禁失败时仍保存观察值，
     * 但每行明确标记attribution_valid=false，不把首次跨阈值当作根因。
     * @return 整层及逐Token报告，包含输入、传播、局部和总误差四组对照。
     */"""
    rows = []
    for field, expected in saved_hf.items():
        rows.append(compare(f'layer{layer}/hf_replay/{field}', references['hf'][field],
                            expected, len(expected), exact=True))
    actual = restore_half_wire(results['native']['output'])
    rows.append(compare(f'layer{layer}/native_replay/output', actual, saved_native, hidden, exact=True))
    valid = all(row['status'] == 'PASS' for row in rows)
    pairs = dict(input_difference=(native_input, hf_input),
        propagated_by_hf=(references['native']['output'], references['hf']['output']),
        same_native_input=(actual, references['native']['output']),
        total_output_difference=(actual, references['hf']['output']))
    tokens = []
    for name, (observed, expected) in pairs.items():
        row = compare(f'layer{layer}/{name}', observed, expected, hidden)
        row['attribution_valid'] = valid
        row['reason'] += f'; attribution_valid={valid}'
        rows.append(row)
        for token, token_id in enumerate(ids):
            begin, end = token*hidden, (token+1)*hidden
            item = compare(f'layer{layer}/{name}/token{token}', observed[begin:end], expected[begin:end], hidden)
            item.update(attribution_valid=valid, layer=layer, comparison=name, token_index=token, token_id=token_id)
            item['reason'] += f'; attribution_valid={valid}'
            tokens.append(item)
    return rows, tokens, valid


def main():
    """/** 验证历史证据，复用正式层入口执行成对输入；失败也保存四格式报告和原始日志。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True, help='原始forward_probe目录，不是forward_recheck目录')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, default=Path('build-attention-ep-cu126'))
    parser.add_argument('--layers', type=int, nargs='+', default=[13, 15])
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    args.probe, args.model, args.build = args.probe.resolve(), args.model.resolve(), args.build.resolve()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_stage_inputs_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    (output / 'tokens').mkdir()
    print(f'独立双输入诊断目录：{output}', flush=True)
    rows, tokens = [], []
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), warmup=0, iterations=1, repeats=1, atol=ATOL, rtol=RTOL,
        probe=str(args.probe), model=str(args.model), layers=args.layers, controlled_rank=0,
        report_title='MoE双输入归因诊断', report_description='保留历史失败及输出头不确定性。'
        '完整Token形状的独立MoE重放，不是完整模型验收或性能测试；门禁失败不作归因。')
    try:
        if os.environ.get(TRACE_ENV):
            raise ValueError(f'unset {TRACE_ENV}; this probe reuses saved traces')
        source = json.loads((args.probe / 'summary.json').read_text())
        source_meta = source['metadata']
        checked, _, evidence = recheck(args.probe, source)
        rows.extend(copy.deepcopy(checked))
        metadata.update(source_evidence=evidence, source_git=source_meta['git'],
                        source_summary_sha256=digest(args.probe / 'summary.json'))
        if not evidence['trace_attribution_valid']:
            raise ValueError('saved trace failed attribution gate')
        config = json.loads((args.model / 'config.json').read_text())
        hidden, layers = config['hidden_size'], config['num_hidden_layers']
        if config.get('model_type') != 'qwen2_moe' or len(set(args.layers)) != len(args.layers) or any(
                not 0 <= i < layers for i in args.layers):
            raise ValueError('invalid model/layer selection')
        ids = source_meta['target']['ids']
        trace_meta, native = load_trace(args.probe / 'trace', ids, hidden, layers)
        if trace_meta != source_meta['trace']:
            raise ValueError('model shape disagrees with saved trace')
        reference = json.loads((Path(source_meta['diagnostic']) / 'reference.json').read_text())
        hf = hf_trajectory(reference, source_meta['source_rank'], len(ids), hidden, layers)
        saved_tasks = {task['name']: task for task in reference['layers']}
        metadata['binary_sha256'] = {name: digest(args.build / name) for name in
                                    ('fastllm_moe_layer_check', 'fastllm_tp_worker')}
        if metadata['binary_sha256']['fastllm_tp_worker'] != source_meta['binary_sha256']['fastllm_tp_worker']:
            raise ValueError('worker changed since saved trace; collect a new forward probe first')
        metadata['build_cache'] = (args.build / 'CMakeCache.txt').read_text()
        metadata['layer_attribution_valid'], metadata['reference_environments'] = {}, {}
        save_report(output, rows, metadata)

        for layer in args.layers:
            inputs = dict(hf=hf[f'layer_{layer}_attention'], native=native[f'layer_{layer}_attention'])
            tasks = [dict(name=f'layer{layer}_{name}_input', layer=layer, hidden=hidden, stage=2,
                inputs=[dict(position_begin=0, hidden=values if rank == 0 else []) for rank in range(4)])
                for name, values in inputs.items()]
            results = []
            for index, chunk in enumerate(task_chunks(tasks)):
                path = output / f'layer{layer}-tasks-{index}.json'
                path.write_text(json.dumps(chunk, allow_nan=False))
                directory = output / f'layer{layer}-native-{index}'
                command = [str(args.build / 'fastllm_moe_layer_check'), '--worker',
                    str(args.build / 'fastllm_tp_worker'), '--model', str(args.model), '--tasks', str(path),
                    '--devices', '0,1,2,3', '--output', str(directory / 'weights'), '--dump-weights', '0']
                code = run_logged(command, directory, args.timeout)
                report = parse_report((directory / 'stdout.log').read_text())
                actual = report.get('results', [])
                if code or report.get('status') != 'PASS' or not report.get('workers_reaped') or \
                        [r['task'] for r in actual] != [t['name'] for t in chunk]:
                    raise RuntimeError(f'incomplete native paired-input run; inspect {directory}')
                results.extend(actual)
            print(f'运行：layer{layer} HF双输入重放', flush=True)
            references, env = hf_pair(args.model, config, layer, inputs, reference['metadata'])
            metadata['reference_environments'][str(layer)] = env
            (output / f'layer{layer}-hf.json').write_text(json.dumps(references, allow_nan=False))
            for task, actual, name in zip(tasks, results, inputs):
                empty = {key: [] for key in references[name]}
                task['reference'] = [references[name], empty, empty, empty]
                fields = compare_layer(task, actual, config)
                rows.extend(fields)
                for row in fields:
                    if row['status'] != 'PASS':
                        print(f'FAIL: {row["case"]}: failed={row["failed_elements"]}; {row["reason"]}', flush=True)
            local = {name: result['ranks'][0] for name, result in zip(inputs, results)}
            saved = saved_tasks[f'real_layer_{layer}_moe']['reference'][source_meta['source_rank']]
            compared, detail, valid = attribution_rows(layer, ids, hidden, inputs['native'], inputs['hf'],
                native[f'layer_{layer}_moe'], saved, references, local)
            rows.extend(compared)
            tokens.extend(detail)
            metadata['layer_attribution_valid'][str(layer)] = valid
            save_report(output, rows, metadata)
            save_report(output / 'tokens', tokens, metadata)
            print(f'layer{layer}: attribution_valid={valid}', flush=True)
            for row in compared:
                if '/hf_replay/' not in row['case'] or row['status'] != 'PASS':
                    print(f'{row["status"]}: {row["case"]} max_abs={row["max_abs_error"]:.9g} '
                          f'mean_abs={row["mean_abs_error"]:.9g} failed={row["failed_elements"]}', flush=True)
            focus = {item['token_index'] for item in evidence['first_failure_by_token']
                     if item['stage'] == f'layer_{layer}_moe'}
            for row in detail:
                if row['token_index'] in focus:
                    print(f'  {row["case"]}: max_abs={row["max_abs_error"]:.9g} '
                          f'mean_abs={row["mean_abs_error"]:.9g} failed={row["failed_elements"]}', flush=True)
    except Exception as error:
        (output / 'error.log').write_text(traceback.format_exc())
        rows.append(dict(case='stage_inputs_error', status='FAIL', failed_elements=1, reason=str(error)))
        print(f'FAIL: stage_inputs_error: {error}', flush=True)
    status = save_report(output, rows, metadata)
    if tokens:
        save_report(output / 'tokens', tokens, metadata)
    print(f'{status}: {output}（保留历史失败；独立层诊断，非完整模型验收）', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
