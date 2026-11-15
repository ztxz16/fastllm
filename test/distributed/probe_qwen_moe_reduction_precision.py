#!/usr/bin/env python3
"""独立专家贡献重放：观察HF真实前向，仅在离线重放中改变累加精度；不是模型验收。"""

import argparse
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import struct
import sys
import traceback
import uuid

from probe_qwen_moe_forward_precision import compare, digest, hf_trajectory, load_trace
from probe_qwen_moe_stage_inputs import attribution_rows, hf_pair, observe_moe, restore_half_wire
from recheck_qwen_moe_forward_precision import recheck
from run_process_tp import ROOT, capture, parse_report, save_report


def rounded(value, dtype='e'):
    """/** 将有限数值舍入为IEEE FP16或FP32；溢出及非有限值均拒绝。 */"""
    result, = struct.unpack('<' + dtype, struct.pack('<' + dtype, value))
    if not math.isfinite(result):
        raise ValueError('nonfinite contribution or reduction')
    return result


def capture_contributions(torch, block, norm, x):
    """/**
     * 沿原版HF公开MoE前向观察每次index_add_的实际专家输出和加权输入。
     * 输入保持完整FP16 [1,tokens,hidden]，不重选专家、不替换GEMM或累加。
     * 专家hook关联实际专家ID，Mode只复制数据；成功和异常均清理hook/Mode。
     * @param torch 已加载的参考PyTorch模块。
     * @param block 原版HF MoE模块，包含逐专家ModuleList。
     * @param norm 本层原版HF post-attention RMSNorm模块。
     * @param x 完整FP16 [1,tokens,hidden] 输入，保持原设备和Token顺序。
     * @return 原观察器的完整层结果，以及实际调用顺序的专家贡献列表。
     */"""
    from torch.overrides import TorchFunctionMode
    records, pending, initial = [], [], []
    shape = (x.shape[0] * x.shape[1], x.shape[2])

    def expert_hook(expert):
        """/** 将真实专家返回值关联到下一次路由累加，拒绝遗漏或嵌套专家调用。 */"""
        def after(module, args, value):
            """/** 保留专家输出原dtype与原行顺序，不改变公开forward返回值。 */"""
            if pending:
                raise ValueError('expert output was not consumed by index_add_')
            pending.append((expert, value.detach().clone()))
        return after

    class Contributions(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            """/** 原算子恰好执行一次，仅记录无重复Token的FP16路由累加。 */"""
            kwargs = kwargs or {}
            is_add = func is torch.Tensor.index_add_
            if is_add:
                if len(args) != 4 or args[1] != 0 or kwargs.get('alpha', 1) != 1 or len(pending) != 1:
                    raise ValueError('unsupported routed index_add_ call')
                target, _, indices, values = args
                if tuple(target.shape) != shape or target.dtype != torch.float16 or \
                        values.dtype != torch.float16 or indices.ndim != 1 or \
                        tuple(values.shape) != (indices.numel(), shape[1]):
                    raise ValueError('unexpected routed contribution shape/dtype')
                if not records:
                    initial.append(target.detach().clone())
            result = func(*args, **kwargs)
            if is_add:
                expert, raw = pending.pop()
                records.append((expert, indices.detach().clone(), values.detach().clone(), raw))
            return result

    with ExitStack() as hooks, Contributions():
        for expert, module in enumerate(block.experts):
            hooks.callback(module.register_forward_hook(expert_hook(expert)).remove)
        observed = observe_moe(torch, block, norm, x)
    if pending or len(initial) != 1 or any(initial[0].reshape(-1).cpu().tolist()):
        raise ValueError('incomplete expert capture or nonzero initial routed accumulator')
    contributions = [dict(expert=expert, tokens=indices.cpu().tolist(),
        weighted=values.reshape(-1).cpu().tolist(), expert_output=raw.reshape(-1).cpu().tolist())
        for expert, indices, values, raw in records]
    return observed, contributions


def replay_contributions(records, observed, inputs, hidden, topk, experts):
    """/**
     * 用同一组真实HF加权贡献重放FP16逐次累加和FP32逐次累加。
     * 校验实际专家/Token覆盖、FP16乘权重结果、形状和有限值；保留真实专家调用顺序。
     * FP32候选不模拟原生专家GEMM、rank分组或NCCL归约树，只用于独立数值对照。
     * @param records 按实际调用次序保存的专家ID、Token索引和FP16贡献。
     * @param observed 同次HF前向的路由、共享专家及层输出记录。
     * @param inputs 按行展开的FP16 [tokens,hidden] 原始层输入。
     * @param hidden 隐藏维度，必须为正整数。
     * @param topk 每个Token选择的专家数，必须为正且不大于experts。
     * @param experts 模型专家总数，用于检查ID范围。
     * @return 两种候选的路由贡献及按FP16共享专家/残差语义合成的层输出。
     */"""
    if hidden <= 0 or topk <= 0 or experts < topk or not inputs or len(inputs) % hidden:
        raise ValueError('invalid replay shape')
    tokens = len(inputs) // hidden
    ids, weights, shared = (observed[key] for key in ('selected_experts', 'routing_weights', 'shared_output'))
    if len(ids) != tokens * topk or len(weights) != len(ids) or len(shared) != len(inputs):
        raise ValueError('incomplete routing/shared fields')
    if any(type(e) is not int or not 0 <= e < experts for e in ids):
        raise ValueError('invalid expert ID')
    if any(rounded(v) != v for v in [*inputs, *weights, *shared]):
        raise ValueError('replay requires exact FP16 values')
    routes = [dict(zip(ids[t*topk:(t+1)*topk], weights[t*topk:(t+1)*topk])) for t in range(tokens)]
    if any(len(route) != topk for route in routes):
        raise ValueError('duplicate selected expert')
    expected = {(t, e) for t, route in enumerate(routes) for e in route}
    seen, expert_order = set(), []
    sums = {kind: [0.] * len(inputs) for kind in ('fp16', 'fp32')}

    for record in records:
        expert, indices = record['expert'], record['tokens']
        values, raw = record['weighted'], record['expert_output']
        if type(expert) is not int or expert in expert_order or not 0 <= expert < experts or \
                not indices or len(set(indices)) != len(indices) or \
                len(values) != len(indices) * hidden or len(raw) != len(values):
            raise ValueError('invalid or duplicate expert contribution')
        expert_order.append(expert)
        for row, token in enumerate(indices):
            if type(token) is not int or (token, expert) not in expected or (token, expert) in seen:
                raise ValueError('contribution does not match actual route')
            seen.add((token, expert))
            for col in range(hidden):
                j, i = row * hidden + col, token * hidden + col
                if rounded(raw[j]) != raw[j] or rounded(values[j]) != values[j] or \
                        rounded(raw[j] * routes[token][expert]) != values[j]:
                    raise ValueError('captured contribution does not match FP16 expert product')
                sums['fp16'][i] = rounded(sums['fp16'][i] + values[j])
                sums['fp32'][i] = rounded(sums['fp32'][i] + values[j], 'f')
    if seen != expected:
        raise ValueError('missing routed contribution')
    return {kind: dict(routed_output=values,
        output=[rounded(x + rounded(rounded(v) + s)) for x, v, s in zip(inputs, values, shared)])
        for kind, values in sums.items()}


def load_pair(pair, model, layer):
    """/**
     * 只读复核历史轨迹、双输入任务及原生结果，门禁失配时拒绝继续HF重放。
     * @param pair 已有双输入诊断的绝对目录。
     * @param model 本地模型绝对目录，必须与原诊断一致。
     * @param layer 从零开始的层索引，必须已有有效重放证据。
     * @return 原报告、配置、Token、完整双输入、HF与原生结果以及证据哈希。
     */"""
    parent = json.loads((pair / 'summary.json').read_text())
    meta = parent['metadata']
    if meta.get('layer_attribution_valid', {}).get(str(layer)) is not True or \
            Path(meta['model']).resolve() != model:
        raise ValueError('paired layer is not attributable or model path differs')
    source_path = Path(meta['probe'])
    source = json.loads((source_path / 'summary.json').read_text())
    if digest(source_path / 'summary.json') != meta['source_summary_sha256']:
        raise ValueError('source forward summary changed')
    _, _, evidence = recheck(source_path, source)
    if not evidence['trace_attribution_valid']:
        raise ValueError('historical trace failed attribution gate')
    config = json.loads((model / 'config.json').read_text())
    hidden, layers = config['hidden_size'], config['num_hidden_layers']
    if config.get('model_type') != 'qwen2_moe' or not 0 <= layer < layers:
        raise ValueError('invalid model/layer')
    source_meta = source['metadata']
    ids = source_meta['target']['ids']
    trace_meta, native = load_trace(source_path / 'trace', ids, hidden, layers)
    if trace_meta != source_meta['trace']:
        raise ValueError('trace descriptor differs from source')
    reference = json.loads((Path(source_meta['diagnostic']) / 'reference.json').read_text())
    hf = hf_trajectory(reference, source_meta['source_rank'], len(ids), hidden, layers)
    inputs = dict(hf=hf[f'layer_{layer}_attention'], native=native[f'layer_{layer}_attention'])
    tasks, results, files = {}, {}, [pair / 'summary.json', pair / f'layer{layer}-hf.json']
    for path in sorted(pair.glob(f'layer{layer}-tasks-*.json')):
        files.append(path)
        for task in json.loads(path.read_text()):
            if task['name'] in tasks:
                raise ValueError('duplicate paired task')
            tasks[task['name']] = task
    for path in sorted(pair.glob(f'layer{layer}-native-*/stdout.log')):
        files.append(path)
        report = parse_report(path.read_text())
        if report.get('status') != 'PASS' or not report.get('workers_reaped'):
            raise ValueError('incomplete saved native run')
        for result in report['results']:
            if result['task'] in results or len(result['ranks']) != 4 or any(r['output'] for r in result['ranks'][1:]):
                raise ValueError('duplicate/incomplete native task or unexpected active rank')
            results[result['task']] = result['ranks'][0]
    names = {f'layer{layer}_{name}_input' for name in inputs}
    if set(tasks) != names or set(results) != names:
        raise ValueError('missing paired tasks/results')
    for name, values in inputs.items():
        task = tasks[f'layer{layer}_{name}_input']
        expected = [dict(position_begin=0, hidden=values if rank == 0 else []) for rank in range(4)]
        if task['layer'] != layer or task['stage'] != 2 or task['hidden'] != hidden or task['inputs'] != expected:
            raise ValueError('paired task changed from saved full input')
    refs = json.loads((pair / f'layer{layer}-hf.json').read_text())
    local = {name: results[f'layer{layer}_{name}_input'] for name in inputs}
    saved = next(task for task in reference['layers'] if task['name'] == f'real_layer_{layer}_moe')
    _, _, valid = attribution_rows(layer, ids, hidden, inputs['native'], inputs['hf'],
        native[f'layer_{layer}_moe'], saved['reference'][source_meta['source_rank']], refs, local)
    if not valid:
        raise ValueError('saved paired replay gate no longer passes')
    hashes = {str(path): digest(path) for path in files}
    return parent, config, ids, inputs, refs, local, hashes


def main():
    """/** 保存独立重放与四格式报告，继承历史失败；不启动原生进程、不构建、不改验收阈值。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--tokens', type=int, nargs='+', default=[6, 11])
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    pair, model = args.pair.resolve(), args.model.resolve()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_moe_reduction_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    print(f'独立专家贡献重放目录：{output}', flush=True)
    rows, metadata = [], dict(command=shlex.join([sys.executable, *sys.argv]),
        git=capture(['git', 'rev-parse', 'HEAD']), gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), warmup=0, iterations=1, repeats=1,
        pair=str(pair), model=str(model), layer=args.layer, attribution_valid=False,
        report_title='HF实际专家贡献的独立精度重放',
        native_expert_outputs_observed=False, native_nccl_reduction_replayed=False,
        report_description='FP32候选按HF实际专家调用顺序累加，不模拟原生GEMM或NCCL归约树。'
        '保留历史失败，不是完整模型验收或性能测试。')
    try:
        parent, config, ids, inputs, saved, native, hashes = load_pair(pair, model, args.layer)
        rows.extend(copy.deepcopy(parent['cases']))
        metadata.update(source_sha256=hashes, source_git=parent['metadata']['git'],
            build_cache=parent['metadata']['build_cache'], tokens=ids, atol=.05, rtol=.01)
        if len(set(args.tokens)) != len(args.tokens) or any(not 0 <= t < len(ids) for t in args.tokens):
            raise ValueError('invalid focus token')
        captured = []

        def observe(torch, block, norm, x):
            """/** 在完整输入上记录同一次公开HF前向的专家贡献，保持两种输入的调用次序。 */"""
            observed, contributions = capture_contributions(torch, block, norm, x)
            captured.append(contributions)
            return observed

        print(f'运行：第{args.layer}层HF双输入专家贡献采集', flush=True)
        refs, env = hf_pair(model, config, args.layer, inputs,
            parent['metadata']['reference_environments'][str(args.layer)], observer=observe)
        metadata['reference_environment'] = env
        (output / 'captured.json').write_text(json.dumps(dict(references=refs,
            contributions=dict(zip(inputs, captured))), allow_nan=False))
        if len(captured) != len(inputs):
            raise ValueError('incomplete HF capture')
        hidden = config['hidden_size']
        gates, replays = [], {}
        for (name, values), contributions in zip(inputs.items(), captured):
            if set(refs[name]) != set(saved[name]):
                raise ValueError('HF observed field set changed')
            for field, expected in saved[name].items():
                gate = compare(f'{name}/hf_capture/{field}', refs[name][field], expected, len(expected), exact=True)
                gates.append(gate)
                rows.append(gate)
            replays[name] = replay_contributions(contributions, refs[name], values, hidden,
                                                config['num_experts_per_tok'], config['num_experts'])
            for field, values16 in replays[name]['fp16'].items():
                gate = compare(f'{name}/fp16_replay/{field}', values16, refs[name][field], hidden, exact=True)
                gates.append(gate)
                rows.append(gate)
        (output / 'replays.json').write_text(json.dumps(replays, allow_nan=False))
        metadata['attribution_valid'] = all(row['status'] == 'PASS' for row in gates)
        for row in gates:
            print(f'{row["status"]}: {row["case"]} max={row["max_abs_error"]:.9g} different={row["exact_different_elements"]}', flush=True)
        if not metadata['attribution_valid']:
            raise ValueError('HF capture/reduction replay gate failed; no attribution')

        metadata['native_routing_and_shared_match'] = {}
        for name, replay in replays.items():
            compatible = []
            topk, experts = config['num_experts_per_tok'], config['num_experts']
            for token in range(len(ids)):
                routes = []
                for record, wire in ((native[name], True), (refs[name], False)):
                    weights = record['routing_weights'][token*topk:(token+1)*topk]
                    if wire:
                        weights = restore_half_wire(weights)
                    selected = record['selected_experts'][token*topk:(token+1)*topk]
                    routes.append(dict(zip(selected, weights)))
                same_router = restore_half_wire(native[name]['router_logits'][token*experts:(token+1)*experts]) == \
                    refs[name]['router_logits'][token*experts:(token+1)*experts]
                same_shared = restore_half_wire(native[name]['shared_output'][token*hidden:(token+1)*hidden]) == \
                    refs[name]['shared_output'][token*hidden:(token+1)*hidden]
                compatible.append(same_router and same_shared and len(routes[0]) == topk and routes[0] == routes[1])
            metadata['native_routing_and_shared_match'][name] = compatible
            print(f'{name}: routing_and_shared_match_all={all(compatible)}; '
                  f'focus={dict((t, compatible[t]) for t in args.tokens)}', flush=True)
            for kind, values in replay.items():
                for field, observed in values.items():
                    expected = native[name][field]
                    if field == 'output':
                        expected = restore_half_wire(expected)
                    for token in [None, *args.tokens]:
                        begin, end = (0, len(observed)) if token is None else (token*hidden, (token+1)*hidden)
                        label = f'{name}/{kind}/{field}/' + ('all' if token is None else f'token{token}')
                        row = compare(label, observed[begin:end], expected[begin:end], hidden)
                        row.update(attribution_valid=True, counterfactual=kind == 'fp32', token_index=token,
                                   native_routing_and_shared_match=all(compatible) if token is None else compatible[token])
                        row['reason'] += f'; different={row["exact_different_elements"]}; compared with saved native'
                        if field == 'routed_output':
                            row['reason'] += '; native FP32 JSON is rounded to 12 decimals; unequal is not a bitwise claim'
                        rows.append(row)
                        if token is not None:
                            print(f'{row["status"]}: {label} max={row["max_abs_error"]:.9g} '
                                  f'mean={row["mean_abs_error"]:.9g} failed={row["failed_elements"]} '
                                  f'different={row["exact_different_elements"]}', flush=True)
        for path, expected_hash in hashes.items():
            if digest(Path(path)) != expected_hash:
                metadata['attribution_valid'] = False
                raise ValueError(f'paired evidence changed during replay: {path}')
    except Exception as error:
        metadata['attribution_valid'] = False
        (output / 'error.log').write_text(traceback.format_exc())
        rows.append(dict(case='reduction_probe_error', status='FAIL', failed_elements=1, reason=str(error)))
        print(f'FAIL: reduction_probe_error: {error}', flush=True)
    status = save_report(output, rows, metadata)
    print(f'attribution_valid={metadata["attribution_valid"]}', flush=True)
    print(f'{status}: {output}（保留历史失败；FP32重放为独立数值候选，非完整模型验收）', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
