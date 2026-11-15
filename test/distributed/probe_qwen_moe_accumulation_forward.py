#!/usr/bin/env python3
"""完整HF前向的独立累加精度实验；保留原验收，不修改FastLLM或正式HF参考。"""

import argparse
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import traceback
import uuid

from probe_qwen_moe_forward_precision import compare, digest, hf_trajectory, load_trace
from qwen_moe_transformers_reference import (
    check_reference_placement, moe_routing_observer, reference_device_map,
)
from recheck_qwen_moe_forward_precision import recheck
from run_process_tp import ROOT, capture, save_report


def reduce_captured(torch, record, contributions):
    """/**
     * 用本次HF真实FP16专家贡献分别执行逐专家FP16和FP32累加。
     * 校验专家覆盖、行索引、乘权重与原FP16累加；FP32候选仍在加共享专家前转FP16。
     * 保持专家真实调用顺序，不模拟原生专家GEMM、rank分组或NCCL归约树。
     * @param torch 已安装的PyTorch模块，不调用编译入口。
     * @param record 本次公开MoE前向的路由、共享专家和FP16累加观测。
     * @param contributions 实际调用顺序的(expert, indices, weighted, raw)张量记录。
     * @return 经门禁验证的FP32累加值，以及FP16 routed+shared候选结果。
     */"""
    selected, weights = record['selected_experts'], record['routing_weights']
    routed, shared = record['routed_output'], record['shared_output']
    if routed.ndim != 2 or routed.dtype != torch.float16 or shared.dtype != routed.dtype or \
            shared.shape != routed.shape or selected.ndim != 2 or weights.shape != selected.shape or \
            weights.dtype != routed.dtype or selected.shape[0] != routed.shape[0]:
        raise ValueError('unsupported routed/shared shape or dtype')
    routes = selected.cpu().tolist()
    expected = {(token, expert) for token, experts in enumerate(routes) for expert in experts}
    if len(expected) != selected.numel():
        raise ValueError('duplicate selected expert')
    seen, experts_seen = set(), set()
    half = torch.zeros_like(routed)
    single = torch.zeros_like(routed, dtype=torch.float32)

    for expert, indices, weighted, raw in contributions:
        tokens = indices.cpu().tolist()
        if expert in experts_seen or not tokens or len(set(tokens)) != len(tokens) or \
                indices.ndim != 1 or weighted.shape != (len(tokens), routed.shape[1]) or \
                raw.shape != weighted.shape or weighted.dtype != torch.float16 or raw.dtype != torch.float16:
            raise ValueError('invalid expert contribution')
        experts_seen.add(expert)
        pairs = {(token, expert) for token in tokens}
        if not pairs <= expected or seen & pairs:
            raise ValueError('contribution does not match selected experts')
        seen.update(pairs)
        slots = (selected[indices] == expert).to(torch.int64).argmax(dim=-1)
        if not torch.equal(raw * weights[indices, slots, None], weighted) or \
                not bool(torch.isfinite(weighted).all()):
            raise ValueError('expert product or finite-value gate failed')
        half.index_add_(0, indices, weighted)
        single.index_add_(0, indices, weighted.float())
    if seen != expected or not torch.equal(half, routed):
        raise ValueError('FP16 contribution replay does not reproduce actual HF accumulation')
    candidate = single.to(torch.float16) + shared
    if not bool(torch.isfinite(candidate).all()):
        raise ValueError('nonfinite FP32 accumulation candidate')
    return single, candidate


def observed_forward(model, torch, ids, *, fp32=False):
    """/**
     * 沿公开完整模型前向捕获真实路由贡献，候选仅替换MoE的专家累加结果。
     * Norm、Attention、Top-K、专家GEMM、乘权重、共享专家及层残差均调用原HF路径。
     * 每层先验证原FP16合成与重放；候选结果再交回原层，后续层自然接收改变后的输入。
     * 成功和异常均移除hook及线程局部Mode，不改模型权重或全局Torch算子。
     * @param model 已加载并eval的FP16完整模型，须有Linear gate与ModuleList专家。
     * @param torch 已安装的参考PyTorch模块。
     * @param ids 单个完整历史的Token ID，保持[1,T]、eager和use_cache=False。
     * @param fp32 false仅观察原版HF；true启用独立FP32累加候选。
     * @return 原始精度的末Token logits、完整隐藏状态轨迹和逐层路由记录。
     */"""
    from torch.overrides import TorchFunctionMode
    layers = model.model.layers
    records, trajectory = [dict() for _ in layers], {}
    observer = moe_routing_observer(torch)
    state = dict(contributions=[], pending=[], target=None)

    class Capture(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            """/** 原index_add_照常执行，只捕获当前MoE专家的实际半精度贡献。 */"""
            is_add = func is torch.Tensor.index_add_ and observer.active is not None
            if is_add:
                if len(args) != 4 or args[1] != 0 or (kwargs or {}).get('alpha', 1) != 1 or \
                        len(state['pending']) != 1:
                    raise ValueError('unsupported expert index_add_ call')
                target, _, indices, weighted = args
                if state['target'] is None:
                    if bool(torch.count_nonzero(target)):
                        raise ValueError('nonzero initial routed accumulator')
                    state['target'] = target
                if state['target'] is not target:
                    raise ValueError('routed accumulator changed within layer')
            result = func(*args, **(kwargs or {}))
            if is_add:
                expert, raw = state['pending'].pop()
                state['contributions'].append((expert, indices.detach().clone(), weighted.detach().clone(), raw))
            return result

    def expert_hook(expert):
        """/** 关联实际专家返回值；拒绝未消费的专家输出。 */"""
        def after(module, args, output):
            """/** 保留原行顺序和FP16数值，不重算专家GEMM。 */"""
            if state['pending']:
                raise ValueError('unconsumed expert output')
            state['pending'].append((expert, output.detach().clone()))
        return after

    def save(record, name, *, before=False):
        """/** 返回只读Tensor hook，兼容层tuple输出和Attention残差输入。 */"""
        def hook(module, args, value):
            """/** 前置hook的value是kwargs，后置hook的value是实际输出。 */"""
            tensor = (args[0] if args else value['hidden_states']) if before else value
            record[name] = (tensor[0] if isinstance(tensor, tuple) else tensor).detach().clone()
        return hook

    def gate_hook(record, index):
        """/** 在当前层gate实际返回后开始观察，不改变路由计算。 */"""
        def after(module, args, value):
            """/** 上一层必须完整结束，才能初始化下一层贡献捕获。 */"""
            if state['contributions'] or state['pending'] or state['target'] is not None:
                raise ValueError('previous layer capture did not finish')
            observer.begin(record, f'layer={index}')
            record['router_logits'] = value.detach().clone()
        return after

    def moe_hook(record):
        """/** 校验原MoE结果，仅候选模式向原层交回FP32累加后的FP16输出。 */"""
        def after(module, args, output):
            """/** 先终止观察，再重放，避免将实验算子误收录为正式专家贡献。 */"""
            observer.finish(output)
            if state['pending'] or not isinstance(output, tuple) or len(output) != 2:
                raise ValueError('incomplete capture or unsupported HF MoE return type')
            single, candidate = reduce_captured(torch, record, state['contributions'])
            record['fp32_routed_output'] = single.detach().clone()
            state.update(contributions=[], pending=[], target=None)
            return (candidate.reshape_as(output[0]), output[1]) if fp32 else None
        return after

    with ExitStack() as hooks, observer, Capture(), torch.inference_mode():
        for index, (layer, record) in enumerate(zip(layers, records)):
            block = layer.mlp
            if not isinstance(block.gate, torch.nn.Linear) or not isinstance(block.experts, torch.nn.ModuleList):
                raise ValueError('requires HF Linear router and ModuleList experts')
            hooks.callback(layer.register_forward_pre_hook(save(record, 'input', before=True), with_kwargs=True).remove)
            hooks.callback(layer.post_attention_layernorm.register_forward_pre_hook(
                save(record, 'attention_input', before=True), with_kwargs=True).remove)
            hooks.callback(layer.register_forward_hook(save(record, 'output')).remove)
            hooks.callback(block.gate.register_forward_hook(gate_hook(record, index)).remove)
            hooks.callback(block.shared_expert.register_forward_hook(save(record, 'shared')).remove)
            hooks.callback(block.shared_expert_gate.register_forward_hook(save(record, 'shared_gate')).remove)
            for expert, module in enumerate(block.experts):
                hooks.callback(module.register_forward_hook(expert_hook(expert)).remove)
            hooks.callback(block.register_forward_hook(moe_hook(record)).remove)
        hooks.callback(model.model.norm.register_forward_hook(save(trajectory, 'final_norm')).remove)
        out = model(input_ids=torch.tensor([ids], device=model.get_input_embeddings().weight.device), use_cache=False)
    if observer.active is not None or state['pending'] or state['contributions']:
        raise ValueError('unfinished full-forward capture')
    values = [{key: tensor.detach().cpu().reshape(-1).tolist() for key, tensor in record.items()} for record in records]
    hidden = model.config.hidden_size
    trajectory = dict(embedding=values[0]['input'], final_norm=trajectory['final_norm'].cpu().reshape(-1).tolist()[-hidden:])
    for index, record in enumerate(values):
        trajectory[f'layer_{index}_attention'] = record['attention_input']
        trajectory[f'layer_{index}_moe'] = record['output']
    return dict(logits=out.logits[0, -1].float().cpu().tolist(), trajectory=trajectory, layers=values)


def load_evidence(directory, model):
    """/**
     * 复核历史完整轨迹和原阈值，记录证据哈希；任何失配均拒绝后续实验。
     * 直接比较保存的logits与完整轨迹，不依赖此前输出头独立重放的归因结论。
     * @param directory 原forward_probe目录，包含完整四rank原生采集证据。
     * @param model 本地模型目录，配置形状必须与保存的FP16轨迹及词表一致。
     * @return 已复核的原报告、原生/HF轨迹、logits、配置和证据哈希。
     */"""
    report = json.loads((directory / 'summary.json').read_text())
    _, _, checked = recheck(directory, report)
    if not checked['trace_attribution_valid']:
        raise ValueError('saved native trace attribution gate failed')
    meta = report['metadata']
    config = json.loads((model / 'config.json').read_text())
    ids = meta['target']['ids']
    if config.get('model_type') != 'qwen2_moe' or config['hidden_size'] != meta['trace']['hidden'] or \
            config['num_hidden_layers'] != meta['trace']['layers']:
        raise ValueError('model config differs from original trace')
    _, native = load_trace(directory / 'trace', ids, config['hidden_size'], config['num_hidden_layers'])
    reference = json.loads((Path(meta['diagnostic']) / 'reference.json').read_text())
    hf = hf_trajectory(reference, meta['source_rank'], len(ids), config['hidden_size'], config['num_hidden_layers'])
    full = json.loads((Path(meta['source_dir']) / 'transformers-reference.json').read_text())
    logits = next(row['logits'] for row in full['rows'] if row['ids'] == ids)
    if len(logits) != config['vocab_size']:
        raise ValueError('model vocabulary differs from original reference')
    files = [directory / 'summary.json', model / 'config.json', *list((directory / 'trace').iterdir())]
    hashes = dict(checked['artifact_sha256'], **{str(path): digest(path) for path in files})
    return dict(report=report, config=config, ids=ids, hf=hf, native=native, reference=reference,
                logits=logits, native_logits=json.loads((directory / 'native-traced.json').read_text()), hashes=hashes)


def load_model(model_path, expected):
    """/**
     * 仅加载本地FP16/eager四卡模型；版本或设备不符即拒绝，不下载或编译。
     * 沿用正式参考的整层分卡映射，检查Accelerate执行设备并保存实际运行环境。
     * @param model_path 本地完整模型目录。
     * @param expected 历史参考的torch、transformers和dtype版本记录。
     * @return torch模块、已eval的完整模型和实际参考环境记录。
     */"""
    os.environ.pop('USE_TORCH', None)
    import torch
    import transformers
    versions = dict(torch=torch.__version__, transformers=transformers.__version__, dtype='float16')
    if torch.cuda.device_count() != 4 or transformers.__version__ != '4.57.6' or \
            any(versions[key] != expected[key] for key in versions):
        raise ValueError(f'reference environment differs: {versions}')
    config = transformers.AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    placement = reference_device_map(config)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, config=config, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float16, attn_implementation='eager', device_map=placement).eval()
    versions.update(cuda=torch.version.cuda, devices=check_reference_placement(model, placement),
        allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    return torch, model, versions


def main():
    """/** 原HF精确门禁通过才运行完整候选；保留历史FAIL、原阈值及四格式报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_accumulation_forward_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    print(f'独立完整前向累加实验目录：{output}', flush=True)
    rows, meta = [], dict(command=shlex.join([sys.executable, *sys.argv]),
        git=capture(['git', 'rev-parse', 'HEAD']), gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), warmup=0, iterations=1, repeats=1, atol=.05, rtol=.01,
        model=str(args.model.resolve()), probe=str(args.probe.resolve()), attribution_valid=False,
        report_title='完整HF前向的独立FP32专家累加实验',
        report_description='候选仍用HF专家GEMM及实际路由，只改变专家贡献累加。'
        '不模拟NCCL归约树，不替换正式参考、不豁免原失败；非完整模型验收或性能测试。')

    def record(label, actual, expected, width, *, exact=False, candidate=False, verbose=True):
        """/** 原阈值比较和零容差门禁分开记录，候选必须显式标识。 */"""
        row = compare(label, actual, expected, width, exact=exact)
        row.update(counterfactual=candidate, backend='independent_hf_accumulation_forward')
        rows.append(row)
        if verbose or row['status'] != 'PASS':
            print(f'{row["status"]}: {label} max={row["max_abs_error"]:.9g} '
                  f'mean={row["mean_abs_error"]:.9g} failed={row["failed_elements"]} '
                  f'different={row["exact_different_elements"]}', flush=True)
        return row['status'] == 'PASS'

    try:
        data = load_evidence(args.probe.resolve(), args.model.resolve())
        rows.extend(copy.deepcopy(data['report']['cases']))
        source_meta = data['report']['metadata']
        meta.update(source_sha256=data['hashes'], source_git=source_meta['git'], ids=data['ids'],
                    build_cache=source_meta['build_cache'], source_head_attribution_valid=source_meta.get('head_attribution_valid'))
        save_report(output, rows, meta)
        torch, model, versions = load_model(args.model.resolve(), data['reference']['metadata'])
        meta['reference_environment'] = versions
        hidden, vocab = data['config']['hidden_size'], data['config']['vocab_size']
        with torch.inference_mode():
            plain = model(input_ids=torch.tensor([data['ids']], device=model.get_input_embeddings().weight.device), use_cache=False)
            plain = plain.logits[0, -1].float().cpu().tolist()
        gates = [record('gate/plain_hf_vs_saved', plain, data['logits'], vocab, exact=True)]
        baseline = observed_forward(model, torch, data['ids'])
        (output / 'baseline.json').write_text(json.dumps(baseline, allow_nan=False))
        gates.append(record('gate/observer_vs_plain', baseline['logits'], plain, vocab, exact=True))
        for stage, expected in data['hf'].items():
            gates.append(record('gate/' + stage, baseline['trajectory'][stage], expected, hidden, exact=True, verbose=False))
        rank = source_meta['source_rank']
        for index, observed in enumerate(baseline['layers']):
            saved = next(task for task in data['reference']['layers'] if task['name'] == f'real_layer_{index}_moe')['reference'][rank]
            for field, expected in saved.items():
                gates.append(record(f'gate/layer{index}/{field}', observed[field], expected, len(expected), exact=True, verbose=False))
        print(f'原HF完整轨迹、路由与logits精确门禁：{all(gates)}', flush=True)
        save_report(output, rows, meta)
        if not all(gates):
            raise ValueError('original HF replay/observation gate failed; no candidate attribution')

        print('运行：所有MoE层FP32专家贡献累加候选', flush=True)
        candidate = observed_forward(model, torch, data['ids'], fp32=True)
        (output / 'candidate.json').write_text(json.dumps(candidate, allow_nan=False))
        with torch.inference_mode():
            after = model(input_ids=torch.tensor([data['ids']], device=model.get_input_embeddings().weight.device), use_cache=False)
            after = after.logits[0, -1].float().cpu().tolist()
        if not record('gate/plain_hf_after_candidate', after, plain, vocab, exact=True):
            raise ValueError('candidate changed subsequent original HF forward')
        record('native_vs_original_hf/logits', data['native_logits'], plain, vocab)
        record('native_vs_fp32_candidate/logits', data['native_logits'], candidate['logits'], vocab, candidate=True)
        record('fp32_candidate_vs_original_hf/logits', candidate['logits'], plain, vocab, candidate=True)
        meta['routing_changes'] = []
        topk = data['config']['num_experts_per_tok']
        for index, (base, changed) in enumerate(zip(baseline['layers'], candidate['layers'])):
            changed_tokens = [token for token in range(len(data['ids'])) if
                set(base['selected_experts'][token*topk:(token+1)*topk]) !=
                set(changed['selected_experts'][token*topk:(token+1)*topk])]
            meta['routing_changes'].append(dict(layer=index, changed_expert_set_tokens=changed_tokens))
            if changed_tokens:
                print(f'layer{index}: HF候选相对原HF的专家集合变化Token={changed_tokens}', flush=True)
        focus_stages = {'final_norm', *(f'layer_{i}_moe' for i in (0, 9, 12, 13, 15, len(baseline['layers']) - 1))}
        for stage, native in data['native'].items():
            for name, values in (('original_hf', baseline), ('fp32_candidate', candidate)):
                record(f'native_vs_{name}/{stage}', native, values['trajectory'][stage], hidden,
                       candidate=name == 'fp32_candidate', verbose=stage in focus_stages)
        for path, expected in data['hashes'].items():
            if digest(Path(path)) != expected:
                raise ValueError(f'source evidence changed during experiment: {path}')
        meta['attribution_valid'] = True
    except Exception as error:
        (output / 'error.log').write_text(traceback.format_exc())
        rows.append(dict(case='accumulation_forward_error', backend='independent_hf_accumulation_forward',
                         status='FAIL', max_abs_error=None, mean_abs_error=None, failed_elements=1, reason=str(error)))
        print(f'FAIL: accumulation_forward_error: {error}', flush=True)
    status = save_report(output, rows, meta)
    print(f'attribution_valid={meta["attribution_valid"]}', flush=True)
    print(f'{status}: {output}（保留原失败；独立累加精度实验，非完整模型验收）', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
