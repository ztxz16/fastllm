#!/usr/bin/env python3
"""独立参考进程，仅加载本地权重，FP16/四卡按层分配；不导入FastLLM或修改其结果。"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import os


def reference_device_map(config):
    """/**
     * 将完整Decoder层连续分配到四卡，显式覆盖Embedding、RoPE与输出层。
     * 不使用根模块兜底映射，避免Accelerate递归保留GPU 0的子模块执行hook。
     * 绑定的Embedding与LM Head必须同卡；不足四层时拒绝生成分配方案。
     * @param config 本地Qwen2 MoE配置，提供层数与权重绑定标志。
     * @return 互不包含的模块路径到CUDA逻辑设备编号的映射。
     */"""
    count = config.num_hidden_layers
    if count < 4:
        raise ValueError('real reference requires at least four decoder layers')

    placement = {'model.embed_tokens': 0, 'model.rotary_emb': 0}
    placement.update({f'model.layers.{i}': i * 4 // count for i in range(count)})
    placement.update({'model.norm': 3, 'lm_head': 0 if config.tie_word_embeddings else 3})
    return placement


def check_reference_placement(model, placement):
    """/**
     * 在首次前向前核对所有参考权重、buffer与Accelerate执行hook的设备。
     * 只读检查，不迁移Tensor、不替换hook；设备冲突或未覆盖的Tensor立即报错。
     * @param model 已由Transformers与Accelerate完成分配的参考模型。
     * @param placement 互不包含的模块到CUDA逻辑设备编号的预期映射。
     * @return 已通过检查的模块设备记录，供日志与参考JSON保存。
     */"""
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        owners = [key for key in placement if name.startswith(key + '.')]
        if len(owners) != 1:
            raise RuntimeError(f'reference tensor {name}: expected one device-map owner, got {owners}')
        expected = f'cuda:{placement[owners[0]]}'
        if str(tensor.device) != expected:
            raise RuntimeError(f'reference tensor {name}: device={tensor.device}, expected={expected}')

    evidence = []
    for name, device in placement.items():
        block = model.get_submodule(name)
        expected = f'cuda:{device}'
        if getattr(block, '_hf_hook', None) is None:
            raise RuntimeError(f'reference module {name}: missing Accelerate device hook')
        for suffix, module in block.named_modules():
            pending = [getattr(module, '_hf_hook', None)]
            while pending:
                hook = pending.pop()
                pending.extend(getattr(hook, 'hooks', ()))
                execution = getattr(hook, 'execution_device', None)
                if execution is not None:
                    execution = f'cuda:{execution}' if isinstance(execution, int) else str(execution)
                    if execution != expected:
                        path = name + ('.' + suffix if suffix else '')
                        raise RuntimeError(f'reference hook {path}: execution_device={execution}, expected={expected}')
                if getattr(hook, 'offload', False):
                    raise RuntimeError(f'reference module {name}: CPU/disk offload is not allowed')
        evidence.append(dict(module=name, device=expected))
    return evidence


def moe_routing_observer(torch):
    """/**
     * 创建仅观察HF MoE实际算子返回值的线程局部上下文，不重算Top-K。
     * 每次调用原算子恰好一次并原样返回结果；保存实际专家ID、转型后的路由权重
     * 和index_add_后的累加值。算子序列或共享专家合成关系不符时拒绝生成参考。
     * @param torch 参考进程的PyTorch模块；主验收环境无需导入PyTorch。
     * @return 需配合with使用的观察器，支持gate处begin与MoE输出处finish。
     */"""
    from torch.overrides import TorchFunctionMode

    class Observer(TorchFunctionMode):
        def __init__(self):
            """/** 初始无活动MoE层，状态仅在一次完整MoE前向期间有效。 */"""
            super().__init__()
            self.active = None

        def begin(self, record, label):
            """/** 关联本层记录；拒绝嵌套或上层未结束的采集。 */"""
            if self.active is not None:
                raise RuntimeError(f'{label}: previous MoE observation did not finish')
            self.active = dict(record=record, label=label, weights=None, topk=0, cast=0, adds=0)

        def __torch_function__(self, func, types, args=(), kwargs=None):
            """/** 原样执行一次公开算子，复制所需输出；不改动输入、返回值或累加精度。 */"""
            result = func(*args, **(kwargs or {}))
            state = self.active
            if state is None:
                return result

            record = state['record']
            if func is torch.topk:
                state['topk'] += 1
                state['weights'] = result[0]
                record['selected_experts'] = result[1].detach().clone()
            elif func is torch.Tensor.to and args and args[0] is state['weights']:
                state['cast'] += 1
                record['routing_weights'] = result.detach().clone()
            elif func is torch.Tensor.index_add_:
                state['adds'] += 1
                record['routed_output'] = result.detach().clone()
            return result

        def finish(self, output):
            """/** 核对实际路由覆盖与MoE合成结果，未识别的实现报出具体层及采集计数。 */"""
            state, self.active = self.active, None
            if state is None:
                raise RuntimeError('MoE output observed without router input')
            record, label = state['record'], state['label']
            if state['topk'] != 1 or state['cast'] != 1 or state['adds'] == 0:
                raise RuntimeError(f'{label}: unsupported HF routing operations: '
                                   f'topk={state["topk"]}, cast={state["cast"]}, index_add={state["adds"]}')
            expected_calls = record['selected_experts'].unique().numel()
            if state['adds'] != expected_calls:
                raise RuntimeError(f'{label}: expert accumulation count={state["adds"]}, expected={expected_calls}')

            # 在模型实际执行设备和dtype上合成共享专家，避免CPU sigmoid的舍入差异。
            shared = torch.sigmoid(record.pop('shared_gate')) * record.pop('shared')
            record['shared_output'] = shared
            actual = output[0] if isinstance(output, tuple) else output
            combined = (record['routed_output'] + shared).reshape_as(actual)
            if not torch.equal(combined, actual):
                raise RuntimeError(f'{label}: captured routed/shared outputs do not reconstruct HF MoE output; '
                                   f'shape={tuple(actual.shape)}, device={actual.device}, dtype={actual.dtype}')

    return Observer()


def layer_reference(model, torch, sequences=None, split_stages=False):
    """/**
     * 用HF真实完整前向记录各层输入、实际路由、共享/路由专家和层输出。
     * 支持Linear gate与ModuleList专家的逐专家index_add_路径；拒绝未识别布局。
     * 仅观察实际算子输出，不在CPU重选专家；退出或异常时卸载全部hook与观察器。
     * @param model 已分配设备的HF参考模型，所有Decoder层均须为MoE。
     * @param torch 参考环境的PyTorch模块。
     * @param sequences 四个rank的Token ID列表；默认使用原短序列，空列表表示空闲rank。
     * @param split_stages 是否额外采集Attention残差边界，生成Attention/独立MoE任务。
     * @return 逐层参考和同一输入；独立MoE使用HF的Attention后输入，不混入上游误差。
     */"""
    sequences = ([1, 2], [2, 3, 4], [3], [4, 5, 6, 7]) if sequences is None else sequences
    if len(sequences) != 4 or not any(sequences) or any(
            not isinstance(ids, (list, tuple)) or len(ids) > 2048 or any(
                type(token) is not int or token < 0 for token in ids) for ids in sequences):
        raise ValueError('layer reference requires four valid token sequences, at least one nonempty')
    layers = model.model.layers
    for index, layer in enumerate(layers):
        if not isinstance(getattr(layer.mlp, 'gate', None), torch.nn.Linear) or not isinstance(
                getattr(layer.mlp, 'experts', None), torch.nn.ModuleList):
            raise RuntimeError(f'layer={index}: reference requires HF Linear gate and ModuleList experts')
        if split_stages and not hasattr(layer, 'post_attention_layernorm'):
            raise RuntimeError(f'layer={index}: missing Attention residual boundary')
    references = [[] for _ in layers]
    for rank, ids in enumerate(sequences):
        if not ids:
            for rows in references:
                rows.append({key: [] for key in ('input', 'output', 'router_logits', 'selected_experts',
                            'routing_weights', 'routed_output', 'shared_output', 'attention_input')})
            continue
        records = [dict() for _ in layers]
        observer = moe_routing_observer(torch)
        with ExitStack() as hooks, observer, torch.inference_mode():
            for index, layer in enumerate(layers):
                record = records[index]
                label = f'layer={index} rank={rank} tokens={len(ids)}'
                def before(module, args, kwargs, record=record):
                    value = args[0] if args else kwargs['hidden_states']
                    record['input'] = value.detach().clone()
                def after_attention(module, args, kwargs, record=record):
                    value = args[0] if args else kwargs['hidden_states']
                    record['attention_input'] = value.detach().clone()
                def output(module, args, value, record=record):
                    record['output'] = (value[0] if isinstance(value, tuple) else value).detach().clone()
                def gate(module, args, value, record=record, label=label):
                    observer.begin(record, label)
                    record['router_logits'] = value.detach().clone()
                def moe_output(module, args, value):
                    observer.finish(value)
                def save(name, record):
                    def hook(module, args, value):
                        record[name] = value.detach().clone()
                    return hook
                hooks.callback(layer.register_forward_pre_hook(before, with_kwargs=True).remove)
                if split_stages:
                    hooks.callback(layer.post_attention_layernorm.register_forward_pre_hook(
                        after_attention, with_kwargs=True).remove)
                hooks.callback(layer.register_forward_hook(output).remove)
                hooks.callback(layer.mlp.gate.register_forward_hook(gate).remove)
                hooks.callback(layer.mlp.shared_expert.register_forward_hook(save('shared', record)).remove)
                hooks.callback(layer.mlp.shared_expert_gate.register_forward_hook(save('shared_gate', record)).remove)
                hooks.callback(layer.mlp.register_forward_hook(moe_output).remove)

            model(input_ids=torch.tensor([ids], device=model.get_input_embeddings().weight.device), use_cache=False)
        for index, record in enumerate(records):
            references[index].append({key: value.detach().cpu().reshape(-1).tolist() for key, value in record.items()})
    tasks = []
    stages = ((0, 'full'), (1, 'attention'), (2, 'moe')) if split_stages else ((0, 'full'),)
    for i, rows in enumerate(references):
        for stage, name in stages:
            tasks.append(dict(layer=i, hidden=model.config.hidden_size, stage=stage,
                name=f'real_layer_{i}_{name}' if split_stages else f'real_layer_{i}',
                inputs=[dict(position_begin=0, hidden=r['attention_input' if stage == 2 else 'input']) for r in rows],
                reference=[dict(output=r['attention_input']) if stage == 1 else
                           {k: v for k, v in r.items() if k not in ('input', 'attention_input')} for r in rows]))
    return tasks


def main():
    """/** 在参考环境重算完整历史最后位置logits，输出原始精度和运行版本。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer-inputs', type=Path,
                        help='可选的四rank Token列表；启用同输入Attention/MoE/完整层定位')
    args = parser.parse_args()
    os.environ.pop('USE_TORCH', None)
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if torch.cuda.device_count() < 4:
        raise RuntimeError('real reference requires four GPUs')
    placement = reference_device_map(config)
    print('Reference device map: ' + json.dumps(placement), flush=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, config=config, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float16, attn_implementation='eager',
        device_map=placement).eval()
    if any(str(v) in ('cpu', 'disk') for v in model.hf_device_map.values()):
        raise RuntimeError('reference must fit on four GPUs without CPU/disk offload')
    devices = check_reference_placement(model, placement)
    print('PASS: reference weights/buffers/execution hooks agree on all four GPUs', flush=True)

    rows = []
    histories = json.loads(args.inputs.read_text())
    for index, ids in enumerate(histories):
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=model.get_input_embeddings().weight.device), use_cache=False)
            rows.append(dict(ids=ids, logits=out.logits[0, -1].float().cpu().tolist()))
        print(f'Reference history {index + 1}/{len(histories)}: tokens={len(ids)}', flush=True)
    print('Reference: collecting per-layer intermediates', flush=True)
    sequences = json.loads(args.layer_inputs.read_text()) if args.layer_inputs else None
    intermediate = layer_reference(model, torch, sequences, split_stages=args.layer_inputs is not None)
    args.output.write_text(json.dumps(dict(rows=rows, layers=intermediate, metadata=dict(
        torch=torch.__version__, transformers=transformers.__version__, dtype='float16',
        attention='eager', use_cache=False, device_map=model.hf_device_map, checked_devices=devices,
        routing_capture='HF actual topk, dtype cast and index_add_; no CPU rerouting')), allow_nan=False))
    print(f'PASS: reference saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
