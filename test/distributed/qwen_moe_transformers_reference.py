#!/usr/bin/env python3
"""独立参考进程，仅加载本地权重，FP16/四卡按层分配；不导入FastLLM或修改其结果。"""

import argparse
import json
from pathlib import Path
import os


def layer_reference(model, torch):
    """/**
     * 用HF真实完整前向的hook记录每层输入、Router、共享/路由专家及层输出。
     * 兼容ModuleList专家与新版集合专家；只读观察，不替换forward或修改输入。
     * @return 四个短序列的逐层参考，以及供正式单层公开入口使用的同一输入。
     */"""
    layers = model.model.layers
    references = [[] for _ in layers]
    for rank, ids in enumerate(([1, 2], [2, 3, 4], [3], [4, 5, 6, 7])):
        records = [dict() for _ in layers]
        handles = []
        for index, layer in enumerate(layers):
            record = records[index]
            def before(module, args, kwargs, record=record):
                value = args[0] if args else kwargs['hidden_states']
                record['input'] = value.detach().cpu()
            def output(module, args, value, record=record):
                record['output'] = (value[0] if isinstance(value, tuple) else value).detach().cpu()
            def gate(module, args, value, record=record):
                logits = (value[0] if isinstance(value, tuple) else value).detach().cpu()
                probs = torch.softmax(logits.float(), dim=-1)
                selected, expert_ids = torch.topk(probs, model.config.num_experts_per_tok, dim=-1)
                if model.config.norm_topk_prob:
                    selected = selected / selected.sum(dim=-1, keepdim=True)
                record.update(router_logits=logits, routing_weights=selected.half(), selected_experts=expert_ids)
            def save(name, record):
                def hook(module, args, value):
                    record[name] = value.detach().cpu()
                return hook
            handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(layer.register_forward_hook(output))
            handles.append(layer.mlp.gate.register_forward_hook(gate))
            handles.append(layer.mlp.shared_expert.register_forward_hook(save('shared', record)))
            handles.append(layer.mlp.shared_expert_gate.register_forward_hook(save('shared_gate', record)))
            if isinstance(layer.mlp.experts, torch.nn.ModuleList):
                record['routed_output'] = torch.zeros(len(ids), model.config.hidden_size, dtype=torch.float32)
                for expert_id, expert in enumerate(layer.mlp.experts):
                    def expert_output(module, args, value, record=record, expert_id=expert_id):
                        slot, token = torch.where(record['selected_experts'].T == expert_id)
                        values = value.detach().cpu()
                        if len(values) != len(token):
                            raise RuntimeError('HF expert token order/shape changed')
                        weighted = (values * record['routing_weights'][token, slot, None]).half().float()
                        record['routed_output'].index_add_(0, token, weighted)
                    handles.append(expert.register_forward_hook(expert_output))
            else:
                handles.append(layer.mlp.experts.register_forward_hook(save('routed_output', record)))
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=model.get_input_embeddings().weight.device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        for index, record in enumerate(records):
            shared = (torch.sigmoid(record['shared_gate']) * record['shared']).half()
            record['shared_output'] = shared
            references[index].append({key: value.reshape(-1).tolist() for key, value in record.items()
                                      if key not in ('shared', 'shared_gate')})
    return [dict(layer=i, hidden=model.config.hidden_size, stage=0, name=f'real_layer_{i}',
                 inputs=[dict(position_begin=0, hidden=r['input']) for r in rows],
                 reference=[{k: v for k, v in r.items() if k != 'input'} for r in rows])
            for i, rows in enumerate(references)]


def main():
    """/** 在参考环境重算完整历史最后位置logits，输出原始精度和运行版本。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.pop('USE_TORCH', None)
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if torch.cuda.device_count() < 4 or config.num_hidden_layers < 4:
        raise RuntimeError('real reference requires four GPUs and at least four layers')
    # 显式按连续层分到四卡，避免auto在32GiB卡上将全部权重塞进单卡。
    placement = {'': 0, 'model.norm': 3, 'lm_head': 0 if config.tie_word_embeddings else 3}
    placement.update({f'model.layers.{i}': i*4//config.num_hidden_layers for i in range(config.num_hidden_layers)})
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, config=config, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float16, attn_implementation='eager',
        device_map=placement).eval()
    if any(str(v) in ('cpu', 'disk') for v in model.hf_device_map.values()):
        raise RuntimeError('reference must fit on four GPUs without CPU/disk offload')
    rows = []
    for ids in json.loads(args.inputs.read_text()):
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=model.get_input_embeddings().weight.device), use_cache=False)
            rows.append(dict(ids=ids, logits=out.logits[0, -1].float().cpu().tolist()))
    intermediate = layer_reference(model, torch)
    args.output.write_text(json.dumps(dict(rows=rows, layers=intermediate, metadata=dict(
        torch=torch.__version__, transformers=transformers.__version__, dtype='float16',
        attention='eager', use_cache=False, device_map=model.hf_device_map)), allow_nan=False))


if __name__ == '__main__':
    main()
