#!/usr/bin/env python3
"""采集正式完整前向的隐藏状态轨迹；独立定位累积误差和输出头，不编译或改变验收阈值。"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import struct
import sys
import traceback
import uuid

from run_process_tp import ROOT, capture, save_report
from run_qwen_moe_precision_diagnostic import ATOL, RTOL, difference, failing_history
from fastllm_pytools.qwen_moe_process_session import QwenMoeProcessSession

TRACE_ENV = 'FASTLLM_MOE_TRACE_DIR'
BACKEND = 'independent_formal_eager_forward_trace'


def digest(path):
    """/** 流式记录已有文件SHA-256，不加载二进制到内存，不使用Python 3.11专有接口。 */"""
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def stage_names(layers):
    """/** 返回真实前向的顺序边界，不把独立子层参考拼接成原生计算。 */"""
    return ['embedding', *[f'layer_{i}_{stage}' for i in range(layers)
                          for stage in ('attention', 'moe')], 'final_norm']


def load_trace(directory, ids, hidden, layers):
    """/**
     * 校验单个活动rank的完整FP16轨迹，拒绝缺失、重复、越界、非有限值或历史不一致。
     * @param directory 本轮新建目录，必须恰有一个JSON描述和一个二进制文件。
     * @param ids 本轮正式请求的完整Token ID列表，强制rank0且从位置零开始。
     * @param hidden 隐藏宽度；普通阶段形状为[len(ids),hidden]，final_norm为[1,hidden]。
     * @param layers 必须包含的Decoder层数。
     * @return 原始描述及按计算顺序排列的FP16精确值列表，不裁剪或重排。
     */"""
    paths = list(directory.glob('*.json'))
    if len(paths) != 1:
        raise ValueError(f'expected one completed rank0 trace, got {len(paths)}')
    meta = json.loads(paths[0].read_text())
    expected = dict(version=1, dtype='float16', byte_order='little', rank=0,
                    tokens=len(ids), hidden=hidden, layers=layers, ids=ids)
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError('trace metadata/history/shape mismatch')
    binary = directory / meta['binary']
    if Path(meta['binary']).name != meta['binary'] or binary.resolve().parent != directory.resolve():
        raise ValueError('trace binary escapes directory')
    if set(directory.iterdir()) != {paths[0], binary}:
        raise ValueError('unexpected or incomplete trace files')
    count = (1 + 2*layers) * len(ids) * hidden + hidden
    if count <= 0 or count*2 > 32*1024*1024 or binary.stat().st_size != count*2:
        raise ValueError('invalid trace binary size')
    data = binary.read_bytes()
    entries = meta['snapshots']
    if [entry.get('name') for entry in entries] != stage_names(layers):
        raise ValueError('incomplete or unordered trace stages')
    result, offset = {}, 0
    for entry in entries:
        size = hidden if entry['name'] == 'final_norm' else len(ids)*hidden
        if type(entry.get('offset')) is not int or type(entry.get('elements')) is not int or \
                entry['offset'] != offset or entry['elements'] != size:
            raise ValueError('invalid trace offset/elements')
        values = list(struct.unpack_from(f'<{size}e', data, offset*2))
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f'nonfinite trace: {entry["name"]}')
        result[entry['name']] = values
        offset += size
    meta.update(binary_sha256=digest(binary), manifest_sha256=digest(paths[0]))
    return meta, result


def hf_trajectory(reference, rank, tokens, hidden, layers):
    """/** 读取同一次HF完整前向的真实边界，逐层验证输入连续性和Attention/MoE对应关系。 */"""
    tasks = reference['layers']
    lookup = {task['name']: task for task in tasks}
    required = {f'real_layer_{i}_{stage}' for i in range(layers) for stage in ('full', 'attention', 'moe')}
    if set(lookup) != required or len(tasks) != len(required):
        raise ValueError('incomplete or duplicate HF layer reference')
    result = {}
    for layer in range(layers):
        full, attention, moe = [lookup[f'real_layer_{layer}_{stage}'] for stage in ('full', 'attention', 'moe')]
        before = full['inputs'][rank]['hidden']
        attended = attention['reference'][rank]['output']
        after = full['reference'][rank]['output']
        values = [before, attended, after]
        if any(len(v) != tokens*hidden or not all(math.isfinite(x) for x in v) for v in values):
            raise ValueError(f'layer {layer}: invalid HF trajectory shape/value')
        if (before != attention['inputs'][rank]['hidden'] or attended != moe['inputs'][rank]['hidden'] or
                after != moe['reference'][rank]['output'] or
                (layer and before != result[f'layer_{layer-1}_moe'])):
            raise ValueError(f'layer {layer}: disconnected HF trajectory')
        if not layer:
            result['embedding'] = before
        result[f'layer_{layer}_attention'], result[f'layer_{layer}_moe'] = attended, after
    return result


def compare(label, actual, expected, width, exact=False):
    """/** 保存全张量及末Token误差；只有重放门禁使用零容差，数值诊断保留原验收阈值。 */"""
    if width <= 0 or not actual or len(actual) % width or len(actual) != len(expected):
        raise ValueError(f'{label}: invalid comparison shape')
    if any(not math.isfinite(v) for v in [*actual, *expected]):
        raise ValueError(f'{label}: nonfinite comparison')
    metrics = difference(actual, expected)
    different = sum(a != b for a, b in zip(actual, expected))
    if exact:
        metrics.update(status='FAIL' if different else 'PASS', failed_elements=different)
    return dict(case=label, backend=BACKEND, **metrics, exact_different_elements=different,
                shape=[len(actual)//width, width], last_token=difference(actual[-width:], expected[-width:]),
                reason=f'shape={[len(actual)//width, width]}; ' +
                       ('exact replay gate' if exact else f'atol={ATOL}; rtol={RTOL}'))


def native_forward(args, output, original, ids, trace_directory=None):
    """/** 正式四rank会话只提交一个完整历史请求；用子进程环境开启采集，不修改调用者环境。 */"""
    request = copy.deepcopy(original['request'])
    request.update(token_ids=ids, max_new_tokens=1, eos_token_ids=[])
    environment = dict(os.environ)
    environment.pop(TRACE_ENV, None)
    if trace_directory is not None:
        environment[TRACE_ENV] = str(trace_directory.resolve())
    with QwenMoeProcessSession(args.build / 'fastllm_moe_generate', args.build / 'fastllm_tp_worker',
            args.model, '0,1,2,3', output, timeout=args.timeout, trace=True,
            cuda_graph=False, prefix_cache=False, environment=environment) as session:
        report = session.generate(request)
    if report['status'] != 'PASS' or report['dp_rank'] != 0 or len(report['steps']) != 1:
        raise RuntimeError('incomplete native full-history request')
    step = report['steps'][0]
    if step['execution'] != 'eager' or step.get('joint_round') != dict(
            actions=[1, 0, 0, 0], submitted_token_counts=[len(ids), 0, 0, 0]):
        raise RuntimeError('unexpected native execution/joint round')
    return step['logits']


def head_values(torch, norm, weight, hf_hidden, native_hidden, native_norm, gemm):
    """/**
     * 独立重放最终归一化与输出投影，返回全部候选，不以候选替换正式输出。
     * @param norm HF原版RMSNorm，权重及dtype与参考一致。
     * @param weight HF FP16 [vocab,hidden]，与各输入同设备。
     * @param hf_hidden 同一次HF完整前向的FP16 [tokens,hidden]末层残差。
     * @param native_hidden 正式原生前向的FP16 [tokens,hidden]末层残差。
     * @param native_norm 正式Norm的FP16 [1,hidden]末Token输出。
     * @param gemm 独立cuBLAS调用器，匹配原生FP32输出LM Head参数。
     * @return 候选张量；保留HF全行/末行GEMM形状差异和原生实际归一化边界。
     */"""
    with torch.inference_mode():
        hf_norm = norm(hf_hidden)
        hf_norm_on_native = norm(native_hidden[-1:].contiguous())
        return dict(hf_norm=hf_norm[-1:], hf_norm_on_native=hf_norm_on_native,
            hf_head_full=torch.nn.functional.linear(hf_norm, weight)[-1:],
            hf_head_last=torch.nn.functional.linear(hf_norm[-1:].contiguous(), weight),
            native_head_on_hf=gemm.linear(torch, hf_norm[-1:].contiguous(), weight, fp32=True),
            native_head_replay=gemm.linear(torch, native_norm, weight, fp32=True),
            hf_head_on_native_norm=torch.nn.functional.linear(native_norm, weight))


def head_probe(args, config, hf, native, output):
    """/** 仅加载最终Norm和LM Head到HF原设备；所有原生会话已退出，不再加载完整模型。 */"""
    os.environ.pop('USE_TORCH', None)
    import torch
    import transformers
    from safetensors import safe_open
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeRMSNorm
    from probe_qwen_moe_attention_precision import MoeGemm
    from probe_qwen_moe_router_precision import linked_cublas

    if torch.cuda.device_count() != 4 or transformers.__version__ != '4.57.6':
        raise ValueError('head probe requires four GPUs and the recorded Transformers 4.57.6 reference')
    if config.get('tie_word_embeddings'):
        raise ValueError('this head probe requires the recorded untied Qwen MoE head on cuda:3')
    device = torch.device('cuda', 3)
    torch.cuda.set_device(device)
    mapping = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']

    def weight(name):
        """/** 只读索引中的单个FP16权重，禁止逃逸模型目录或自动下载。 */"""
        path = (args.model / mapping[name]).resolve()
        if not path.is_relative_to(args.model):
            raise ValueError('weight shard escapes model directory')
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            return handle.get_tensor(name).to(device=device, dtype=torch.float16).contiguous()

    hidden, layers = config['hidden_size'], config['num_hidden_layers']
    norm = Qwen2MoeRMSNorm(hidden, eps=config['rms_norm_eps']).to(device=device, dtype=torch.float16).eval()
    norm.weight = torch.nn.Parameter(weight('model.norm.weight'), requires_grad=False)
    head = weight('lm_head.weight')
    if tuple(head.shape) != (config['vocab_size'], hidden) or tuple(norm.weight.shape) != (hidden,):
        raise ValueError('invalid final norm/head weight shape')
    final = f'layer_{layers-1}_moe'
    tensors = [torch.tensor(values, dtype=torch.float16, device=device).reshape(-1, hidden)
               for values in (hf[final], native[final], native['final_norm'])]
    library = linked_cublas(args.build / 'fastllm_tp_worker')
    gemm = MoeGemm(library, torch.cuda.current_stream().cuda_stream)
    try:
        values = head_values(torch, norm, head, *tensors, gemm)
        result = {key: value.float().cpu().reshape(-1).tolist() for key, value in values.items()}
        metadata = dict(torch=torch.__version__, transformers=transformers.__version__, device=str(device),
            cuda=torch.version.cuda, cublas=str(library), cublas_version=gemm.version,
            capability=torch.cuda.get_device_capability(device),
            allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            description='Independent final norm/head replay, not a replacement backend or performance result.')
        (output / 'head-replay.json').write_text(json.dumps(dict(metadata=metadata, values=result), allow_nan=False))
        return metadata, result
    finally:
        gemm.close()


def main():
    """/** 保留历史失败，先检查采集不扰动结果，再比较真实轨迹及输出头；失败也保存四格式报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, default=Path('build-attention-ep-cu126'))
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    args.model, args.build, args.diagnostic = args.model.resolve(), args.build.resolve(), args.diagnostic.resolve()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_forward_probe_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    print(f'独立诊断目录：{output}', flush=True)
    rows = []
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), warmup=0, iterations=1, repeats=1, atol=ATOL, rtol=RTOL,
        diagnostic=str(args.diagnostic), model=str(args.model), controlled_rank=0,
        report_title='MoE正式完整前向轨迹与输出头诊断',
        report_description='保留历史失败。仅rank0完整历史Eager请求，其余rank参与EP。'
                           '不是原调度重放、完整模型验收或性能测试；不改变原阈值。')

    def record(label, actual, expected, width, exact=False):
        """/** 每项立即输出并保存，包括精确差异数与末Token误差，避免后续异常丢失证据。 */"""
        row = compare(label, actual, expected, width, exact)
        rows.append(row)
        save_report(output, rows, metadata)
        print(f'{row["status"]}: {label}: max_abs={row["max_abs_error"]:.9g} '
              f'mean_abs={row["mean_abs_error"]:.9g} failed={row["failed_elements"]} '
              f'exact_diff={row["exact_different_elements"]} last_max={row["last_token"]["max_abs_error"]:.9g}', flush=True)
        return row['status'] == 'PASS'

    try:
        if os.environ.get(TRACE_ENV):
            raise ValueError(f'unset {TRACE_ENV}; the probe sets it only for its traced child session')
        config = json.loads((args.model / 'config.json').read_text())
        hidden, layers, vocab = [config[key] for key in ('hidden_size', 'num_hidden_layers', 'vocab_size')]
        if config.get('model_type') != 'qwen2_moe':
            raise ValueError('probe requires Qwen2 MoE')
        source = json.loads((args.diagnostic / 'summary.json').read_text())
        source_meta = source['metadata']
        source_dir = Path(source_meta['source_dir'])
        original = json.loads((source_dir / source_meta['request']).read_text())
        full_reference = json.loads((source_dir / 'transformers-reference.json').read_text())
        target = failing_history(original, full_reference)
        ids, rank = target['ids'], original['dp_rank']
        if ids != source_meta['target']['ids'] or rank != source_meta['rank'] or not 0 <= rank < 4:
            raise ValueError('source history/rank disagrees with layer diagnostic')
        sequences = json.loads((args.diagnostic / 'layer-inputs.json').read_text())
        if sequences != [ids if i == rank else [] for i in range(4)]:
            raise ValueError('saved HF layer inputs disagree with source history')
        reference = json.loads((args.diagnostic / 'reference.json').read_text())
        hf = hf_trajectory(reference, rank, len(ids), hidden, layers)
        saved_hf = next(row['logits'] for row in full_reference['rows'] if row['ids'] == ids)
        metadata.update(source_dir=str(source_dir), source_git=source_meta.get('git'), target=target,
            source_rank=rank, reference=reference['metadata'], build_cache=(args.build / 'CMakeCache.txt').read_text(),
            binary_sha256={name: digest(args.build / name) for name in ('fastllm_moe_generate', 'fastllm_tp_worker')})
        previous = [row for row in source['cases'] if row['status'] != 'PASS' and row['case'] != 'saved_full_model_logits']
        metadata['previous_independent_failures'] = previous
        print(f'此前独立子层非PASS项：{len(previous)}；completed_tasks={source_meta.get("completed_tasks")}', flush=True)
        for row in previous:
            print(f'  {row["case"]}: {row.get("reason", "")}', flush=True)
        record('saved_native_vs_saved_hf', original['steps'][target['step']]['logits'], saved_hf, vocab)

        baseline = []
        for name in ('native-plain-1', 'native-plain-2', 'native-traced'):
            trace = output / 'trace' if name == 'native-traced' else None
            if trace is not None:
                trace.mkdir()
            print(f'运行：{name}', flush=True)
            values = native_forward(args, output / name, original, ids, trace)
            (output / f'{name}.json').write_text(json.dumps(values, allow_nan=False))
            baseline.append(values)
        first, second, traced = baseline
        stable = record('plain_repeat_exact', second, first, vocab, exact=True)
        unchanged = record('trace_vs_plain_exact', traced, second, vocab, exact=True)
        metadata['trace_attribution_valid'] = stable and unchanged
        print(f'采集归因门禁：{metadata["trace_attribution_valid"]}；不通过时仅保留观察，不能归因原失败。', flush=True)
        record('plain_fresh_vs_saved_hf', second, saved_hf, vocab)
        record('traced_fresh_vs_saved_hf', traced, saved_hf, vocab)
        trace_meta, native = load_trace(output / 'trace', ids, hidden, layers)
        metadata['trace'] = trace_meta
        metadata['first_exact_difference'] = metadata['first_tolerance_failure'] = None
        for name, expected in hf.items():
            record(name, native[name], expected, hidden)
            row = rows[-1]
            if row['exact_different_elements'] and metadata['first_exact_difference'] is None:
                metadata['first_exact_difference'] = name
            if row['status'] != 'PASS' and metadata['first_tolerance_failure'] is None:
                metadata['first_tolerance_failure'] = name

        head_meta, head = head_probe(args, config, hf, native, output)
        metadata['head_reference'] = head_meta
        hf_valid = record('hf_head_reproduces_saved_logits', head['hf_head_full'], saved_hf, vocab, exact=True)
        native_valid = record('native_head_reproduces_traced_logits', head['native_head_replay'], traced, vocab, exact=True)
        metadata['head_attribution_valid'] = hf_valid and native_valid and metadata['trace_attribution_valid']
        record('final_norm_accumulated', native['final_norm'], head['hf_norm'], hidden)
        record('final_norm_same_native_input', native['final_norm'], head['hf_norm_on_native'], hidden)
        record('head_hf_single_row_vs_full_rows', head['hf_head_last'], head['hf_head_full'], vocab)
        record('head_native_vs_hf_same_hf_norm', head['native_head_on_hf'], head['hf_head_last'], vocab)
        record('head_native_vs_hf_same_native_norm', traced, head['hf_head_on_native_norm'], vocab)
        print(f'首次精确差异：{metadata["first_exact_difference"]}；'
              f'首次原阈值超差：{metadata["first_tolerance_failure"]}；'
              f'输出头归因门禁：{metadata["head_attribution_valid"]}', flush=True)
    except Exception as error:
        (output / 'error.log').write_text(traceback.format_exc())
        rows.append(dict(case='probe_error', backend=BACKEND, status='FAIL', max_abs_error=None,
                         mean_abs_error=None, failed_elements=1, reason=str(error)))
        print(f'FAIL: probe_error: {error}', flush=True)
    status = save_report(output, rows, metadata)
    print(f'{status}: {output}（保留历史失败和精确重放门禁；非完整模型验收）', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
