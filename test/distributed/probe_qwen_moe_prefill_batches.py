#!/usr/bin/env python3
"""比较四请求与两请求Prefill的完整逐层轨迹；先严格复现已保存logits，不运行HF或性能测试。"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import traceback
import uuid

import numpy as np

from probe_qwen_moe_concurrency import file_hash
from run_process_tp import ROOT, capture, save_report
from fastllm_pytools.qwen3_process_chat import validate_request
from fastllm_pytools.qwen_moe_process_session import RoundTransport

TRACE_ENV = 'FASTLLM_MOE_TRACE_DIR'
LIMIT_ENV = 'FASTLLM_MOE_TRACE_MAX_MIB'
BLOCK_ENV = 'FASTLLM_MOE_BLOCK_TRACE_DIR'


def stage_names(layers):
    """/** 按正式前向顺序列出完整张量边界；不能用末Token切片替代整段Prompt比较。 */"""
    return ['embedding', *[f'layer_{i}_{stage}' for i in range(layers)
                          for stage in ('attention', 'moe')], 'final_norm']


def difference(actual, expected):
    """/** 分块比较完整数组，拒绝非有限值；零容差仅用于定位差异，不自动认定为算子错误。 */"""
    a, b = np.asarray(actual), np.asarray(expected)
    if a.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError('different or empty trace shapes')
    maximum, total, failed, first = 0., 0., 0, None
    for start in range(0, a.size, 65536):
        left, right = a[start:start+65536].astype(np.float64), b[start:start+65536].astype(np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('nonfinite trace/logits')
        errors = np.abs(left - right)
        indices = np.flatnonzero(errors)
        maximum, total = max(maximum, float(errors.max())), total + float(errors.sum())
        failed += int(indices.size)
        if first is None and indices.size:
            i = int(indices[0])
            first = dict(element_0based=start+i, actual=float(left[i]), expected=float(right[i]))
    return dict(status='FAIL' if failed else 'PASS', max_abs_error=maximum, mean_abs_error=total/int(a.size),
                failed_elements=failed, compared_elements=int(a.size), first_difference=first)


def load_traces(directory, ranks, ids, hidden, layers):
    """/**
     * 校验所有活动rank的warmup/正式快照，内存映射第2次Begin的完整FP16张量。
     * @param directory 本会话独占快照目录；只允许每rank每轮一对JSON/F16文件。
     * @param ranks 实际Begin的rank列表，空闲rank不得生成快照。
     * @param ids 原始完整Prompt Token ID，不接受截断、重排或其他输入。
     * @param hidden 模型隐藏宽度；一般阶段为[len(ids),hidden]，final_norm为[1,hidden]。
     * @param layers 模型总层数，必须覆盖全部Attention与MoE边界。
     * @return 按rank索引的描述、只读映射和阶段信息；缺失、重复、越界或布局错误直接失败。
     */"""
    found, allowed, result = set(), set(), {}
    names = stage_names(layers)
    elements = (1 + 2*layers)*len(ids)*hidden + hidden
    if not 0 < elements*2 <= 256*1024*1024:
        raise ValueError('full prompt trace exceeds diagnostic limit')
    for path in directory.glob('*.json'):
        meta = json.loads(path.read_text())
        key = (meta.get('rank'), meta.get('round'))
        if any(type(v) is not int for v in key) or key in found or key[0] not in ranks or key[1] not in (1, 2):
            raise ValueError('unexpected/duplicate rank or trace round')
        found.add(key)
        expected = dict(version=1, dtype='float16', byte_order='little', tokens=len(ids),
                        hidden=hidden, layers=layers, ids=ids, max_mib=256)
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError('trace input/layout/limit metadata mismatch; rebuild the worker')
        name = meta['binary']
        binary = directory / name
        if Path(name).name != name or binary.resolve().parent != directory.resolve() or not binary.is_file():
            raise ValueError('trace binary escapes directory or is missing')
        if binary.stat().st_size != elements*2:
            raise ValueError('truncated/oversized trace binary')
        entries, offset = meta['snapshots'], 0
        if [s['name'] for s in entries] != names:
            raise ValueError('missing/reordered trace stages')
        for entry in entries:
            count = hidden if entry['name'] == 'final_norm' else len(ids)*hidden
            if type(entry['offset']) is not int or type(entry['elements']) is not int or \
                    entry['offset'] != offset or entry['elements'] != count:
                raise ValueError('invalid trace offsets/shapes')
            offset += count
        if {path, binary} & allowed:
            raise ValueError('trace files reused across ranks/rounds')
        allowed.update((path, binary))
        if key[1] == 2:
            meta.update(manifest_sha256=file_hash(path), binary_sha256=file_hash(binary))
            result[key[0]] = (meta, np.memmap(binary, mode='r', dtype='<f2', shape=(elements,)))
    if found != {(rank, repeat) for rank in ranks for repeat in (1, 2)} or set(directory.iterdir()) != allowed:
        raise ValueError('missing/incomplete or unexpected trace files')
    return result


def prefill(args, output, job, active_ranks, vocab, *, block_trace=False):
    """/**
     * 经正式联合协议执行一次warmup及一次Prefill，随后End并核验KV释放。
     * 保留原请求capacity/Prompt/采样参数；只运行首步，不执行后续Decode或混合准入。
     * 快照环境只传给本次子进程；既有原生算子、通信和总工作区预算保持正式配置。
     * @param block_trace true只采集第0层MoE内部；false保持原完整逐层快照。
     * @return 第2次Begin的原始报告、两轮快照目录及关闭确认；异常由transport回收整组进程。
     */"""
    traces = output / 'trace'
    traces.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    for key in (TRACE_ENV, LIMIT_ENV, BLOCK_ENV):
        environment.pop(key, None)
    environment.update({BLOCK_ENV: str(traces)} if block_trace else {TRACE_ENV: str(traces), LIMIT_ENV: '256'})
    with RoundTransport(args.build / 'fastllm_moe_generate', args.build / 'fastllm_tp_worker',
            args.model, '0,1,2,3', output / 'native', timeout=args.timeout,
            cuda_graph=False, prefix_cache=False, environment=environment) as session:
        for repeat in (1, 2):
            steps = [dict(action=1, token_ids=job['token_ids'], position_begin=0,
                capacity=len(job['token_ids']) + job['max_new_tokens'] - 1,
                sampling={k: job[k] for k in ('seed', 'top_k', 'top_p', 'temperature')}) if rank in active_ranks
                else dict(action=0, token_ids=[]) for rank in range(4)]
            result = session.exchange('step', steps=steps, trace=True)
            (output / f'prefill{repeat}.json').write_text(json.dumps(result, allow_nan=False))
            if result['execution'] != 'eager' or any(len(result[k]) != 4 for k in ('states', 'token_ids', 'logits')):
                raise ValueError('invalid prefill rank count/path')
            for rank, state in enumerate(result['states']):
                active = rank in active_ranks
                if (state['request_active'] != active or state['captures'] or state['replays'] or
                        state['prefix_cache']['last_reused_tokens'] or
                        state['request_tokens'] != (len(job['token_ids']) if active else 0)):
                    raise ValueError(f'prefill rank {rank}: invalid request/cache/Graph evidence')
                logits, token = result['logits'][rank], result['token_ids'][rank]
                if active:
                    if len(logits) != vocab or type(token) is not int or not 0 <= token < vocab:
                        raise ValueError(f'prefill rank {rank}: invalid output')
                    difference(logits, logits)
                elif token is not None or logits:
                    raise ValueError('inactive rank returned output')
            ended = session.exchange('step', steps=[dict(action=3 if r in active_ranks else 0,
                                                        token_ids=[]) for r in range(4)], trace=False)
            (output / f'end{repeat}.json').write_text(json.dumps(ended, allow_nan=False))
            if (len(ended['states']) != 4 or ended['execution'] != 'eager' or
                    ended['token_ids'] != [None]*4 or ended['logits'] != [[]]*4 or any(
                        s[k] for s in ended['states'] for k in ('request_active', 'request_tokens',
                                                               'request_capacity', 'request_kv_bytes'))):
                raise ValueError('prefill End retained output or active KV')
    return result, traces, session.shutdown


def main():
    """/** 先校验旧报告及新采集的精确重放，再报告各rank首个不同阶段；失败保留部分四格式报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='已采集第1步logits的moe_admission_probe目录')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, default=Path('build-attention-ep-cu126'))
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be positive and finite')
    args.model, args.build, args.baseline = args.model.resolve(), args.build.resolve(), args.baseline.resolve()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_moe_prefill_batches_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f'Prefill逐层诊断：{output}', flush=True)
    rows, metadata = [], dict(command=shlex.join([sys.executable, *sys.argv]), baseline=str(args.baseline),
        scope='formal_prefill_batch_trace_diagnostic', execution_complete=False, trace_attribution_valid=False,
        warmup=1, iterations=1, repeats=1, cuda_graph=False, prefix_cache=False, trace_max_mib=256,
        git=capture(['git', 'rev-parse', 'HEAD']), git_status=capture(['git', 'status', '--short']),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cpu=capture(['lscpu']), cuda=capture(['nvcc', '--version']),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), report_title='MoE Prefill并发组合逐层诊断',
        report_description='独立归因，不作性能或HF验收。仅运行首步Prefill；原capacity不变。'
            '先要求正式采集logits精确复现旧报告；通过后比较完整Prompt各阶段FP16值。'
            '阶段FAIL表示出现数值差异，不单独证明算子错误；原始logits与二进制轨迹均保留。')
    try:
        baseline = json.loads((args.baseline / 'summary.json').read_text())
        old = baseline['metadata']
        if (not baseline['complete'] or old['scope'] != 'controlled_admission_diagnostic' or
                old['prefix_cache'] or 1 not in old['trace_tokens_1based'] or
                not {2, 4}.issubset(old['first_waves'])):
            raise ValueError('baseline must be a completed 4/2 admission diagnostic capturing token 1')
        config = json.loads((args.model / 'config.json').read_text())
        hidden, layers, vocab = config['hidden_size'], config['num_hidden_layers'], config['vocab_size']
        job = old['request']
        validate_request(job, vocab)
        if job['temperature'] != 0 or job['eos_token_ids']:
            raise ValueError('baseline must use greedy sampling and no EOS')
        trace_bytes = ((1 + 2*layers)*len(job['token_ids'])*hidden + hidden)*2
        if not 0 < trace_bytes <= 256*1024*1024:
            raise ValueError('full prompt trace exceeds 256 MiB per active rank')
        metadata.update(request=job, model_path=str(args.model), hidden_size=hidden, layers=layers,
                        trace_bytes_per_rank=trace_bytes, expected_case_count=6 + 2*(2*layers+3), baseline_git=old['git'])
        for filename, key in (('config.json', 'model_config_sha256'),
                              ('model.safetensors.index.json', 'model_index_sha256')):
            metadata[key] = file_hash(args.model / filename)
            if metadata[key] != old[key]:
                raise ValueError(f'model provenance mismatch: {filename}')
        metadata['build_cache'] = (args.build / 'CMakeCache.txt').read_text()
        metadata['binary_sha256'] = {n: file_hash(args.build / n)
                                    for n in ('fastllm_moe_generate', 'fastllm_tp_worker')}
        metadata['baseline_binary_sha256'] = old['binary_sha256']
        saved = {}
        for wave in (4, 2):
            for rank in range(wave):
                path = args.baseline / f'eager_first{wave}_then{4-wave}/repeat01/rank{rank}.json'
                report = json.loads(path.read_text())
                if report['status'] != 'PASS' or report['request'] != job or report['dp_rank'] != rank or \
                        report['steps'][0]['execution'] != 'eager' or len(report['steps'][0]['logits']) != vocab:
                    raise ValueError('baseline raw request/first-step evidence mismatch')
                saved[(wave, rank)] = report['steps'][0]
        save_report(output, rows, metadata)

        results, traces = {}, {}
        for wave in (4, 2):
            result, directory, shutdown = prefill(args, output / f'first{wave}', job, list(range(wave)), vocab)
            results[wave] = result
            metadata[f'first{wave}_shutdown'] = shutdown
            for rank in range(wave):
                metrics = difference(result['logits'][rank], saved[(wave, rank)]['logits'])
                token_match = result['token_ids'][rank] == saved[(wave, rank)]['token_id']
                rows.append(dict(case=f'first{wave}_rank{rank}_baseline_replay', backend='formal_eager_prefill',
                    **metrics, token_match=token_match, reason='精确复现旧报告第1步logits和采样Token'))
                if not token_match:
                    rows[-1]['status'] = 'FAIL'
                print(f'{rows[-1]["status"]}: {rows[-1]["case"]}; max_abs={metrics["max_abs_error"]:.6g}', flush=True)
            save_report(output, rows, metadata)
            traces[wave] = load_traces(directory, list(range(wave)), job['token_ids'], hidden, layers)
        if any(r['status'] != 'PASS' for r in rows):
            raise ValueError('traced prefill did not exactly replay baseline; stage attribution refused')
        metadata['trace_attribution_valid'] = True
        metadata['first_different_stage'] = {}
        metadata['traces'] = {str(w): {str(r): m for r, (m, _) in traces[w].items()} for w in traces}

        for rank in (0, 1):
            actual_meta, actual = traces[2][rank]
            expected_meta, expected = traces[4][rank]
            for a, b in zip(actual_meta['snapshots'], expected_meta['snapshots']):
                av = actual[a['offset']:a['offset']+a['elements']]
                bv = expected[b['offset']:b['offset']+b['elements']]
                metrics = difference(av, bv)
                first = metrics['first_difference']
                if first:
                    first.update(prompt_token_1based=first['element_0based']//hidden+1,
                                 hidden_column_0based=first['element_0based'] % hidden)
                    if a['name'] == 'final_norm':
                        first['prompt_token_1based'] = len(job['token_ids'])
                row = dict(case=f'rank{rank}_{a["name"]}', backend='formal_eager_prefill', **metrics,
                    shape=[a['elements']//hidden, hidden], last_token=difference(av[-hidden:], bv[-hidden:]),
                    reason='两请求与四请求完整张量比较；零容差用于定位，不是新验收阈值')
                rows.append(row)
                if metrics['failed_elements']:
                    metadata['first_different_stage'].setdefault(str(rank), a['name'])
                print(f'{row["status"]}: {row["case"]}; max_abs={row["max_abs_error"]:.6g}; '
                      f'mean_abs={row["mean_abs_error"]:.6g}; different={row["failed_elements"]}', flush=True)
            rows.append(dict(case=f'rank{rank}_logits', backend='formal_eager_prefill',
                **difference(results[2]['logits'][rank], results[4]['logits'][rank]), reason='首步完整词表logits'))
            save_report(output, rows, metadata)
        metadata['execution_complete'] = len(rows) == metadata['expected_case_count']
    except BaseException as error:
        metadata['trace_attribution_valid'] = False
        (output / 'diagnostic.traceback.log').write_text(traceback.format_exc())
        rows.append(dict(case='diagnostic_error_or_interrupt', status='FAIL', failed_elements=1,
                         reason=f'{type(error).__name__}: {error}'))
        print(f'FAIL: {type(error).__name__}: {error}', flush=True)
    finally:
        status = save_report(output, rows, metadata)
    if metadata['trace_attribution_valid']:
        print('首个不同阶段：' + json.dumps(metadata['first_different_stage'], ensure_ascii=False), flush=True)
    else:
        print('逐层归因未完成；请先处理采集或精确重放失败。', flush=True)
    print(f'{status}: {output}', flush=True)
    return int(status != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
