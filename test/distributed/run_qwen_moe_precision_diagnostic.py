#!/usr/bin/env python3
"""从已保存的失败历史定位MoE逐层误差；使用已有产物，不编译、不安装、不下载。"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import sys
import traceback
import uuid

from run_process_tp import ROOT, capture, case_process, parse_report, save_report, wait_case
from run_qwen_moe_full import align_tied_routing

ATOL, RTOL = .05, .01
BACKEND = 'qwen2_moe_attention_dp_ep_fp16_nccl'


def difference(actual, expected):
    """/** 按原真实模型阈值比较完整张量；保存最坏元素，形状和非有限值均失败。 */"""
    if len(actual) != len(expected):
        raise ValueError(f'shape mismatch: {len(actual)} != {len(expected)}')
    errors = [abs(a-b) if math.isfinite(a) and math.isfinite(b) else float('inf')
              for a, b in zip(actual, expected)]
    bad = sum(not math.isfinite(e) or e > ATOL + RTOL*abs(b) for e, b in zip(errors, expected))
    worst = max(range(len(errors)), key=errors.__getitem__) if errors else None
    return dict(status='FAIL' if bad else 'PASS', max_abs_error=max(errors, default=0.0),
                mean_abs_error=sum(errors)/len(errors) if errors else 0.0, failed_elements=bad,
                worst_index=worst, actual=None if worst is None else actual[worst],
                expected=None if worst is None else expected[worst])


def failing_history(report, reference):
    """/** 重建指定请求首次超差的实际历史，保留teacher forcing语义，不重新生成Token。 */"""
    lookup = {tuple(row['ids']): row['logits'] for row in reference['rows']}
    ids = list(report['request']['token_ids'])
    forced = report.get('teacher_forced_tokens')
    for step, trace in enumerate(report['steps']):
        expected = lookup[tuple(ids)]
        metrics = difference(trace['logits'], expected)
        if metrics['status'] == 'FAIL':
            return dict(ids=ids, step=step, execution=trace['execution'], metrics=metrics)
        if step + 1 < len(report['steps']):
            ids.append(forced[step] if forced is not None else trace['token_id'])
    raise ValueError('specified request has no logits exceeding the original tolerance')


def task_chunks(layers, byte_limit=8*1024*1024, count_limit=64):
    """/** 仅分割诊断任务文件；保持张量完整，低于已有checker的16MiB/64任务上限。 */"""
    chunks, chunk, size = [], [], 2
    for layer in layers:
        task = {k: v for k, v in layer.items() if k != 'reference'}
        length = len(json.dumps(task, allow_nan=False).encode())
        if length + 2 > byte_limit:
            raise ValueError(f'{layer["name"]}: a single task exceeds diagnostic input limit')
        if chunk and (size + length + 2 > byte_limit or len(chunk) == count_limit):
            chunks.append(chunk)
            chunk, size = [], 2
        chunk.append(layer)
        size += length + 2
    if chunk:
        chunks.append(chunk)
    return chunks


def compare_layer(layer, actual, config):
    """/** 分别记录每rank每字段；同分路由只对齐槽位，路由失败后仍保留数值误差。 */"""
    if actual['task'] != layer['name'] or len(actual['ranks']) != len(layer['reference']):
        raise ValueError('native layer name/rank count mismatch')
    rows = []
    for rank, (values, expected) in enumerate(zip(actual['ranks'], layer['reference'])):
        tokens = len(layer['inputs'][rank]['hidden']) // layer['hidden']
        label = f'{layer["name"]}/rank{rank}/tokens{tokens}'
        aligned = None
        if layer['stage'] != 1:
            row = dict(case=label+'/selected_experts', backend=BACKEND, status='PASS',
                       max_abs_error=None, mean_abs_error=None, failed_elements=0, reason='')
            try:
                aligned, row['tied_routing_slots'] = align_tied_routing(
                    values, expected, tokens, config['num_experts'], config['num_experts_per_tok'])
            except AssertionError as error:
                row.update(status='FAIL', failed_elements=1, reason=str(error))
            rows.append(row)
        for field, target in expected.items():
            if field == 'selected_experts':
                continue
            observed = aligned if field == 'routing_weights' and aligned is not None else values[field]
            metrics = difference(observed, target)
            rows.append(dict(case=label+'/'+field, backend=BACKEND, **metrics,
                reason=f'worst_index={metrics["worst_index"]}, actual={metrics["actual"]:.3f}, '
                       f'expected={metrics["expected"]:.3f}' if metrics['failed_elements'] else ''))
    return rows


def run_logged(command, directory, timeout):
    """/** 独立进程组运行已有公开入口；日志先落盘，超时由共同清理逻辑回收进程组。 */"""
    directory.mkdir()
    (directory / 'command.txt').write_text(shlex.join(command))
    print(f'运行：{directory.name}；日志：{directory}', flush=True)
    with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        with case_process(command, stdout=out, stderr=err) as process:
            return wait_case(process, timeout)


def main():
    """/** 先保存原失败，再采集同一历史的HF分层参考并逐块验收；异常也输出四种报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--request', default='session-0002/request-000001.json')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, default=Path('build-attention-ep-cu126'))
    parser.add_argument('--reference-python', required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                      '_moe_precision_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    print(f'诊断目录：{output}', flush=True)
    rows = []
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), warmup=0, iterations=1, repeats=1, atol=ATOL, rtol=RTOL,
        report_title='MoE失败历史的独立子层定位', source_dir=str(args.result_dir.resolve()),
        report_description='保留原完整模型失败；其余行为同输入独立子层诊断，不是完整模型或性能验收。',
        scope='Only the failing DP rank is active. Other ranks are idle but participate in EP. '
              'Each task starts from HF hidden states; no chained hidden state or original batch scheduling replay.')
    try:
        metadata['build_cache'] = (args.build / 'CMakeCache.txt').read_text()
        source = json.loads((args.result_dir / 'summary.json').read_text())
        metadata['source_git'] = source.get('metadata', {}).get('git')
        report = json.loads((args.result_dir / args.request).read_text())
        reference = json.loads((args.result_dir / 'transformers-reference.json').read_text())
        target = failing_history(report, reference)
        rank = report['dp_rank']
        if type(rank) is not int or not 0 <= rank < 4:
            raise ValueError(f'invalid source DP rank: {rank}')
        metadata.update(target=target, request=args.request, rank=rank)
        rows.append(dict(case='saved_full_model_logits', backend=target['execution'], **target['metrics'],
                         reason=f'saved failure step={target["step"]} tokens={len(target["ids"])}'))
        save_report(output, rows, metadata)
        print(f'定位原失败：rank={rank} step={target["step"]} tokens={len(target["ids"])}', flush=True)

        sequences = [[] for _ in range(4)]
        sequences[rank] = target['ids']
        (output / 'layer-inputs.json').write_text(json.dumps(sequences))
        (output / 'histories.json').write_text('[]')
        reference_path = output / 'reference.json'
        command = [args.reference_python, str(ROOT / 'test/distributed/qwen_moe_transformers_reference.py'),
            '--model', str(args.model.resolve()), '--inputs', str(output / 'histories.json'),
            '--layer-inputs', str(output / 'layer-inputs.json'), '--output', str(reference_path)]
        if run_logged(command, output / 'reference', args.timeout) != 0:
            raise RuntimeError(f'HF reference failed; inspect {output / "reference/stderr.log"}')
        reference = json.loads(reference_path.read_text())
        metadata['reference'] = reference['metadata']
        config = json.loads((args.model / 'config.json').read_text())
        layers = reference['layers']
        if len(layers) != 3*config['num_hidden_layers']:
            raise ValueError('incomplete split-stage reference')
        chunks = task_chunks(layers)
        metadata['task_count'] = len(layers)
        metadata['completed_tasks'] = 0
        for number, chunk in enumerate(chunks):
            directory = output / f'native-{number:03d}'
            tasks = output / f'tasks-{number:03d}.json'
            tasks.write_text(json.dumps([{k: v for k, v in row.items() if k != 'reference'} for row in chunk],
                                       allow_nan=False))
            command = [str((args.build / 'fastllm_moe_layer_check').resolve()),
                '--worker', str((args.build / 'fastllm_tp_worker').resolve()), '--model', str(args.model.resolve()),
                '--tasks', str(tasks), '--devices', '0,1,2,3', '--output', str(directory / 'weights'),
                '--dump-weights', '0']
            code = run_logged(command, directory, args.timeout)
            result = parse_report((directory / 'stdout.log').read_text())
            results = result.get('results', [])
            if len(results) > len(chunk):
                raise ValueError('unexpected extra native layer results')
            for layer, actual in zip(chunk, results):
                compared = compare_layer(layer, actual, config)
                rows.extend(compared)
                metadata['completed_tasks'] += 1
                save_report(output, rows, metadata)
                active = [row for row in compared if f'/rank{rank}/' in row['case']]
                worst = max((row['max_abs_error'] for row in active if row['max_abs_error'] is not None), default=0)
                failed = [row for row in active if row['status'] == 'FAIL']
                print(f'{"FAIL" if failed else "PASS"}: {layer["name"]} max_abs={worst:.6g} '
                      f'failed_fields={len(failed)}', flush=True)
                for row in failed:
                    print(f'  {row["case"]}: failed_elements={row["failed_elements"]}; '
                          f'{row["reason"]}', flush=True)
            if code or result.get('status') != 'PASS' or not result.get('workers_reaped') or len(results) != len(chunk):
                raise RuntimeError(f'native chunk {number} failed/incomplete; inspect {directory}; {result.get("error", "")}')
    except Exception as error:
        (output / 'error.log').write_text(traceback.format_exc())
        rows.append(dict(case='diagnostic_setup_or_execution', backend=BACKEND, status='FAIL',
                         failed_elements=1, reason=str(error)))
        print(f'FAIL: {error}', flush=True)
    finally:
        status = save_report(output, rows, metadata)
        print(f'{status}: {output}（包含已保存的完整模型失败，分层结果见summary.json）', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
