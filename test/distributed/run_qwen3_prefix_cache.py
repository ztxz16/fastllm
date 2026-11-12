#!/usr/bin/env python3
"""通过生产进程入口验证前缀快照数值、LRU、清理和模型隔离；仅使用已有产物。"""

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import uuid

from run_process_tp import ROOT, capture, save_report
from run_qwen3_prefill import make_model, require, report_error, compare_logits
from run_qwen3_decode import invoke, verify, verify_graph, verify_allreduce, compare_results, expected_cache, expectations, verify_failure
from run_qwen3_shards import decode, encode, save_fixture
from fastllm_pytools.tp_graph import validate_prefix
from qwen3_layer_reference import prefill


def make_tasks(kind='normal'):
    tasks = [dict(name='initial', op='query'), dict(name='initial-clear', op='clear_prefix')]

    def request(name, ids, position=0, decode_steps=0, clear_active=False):
        tasks.append(dict(name=name, op='begin', token_ids=ids, position_begin=position,
                          capacity=len(ids)+decode_steps))
        if name == 'full-hit':
            tasks.append(dict(name='stateless-interleave', op='prefill', token_ids=[3, 8], position_begin=5))
        for step in range(decode_steps):
            tasks.append(dict(name=f'{name}-decode-{step}', op='decode', token_id=2+step))
            if clear_active and step == 0:
                tasks.append(dict(name=name+'-clear-active', op='clear_prefix'))
        tasks.append(dict(name=name+'-end', op='end'))

    if kind == 'budget':
        # 256 layers * 2 KV heads * 256 dim * K/V * FP16 = 512 KiB/token（全局保守计费）。
        # 127 -> 128 -> 129 tokens，分别在 64 MiB 边界下、边界上、边界外。
        request('budget-below', [1]*127)
        request('budget-exact', [2])
        request('budget-over-evict', [3])
        request('budget-evicted-miss', [1]*127)
        request('budget-survivor-hit', [1]*127)
        return tasks
    if kind == 'oversize':
        request('snapshot-too-large', [1]*128)
        request('snapshot-not-retained', [1]*128)
        return tasks
    if kind == 'wide':
        request('large-nccl-prefill', [1]*65)
        request('hit-small-custom-prefill', [1]*65, decode_steps=1)
        return tasks
    if kind == 'bad_suffix':
        request('valid-prefix', [1,4,7], decode_steps=1)
        tasks.append(dict(name='invalid-suffix', op='begin', token_ids=[1,4,17], position_begin=0, capacity=3))
        return tasks
    if kind == 'long':
        request('long-cold', [1]*512, decode_steps=2)
        request('long-hit', [1]*512, decode_steps=2)
        request('long-max', [2]*2048)
        request('long-max-hit', [2]*2048)
        tasks.append(dict(name='long-clear', op='clear_prefix'))
        request('long-after-clear', [1]*129)
        return tasks

    a = [1, 4, 7, 9]
    request('cold', a, decode_steps=2)
    request('full-hit', a, decode_steps=2)
    request('extend', a+[2, 3], decode_steps=1)
    request('shorter', a[:2], decode_steps=1)
    request('partial', [1, 4, 8, 9], decode_steps=1)
    request('miss', [2, 4, 7, 9])
    request('position-miss', a, 23)
    request('position-hit', a, 23, 2, True)
    request('after-clear', a)
    tasks.append(dict(name='before-lru', op='clear_prefix'))
    for i in range(8):
        request(f'lru-{i}', [i+1, 15, 16])
    request('lru-touch-oldest', [1, 15, 16])
    request('lru-overflow', [9, 15, 16])
    request('lru-survivor', [1, 15, 16])
    request('lru-evicted-miss', [2, 15, 16])
    request('one-token', [1])
    request('one-token-repeat', [1])
    request('max-prompt', [1, 4]*64, decode_steps=1)
    request('max-prompt-hit', [1, 4]*64, decode_steps=1)
    tasks += [dict(name='unload', op='unload'), dict(name='reload', op='load', reload=True)]
    request('after-reload', a, decode_steps=1)
    return tasks


def verify_prefix(report, tasks, config, world, *, unloaded=True):
    """独立按 ID/位置推导最长前缀及 LRU，核对真实 GPU 字节和执行长度。"""
    entries = []
    require(len(report['results']) == len(tasks), 'missing prefix lifecycle results')
    counters = dict(lookups=0, hits=0, reused_tokens=0, last_reused_tokens=0,
                    last_computed_tokens=0, evictions=0, clears=0)
    dim = config.get('head_dim', config['hidden_size']//config['num_attention_heads'])
    charge = config['num_hidden_layers'] * 2 * config['num_key_value_heads'] * dim * 2
    for task, result in zip(tasks, report['results']):
        op = task['op']
        if op == 'begin':
            ids, position = tuple(task['token_ids']), task['position_begin']
            matches = []
            for saved, pos in entries:
                common = 0
                if pos == position:
                    for left, right in zip(saved, ids[:-1]):
                        if left != right:
                            break
                        common += 1
                matches.append(common)
            reused = max(matches, default=0)
            if reused:
                entries.append(entries.pop(matches.index(reused)))
            counters.update(lookups=counters['lookups']+1, hits=counters['hits']+int(reused > 0),
                            reused_tokens=counters['reused_tokens']+reused,
                            last_reused_tokens=reused, last_computed_tokens=len(ids)-reused)
            key = (ids, position)
            if key in entries:
                entries.remove(key)
                entries.append(key)
            elif len(ids)*charge <= 64*1024*1024:
                while entries and (len(entries) >= 8 or (sum(len(e[0]) for e in entries)+len(ids))*charge > 64*1024*1024):
                    entries.pop(0)
                    counters['evictions'] += 1
                entries.append(key)
        elif op in ('clear_prefix', 'unload'):
            entries.clear()
            if op == 'unload':
                counters = dict.fromkeys(counters, 0)
            else:
                counters['clears'] += 1
        observed = validate_prefix(result['graph'], world, True)
        for rank, p in enumerate(observed):
            tokens = sum(len(e[0]) for e in entries)
            size = expected_cache(config, dict(active=bool(tokens), tokens=tokens, capacity=tokens, position_begin=0), rank, world)['device_bytes']
            require(all(p[k] == v for k, v in counters.items()) and p['entries'] == len(entries) and p['device_bytes'] == size,
                    f'{task["name"]}: prefix lifecycle mismatch rank={rank}: {p}, expected={counters}, entries={len(entries)}, bytes={size}')
    if unloaded:
        validate_prefix(report['graph_after_unload'], world, True, reset=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('check', 'worker', 'build-cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--long-context', action='store_true')
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--custom-allreduce', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'test/distributed/results')
    args = parser.parse_args()
    if args.long_context and args.model:
        parser.error('--long-context uses generated models for independent CPU verification')
    devices = [int(v) for v in args.devices.split(',')]
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        parser.error('requires two distinct GPUs')
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_prefix_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    specs = [('real', {})] if args.model else [(d, dict(dtype=d)) for d in ('F16', 'BF16', 'F32')]+[
        ('bias', dict(bias=True)), ('tied', dict(tied=True)), ('kv_replication', dict(kv=1)),
        ('budget', dict(hidden=4, layers=256, head_dim=256, kv=2, kind='budget', gpu_reference=True)),
        ('oversize', dict(hidden=4, layers=257, head_dim=256, kv=2, kind='oversize', gpu_reference=True)),
        ('wide', dict(hidden=1024, layers=1, kind='wide', gpu_reference=True)),
        ('bad_suffix', dict(kind='bad_suffix'))]
    if args.long_context:
        specs = [('long_prefix', dict(kind='long', layers=1, hidden=4, head_dim=2, heads=2, kv=1))]
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
        git_status=capture(['git', 'status', '--short']), seed=0, warmup=0, iterations=1, repeats=1,
        gpu=capture(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']),
        cuda=capture(['nvcc', '--version']), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), devices=devices,
        execution='decode_cuda_graph' if args.cuda_graph else 'eager', custom_allreduce=args.custom_allreduce,
        prefix_cache=True, kv_cache='FP16 request-local plus immutable Prompt snapshots',
        expected_case_count=len(specs), report_title='Qwen3 cross-request prefix cache correctness',
        report_description='同一公开 Begin/Decode/Clear/End 路径；逐元素对照无前缀 NCCL Eager TP、单卡和合成模型 CPU；非性能测试。')
    rows = []
    try:
        metadata['build_cache'] = args.build_cache.read_text()
        metadata['compile_architecture'] = [s for s in metadata['build_cache'].splitlines() if s.startswith(('CUDA_ARCH:', 'CMAKE_CUDA_ARCHITECTURES:'))]
        for name, spec in specs:
            print('RUN: ' + name, flush=True)
            directory = output / name
            directory.mkdir()
            row = dict(case=name, status='FAIL', reason='', backend='qwen3_process/prefix/'+metadata['execution'],
                       max_abs_error=None, mean_abs_error=None, failed_elements=None)
            metrics = dict(max=0.0, sum=0.0, count=0, failed=0)
            try:
                model = args.model.resolve() if args.model else directory/'model'
                config, tensors = (json.loads((model/'config.json').read_text()), None) if args.model else make_model(model, spec)
                updated, reload_model = None, model
                if tensors is not None and not spec.get('gpu_reference') and spec.get('kind', 'normal') == 'normal':
                    updated = copy.deepcopy(tensors)
                    norm = updated['model.norm.weight']
                    norm['data'] = encode([-v for v in decode(norm['data'], norm['dtype'])], norm['dtype'])
                    reload_model = directory/'reloaded-model'
                    reload_model.mkdir()
                    save_fixture(reload_model, config, updated)
                tasks = make_tasks(spec.get('kind', 'normal'))
                (directory/'tasks.json').write_text(json.dumps(tasks))
                row.update(dtype=spec.get('dtype', 'model'), task_count=len(tasks), hidden=config['hidden_size'],
                           layers=config['num_hidden_layers'], input_tasks='tasks.json')
                flags = ['--prefix-cache', '1', '--cuda-graph', str(int(args.cuda_graph)), '--custom-allreduce', str(int(args.custom_allreduce))]
                code, tp = invoke(args.check.resolve(), args.worker.resolve(), model, directory, devices, 'prefix', reload_model=reload_model, extra_args=flags)
                if spec.get('kind') == 'bad_suffix':
                    verify_failure(code, tp, dict(reason='token ID outside vocabulary', group_failed=True), tasks)
                    completed = tp.get('results', [])
                    require(len(completed) == len(tasks)-1, 'failed before invalid suffix')
                    # 失败组已经被回收，没有卸载后响应；仅核对失败前实际收到的状态。
                    verify_prefix(tp, tasks[:-1], config, 2, unloaded=False)
                    for result, state in zip(completed, expectations(tasks[:-1])):
                        if 'logits' in result:
                            compare_logits(result['logits'], prefill(config, tensors, state['ids'], state['rope_position']),
                                           metrics, result['task']+' before rejection vs CPU', 0.008, 0.008)
                    row.update(status='PASS', workers=tp['workers'], reason='invalid suffix rejected with snapshots resident; whole group reaped')
                    continue
                require(code == 0, report_error(tp))
                row['workers'] = tp['workers']
                verify(tp, devices, tasks, config)
                verify_graph(tp, tasks, len(devices), args.cuda_graph)
                verify_prefix(tp, tasks, config, len(devices))
                if args.custom_allreduce:
                    verify_allreduce(tp, tasks, config, args.cuda_graph)
                atol, rtol = (0.05, 0.01) if args.model else (0.008, 0.008)
                code, eager = invoke(args.check.resolve(), args.worker.resolve(), model, directory, devices, 'eager-no-prefix', reload_model=reload_model)
                require(code == 0, report_error(eager))
                verify(eager, devices, tasks, config)
                for actual, baseline in zip(tp['results'], eager['results']):
                    if 'logits' in actual:
                        compare_logits(actual['logits'], baseline['logits'], metrics, actual['task']+' prefix vs uncached TP', atol, rtol)
                code, single = invoke(args.check.resolve(), args.worker.resolve(), model, directory, devices[:1], 'single', reload_model=reload_model)
                require(code == 0, report_error(single))
                verify(single, devices[:1], tasks, config)
                compare_results(tp, single, tasks, config, None if spec.get('gpu_reference') else tensors,
                                updated, metrics, atol, rtol, {}, directory/'reference.json')
                if updated is not None:
                    first = next(r['logits'] for r in tp['results'] if r['task'] == 'cold')
                    after = next(r['logits'] for r in tp['results'] if r['task'] == 'after-reload')
                    require(first != after, 'reload returned stale model logits')
                row.update(status='PASS', prefix_states=[dict(task=r['task'], ranks=r['graph']) for r in tp['results']],
                           allreduce=tp['graph_after_unload'][0]['allreduce'])
                prefix_states = [r['graph'][0]['prefix_cache'] for r in tp['results']]
                row.update(prefix_hits=max(p['hits'] for p in prefix_states),
                           reused_tokens=max(p['reused_tokens'] for p in prefix_states),
                           evictions=max(p['evictions'] for p in prefix_states),
                           prefix_bytes=max(p['device_bytes'] for p in prefix_states),
                           reference='NCCL Eager TP + single GPU' if spec.get('gpu_reference') or args.model else 'CPU + NCCL Eager TP + single GPU')
            except Exception as error:
                row['reason'] = f'{type(error).__name__}: {error}'
            finally:
                if metrics['count']:
                    row.update(max_abs_error=metrics['max'] if math.isfinite(metrics['max']) else 'inf',
                               mean_abs_error=metrics['sum']/metrics['count'] if math.isfinite(metrics['sum']) else 'inf',
                               failed_elements=metrics['failed'], compared_elements=metrics['count'])
                rows.append(row)
                save_report(output, rows, metadata)
    except Exception as error:
        rows.append(dict(case='runner', backend='python', status='FAIL', reason=f'{type(error).__name__}: {error}'))
    finally:
        status = save_report(output, rows, metadata)
    print(f'{status}: {output}')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
