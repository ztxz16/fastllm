#!/usr/bin/env python3
"""正式HTTP/SSE混合长度诊断；验证输出，观察端到端延迟、服务队列和GPU显存。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import signal
import statistics
import sys
import threading
import time
import uuid

from run_process_tp import ROOT, capture, write_xlsx
from benchmark_observer import gpu_sample
from benchmark_qwen3_dp import percentile
from run_qwen3_dp_service import ServiceRun, ZERO, exchange, payload, process_snapshot
from run_qwen3_prefill import require
from fastllm_pytools.qwen3_process_chat import prepare_request
from fastllm_pytools.dp_baseline import PROFILES, model_flags
from fastllm_pytools.qwen3_process_session import BACKEND, EMPTY
from fastllm_pytools.util import make_normal_llm_model, make_normal_parser
from qwen3_request_inputs import read_requests, prepare_inputs


def make_inputs(model, tokenizer, lengths, outputs):
    """/** 用实际聊天模板构造接近目标的文本；保留完整Token，不截断或伪造请求长度。 */"""
    inputs = []
    for index, target in enumerate(lengths):
        def text(repeats):
            return f'资料{index}：' + '苹果是水果。' * repeats + '\n请按编号列出1000种常见物品并说明用途，不要省略。'

        def ids(repeats):
            return tokenizer.apply_chat_template([dict(role='user', content=text(repeats))], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=False)

        require(len(ids(0)) <= target, f'chat template exceeds target {target}')
        low, high = 0, target * 2
        require(len(ids(high)) > target, 'tokenizer cannot construct requested prompt lengths')
        while low + 1 < high:
            middle = (low + high) // 2
            if len(ids(middle)) <= target:
                low = middle
            else:
                high = middle
        for cap in outputs:
            request = prepare_request(model, tokenizer, text(low), max_new_tokens=cap)
            require(target - 16 <= len(request['token_ids']) <= target, f'cannot approach target {target}')
            for stream in (False, True):
                body = payload(cap, stream)
                body['messages'][0]['content'] = text(low)
                inputs.append(dict(name=f'p{target}_o{cap}_{"sse" if stream else "http"}', target=target,
                                   request=request, payload=body))
    require(len({len(i['request']['token_ids']) for i in inputs}) == len(lengths), 'prompt lengths did not vary')
    return inputs


def input_key(body):
    """/** 基准身份覆盖完整对话及采样参数；HTTP与SSE仅传输形式不同，可共用Token参考。 */"""
    return json.dumps({k: v for k, v in body.items() if k not in ('stream', 'stream_options')}, sort_keys=True, ensure_ascii=False)


class MixedService(ServiceRun):
    def reference_for(self, result, report):
        """/** 每种Prompt/输出上限独立建立同TP的DP1基准；关闭缓存避免混淆部分命中语义。 */"""
        return self.references[input_key(result['payload'])]


class MemorySampler:
    """/** 周期只读nvidia-smi；记录整卡采样值，不将其宣称为本进程精确峰值。 */"""
    def __init__(self):
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        started = time.monotonic()
        try:
            rows = gpu_sample(['uuid', 'index', 'memory.used', 'memory.total'])
            for row in rows:
                require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                            for k in ('memory.used', 'memory.total')), f'invalid GPU memory sample: {row}')
                require(0 < row['memory.total'] and row['memory.used'] <= row['memory.total'], 'invalid memory capacity')
            self.samples.append(dict(start=started, end=time.monotonic(), gpu=rows))
        except Exception as error:
            self.errors.append(f'{type(error).__name__}: {error}')

    def __enter__(self):
        self.sample()
        require(not self.errors, f'GPU memory sampling unavailable: {self.errors}')
        def loop():
            while not self.stop.wait(0.5):
                self.sample()
                if self.errors:
                    break
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(5)
            require(not self.thread.is_alive(), 'GPU memory observer did not stop')
        self.sample()


def run_wave(service, inputs, concurrency, label):
    """/** 固定工作量按客户端通道提交；计时包含服务分词，测试输入构造、校验和落盘在计时外。 */"""
    barrier = threading.Barrier(concurrency + 1)
    records, lock = [], threading.Lock()
    def client(lane):
        barrier.wait(timeout=10)
        for index in range(lane, len(inputs), concurrency):
            item = inputs[index]
            result = exchange(service.address, item['payload'], f'{label}-{index}', timeout=service.args.timeout + 10)
            result.update(input_name=item['name'], index=index)
            with lock:
                records.append(result)
            if result['status'] != 200 or result['error']:
                break
    with ThreadPoolExecutor(concurrency) as pool:
        futures = [pool.submit(client, lane) for lane in range(concurrency)]
        start = time.monotonic()
        try:
            barrier.wait(timeout=10)
            for future in futures:
                future.result()
        except BaseException:
            service.close()
            raise
        end = time.monotonic()
    wave = dict(start=start, end=end, elapsed_s=end-start, records=sorted(records, key=lambda r: r['index']))
    service.save(label, wave)
    require(len(records) == len(inputs), f'{label}: incomplete workload')
    for result in wave['records']:
        service.check(result)
        audit = service.audit(result['id'])
        timing = audit['timing']
        require(timing['enqueued_at'] <= timing['dequeued_at'] <= timing['finished_at'], 'invalid audit clock order')
        result.update(queue_ms=timing['queue_ms'], service_ms=timing['service_ms'], dp_rank=audit['dp_rank'],
                      generated_tokens=audit['response']['usage']['completion_tokens'],
                      prompt_tokens=audit['response']['usage']['prompt_tokens'])
        require(start <= result['start'] <= result['end'] <= end, 'invalid client clock order')
        # 同一轮的输入均已就绪；单独记录尚未提交的等待，避免将排队移到客户端误判为改善。
        result.update(client_wait_ms=1000*(result['start']-start), completion_ms=1000*(result['end']-start))
        if result['payload']['stream'] and audit['response']['choices'][0]['message']['content']:
            require(result['first_content'] is not None, 'missing first content timestamp')
    service.idle()
    wave['processes_after'] = process_snapshot(service.process.pid)
    service.save(label, wave)
    return wave


def stats(samples):
    """/** 轮耗时以Median/P95/CV汇总；吞吐使用实际输出数，首内容延迟仅覆盖SSE。 */"""
    elapsed = [s['elapsed_s'] for s in samples]
    records = [r for s in samples for r in s['records']]
    result = dict(median_batch_ms=1000*statistics.median(elapsed), p95_batch_ms=1000*percentile(elapsed, .95),
                  batch_cv_pct=100*statistics.pstdev(elapsed)/statistics.mean(elapsed),
                  output_tokens_per_s=statistics.median(sum(r['generated_tokens'] for r in s['records'])/s['elapsed_s'] for s in samples),
                  requests=len(records))
    for name, values in [('request', [1000*(r['end']-r['start']) for r in records]),
                         ('client_wait', [r['client_wait_ms'] for r in records]),
                         ('completion', [r['completion_ms'] for r in records]),
                         ('queue', [r['queue_ms'] for r in records]),
                         ('service', [r['service_ms'] for r in records]),
                         ('first_content', [1000*(r['first_content']-r['start']) for r in records if r['first_content'] is not None])]:
        result['median_' + name + '_ms'] = statistics.median(values) if values else None
        result['p95_' + name + '_ms'] = percentile(values, .95) if values else None
    return result


FIELDS = ['case', 'status', 'dp', 'tp', 'concurrency', 'requests', 'median_batch_ms', 'p95_batch_ms', 'batch_cv_pct',
          'median_request_ms', 'p95_request_ms', 'median_queue_ms', 'p95_queue_ms', 'median_service_ms', 'p95_service_ms',
          'median_client_wait_ms', 'p95_client_wait_ms', 'median_completion_ms', 'p95_completion_ms',
          'median_first_content_ms', 'p95_first_content_ms', 'output_tokens_per_s', 'speedup_vs_c1',
          'max_abs_error', 'mean_abs_error', 'failed_elements', 'backend', 'reason']


def save(output, rows, metadata):
    """/** 即使失败也保存四种报告和原始样本；未完成五轮不能报告整体PASS。 */"""
    complete = ([r['concurrency'] for r in rows] == metadata['concurrency_levels'] and
                all(len(r['samples']) == metadata['repeats'] for r in rows))
    status = 'PASS' if complete and all(r['status'] == 'PASS' for r in rows) else 'FAIL'
    (output / 'summary.json').write_text(json.dumps(dict(status=status, complete=complete, metadata=metadata, cases=rows), ensure_ascii=False, indent=2))
    def display(value):
        return f'{value:.3f}' if isinstance(value, float) else ('n/a' if value is None else str(value))
    values = [[display(r.get(k)) for k in FIELDS] for r in rows]
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(values)
    title = 'DP请求集服务验证' if metadata.get('profile') else 'DP混合长度服务诊断'
    lines = [f'# {title}：{status}', '', metadata['scope'], '', '| ' + ' | '.join(FIELDS) + ' |',
             '| ' + ' | '.join(['---']*len(FIELDS)) + ' |']
    lines += ['| ' + ' | '.join(v.replace('|', '\\|').replace('\n', ' ') for v in row) + ' |' for row in values]
    lines += ['', '```json', json.dumps(metadata, ensure_ascii=False, indent=2), '```']
    (output / 'summary.md').write_text('\n'.join(lines))
    sample_fields = ['case', 'repeat', 'input', 'prompt_tokens', 'output_tokens', 'latency_ms', 'queue_ms', 'service_ms',
                     'client_wait_ms', 'completion_ms', 'first_content_ms', 'dp_rank']
    samples = [[row['case'], i, r['input_name'], r['prompt_tokens'], r['generated_tokens'], 1000*(r['end']-r['start']),
                r['queue_ms'], r['service_ms'], r['client_wait_ms'], r['completion_ms'],
                None if r['first_content'] is None else 1000*(r['first_content']-r['start']), r['dp_rank']]
               for row in rows for i, sample in enumerate(row['samples']) for r in sample['records']]
    group_fields = ['case', 'input', 'requests', 'median_request_ms', 'p95_request_ms', 'median_queue_ms', 'p95_queue_ms',
                    'median_client_wait_ms', 'p95_client_wait_ms', 'median_completion_ms', 'p95_completion_ms',
                    'median_service_ms', 'p95_service_ms', 'median_first_content_ms', 'p95_first_content_ms']
    groups = [[row['case'], group, *[metrics[k] for k in group_fields[2:]]]
              for row in rows for group, metrics in row.get('by_input', {}).items()]
    with (output / 'inputs.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(group_fields)
        writer.writerows([[display(v) for v in row] for row in groups])
    write_xlsx(output / 'summary.xlsx', [('Cases', FIELDS, values), ('Requests', sample_fields, samples), ('By input', group_fields, groups),
        ('Metadata', ['key', 'value'], [[k, json.dumps(v, ensure_ascii=False)] for k, v in metadata.items()])])
    return status


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'generate', 'worker', 'build-cache'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--dp', type=int, choices=(2, 4))
    parser.add_argument('--tp', type=int, choices=(1, 2))
    parser.add_argument('--profile', choices=PROFILES, help='使用固定服务配置，仅比较C1和建议的C=DP；必须同时提供--requests')
    parser.add_argument('--requests', type=Path, help='JSONL请求文件；必须同时提供--profile')
    parser.add_argument('--prompt-lengths', type=int, nargs='+')
    parser.add_argument('--output-lengths', type=int, nargs='+')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args(argv)
    if args.profile:
        if not args.requests or args.dp is not None or args.tp is not None or args.prompt_lengths or args.output_lengths:
            parser.error('--profile requires --requests and cannot combine with --dp/--tp/length options')
        args.dp, args.tp = PROFILES[args.profile]
    elif args.requests or args.dp is None or args.tp is None:
        parser.error('use --dp and --tp for mixed lengths, or --profile and --requests for JSONL validation')
    args.prompt_lengths = args.prompt_lengths or [64, 512, 1024]
    args.output_lengths = args.output_lengths or [16, 96]
    if (args.repeats < 5 or not math.isfinite(args.timeout) or args.timeout < 5 or
            len(args.prompt_lengths) < 2 or len(set(args.prompt_lengths)) != len(args.prompt_lengths) or
            min(args.prompt_lengths) < 32 or max(args.prompt_lengths) > 2048 or
            min(args.output_lengths) < 2 or len(set(args.output_lengths)) != len(args.output_lengths) or
            max(args.prompt_lengths) + max(args.output_lengths) - 1 > 2048 or
            2*len(args.prompt_lengths)*len(args.output_lengths) < 2*args.dp):
        parser.error('requires distinct mixed lengths within capacity 2048, enough requests for 2*DP clients, repeats>=5 and timeout>=5')
    args.routing, args.prefix_cache = 'round_robin', False
    return args


def main(argv=None):
    args = parse_args(argv)
    levels = [1, args.dp] if args.profile else [1, args.dp, 2*args.dp]
    # 轮换执行位置，避免新增的C=DP始终夹在另两组之间；完整顺序写入报告。
    orders = [levels[i % len(levels):] + levels[:i % len(levels)] for i in range(args.repeats)]
    kind = 'dp_requests' if args.requests else 'dp_lengths'
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_{kind}_' + uuid.uuid4().hex[:8])
    (output / 'raw').mkdir(parents=True)
    print(f'REPORT: {output}', flush=True)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv] if argv is None else [sys.executable, __file__, *argv]),
        git=capture(['git', 'rev-parse', 'HEAD']), gpu=capture(['nvidia-smi', '-L']), cuda=capture(['nvcc', '--version']),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), seed=0, warmup=1, iterations=1, repeats=args.repeats,
        dp=args.dp, tp=args.tp, prompt_targets=args.prompt_lengths, output_limits=args.output_lengths,
        profile=args.profile, request_source=str(args.requests.resolve()) if args.requests else 'synthetic mixed lengths',
        concurrency_levels=levels, expected_case_count=len(levels), repeat_orders=orders,
        client_dispatch='fixed lanes: shuffled input indices lane, lane+C, ...; all inputs ready at wave start',
        dtype='FP16', kv_cache='FP16', prefix_cache=False, cuda_graph=True, custom_allreduce=args.tp == 2,
        routing=args.routing, request_log='on', eos='formal model config unchanged',
        scope='正式HTTP/SSE端到端诊断，含生产分词/排队/日志，不含测试校验与落盘。SSE首内容延迟不是原生首Token时间；'
              'queue_ms仅为正式HTTP等待队列，service_ms含分词及生成、不含最终审计落盘。每轮相同输入按repeat_orders轮换并发，'
              'client_wait_ms为整轮开始到客户端提交的时间，completion_ms为整轮开始到该请求完整接收的时间，'
              '后者含客户端等待，适用于本轮同时就绪的固定请求集合，不代表线上独立到达流量的响应延迟。'
              'speedup_vs_c1为同配置C1 Median轮耗时/当前耗时，不是DP扩展比。'
              '整卡显存是0.5秒周期采样的观测最大值，可能含其他进程且会漏掉瞬时峰值，不证明无泄漏。')
    if args.requests:
        metadata.update(prompt_targets=None, output_limits=None, correctness='exact tokens vs same-TP DP1; not answer quality')
        metadata['scope'] += 'JSONL请求验证只确认与DP1输出一致、协议正确和本输入集性能，不评价业务回答质量。'
    rows = [dict(case=f'dp{args.dp}_tp{args.tp}_c{c}', status='FAIL', reason='not completed', backend=BACKEND,
                 dp=args.dp, tp=args.tp, concurrency=c, samples=[]) for c in levels]
    service, memory = None, MemorySampler()
    current = rows[0]
    def interrupt(*_):
        raise KeyboardInterrupt('length diagnostic interrupted')
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        items = None
        if args.requests:
            source = args.requests.read_bytes()
            (output / 'requests.jsonl').write_bytes(source)
            metadata['requests_sha256'] = hashlib.sha256(source).hexdigest()
            items = read_requests(source.decode('utf-8'))
            require(len(items) >= args.dp, f'JSONL requires at least {args.dp} requests to exercise recommended concurrency')

        cache = args.build_cache.read_text()
        (output / 'CMakeCache.txt').write_text(cache)
        metadata['compile_architecture'] = [l for l in cache.splitlines() if l.startswith(('CUDA_ARCH:', 'CMAKE_CUDA_ARCHITECTURES:'))]
        flags = model_flags(args.model, args.generate, args.worker, output / 'reference-model', args.timeout, args.tp)
        references = {}
        print('RUN: independent_dp1_references', flush=True)
        with make_normal_llm_model(make_normal_parser('mixed lengths').parse_args(flags)) as model:
            tokenizer = model.hf_tokenizer
            inputs = (prepare_inputs(items, args.model.resolve(), tokenizer) if items is not None else
                      make_inputs(args.model.resolve(), tokenizer, args.prompt_lengths, args.output_lengths))
            random.Random(0).shuffle(inputs)
            (output / 'inputs.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2))
            for item in inputs:
                key = input_key(item['payload'])
                if key not in references:
                    print('REFERENCE: ' + item['name'], flush=True)
                    references[key] = model.generate(item['request'])
            (output / 'references.json').write_text(json.dumps(list(references.values()), ensure_ascii=False))
            if items is not None:
                answers = [dict(name=item['name'], text=tokenizer.decode(references[input_key(item['payload'])]['token_ids'],
                               skip_special_tokens=True, clean_up_tokenization_spaces=False)) for item in inputs]
                (output / 'reference_texts.json').write_text(json.dumps(answers, ensure_ascii=False, indent=2))
        service = MixedService(args, output, flags, references, tokenizer)
        print('RUN: formal_cli_ready', flush=True)
        metadata['ready'] = service.start()
        with memory:
            for row in rows:
                current = row
                print('WARMUP: ' + row['case'], flush=True)
                run_wave(service, inputs, row['concurrency'], 'warmup-' + row['case'])
            for repeat, order in enumerate(orders):
                for concurrency in order:
                    row = next(row for row in rows if row['concurrency'] == concurrency)
                    current = row
                    print(f'RUN: {row["case"]} {repeat+1}/{args.repeats}', flush=True)
                    wave = run_wave(service, inputs, row['concurrency'], f'r{repeat}-' + row['case'])
                    row['samples'].append(wave)
                    row.update(stats(row['samples']), **ZERO)
                    row['by_input'] = {item['name']: {key: value for key, value in
                        stats([dict(sample, records=[r for r in sample['records'] if r['input_name'] == item['name']])
                               for sample in row['samples']]).items()
                        if key == 'requests' or (key.endswith('_ms') and 'batch' not in key)} for item in inputs}
                    save(output, rows, metadata)
        require(not memory.errors, f'GPU memory sampling failed: {memory.errors}')
        for row in rows:
            row.update(status='PASS', reason='', speedup_vs_c1=rows[0]['median_batch_ms']/row['median_batch_ms'])
    except (Exception, KeyboardInterrupt) as error:
        current.update(status='FAIL', reason=f'{type(error).__name__}: {error}', **getattr(error, 'metrics', {}))
    finally:
        try:
            if service is not None:
                service.close()
                if all(row['status'] == 'PASS' for row in rows):
                    require(service.process.exitcode == 0, 'formal CLI did not exit cleanly')
                    for replica in service.ready['replicas']:
                        shutdown = json.loads((Path(replica['output']) / 'session/shutdown.json').read_text())
                        require(shutdown['shutdown_complete'] and shutdown['cache_after_unload'] == [EMPTY]*args.tp,
                                'native unload incomplete or leaked request KV')
                    metadata['shutdown_complete'] = True
        except Exception as error:
            current.update(status='FAIL', reason=current['reason'] + f'; cleanup: {error}')
        finally:
            signal.signal(signal.SIGTERM, previous)
        (output / 'memory.json').write_text(json.dumps(dict(samples=memory.samples, errors=memory.errors), indent=2))
        metadata['gpu_memory_observed'] = [dict(uuid=uid, samples=len(values),
            min_used_mib=min(v['memory.used'] for v in values), max_used_mib=max(v['memory.used'] for v in values),
            total_mib=values[0]['memory.total']) for uid in sorted({g['uuid'] for s in memory.samples for g in s['gpu']})
            for values in [[g for s in memory.samples for g in s['gpu'] if g['uuid'] == uid]]]
        status = save(output, rows, metadata)
    for row in rows:
        summary = {k: row.get(k) for k in ('case', 'status', 'reason', 'batch_cv_pct', 'p95_queue_ms', 'p95_request_ms',
                                         'p95_client_wait_ms', 'p95_completion_ms', 'p95_first_content_ms', 'output_tokens_per_s')}
        summary['by_input' if args.requests else 'short_requests'] = {
            name: {k: value[k] for k in ('p95_queue_ms', 'p95_request_ms', 'p95_client_wait_ms', 'p95_completion_ms')}
            for name, value in row.get('by_input', {}).items() if args.requests or name.startswith(f'p{min(args.prompt_lengths)}_')}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f'{status}: {output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
