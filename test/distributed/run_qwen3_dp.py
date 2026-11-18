#!/usr/bin/env python3
"""DP 真实模型验收：正式工厂/CLI、并发 HTTP、输出对照及回收；只运行已有产物。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import uuid

from run_process_tp import ROOT, capture, save_report
from run_qwen3_entrypoint import verify_native
from run_qwen3_http import call
from run_qwen3_prefill import require
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model
from fastllm_pytools.qwen3_process_chat import prepare_request
from fastllm_pytools.qwen3_process_session import BACKEND, GenerationCancelled, EMPTY
from fastllm_pytools.qwen3_process_server import ChatHTTPServer, ChatService
from fastllm_pytools import cli


def run_cli(flags, output, expected):
    """/** 通过当前源码的正式 CLI 生成文本，保留命令和原始输出用于复核。 */"""
    command = [sys.executable, '-m', 'fastllm_pytools.cli', 'run', *flags,
               '--prompt', '请用一句话介绍你自己。', '--max-new-tokens', '4']
    (output / 'cli.command.txt').write_text(shlex.join(command) + '\n')
    previous = sys.argv
    try:
        sys.argv = ['ftllm', *command[3:]]
        with (output / 'cli.stdout.log').open('w') as stdout, (output / 'cli.stderr.log').open('w') as stderr:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = cli.main()
        require(result in (0, None), 'CLI failed; inspect cli.stderr.log')
    finally:
        sys.argv = previous
    require((output / 'cli.stdout.log').read_text().rstrip('\n') == expected.rstrip('\n'), 'CLI text mismatch')
    return dict(max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0)


def main():
    """/** 对同 TP 大小的 DP=1 与 DP>1 使用同一输入、采样、缓存和 Graph 设置。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'generate', 'worker', 'build-cache'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dp', type=int, default=2)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--custom-allreduce', action='store_true')
    parser.add_argument('--prefix-cache', action='store_true')
    parser.add_argument('--dp-routing', choices=('round_robin', 'prefix', 'load'), default='load')
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    if args.dp_routing == 'prefix' and not args.prefix_cache:
        parser.error('prefix routing requires --prefix-cache')
    if args.dp < 2 or args.tp < 1:
        parser.error('requires dp>=2 and tp>=1')
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_dp_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
                    gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']),
                    cuda=capture(['nvcc', '--version']), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    seed=0, warmup=0, iterations=1, repeats=1, dp=args.dp, tp=args.tp,
                    cuda_graph=args.cuda_graph, custom_allreduce=args.custom_allreduce, prefix_cache=args.prefix_cache,
                    expected_case_count=12 if args.dp_routing == 'load' else (11 if args.dp_routing == 'prefix' else 10), dp_routing=args.dp_routing, report_title='Process DP public entrypoint',
                    report_description='同一正式工厂 DP=1 与多副本输出精确对照；固定 Token ID、EOS、采样和执行设置。'
                                       '生命周期检查误差为 n/a，不提供性能或独立数学精度结论。')
    rows = []

    def case(name, action):
        row = dict(case=name, backend=BACKEND, status='FAIL', reason='',
                   max_abs_error=None, mean_abs_error=None, failed_elements=None)
        try:
            row.update(action() or {})
            row['status'] = 'PASS'
        except Exception as error:
            row.update(getattr(error, 'metrics', {}))
            row['reason'] = f'{type(error).__name__}: {error}'
        rows.append(row)
        save_report(output, rows, metadata)
        require(row['status'] == 'PASS', f'{name}: {row["reason"]}')

    try:
        cache = args.build_cache.read_text()
        metadata['compile_architecture'] = [line for line in cache.splitlines()
                                            if line.startswith(('CUDA_ARCH:', 'CMAKE_CUDA_ARCHITECTURES:'))]
        flags = [str(args.model.resolve()), '--tp-backend', 'process', '--tp', str(args.tp),
                 '--tp-binary', str(args.generate.resolve()), '--tp-worker', str(args.worker.resolve()),
                 '--tp-output', str(output / 'models'), '--tp-timeout', '300']
        if args.cuda_graph:
            flags += ['--cuda-graph']
        if args.custom_allreduce:
            flags += ['--custom-allreduce']
        if args.prefix_cache:
            flags += ['--prefix-cache', 'true', '--page_size', '4']
        if args.dp_routing == 'load':
            # 同一真实页池容量用于单副本基准及DP，稳定覆盖50%压力阈值两侧。
            flags += ['--page_size', '16', '--tokens', '512']
            metadata.update(page_size=16, cache_tokens=512)
        prompt = '请用一句话介绍你自己。'
        reference = request = expected_text = None
        pressure_requests, pressure_references = {}, {}

        def baseline():
            nonlocal reference, request, expected_text
            with make_normal_llm_model(make_normal_parser('baseline').parse_args(flags)) as model:
                request = prepare_request(model.model_path, model.hf_tokenizer, prompt, max_new_tokens=4)
                reference = model.generate(request)
                expected_text = model.hf_tokenizer.decode(reference['token_ids'], skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False)
                metadata['request'] = request
                (output / 'reference.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2))
                if args.dp_routing == 'load':
                    for name, length in (('high', 384), ('low', 32)):
                        ids = (request['token_ids'] * (length // len(request['token_ids']) + 1))[:length]
                        pressure_requests[name] = dict(request, token_ids=ids, max_new_tokens=4, eos_token_ids=[])
                        pressure_references[name] = model.generate(pressure_requests[name])
                        (output / f'reference-{name}.json').write_text(json.dumps(pressure_references[name], indent=2))
                return verify_native(reference, reference)
        case('single_replica_reference', baseline)

        flags += ['--dp', str(args.dp), '--dp-routing', args.dp_routing]
        model = None
        try:
            def startup():
                nonlocal model
                model = make_normal_llm_model(make_normal_parser('DP').parse_args(flags))
                require(model.dp_size == args.dp and model.tp_size == args.tp, 'wrong DP/TP sizes')
                require([r.session.devices for r in model.replicas] ==
                        [list(range(i * args.tp, (i + 1) * args.tp)) for i in range(args.dp)], 'wrong device assignment')
                require(len({w['pid'] for w in model.ready['workers']}) == args.dp * args.tp, 'workers are shared')
                require(all(r.ready['model_load_count'] == 1 for r in model.replicas), 'replica loaded more than once')
                metadata['allocation'] = model.plan
                return dict(workers=model.ready['workers'], logs=str(model.output))
            case('pool_startup', startup)

            def sequential():
                reports = [model.generate(request) for _ in range(2 * args.dp)]
                for i, report in enumerate(reports):
                    if args.dp_routing != 'prefix':
                        require(report['dp_rank'] == i % args.dp, 'idle replica rotation mismatch')
                    verify_native(report, reference)
                return dict(max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0,
                            dp_ranks=[r['dp_rank'] for r in reports])
            case('sequential_reuse', sequential)

            def concurrent():
                barrier = threading.Barrier(args.dp, timeout=30)
                def generate():
                    barrier.wait()
                    return model.generate(request)
                with ThreadPoolExecutor(args.dp) as pool:
                    reports = list(pool.map(lambda _: generate(), range(args.dp)))
                for report in reports:
                    verify_native(report, reference)
                require(len({r['dp_rank'] for r in reports}) == args.dp, 'not all replicas handled concurrent requests')
                require(len({r['dp_request_id'] for r in reports}) == args.dp, 'DP request IDs collided')
                return dict(max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0,
                            dp_ranks=[r['dp_rank'] for r in reports])
            case('concurrent_outputs', concurrent)

            if args.dp_routing == 'load':
                def pressure_case(name):
                    barrier = threading.Barrier(args.dp * 2, timeout=30)
                    def generate():
                        barrier.wait()
                        return model.generate(pressure_requests[name])
                    with ThreadPoolExecutor(args.dp * 2) as pool:
                        reports = list(pool.map(lambda _: generate(), range(args.dp * 2)))
                    for report in reports:
                        verify_native(report, pressure_references[name])
                        samples = report.get('load_samples', [])
                        require(len(samples) >= 2 and samples[-1]['phase'] == 'complete', 'missing live KV evidence')
                        require(samples[-1]['kv_cache_usage'] == 0, 'KV pressure did not reset after cleanup')
                        peak = max(sample['kv_cache_usage'] for sample in samples)
                        require(peak > 0.5 if name == 'high' else 0 < peak <= 0.5, f'{name} KV threshold not exercised: {peak}')
                        for sample in samples:
                            workers = sample['workers']
                            require(len(workers) == args.tp, 'KV sample missing TP rank')
                            expected = max(1-w['free_pages']/w['total_pages'] if w['total_pages'] else 0 for w in workers)
                            require(abs(expected-sample['kv_cache_usage']) < 1e-9, 'KV usage differs from actual worker pages')
                        decision = report['dp_routing']
                        for snapshot in decision['replicas']:
                            w, r, u = snapshot['waiting'], snapshot['running'], snapshot['kv_cache_usage']
                            require(abs(snapshot['score']-(w+r+w*6*max(0, u-0.5))) < 1e-9, 'routing score mismatch')
                        require(decision['replicas'][report['dp_rank']]['score'] ==
                                min(s['score'] for s in decision['replicas']), 'request was not routed to minimum load')
                    require(all(s['waiting'] == s['running'] == 0 for s in model.load_snapshot()), 'request accounting leaked')
                    return dict(max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0,
                                prompt_tokens=len(pressure_requests[name]['token_ids']),
                                reports=[dict(dp_rank=r['dp_rank'], token_ids=r['token_ids'],
                                              decision=r['dp_routing'], load_samples=r['load_samples']) for r in reports])
                # 高压后再测低压，同时验证可回收的前缀页不被误计为占用。
                case('live_kv_high_pressure', lambda: pressure_case('high'))
                case('live_kv_low_pressure', lambda: pressure_case('low'))

            if args.dp_routing == 'prefix':
                def prefix_route():
                    model.clear_prefix_cache()
                    cold = model.generate(request)
                    warm = model.generate(request)
                    metrics = verify_native(warm, reference)
                    p = warm['graph'][0]['prefix_cache']
                    expected = (len(request['token_ids']) - 1) // p['page_len'] * p['page_len']
                    require(p['directory_version'] == 2 and expected > 0 and p['last_reused_tokens'] == expected,
                            'DP route did not restore complete FT pages')
                    require(warm['dp_rank'] == cold['dp_rank'] and
                            warm['dp_routing']['estimated_reused_tokens'] == expected,
                            'DP route directory differs from TP common reuse')
                    return metrics
                case('paged_prefix_route', prefix_route)

            def cancellation():
                event = threading.Event()
                try:
                    model.generate(dict(request, max_new_tokens=128, eos_token_ids=[]), cancel_event=event,
                                   on_token=lambda _: event.set())
                except GenerationCancelled as error:
                    require(error.report['cache_after_error'] == [EMPTY] * args.tp, 'cancelled request leaked KV')
                else:
                    raise AssertionError('long request finished without cancellation acknowledgement')
                return verify_native(model.generate(request), reference)
            case('cancel_and_reuse', cancellation)

            def text_stream():
                stream = model.stream_response(prompt, max_length=4)
                try:
                    text = ''.join(stream)
                finally:
                    stream.close()
                require(text == expected_text, 'stream text differs from baseline')
                return dict(max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0)
            case('text_stream', text_stream)
            case('clear_prefix_cache', lambda: dict(replica_count=len(model.clear_prefix_cache())))

            def http():
                service = ChatService(model, model.hf_tokenizer, model.model_path, 'qwen3', model.output,
                                      queue_size=args.dp * 2, request_timeout=240)
                server = thread = None
                try:
                    server = ChatHTTPServer(('127.0.0.1', 0), service)
                    thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=0.05))
                    thread.start()
                    payload = dict(model='qwen3', messages=[dict(role='user', content=prompt)], max_tokens=4)
                    with ThreadPoolExecutor(args.dp) as pool:
                        results = list(pool.map(lambda i: call(server.server_address, 'POST', '/v1/chat/completions',
                                                               payload, f'dp-{i}'), range(args.dp)))
                    for i, (status, response) in enumerate(results):
                        require(status == 200 and response['id'] == f'chatcmpl-dp-{i}', 'wrong HTTP caller/status')
                        require(response['choices'][0]['message']['content'] == expected_text, 'HTTP output mismatch')
                    require(service.health()['model_load_count'] == args.dp, 'HTTP lost DP metadata')
                    return dict(max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0)
                finally:
                    try:
                        service.close()
                    finally:
                        if thread is not None:
                            server.shutdown()
                            thread.join(10)
                        if server is not None:
                            server.server_close()
            case('http_concurrent', http)
        finally:
            if model is not None:
                model.close()

        def shutdown():
            require(model.state == 'closed', 'pool did not close')
            require(all(not Path(f"/proc/{w['pid']}").exists() for w in model.ready['workers']), 'worker not reaped')
            require(all(r.session.shutdown['shutdown_complete'] for r in model.replicas), 'replica shutdown incomplete')
        case('pool_shutdown', shutdown)
        case('formal_cli_run', lambda: run_cli(flags, output, expected_text))
    except (Exception, KeyboardInterrupt) as error:
        if not rows or rows[-1]['status'] == 'PASS':
            rows.append(dict(case='runner_failure', backend=BACKEND, status='FAIL', reason=f'{type(error).__name__}: {error}'))
    status = save_report(output, rows, metadata)
    print(f'{status}: {output}')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
