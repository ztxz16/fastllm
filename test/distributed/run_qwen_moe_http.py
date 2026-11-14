#!/usr/bin/env python3
"""通过正式CLI启动真实MoE HTTP/SSE服务，验证并发、队列、取消、断连和SIGTERM。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import sys
import time

from run_qwen_moe_full import Suite, require
from run_process_tp import ROOT, case_process, save_report, wait_case
from run_qwen3_http import call, stream_call, wait_for
from fastllm_pytools.qwen_moe_process_session import BACKEND


def main():
    """/** 真实HTTP不走测试专用推理入口；失败也保存已完成case及原生进程日志。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--build-cache', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    suite = Suite(args)
    directory = suite.output / 'service'
    command = [sys.executable, '-m', 'fastllm_pytools.cli', 'serve', str(args.model.resolve()),
        '--tp-backend', 'process', '--tp', '1', '--attention-dp', '4', '--ep', '4', '--dp', '1',
        '--tp-binary', str(args.binary.resolve()), '--tp-worker', str(args.worker.resolve()),
        '--cuda-graph', '--prefix-cache', 'true', '--dp-routing', 'prefix',
        '--tp-output', str(directory), '--tp-timeout', str(args.timeout), '--tp-request-timeout', '30',
        '--tp-queue-size', '2', '--host', '127.0.0.1', '--port', '0', '--model_name', 'moe']
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'tools'), PYTHONNOUSERSITE='1')
    payload = dict(model='moe', messages=[dict(role='user', content='请用一句话介绍北京。')],
                   max_tokens=8, temperature=0, top_k=0, top_p=1, seed=20261114)
    try:
        with (suite.output / 'stdout.log').open('wb') as stdout, (suite.output / 'stderr.log').open('wb') as stderr:
            with case_process(command, env=environment, stdout=stdout, stderr=stderr) as process:
                def ready():
                    require(process.poll() is None, 'CLI exited before ready; inspect stderr.log')
                    for path in directory.glob('*/ready.json'):
                        try:
                            return json.loads(path.read_text())
                        except json.JSONDecodeError:
                            pass
                info = wait_for(ready, args.timeout)
                require(info['backend'] == BACKEND and info['model_factory'] == 'make_normal_llm_model', 'wrong factory/backend')
                require(info['attention_dp_size'] == info['ep_size'] == 4 and len(info['workers']) == 4, 'wrong GPU mapping')
                address, logs = (info['host'], info['port']), Path(info['logs'])
                suite.metadata['workers'] = info['workers']

                def healthy():
                    status, body = call(address, 'GET', '/health')
                    require(status == 200 and body['workers'] == info['workers'], 'service lost workers')
                    return body

                def idle():
                    return wait_for(lambda: not healthy()['active_request_ids'] and not healthy()['queued_request_ids'], 90)

                baseline = {}
                def normal(_):
                    status, body = call(address, 'POST', '/v1/chat/completions', payload, 'normal')
                    require(status == 200 and body['choices'][0]['message']['content'], f'HTTP failed: {status} {body}')
                    baseline.update(body)
                    return body
                suite.case('formal_chinese_http', normal)

                def sse(_):
                    idle()
                    status, events = stream_call(address, dict(payload, stream=True, stream_options=dict(include_usage=True)),
                                                 'sse', suite.output)
                    require(status == 200 and events[-1] == '[DONE]', 'SSE missing DONE')
                    text = ''.join(e['choices'][0]['delta'].get('content', '') for e in events if isinstance(e, dict) and e.get('choices'))
                    require(text == baseline['choices'][0]['message']['content'], 'HTTP/SSE output differs')
                    return dict(events=events)
                suite.case('formal_sse_exact_text', sse)

                def queue_test(_):
                    idle()
                    with ThreadPoolExecutor(8) as pool:
                        jobs = [pool.submit(call, address, 'POST', '/v1/chat/completions', payload, f'burst-{i}') for i in range(8)]
                        results = [f.result() for f in jobs]
                    require(any(s == 429 for s, _ in results) and any(s == 200 for s, _ in results), 'queue admission/429 not exercised')
                    require(all(s in (200, 429) for s, _ in results), 'unexpected concurrent service error')
                    for status, body in results:
                        if status == 200:
                            require(body['choices'][0]['message']['content'] == baseline['choices'][0]['message']['content'], 'concurrent output corrupted')
                    return results
                suite.case('bounded_queue_concurrent_outputs', queue_test)

                def disconnect(_):
                    idle()
                    result = stream_call(address, dict(payload, max_tokens=128, stream=True), 'disconnect', suite.output,
                                         on_content=lambda: 'disconnect')
                    idle()
                    healthy()
                    return dict(stream=result)
                suite.case('client_disconnect_releases_rank', disconnect)

                def shutdown(_):
                    idle()
                    with ThreadPoolExecutor(1) as pool:
                        future = pool.submit(call, address, 'POST', '/v1/chat/completions', dict(payload, max_tokens=128), 'sigterm')
                        wait_for(lambda: 'sigterm' in healthy()['active_request_ids'], 30)
                        os.kill(process.pid, signal.SIGTERM)
                        require(wait_case(process, 90) == 0, 'SIGTERM exit failed')
                        try:
                            result = future.result(30)
                        except (OSError, ConnectionError) as error:
                            result = str(error)
                    require(json.loads((logs / 'shutdown.json').read_text())['shutdown_complete'], 'missing shutdown acknowledgment')
                    require(all(not Path(f'/proc/{w["pid"]}').exists() for w in info['workers']), 'worker survived SIGTERM')
                    return dict(result=result, workers=info['workers'])
                suite.case('sigterm_during_active_request', shutdown)

        # 超时由独立服务的极短截止时间确定触发，不依赖模型是否会提前产生EOS。
        timeout_directory = suite.output / 'deadline-service'
        command[command.index('--tp-output') + 1] = str(timeout_directory)
        command[command.index('--tp-request-timeout') + 1] = '0.000001'
        def deadline(_):
            with (suite.output / 'deadline.stdout.log').open('wb') as log, (suite.output / 'deadline.stderr.log').open('wb') as errors:
                with case_process(command, env=environment, stdout=log, stderr=errors) as proc:
                    def deadline_ready():
                        require(proc.poll() is None, 'deadline service startup failed')
                        for path in timeout_directory.glob('*/ready.json'):
                            try:
                                return json.loads(path.read_text())
                            except json.JSONDecodeError:
                                pass
                    ready_info = wait_for(deadline_ready, args.timeout)
                    endpoint = (ready_info['host'], ready_info['port'])
                    results = [call(endpoint, 'POST', '/v1/chat/completions', payload, f'deadline-{i}') for i in range(2)]
                    require(all(status == 504 for status, body in results), 'request deadline did not reject promptly')
                    require(call(endpoint, 'GET', '/health')[0] == 200, 'deadline poisoned service')
                    os.kill(proc.pid, signal.SIGTERM)
                    require(wait_case(proc, 90) == 0, 'deadline service cleanup failed')
                    require(all(not Path(f'/proc/{w["pid"]}').exists() for w in ready_info['workers']), 'deadline worker survived')
                    return results
        suite.case('request_deadline_does_not_poison_service', deadline)
    except Exception as error:
        suite.rows.append(dict(case='service_start_or_cleanup', backend=BACKEND, status='FAIL', failed_elements=1, reason=str(error)))
    finally:
        status = save_report(suite.output, suite.rows, suite.metadata)
        print(f'{status}: {suite.output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
