#!/usr/bin/env python3
"""正式CLI的DP HTTP/SSE持续稳定性验收；只使用已有原生产物，不作为性能基准。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import http.client
import json
import math
import multiprocessing
import os
from pathlib import Path
import shlex
import signal
import socket
import sys
import threading
import time
import traceback
import uuid

from run_process_tp import ROOT, capture, save_report
from run_qwen3_entrypoint import verify_native
from run_qwen3_http import call, read_event, verify_stream_error, wait_for
from run_qwen3_prefill import require
from fastllm_pytools import cli
from fastllm_pytools.qwen3_process_chat import prepare_request
from fastllm_pytools.qwen3_process_session import BACKEND, EMPTY
from fastllm_pytools.tp_graph import validate_graph, validate_allreduce, validate_prefix
from fastllm_pytools.util import make_normal_llm_model, make_normal_parser


PROMPT = '请从1开始按顺序列出1000种常见物品，每行一种，写出编号、名称并解释它的用途，不要省略。'
ZERO = dict(max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0)


def payload(tokens=8, stream=False):
    """/** 固定同一Prompt，仅混合8/32个输出Token上限及HTTP/SSE；保留正式EOS策略。 */"""
    value = dict(model='qwen3', messages=[dict(role='user', content=PROMPT)], max_tokens=tokens,
                 temperature=0.0, top_k=0, top_p=1.0, seed=0, stream=stream)
    if stream:
        value['stream_options'] = dict(include_usage=True)
    return value


def exchange(address, data, request_id, on_content=None):
    """/** 使用真实HTTP连接，保留完整响应；SSE按帧读取，回调可模拟客户端主动断连。 */"""
    connection = http.client.HTTPConnection(*address, timeout=180)
    response = None
    result = dict(id=request_id, payload=data, status=None, body=None, error=None)
    try:
        connection.request('POST', '/v1/chat/completions', json.dumps(data).encode(),
                           {'Content-Type': 'application/json', 'X-Request-ID': request_id})
        response = connection.getresponse()
        result['status'] = response.status
        if not data.get('stream') or response.status != 200:
            result['body'] = json.loads(response.read())
        else:
            require('text/event-stream' in response.getheader('Content-Type', ''), 'missing SSE content type')
            result['body'] = events = []
            while True:
                event = read_event(response)
                if event is None:
                    break
                events.append(event)
                if on_content and isinstance(event, dict) and event.get('choices') and event['choices'][0]['delta'].get('content'):
                    callback, on_content = on_content, None
                    if callback() == 'disconnect':
                        break
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        if response is not None:
            response.close()
        connection.close()
    return result


def launch_cli(flags, output):
    """/** 在独立Linux子进程执行正式CLI；继承环境，SIGTERM由正式服务处理。 */"""
    os.setsid()
    sys.argv = ['ftllm', 'serve', *flags]
    with (output / 'server.stdout.log').open('w') as stdout, (output / 'server.stderr.log').open('w') as stderr:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = cli.main() or 0
            except BaseException:
                traceback.print_exc()
                raise
            raise SystemExit(code)


def process_snapshot(root):
    """/** 只读本服务及其后代的PID身份、RSS、线程和FD；starttime用于避免清理已复用PID。 */"""
    processes = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            pid = int(path.parent.name)
            processes[pid] = dict(pid=pid, ppid=int(fields[1]), starttime=fields[19],
                                  rss_bytes=int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
                                  threads=int(fields[17]), fds=len(list((path.parent / 'fd').iterdir())))
        except (OSError, ValueError, IndexError):
            continue
    owned = {root}
    while True:
        expanded = owned | {pid for pid, row in processes.items() if row['ppid'] in owned}
        if expanded == owned:
            return [processes[pid] for pid in sorted(owned) if pid in processes]
        owned = expanded


def alive(row):
    """/** 以PID及启动时钟确认原进程仍存在；PID复用不视作泄漏，也不向新进程发送信号。 */"""
    try:
        return Path(f'/proc/{row["pid"]}/stat').read_text().rsplit(')', 1)[1].split()[19] == row['starttime']
    except (OSError, IndexError):
        return False


class ServiceRun:
    """/** 管理正式CLI、真实HTTP及审计证据；异常时只清理本次启动的进程。 */"""

    def __init__(self, args, output, flags, references, tokenizer):
        self.args, self.output, self.flags = args, output, flags
        self.references, self.tokenizer = references, tokenizer
        self.process = None
        self.ready = None
        self.audits, self.audit_paths, self.owned = {}, set(), []

    def start(self):
        """/** 使用端口0避免冲突，从正式ready文件取得监听地址并核对工厂/副本。 */"""
        flags = self.flags + ['--dp', str(self.args.dp), '--dp-routing', self.args.routing,
                             '--tp-output', str(self.output / 'server'), '--host', '127.0.0.1', '--port', '0',
                             '--model_name', 'qwen3', '--tp-queue-size', str(2*self.args.dp),
                             '--tp-request-timeout', str(self.args.timeout)]
        (self.output / 'server.command.txt').write_text(shlex.join([sys.executable, '-m', 'fastllm_pytools.cli', 'serve', *flags]))
        self.process = multiprocessing.get_context('fork').Process(target=launch_cli, args=(flags, self.output))
        self.process.start()

        def ready():
            require(self.process.is_alive(), f'CLI exited {self.process.exitcode}; inspect server.stderr.log')
            for path in (self.output / 'server').glob('*/ready.json'):
                try:
                    value = json.loads(path.read_text())
                    if 'port' in value:
                        return value
                except json.JSONDecodeError:
                    pass
        self.ready = wait_for(ready, self.args.timeout)
        self.logs = Path(self.ready['logs'])
        self.address = ('127.0.0.1', self.ready['port'])
        require(self.ready['model_factory'] == 'make_normal_llm_model', 'not the formal factory')
        require(self.ready['dp_size'] == self.args.dp and self.ready['dp_routing'] == self.args.routing, 'wrong DP/routing')
        self.owned = process_snapshot(self.process.pid)
        require({w['pid'] for w in self.ready['workers']} <= {row['pid'] for row in self.owned}, 'worker outside service process tree')
        return dict(server_pid=self.process.pid, workers=self.ready['workers'], logs=str(self.logs))

    def health(self):
        status, value = call(self.address, 'GET', '/health')
        require(status == 200 and value['status'] == 'ready', f'unhealthy service: {status}, {value}')
        require(value['workers'] == self.ready['workers'] and value['model_load_count'] == self.args.dp,
                'worker identity or model load count changed')
        return value

    def idle(self):
        """/** 等待HTTP队列及执行槽清空，再确认所有原始子进程仍存活。 */"""
        def drained():
            value = self.health()
            return not value['active_request_ids'] and not value['queued_request_ids']
        wait_for(drained, self.args.timeout)
        require(all(alive(row) for row in self.owned), 'service lost a process')

    def audit(self, request_id):
        """/** 增量读取正式审计，拒绝重复外部ID；不扫描或读取测试之外的日志。 */"""
        for path in self.logs.glob('http-*.json'):
            if path in self.audit_paths:
                continue
            try:
                row = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            require(row['id'] not in self.audits, f'duplicate completion: {row["id"]}')
            self.audits[row['id']] = row
            self.audit_paths.add(path)
        return self.audits.get(request_id)

    def native(self, record):
        """/** 按正式审计中的副本号/原生请求号定位结果，防止跨副本或跨请求混配。 */"""
        require(type(record['native_id']) is int and type(record['dp_rank']) is int, 'missing native request identity')
        replica = self.ready['replicas'][record['dp_rank']]
        return json.loads((Path(replica['output']) / 'session' / f'request-{record["native_id"]:06d}.response.json').read_text())

    def check(self, result):
        """/** 比较实际Token、文本、usage及SSE终帧；按实际冷热状态选取独立DP1基准。 */"""
        require(result['error'] is None and result['status'] == 200, f'HTTP failure: {result}')
        record = wait_for(lambda: self.audit(result['id']), 5)
        require(record['status'] == 200 and record['request'] == result['payload'], 'audit identity/payload mismatch')
        report = self.native(record)
        graph = report['graph']
        validate_graph(graph, self.args.tp, True)
        validate_allreduce(graph, self.args.tp, self.args.tp == 2, require_progress=self.args.tp == 2)
        validate_prefix(graph, self.args.tp, True)
        reused = graph[0]['prefix_cache']['last_reused_tokens']
        require(reused in (0, len(report['request']['token_ids'])-1), 'unexpected partial prefix')
        reference = self.references[result['payload']['max_tokens']]['warm' if reused else 'cold']
        verify_native(report, reference)
        require(report['status'] == 'PASS' and report['workers'] == self.ready['replicas'][record['dp_rank']]['ready']['workers'],
                'native backend/workers/status mismatch')
        body = record['response']
        require(body['id'] == 'chatcmpl-' + result['id'] and body['model'] == 'qwen3', 'response identity mismatch')
        tokens = report['token_ids']
        expected = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        require(body['choices'][0]['message'] == dict(role='assistant', content=expected), 'decoded output mismatch')
        require(body['choices'][0]['finish_reason'] == ('stop' if report['finish_reason'] == 'eos' else 'length'), 'finish mismatch')
        count = len(report['request']['token_ids'])
        require(body['usage'] == dict(prompt_tokens=count, completion_tokens=len(tokens), total_tokens=count+len(tokens)), 'usage mismatch')
        if result['payload']['stream']:
            events = result['body']
            require(events and events[-1] == '[DONE]' and events.count('[DONE]') == 1, 'missing/duplicate SSE DONE')
            chunks = events[:-1]
            require(all(isinstance(e, dict) and e.get('object') == 'chat.completion.chunk' and
                        e.get('id') == body['id'] and e.get('model') == 'qwen3' and 'error' not in e for e in chunks), 'SSE identity/error')
            require(len({e['created'] for e in chunks}) == 1, 'SSE creation identity changed')
            choices = [e['choices'][0] for e in chunks if e['choices']]
            require(choices[0]['delta'].get('role') == 'assistant', 'missing SSE role')
            require(all(c['index'] == 0 and c['finish_reason'] is None for c in choices[:-1]), 'premature SSE finish')
            require(choices[-1]['delta'] == {} and choices[-1]['finish_reason'] == body['choices'][0]['finish_reason'], 'SSE final mismatch')
            require(''.join(c['delta'].get('content', '') for c in choices) == expected, 'SSE text mismatch')
            require([e['usage'] for e in chunks if not e['choices']] == [body['usage']], 'SSE usage mismatch')
        else:
            require(result['body'] == body, 'HTTP response differs from audit')
        return len(tokens)

    def save(self, name, value):
        (self.output / 'raw' / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2))

    def control(self, name, method, path, data=None, request_id=None):
        """/** 保存拒绝与取消操作的原始HTTP证据，断言失败也能定位具体控制请求。 */"""
        result = call(self.address, method, path, data, request_id)
        self.save(name, dict(method=method, path=path, payload=data, id=request_id,
                             status=result[0], response=result[1]))
        return result

    def recovery(self, name):
        """/** 异常后使用相同正式入口检查输出和空队列，不只检查health。 */"""
        self.idle()
        result = exchange(self.address, payload(8, True), name + '-recovery')
        self.save(name + '-recovery', result)
        self.check(result)
        self.idle()

    def soak(self, index, seconds):
        """/** 持续并发混合HTTP/SSE和短长输出；保留生产日志，墙钟只作稳定性持续时长。 */"""
        concurrency = 2*self.args.dp
        barrier = threading.Barrier(concurrency)
        started = time.monotonic()
        deadline = started + seconds

        def client(lane):
            barrier.wait(timeout=10)
            records = []
            turn = 0
            while turn == 0 or time.monotonic() < deadline:
                choice = (lane + turn) % 4
                records.append(exchange(self.address, payload(8 if choice < 2 else 32, bool(choice % 2)),
                                        f'round-{index}-lane-{lane}-{turn}'))
                if records[-1]['error'] or records[-1]['status'] != 200:
                    break
                turn += 1
            return records
        with ThreadPoolExecutor(concurrency) as pool:
            records = [record for group in pool.map(client, range(concurrency)) for record in group]
        elapsed = time.monotonic() - started
        self.save(f'round-{index}', records)
        generated = sum(self.check(record) for record in records)
        self.idle()
        observed = process_snapshot(self.process.pid)
        self.save(f'processes-{index}', observed)
        return dict(**ZERO, requests=len(records), generated_tokens=generated, elapsed_s=elapsed,
                    http_requests=sum(not r['payload']['stream'] for r in records),
                    sse_requests=sum(r['payload']['stream'] for r in records), processes=observed)

    @contextmanager
    def occupied(self, name):
        """/** 以真实长SSE占满副本，观测首段内容后才施压；不修改调度或阻塞服务回调。 */"""
        ids = [f'{name}-active-{i}' for i in range(self.args.dp)]
        signals = [threading.Event() for _ in ids]
        with ThreadPoolExecutor(self.args.dp + 2*self.args.dp) as pool:
            futures = [pool.submit(exchange, self.address, payload(1024, True), rid,
                                   lambda i=i: signals[i].set()) for i, rid in enumerate(ids)]
            try:
                require(all(event.wait(30) for event in signals), 'long SSE did not emit content')
                require(all(not f.done() for f in futures) and set(ids) <= set(self.health()['active_request_ids']),
                        'long request finished before pressure; cannot certify overlap')
                yield pool, ids, futures
            finally:
                if self.process.is_alive():
                    for rid in ids:
                        call(self.address, 'DELETE', '/v1/requests/' + rid)
                for rid, future in zip(ids, futures):
                    self.save(rid, future.result(timeout=self.args.timeout))

    def cancelled(self, rid, status, *, queued=False):
        """/** 验证取消审计及原生确认/KV回收；排队取消必须没有原生请求号。 */"""
        record = wait_for(lambda: self.audit(rid), self.args.timeout)
        require(record['status'] == status, f'wrong cancellation status: {record}')
        if queued:
            require(record['native_id'] is None and record['dp_rank'] is None, 'queued cancellation reached native')
        else:
            native = self.native(record)
            require(native['status'] == 'FAIL' and native.get('cancelled') is True and native['reusable'] is True,
                    'native cancellation not acknowledged')
            require(native['cache_after_error'] == [EMPTY]*self.args.tp, 'cancellation leaked request KV')

    def overload(self):
        """/** 验证队列满429、重复ID409、排队取消和全部活动请求取消，随后确认服务可继续输出。 */"""
        with self.occupied('overload') as (pool, ids, active):
            waiting = [f'overload-queued-{i}' for i in range(2*self.args.dp)]
            pending = [pool.submit(exchange, self.address, payload(8), rid) for rid in waiting]
            wait_for(lambda: set(self.health()['queued_request_ids']) == set(waiting), 10)
            require(self.control('queue-overflow', 'POST', '/v1/chat/completions', payload(8), 'overload-reject')[0] == 429, 'queue overflow accepted')
            require(self.control('duplicate-active', 'POST', '/v1/chat/completions', payload(8), ids[0])[0] == 409, 'duplicate active ID accepted')
            require(self.control('cancel-queued', 'DELETE', '/v1/requests/' + waiting[0])[0] == 202, 'queued cancellation rejected')
            cancelled = pending[0].result(timeout=10)
            self.save(waiting[0], cancelled)
            require(cancelled['status'] == 499, 'queued caller not cancelled')
            self.cancelled(waiting[0], 499, queued=True)
            for rid in ids:
                require(self.control('cancel-' + rid, 'DELETE', '/v1/requests/' + rid)[0] == 202, 'active cancellation rejected')
            for rid, future in zip(ids, active):
                result = future.result(timeout=self.args.timeout)
                verify_stream_error((result['status'], result['body']), 499)
                self.cancelled(rid, 499)
            for rid, future in zip(waiting[1:], pending[1:]):
                result = future.result(timeout=self.args.timeout)
                self.save(rid, result)
                self.check(result)
        self.recovery('overload')
        require(self.audit('overload-reject') is None, 'rejected request acquired a job')
        return dict(**ZERO, active_cancelled=len(ids), queued_cancelled=1, rejected_status=429)

    def disconnect(self, stream):
        """/** 客户端主动关闭连接；SSE在收到内容后关闭，普通HTTP在活动状态关闭。 */"""
        rid = 'disconnect-sse' if stream else 'disconnect-http'
        if stream:
            result = exchange(self.address, payload(1024, True), rid, lambda: 'disconnect')
            self.save(rid, result)
        else:
            data = json.dumps(payload(1024)).encode()
            with socket.create_connection(self.address, timeout=10) as connection:
                connection.sendall((f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n'
                                    f'X-Request-ID: {rid}\r\nContent-Length: {len(data)}\r\n\r\n').encode() + data)
                wait_for(lambda: rid in self.health()['active_request_ids'], 10)
                connection.shutdown(socket.SHUT_RDWR)
            self.save(rid, dict(payload=payload(1024), id=rid, event='client socket closed after active health observation'))
        record = wait_for(lambda: self.audit(rid), self.args.timeout)
        require(record['status'] == 499, 'disconnect not cancelled')
        dispatched = type(record['native_id']) is int and record['native_id'] > 0 and type(record['dp_rank']) is int
        if stream or dispatched:
            self.cancelled(rid, 499)
        else:
            require(record['native_id'] in (None, 0) and record['dp_rank'] is None, 'invalid pre-dispatch cancellation identity')
        self.recovery(rid)
        return dict(**ZERO, native_cancelled=dispatched)

    def shutdown(self):
        """/** 带活动请求、排队请求及慢请求体发送SIGTERM；检查审计、原生释放和PID身份消失。 */"""
        with self.occupied('shutdown') as (pool, ids, active):
            queued = pool.submit(exchange, self.address, payload(8), 'shutdown-queued')
            wait_for(lambda: 'shutdown-queued' in self.health()['queued_request_ids'], 10)
            with socket.create_connection(self.address, timeout=10) as slow:
                slow.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n'
                             b'Content-Type: application/json\r\nExpect: 100-continue\r\n\r\n')
                accepted = b''
                while b'\r\n\r\n' not in accepted and len(accepted) < 2048:
                    part = slow.recv(2048)
                    require(part, 'slow-body connection closed before acceptance')
                    accepted += part
                require(accepted.startswith(b'HTTP/1.1 100 '), 'slow-body handler did not accept request')
                slow.sendall(b'{')
                self.process.terminate()
                self.process.join(self.args.timeout + 45)
            require(self.process.exitcode == 0, f'CLI shutdown failed/hung: {self.process.exitcode}')
            self.save('shutdown-queued', queued.result(timeout=10))
            self.cancelled('shutdown-queued', 503, queued=True)
            for rid in ids:
                self.cancelled(rid, 503)
            require(json.loads((self.logs / 'shutdown.json').read_text())['shutdown_complete'], 'missing CLI shutdown')
            for replica in self.ready['replicas']:
                native = json.loads((Path(replica['output']) / 'session/shutdown.json').read_text())
                require(native['shutdown_complete'] is True, 'incomplete native shutdown')
                require(native['cache_after_unload'] == [EMPTY]*self.args.tp, 'shutdown leaked KV')
                require(all(p['prefix_cache']['entries'] == 0 and p['prefix_cache']['device_bytes'] == 0
                            for p in native['graph_after_unload']), 'shutdown retained prefix snapshots')
            wait_for(lambda: not any(alive(row) for row in self.owned), 10)
        return dict(reaped_processes=len(self.owned), shutdown_complete=True)

    def close(self):
        """/** 失败清理先请求正常停服；超时仅向仍匹配身份的本次后代发信号，不能覆盖原失败。 */"""
        if self.process is None:
            return
        owned = self.owned or process_snapshot(self.process.pid)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(self.args.timeout + 45)
        for signum in (signal.SIGTERM, signal.SIGKILL):
            for row in reversed(owned):
                if alive(row):
                    try:
                        os.kill(row['pid'], signum)
                    except ProcessLookupError:
                        pass
            self.process.join(2)
        require(not self.process.is_alive() and not any(alive(row) for row in owned), 'cleanup left owned processes')


def parse_args(argv):
    """/** 限定已验收后端；默认五轮各至少30秒，时间参数只控制稳定性负载，不改变推理。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'generate', 'worker', 'build-cache'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dp', type=int, choices=(2, 4), default=4)
    parser.add_argument('--tp', type=int, choices=(1, 2), default=1)
    parser.add_argument('--routing', choices=('round_robin', 'prefix'), default='prefix')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--round-seconds', type=float, default=30)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args(argv)
    if args.repeats < 5 or not math.isfinite(args.round_seconds) or args.round_seconds <= 0 or not math.isfinite(args.timeout) or args.timeout < 5:
        parser.error('requires repeats>=5, finite round-seconds>0 and timeout>=5')
    return args


def main(argv=None):
    """/** 建立独立冷/热输出基准后启动正式CLI；逐case保存四种报告，失败与中断均回收。 */"""
    args = parse_args(argv)
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_dp_service_' + uuid.uuid4().hex[:8])
    (output / 'raw').mkdir(parents=True, exist_ok=False)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv] if argv is None else [sys.executable, __file__, *argv]),
                    git=capture(['git', 'rev-parse', 'HEAD']), gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']),
                    cuda=capture(['nvcc', '--version']), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    dp=args.dp, tp=args.tp, routing=args.routing, seed=0, warmup=1, iterations=1,
                    concurrency=2*args.dp, queue_capacity=2*args.dp,
                    repeats=args.repeats, round_seconds=args.round_seconds, expected_case_count=args.repeats+7,
                    dtype='FP16', kv_cache='FP16', cuda_graph=True, custom_allreduce=args.tp == 2,
                    cache_lifecycle='empty at startup; one shared prompt; no clears between rounds',
                    output_limits=[8, 32], fault_output_limit=1024, transport='loopback HTTP/1.1',
                    prefix_cache=True, request_log='on', eos='formal model configuration, unchanged',
                    report_title='DP HTTP/SSE service stability',
                    report_description='正式CLI及真实HTTP/SSE；固定Prompt、混合8/32输出上限，按实际冷热状态对照DP1。'
                                       '保留生产落盘，不作为性能基准；elapsed_s仅记录持续时长，资源样本不证明不存在任何泄漏。')
    rows, service = [], None

    def interrupt(*_):
        """/** 将验收器SIGTERM转为可回收的中断；子进程仍由正式CLI安装自己的停服处理器。 */"""
        raise KeyboardInterrupt('service acceptance interrupted by SIGTERM')
    previous_term = signal.signal(signal.SIGTERM, interrupt)

    def case(name, action):
        print('RUN: ' + name, flush=True)
        row = dict(case=name, backend=BACKEND, status='FAIL', reason='', max_abs_error=None, mean_abs_error=None, failed_elements=None)
        rows.append(row)
        try:
            row.update(action() or {})
            row['status'] = 'PASS'
        except BaseException as error:
            row.update(reason=f'{type(error).__name__}: {error}', **getattr(error, 'metrics', {}))
            raise
        finally:
            save_report(output, rows, metadata)

    try:
        cache = args.build_cache.read_text()
        (output / 'CMakeCache.txt').write_text(cache)
        metadata['compile_architecture'] = [line for line in cache.splitlines() if line.startswith(('CUDA_ARCH:', 'CMAKE_CUDA_ARCHITECTURES:'))]
        flags = [str(args.model.resolve()), '--tp-backend', 'process', '--tp', str(args.tp),
                 '--tp-binary', str(args.generate.resolve()), '--tp-worker', str(args.worker.resolve()),
                 '--tp-timeout', str(args.timeout), '--tp-output', str(output / 'reference-model'),
                 '--tp-request-timeout', str(args.timeout),
                 '--prefix-cache', 'true', '--cuda-graph']
        if args.tp == 2:
            flags += ['--custom-allreduce']
        references, tokenizer = {}, None

        def reference():
            nonlocal tokenizer
            with make_normal_llm_model(make_normal_parser('service reference').parse_args(flags)) as model:
                tokenizer = model.hf_tokenizer
                for count in (8, 32):
                    request = prepare_request(model.model_path, tokenizer, PROMPT, max_new_tokens=count)
                    model.clear_prefix_cache()
                    references[count] = {state: model.generate(request) for state in ('cold', 'warm')}
            (output / 'references.json').write_text(json.dumps(references, ensure_ascii=False))
            return dict(**ZERO, prompt_tokens=len(request['token_ids']))
        case('dp1_cold_warm_reference', reference)
        service = ServiceRun(args, output, flags, references, tokenizer)
        case('formal_cli_ready', service.start)
        case('warmup_http_sse', lambda: service.soak('warmup', 0))
        for index in range(args.repeats):
            case(f'mixed_http_sse_round_{index+1:02d}', lambda index=index: service.soak(index, args.round_seconds))
        case('queue_full_duplicate_and_cancel', service.overload)
        case('http_disconnect_and_recovery', lambda: service.disconnect(False))
        case('sse_disconnect_and_recovery', lambda: service.disconnect(True))
        case('sigterm_inflight_queued_and_slow_client', service.shutdown)
    except (Exception, KeyboardInterrupt) as error:
        if not rows or rows[-1]['status'] == 'PASS':
            rows.append(dict(case='runner_failure', backend=BACKEND, status='FAIL', reason=f'{type(error).__name__}: {error}'))
    finally:
        try:
            if service is not None:
                try:
                    service.close()
                except Exception as error:
                    rows.append(dict(case='cleanup_failure', backend=BACKEND, status='FAIL', reason=str(error)))
        finally:
            signal.signal(signal.SIGTERM, previous_term)
    status = save_report(output, rows, metadata)
    print(f'{status}: {output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
