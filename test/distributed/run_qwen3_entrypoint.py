#!/usr/bin/env python3
"""验证正式 ftllm 模型工厂、run/server 入口；使用已有产物，不编译。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import uuid

from run_process_tp import ROOT, capture, case_process, save_report, wait_case
from run_qwen3_prefill import require
from run_qwen3_http import call, stream_call, verify_completion, verify_stream, wait_for
from qwen3_process_chat import BACKEND, prepare_request, run_native
from qwen3_process_session import EMPTY


def verify_native(report, reference):
    require(report['backend'] == BACKEND and report['request'] == reference['request'], 'wrong backend/request')
    require(report['cache_after_generate'] == [EMPTY for _ in report['workers']], 'request KV leaked')
    actual, expected = report['token_ids'], reference['token_ids']
    require(len(actual) == len(expected), 'wrong output length')
    errors = [abs(a-b) for a, b in zip(actual, expected)]
    metrics = dict(max_abs_error=float(max(errors, default=0)), mean_abs_error=sum(errors)/max(1, len(errors)),
                   failed_elements=sum(e != 0 for e in errors))
    if metrics['failed_elements']:
        error = AssertionError('generated token IDs differ from the native baseline')
        error.metrics = metrics
        raise error
    require(report['finish_reason'] == reference['finish_reason'], 'wrong finish reason')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('generate', 'worker', 'model', 'build-cache', 'package-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--custom-allreduce', action='store_true')
    parser.add_argument('--prefix-cache', action='store_true')
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_entry_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
        git_status=capture(['git', 'status', '--short']), seed=0, warmup=0, iterations=1, repeats=1,
        gpu=capture(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']),
        cuda=capture(['nvcc', '--version']),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), devices=args.devices,
        expected_case_count=13 if args.prefix_cache else 12, execution='decode_cuda_graph' if args.cuda_graph else 'eager',
        custom_allreduce=args.custom_allreduce, graph_warmup_per_request=1 if args.cuda_graph else 0, prefix_cache=args.prefix_cache, kv_cache='FP16 request-local + optional immutable snapshots',
        report_title='FastLLM process TP public entrypoint',
        report_description='正式工厂/CLI 与原生生成的 Token ID 精确对照；不代表性能或独立数学精度验证。')
    rows = []

    def case(name, action):
        row = dict(case=name, backend=BACKEND, status='FAIL', reason='')
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
        metadata['build_cache'] = args.build_cache.read_text()
        metadata['compile_architecture'] = [line for line in metadata['build_cache'].splitlines()
                                            if line.startswith(('CUDA_ARCH:', 'CMAKE_CUDA_ARCHITECTURES:'))]
        package_root = args.package_root.resolve()
        require((package_root / 'ftllm/process_tp.py').is_file(), 'build fastllm_process_tp_package first')
        sys.path.insert(0, str(package_root))
        from ftllm.util import make_normal_parser, make_normal_llm_model
        metadata['model_factory_module'] = make_normal_llm_model.__module__
        os.environ['USE_TORCH'] = '0'
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        prompt = '请用一句话介绍你自己。'
        request = prepare_request(args.model.resolve(), tokenizer, prompt, max_new_tokens=32)
        metadata['prompt_token_ids'] = request['token_ids']
        metadata['input_shape'] = dict(batch=1, prompt_tokens=len(request['token_ids']), max_new_tokens=32)
        reference_dir = output / 'reference'
        reference_dir.mkdir()
        reference = {}

        def baseline():
            reference.update(run_native(args.generate.resolve(), args.worker.resolve(), args.model.resolve(),
                                        request, args.devices, reference_dir))
        case('native_reference', baseline)
        expected_text = tokenizer.decode(reference['token_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        common = [str(args.model.resolve()), '--tp_backend', 'process', '--tp', args.devices]
        if args.cuda_graph:
            common += ['--cuda-graph']
        if args.custom_allreduce:
            common += ['--custom-allreduce']
        if args.prefix_cache:
            common += ['--prefix-cache', 'true']
        # 包目录自动发现产物，故意不传显式 binary/worker，验证正式打包布局。
        model = make_normal_llm_model(make_normal_parser('accept').parse_args(common + ['--tp-output', str(output / 'factory')]))
        try:
            workers = model.ready['workers']

            def check_text(stream=False):
                text = ''.join(model.stream_response(prompt, max_length=32)) if stream else model.response(prompt, max_length=32)
                require(text == expected_text, 'decoded text mismatch')
                path = model.output / 'session' / f'request-{model.session._next_id-1:06d}.response.json'
                report = json.loads(path.read_text())
                require(report['workers'] == workers and model.ready['model_load_count'] == 1, 'model was reloaded')
                metrics = verify_native(report, reference)
                if args.custom_allreduce:
                    ar = report['graph'][0]['allreduce']
                    metrics.update(backend='qwen3_process/custom-ipc-fp32-tp2+nccl-large',
                                   custom_calls=ar['custom_calls'], nccl_calls=ar['nccl_calls'])
                return metrics
            case('factory_response', check_text)
            case('factory_stream_reuse', lambda: check_text(True))
            if args.prefix_cache:
                def clear_prefix():
                    p = model.session._graph_state[0]['prefix_cache']
                    require(p['last_reused_tokens'] == len(request['token_ids'])-1 and p['hits'] == 1,
                            'formal factory did not reuse the actual prompt')
                    model.clear_prefix_cache()
                    result = check_text()
                    require(model.session._graph_state[0]['prefix_cache']['last_reused_tokens'] == 0,
                            'formal clear left a stale snapshot')
                    return result
                case('factory_prefix_clear', clear_prefix)

            def reject_recover():
                before = model.session._next_id
                try:
                    model.generate(dict(request, token_ids=[request['token_ids'][0]] * 2049))
                except ValueError:
                    pass
                else:
                    raise AssertionError('overlong prompt accepted')
                require(model.session._next_id == before, 'invalid request reached GPU')
                return check_text()
            case('factory_reject_recover', reject_recover)
        finally:
            model.close()

        def factory_closed():
            require(model.state == 'closed' and model.session.shutdown['shutdown_complete'], 'model did not close')
            require(all(not Path(f'/proc/{w["pid"]}').exists() for w in workers), 'worker survived model close')
        case('factory_shutdown', factory_closed)

        env = dict(os.environ, PYTHONPATH=str(package_root) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        cli = [sys.executable, '-m', 'ftllm.cli']

        def command_case(name, command):
            (output / f'{name}.command.txt').write_text(shlex.join(command) + '\n')
            with (output / f'{name}.stdout.log').open('wb') as stdout, (output / f'{name}.stderr.log').open('wb') as stderr:
                with case_process(command, env=env, stdout=stdout, stderr=stderr) as process:
                    return wait_case(process, 240)

        def cli_run():
            code = command_case('cli-run', cli + ['run', *common, '--prompt', prompt, '--max-new-tokens', '32',
                                                 '--tp-output', str(output / 'cli-run')])
            require(code == 0, 'CLI run failed; see cli-run.stderr.log')
            require((output / 'cli-run.stdout.log').read_text() == expected_text + '\n', 'CLI text mismatch')
            directory = next((output / 'cli-run').iterdir()) / 'session'
            require(json.loads((directory / 'shutdown.json').read_text())['shutdown_complete'], 'CLI did not shut down')
            return verify_native(json.loads((directory / 'request-000001.response.json').read_text()), reference)
        case('cli_run', cli_run)

        command = cli + ['server', *common, '--host', '127.0.0.1', '--port', '0', '--model_name', 'qwen3',
                         '--tp-output', str(output / 'server')]
        (output / 'cli-server.command.txt').write_text(shlex.join(command) + '\n')
        with (output / 'cli-server.stdout.log').open('wb') as stdout, (output / 'cli-server.stderr.log').open('wb') as stderr:
            with case_process(command, env=env, stdout=stdout, stderr=stderr) as process:
                info = {}

                def health():
                    def read_ready():
                        paths = list((output / 'server').glob('*/ready.json'))
                        if not paths:
                            exited = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                            require(exited is None, 'CLI server exited before ready; see cli-server.stderr.log')
                            return None
                        try:
                            return json.loads(paths[0].read_text())
                        except json.JSONDecodeError:
                            return None
                    info.update(wait_for(read_ready, timeout=180))
                    require(info['backend'] == BACKEND and info['model_factory'] == 'make_normal_llm_model', 'wrong model factory/backend')
                    status, body = call((info['host'], info['port']), 'GET', '/health')
                    require(status == 200 and body['model_load_count'] == 1 and body['workers'] == info['workers'], 'wrong health response')
                case('cli_server_health', health)
                address, directory = (info['host'], info['port']), Path(info['logs'])
                payload = dict(model='qwen3', messages=[dict(role='user', content=prompt)], max_tokens=32,
                               temperature=0, top_k=0, top_p=1, seed=0)
                case('cli_http', lambda: verify_completion(directory, 'normal',
                    call(address, 'POST', '/v1/chat/completions', payload, 'normal'), reference, tokenizer, info['workers']))
                case('cli_sse', lambda: verify_stream(directory, 'stream',
                    stream_call(address, dict(payload, stream=True, stream_options=dict(include_usage=True)), 'stream', directory),
                    reference, tokenizer, info['workers'], True))

                def reject_http():
                    status, body = call(address, 'POST', '/v1/chat/completions', dict(payload, tools=[]))
                    require(status == 400 and 'error' in body, 'unsupported HTTP option accepted')
                    return verify_completion(directory, 'recover', call(address, 'POST', '/v1/chat/completions', payload, 'recover'),
                                             reference, tokenizer, info['workers'])
                case('cli_http_reject_recover', reject_http)

                def shutdown():
                    os.kill(process.pid, signal.SIGTERM)
                    require(wait_case(process, 60) == 0, 'server exit failed')
                    require(json.loads((directory / 'shutdown.json').read_text())['shutdown_complete'], 'missing shutdown acknowledgment')
                    require(all(not Path(f'/proc/{w["pid"]}').exists() for w in info['workers']), 'server worker survived shutdown')
                case('cli_server_shutdown', shutdown)

        def reject_cli():
            log_root = output / 'rejected-cli'
            code = command_case('reject-cli', cli + ['server', *common, '--api_key', 'must-not-ignore', '--tp-output', str(log_root)])
            require(code != 0 and 'does not support api_key' in (output / 'reject-cli.stderr.log').read_text(), 'auth flag silently ignored')
            require(not log_root.exists(), 'rejected configuration started model')
        case('cli_reject_unsupported', reject_cli)
    except Exception as error:
        metadata['error'] = f'{type(error).__name__}: {error}'
    finally:
        status = save_report(output, rows, metadata)
        print(f'{status}: {output}')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
