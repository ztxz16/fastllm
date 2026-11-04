"""PD deployment controller and streaming ingress for Fast KV Manager.

Each configured group is an execution group, not one KV replica. Requests
enter D's normal API/tokenizer; native D scheduling submits prefill to P and
commits the transferred state before decoding. The HTTP router carries no KV.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def add_router_args(parser):
    parser.add_argument('action', choices=('render', 'start', 'status', 'stop', 'serve'))
    parser.add_argument('--config', required=True, help='PD deployment JSON file')


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('version') != 1 or not config.get('model_id'):
        raise ValueError('PD configuration requires version=1 and a checkpoint model_id')
    if not config.get('prefill') or not config.get('decode'):
        raise ValueError('Configure at least one prefill group and one decode group')
    directory = Path(config['state_dir']).expanduser().resolve()
    config['state_dir'] = str(directory)
    peers = [group['endpoint'] for group in config['prefill']]
    if len(set(peers)) != len(peers):
        raise ValueError('Prefill endpoints must be unique')
    names = set()
    transport = config.get('transport', {'type': 'tcp'})
    if transport.get('type', 'tcp') not in ('tcp', 'rdma', 'auto'):
        raise ValueError('transport.type must be tcp, rdma or auto')
    for role in ('prefill', 'decode'):
        for group in config[role]:
            name = group['name']
            if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
                raise ValueError('Execution group names must be unique simple identifiers')
            names.add(name)
            if not group.get('url', '').startswith('http://'):
                raise ValueError('Each group needs its reachable internal HTTP url for readiness checks')
            command = group['command']
            if not isinstance(command, list) or not all(isinstance(v, str) for v in command):
                raise ValueError('command must be an argv array, not a shell command')
            if 'serve' not in command and 'server' not in command:
                raise ValueError('Execution groups must start with ftllm serve')
            if any(arg == '--pd' or arg.startswith('--pd=') for arg in command):
                raise ValueError('The router owns --pd configuration')
            pd = {'role': role, 'model_id': config['model_id'],
                  'transport': transport.get('type', 'tcp'),
                  'rdma_devices': group.get('rdma_devices', transport.get('rdma_devices', [])),
                  'max_inflight_bytes': transport.get('max_inflight_bytes', 8 << 30)}
            if role == 'prefill':
                pd['listen'] = group.get('listen', group['endpoint'])
                pd['chunk_tokens'] = group.get('chunk_tokens', 2048)
                pd['cache_bytes'] = group.get('cache_bytes', 1 << 30)
            else:
                pd['prefill'] = group.get('prefill', peers)
                if not pd['prefill'] or not set(pd['prefill']).issubset(peers):
                    raise ValueError('Decode prefill endpoints must name configured prefill groups')
            group['command'] = command + ['--pd', json.dumps(pd, separators=(',', ':'))]
            group.setdefault('state_dir', str(directory / name))
            group.setdefault('python', sys.executable)
            group.setdefault('env', {})
            if config.get('api_key'):
                group['health_headers'] = {'Authorization': 'Bearer ' + config['api_key']}
    proxy = config.setdefault('router', {})
    defaults = {'host': '0.0.0.0', 'port': 8000, 'queue_limit_per_replica': 128,
                'request_timeout_seconds': 10800, 'affinity_ttl_seconds': 3600,
                'max_sessions': 10000, 'max_body_bytes': 64 << 20,
                'health_interval_seconds': 5, 'queue_timeout_seconds': 300}
    for key, value in defaults.items():
        proxy.setdefault(key, value)
    config.setdefault('api_key', '')
    config.setdefault('name', 'fastkv-pd')
    config.setdefault('stop_timeout_seconds', 20)
    # Reuse the tested HTTP streaming, cancellation and bounded queue code.
    config['replicas'] = [{'name': node['name'], 'url': node['url'], 'max_inflight': node.get('max_inflight', 64)}
                          for node in config['decode']]
    if any(type(node['max_inflight']) is not int or node['max_inflight'] <= 0 for node in config['replicas']):
        raise ValueError('max_inflight must be a positive integer')
    return config


def create_router(config, directory):
    import asyncio
    from aiohttp import ClientTimeout, web
    from .deploy.router import Router

    class PDRouter(Router):
        async def check_health(self):
            await super().check_health()
            headers = {'Authorization': 'Bearer ' + config['api_key']} if config['api_key'] else {}

            async def check(node):
                try:
                    async with self.client.get(node['url'] + '/v1/models', headers=headers,
                                               timeout=ClientTimeout(total=3)) as response:
                        response.raise_for_status()
                        await response.json()
                    return node['endpoint'], True
                except Exception:
                    return node['endpoint'], False

            self.prefill_health = dict(await asyncio.gather(*(check(n) for n in config['prefill'])))
            for index, node in enumerate(config['decode']):
                peers = node.get('prefill', self.prefill_health)
                # Native scheduling rotates through this static set. Until
                # membership changes are supported, every assigned P must be
                # ready before this D group can safely accept new requests.
                self.healthy[index] &= all(self.prefill_health[peer] for peer in peers)

        async def health(self, request):
            ready = all(self.healthy)
            return web.json_response({'status': 'ok' if ready else 'degraded',
                'healthy_decode_groups': sum(self.healthy), 'decode_groups': len(self.backends),
                'prefill': self.prefill_health}, status=200 if ready else 503)

    return PDRouter(config, directory)


def node_operation(action, node):
    command = [node['python'], '-m', 'ftllm.pdnode', action]
    ssh = node.get('ssh')
    if ssh:
        remote = ['env'] + [str(k) + '=' + str(v) for k, v in node['env'].items()] + command
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                   '-p', str(ssh.get('port', 22))]
        if ssh.get('identity'):
            command += ['-i', str(Path(ssh['identity']).expanduser())]
        command += [ssh['host'], shlex.join(remote)]
    result = subprocess.run(command, input=json.dumps(node), text=True,
                            capture_output=True, timeout=60,
                            env={**os.environ, **node.get('env', {})})
    if result.returncode:
        raise RuntimeError(f"{node['name']}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def main(args):
    path = str(Path(args.config).expanduser().resolve())
    config = load_config(path)
    if args.action == 'render':
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    if args.action == 'serve':
        from aiohttp import web
        from .deploy.process import configure_nofile
        configure_nofile(config, apply=True)
        router = create_router(config, Path(config['state_dir']) / 'routing')
        web.run_app(router.app(), host=config['router']['host'], port=config['router']['port'],
                    handler_cancellation=True, access_log=None, shutdown_timeout=config['stop_timeout_seconds'])
        return 0
    ingress = config.get('ingress', {})
    proxy_directory = str(Path(ingress.get('state_dir', config['state_dir'])) / 'router')
    proxy_python = ingress.get('python', sys.executable)
    proxy = {'name': 'router', 'python': proxy_python, 'state_dir': proxy_directory,
             'command': [proxy_python, '-m', 'ftllm.cli', 'router', 'serve', '--config',
                         str(Path(proxy_directory) / 'config.json')],
             'config': json.loads(Path(path).read_text()),
             'env': ingress.get('env', {key: os.environ[key] for key in
                                        ('PYTHONPATH', 'LD_LIBRARY_PATH') if key in os.environ})}
    proxy['url'] = ingress.get('url', 'http://127.0.0.1:' + str(config['router']['port']))
    proxy['health_path'] = '/health'
    if ingress.get('ssh'):
        proxy['ssh'] = ingress['ssh']
    nodes = config['prefill'] + config['decode'] + [proxy]
    if args.action == 'stop':
        nodes.reverse()
    results = []
    for node in nodes:
        try:
            results.append(node_operation(args.action, node))
        except Exception as error:
            # Keep successful nodes recorded so a later stop is always scoped
            # to owned processes, including partially completed deployment.
            results.append({'name': node['name'], 'error': str(error)})
            if args.action == 'start':
                break
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return int(any('error' in result for result in results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_router_args(parser)
    raise SystemExit(main(parser.parse_args()))
