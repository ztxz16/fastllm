"""Deployment-only schema; model options are parsed by the existing serve parser."""
import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys

from ..util import make_normal_parser, add_server_args

PACKAGE = __package__.rsplit('.', 1)[0]


def fields(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'{label}: unknown fields: {sorted(unknown)}')


def integer(value, name, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name}: expected integer in [{minimum}, {maximum or "infinity"}]')
    return value


def seconds(value, name):
    if type(value) not in (int, float) or not 0 < value < float('inf'):
        raise ValueError(f'{name}: expected positive finite seconds')
    return value


def environment(value, name):
    if not isinstance(value, dict) or any(not isinstance(k, str) or not k or '=' in k or '\0' in k
            or not isinstance(v, str) or '\0' in v for k, v in value.items()):
        raise ValueError(f'{name}: expected string environment variables')
    if 'CUDA_VISIBLE_DEVICES' in value:
        raise ValueError(f'{name}: use replicas[].gpus for CUDA_VISIBLE_DEVICES')
    return dict(value)


def path_from(value, base):
    if not isinstance(value, str) or not value or '\0' in value:
        raise ValueError('Expected a nonempty path')
    return str((base / Path(value).expanduser()).resolve())


def parse_serve_args(argv):
    if not isinstance(argv, list) or not argv or any(not isinstance(s, str) or '\0' in s for s in argv):
        raise ValueError('serve_args must be a nonempty argv list')
    # No second copy of model flags, defaults, aliases or type conversions.
    parser = make_normal_parser('FastLLM deployment worker', add_help=False)
    parser.allow_abbrev = False
    add_server_args(parser)
    def error(message):
        raise ValueError('serve_args: ' + message)
    parser.error = error
    args = parser.parse_args(argv)
    explicit = set(args._auto_explicit)
    if explicit & {'host', 'port', 'startup_progress'}:
        raise ValueError('Worker host/port/startup-progress are owned by deploy; configure replica ports separately')
    if not args.model and not args.path:
        raise ValueError('serve_args must specify a model or --path')
    # Legacy positional JSON is applied during model loading and can overwrite
    # even explicit CLI bind ports. Do not let it bypass deployment ownership.
    if args.model and args.model.lower().endswith('.json'):
        raise ValueError('Use model path and explicit serve_args; legacy positional JSON can override worker ports')
    return args


def load_config(filename):
    source = Path(filename).expanduser().resolve()
    data = json.loads(source.read_text())
    return resolve_config(data, source)


def load_target(filename):
    # Stop/status must work after a model directory disappears or serve_args
    # are edited. Their authority is the recorded deployment, not new argv.
    source = Path(filename).expanduser().resolve()
    data = json.loads(source.read_text())
    name = data.get('name', '')
    if data.get('schema_version') != 1 or not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Expected schema_version=1 and a simple deployment name')
    return {'name': name, 'state_dir': path_from(data.get('state_dir', '.ftllm-deploy/'+name), source.parent)}


def resolve_config(data, source):
    source = Path(source).resolve()
    base = source.parent
    fields(data, ('schema_version', 'name', 'state_dir', 'serve_args', 'env', 'worker', 'replicas',
                  'router', 'startup_timeout_seconds', 'stop_timeout_seconds', 'sequential_start', 'nofile'), 'deployment')
    if data.get('schema_version') != 1 or not re.fullmatch(r'[A-Za-z0-9_-]+', data.get('name', '')):
        raise ValueError('Expected schema_version=1 and a simple deployment name')
    args = parse_serve_args(data['serve_args'])
    worker = data.get('worker', {})
    fields(worker, ('python', 'script'), 'worker')
    python = worker.get('python', sys.executable)
    if not isinstance(python, str) or not python:
        raise ValueError('worker.python must be a Python executable')
    if '/' in python:
        # Resolving a venv's Python symlink escapes its site-packages.
        # Make the invocation absolute while preserving the executable path.
        python = os.path.abspath(base / Path(python).expanduser())
    else:
        python = shutil.which(python)
    if not python or not os.access(python, os.X_OK):
        raise ValueError('worker.python is not executable')
    entry = [path_from(worker['script'], base)] if worker.get('script') else ['-m', PACKAGE + '.server']
    if worker.get('script') and not Path(entry[0]).is_file():
        raise ValueError('worker.script does not exist')
    common_env = environment(data.get('env', {}), 'env')
    router = copy.deepcopy(data.get('router', {}))
    fields(router, ('host', 'port', 'max_inflight_per_replica', 'queue_limit_per_replica',
                    'queue_timeout_seconds', 'request_timeout_seconds', 'max_sessions',
                    'affinity_ttl_seconds', 'health_interval_seconds', 'max_body_bytes'), 'router')
    host = router.setdefault('host', '127.0.0.1')
    if not isinstance(host, str) or not host or any(c in host for c in ('/', '\0', '\n')):
        raise ValueError('Invalid router host')
    port = integer(router.get('port'), 'router.port', maximum=65535)
    router.setdefault('queue_limit_per_replica', 0)
    router.setdefault('queue_timeout_seconds', 300)
    router.setdefault('request_timeout_seconds', 3600)
    router.setdefault('affinity_ttl_seconds', 3600)
    router.setdefault('max_sessions', 10000)
    router.setdefault('health_interval_seconds', 5)
    router.setdefault('max_body_bytes', 128 * 1024 * 1024)
    for key in ('queue_timeout_seconds', 'request_timeout_seconds', 'affinity_ttl_seconds', 'health_interval_seconds'):
        seconds(router[key], 'router.' + key)
    integer(router['queue_limit_per_replica'], 'router.queue_limit_per_replica', minimum=0)
    integer(router['max_sessions'], 'router.max_sessions')
    integer(router['max_body_bytes'], 'router.max_body_bytes')
    capacity = router.get('max_inflight_per_replica', args.max_batch)
    integer(capacity, 'router.max_inflight_per_replica (required when --max_batch is automatic)')
    router['max_inflight_per_replica'] = capacity
    replicas = data.get('replicas')
    if not isinstance(replicas, list) or not replicas:
        raise ValueError('At least one replica is required')
    roles, names, ports, devices = [], set(), {port}, set()
    for index, replica in enumerate(replicas):
        fields(replica, ('name', 'gpus', 'port', 'env', 'max_inflight'), 'replica')
        name = replica.get('name', f'replica{index}')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names or name == 'router':
            raise ValueError('Replica names must be distinct simple names other than router')
        names.add(name)
        gpu_ids = replica.get('gpus')
        if not isinstance(gpu_ids, list) or any(type(v) not in (int, str) for v in gpu_ids):
            raise ValueError('replica.gpus must be a list of device IDs (empty for CPU)')
        gpu_ids = [str(v) for v in gpu_ids]
        if any(not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)', v) for v in gpu_ids):
            raise ValueError('Invalid GPU device ID')
        gpu_ids = [str(int(v)) if v.isdecimal() else v for v in gpu_ids]
        if len(set(gpu_ids)) != len(gpu_ids) or devices.intersection(gpu_ids):
            raise ValueError('Replica GPU assignments overlap')
        devices.update(gpu_ids)
        port = integer(replica.get('port'), 'replica.port', maximum=65535)
        if port in ports:
            raise ValueError('Replica and router ports must be distinct')
        ports.add(port)
        limit = integer(replica.get('max_inflight', capacity), 'replica.max_inflight')
        if args.max_batch > 0 and limit > args.max_batch:
            raise ValueError('Router max_inflight exceeds worker --max_batch')
        env = {**common_env, **environment(replica.get('env', {}), 'replica.env'),
               'CUDA_VISIBLE_DEVICES': ','.join(gpu_ids)}
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = os.pathsep.join(path_from(p or '.', base) for p in env['PYTHONPATH'].split(os.pathsep))
        else:
            env['PYTHONPATH'] = control_pythonpath()
        roles.append({'name': name, 'gpus': gpu_ids, 'port': port, 'max_inflight': limit,
                      'url': f'http://127.0.0.1:{port}', 'env': env, 'cwd': str(base),
                      'command': [python, *entry, *data['serve_args'], '--host', '127.0.0.1',
                                  '--port', str(port), '--startup-progress', 'ndjson']})
    sequential = data.get('sequential_start', True)
    if type(sequential) is not bool:
        raise ValueError('sequential_start must be a boolean')
    nofile = data.get('nofile')
    if nofile is not None:
        integer(nofile, 'nofile')
    return {'schema_version': 1, 'name': data['name'], 'source_config': str(source),
            'state_dir': path_from(data.get('state_dir', '.ftllm-deploy/' + data['name']), base),
            'replicas': roles, 'router': router, 'api_key': args.api_key,
            'serve_args': list(data['serve_args']), 'sequential_start': sequential, 'nofile': nofile,
            'startup_timeout_seconds': seconds(data.get('startup_timeout_seconds', 900), 'startup_timeout_seconds'),
            'stop_timeout_seconds': seconds(data.get('stop_timeout_seconds', 30), 'stop_timeout_seconds')}


def public_plan(config):
    """Avoid publishing authentication values in render/status output."""
    result = copy.deepcopy(config)
    if result['api_key']:
        result['api_key'] = '<redacted>'
    def hide(argv):
        result = list(argv)
        for i, value in enumerate(result):
            if value == '--api_key' and i+1 < len(result):
                result[i+1] = '<redacted>'
            elif value.startswith('--api_key='):
                result[i] = '--api_key=<redacted>'
        return result
    result['serve_args'] = hide(result['serve_args'])
    for replica in result['replicas']:
        replica['command'] = hide(replica['command'])
        replica['env'] = {k: '<redacted>' if re.search(r'PASSWORD|TOKEN|SECRET|API_KEY|PRIVATE_KEY|ACCESS_KEY|AUTHORIZATION|CREDENTIAL', k, re.I) else v
                          for k, v in replica['env'].items()}
    result['aggregate_max_inflight'] = sum(r['max_inflight'] for r in config['replicas'])
    result['capacity_scope'] = 'max_inflight is per replica; model options retain native serve semantics'
    return result


def control_pythonpath():
    # Keep detached children importable even when the caller used relative
    # PYTHONPATH entries or only added the source checkout to sys.path.
    parent = str(Path(__file__).resolve().parents[2])
    inherited = [str(Path(p or '.').resolve()) for p in os.environ.get('PYTHONPATH', '').split(os.pathsep)]
    return os.pathsep.join(dict.fromkeys([parent, *inherited]))
