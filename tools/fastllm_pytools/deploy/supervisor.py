"""Detached, run-owned process supervision for a single-host deployment."""
import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from .process import process_identity, owned, group_members, signal_owned, stop_records, write_json
from .config import control_pythonpath


def require_linux():
    if not sys.platform.startswith('linux'):
        raise RuntimeError('deploy process management currently requires Linux; render works on other platforms')


@contextlib.contextmanager
def locked(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root/'deployment.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def records_for(state):
    if not state:
        return []
    manifest = read(Path(state['run_dir'])/'manifest.json') or {}
    return list(manifest.get('processes', {}).values())


def status(config):
    require_linux()
    root = Path(config['state_dir'])
    state = read(root/'state.json')
    if not state:
        return {'name': config['name'], 'phase': 'not_started', 'state_dir': str(root)}
    manifest = read(Path(state['run_dir'])/'manifest.json') or {}
    alive = {name: owned(record) for name, record in manifest.get('processes', {}).items()}
    supervisor_alive = owned(state['supervisor'])
    phase = manifest.get('phase', 'starting')
    if not supervisor_alive and phase not in ('stopped', 'failed'):
        phase = 'orphaned' if any(alive.values()) else 'failed'
    error = manifest.get('error')
    if not error and manifest.get('cleanup_error'):
        # Cleanup exceptions can embed worker argv (including credentials).
        # Keep those details in the private manifest rather than status output.
        error = f"Process cleanup failed; see {state['run_dir']}/manifest.json"
    return {'name': state['name'], 'phase': phase, 'run_dir': state['run_dir'],
            'supervisor_alive': supervisor_alive, 'replica_processes': alive,
            'error': error, 'started_at': manifest.get('started_at'),
            'finished_at': manifest.get('finished_at')}


def stop_state(state, timeout):
    if not state:
        return
    supervisor = state['supervisor']
    if owned(supervisor):
        signal_owned(supervisor, signal.SIGTERM)
        deadline = time.monotonic() + 2 * timeout + 15
        while owned(supervisor) and time.monotonic() < deadline:
            time.sleep(.1)
    # The registry is run-specific. If the supervisor crashed, clean only its
    # recorded process groups, including children outliving their group leader.
    if owned(supervisor):
        signal_owned(supervisor, signal.SIGKILL)
    stop_records(records_for(state), timeout)
    stop_records([supervisor], timeout)
    manifest_path = Path(state['run_dir'])/'manifest.json'
    manifest = read(manifest_path)
    if manifest and manifest.get('phase') not in ('stopped', 'failed'):
        manifest.update(phase='stopped', finished_at=time.time())
        write_json(manifest_path, manifest)


def stop(config):
    require_linux()
    root = Path(config['state_dir'])
    with locked(root):
        state = read(root/'state.json')
        if state:
            # Use frozen stop policy, even if the editable config changed.
            frozen = read(Path(state['run_dir'])/'config.json')
            stop_state(state, frozen['stop_timeout_seconds'])
    return status(config)


def check_ports(config):
    for host, port in [('127.0.0.1', r['port']) for r in config['replicas']] + [
            (config['router']['host'], config['router']['port'])]:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
        for family, kind, proto, _, address in infos:
            with socket.socket(family, kind, proto) as sock:
                try:
                    sock.bind(address)
                except OSError as error:
                    raise RuntimeError(f'Cannot bind deployment address {host}:{port}: {error}') from error


def start(config):
    require_linux()
    if importlib.util.find_spec('aiohttp') is None:
        raise RuntimeError('The deploy router requires aiohttp; install ftllm[deploy] in this Python environment')
    root = Path(config['state_dir'])
    with locked(root):
        previous = read(root/'state.json')
        if previous and (owned(previous['supervisor']) or any(group_members(r) for r in records_for(previous))):
            raise RuntimeError('Deployment is already active; inspect status or stop it first')
        check_ports(config)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory = root/'runs'/f'{stamp}_{uuid.uuid4().hex[:8]}'
        directory.mkdir(parents=True, mode=0o700)
        write_json(directory/'config.json', config)
        write_json(directory/'commands.json', {r['name']: {'argv': r['command'], 'cwd': r['cwd'], 'env': r['env']}
                                             for r in config['replicas']})
        source_dir = Path(__file__).parent
        sources = directory/'sources'
        sources.mkdir(mode=0o700)
        hashes = {}
        for source in source_dir.glob('*.py'):
            data = source.read_bytes()
            (sources/source.name).write_bytes(data)
            hashes[source.name] = hashlib.sha256(data).hexdigest()
        write_json(directory/'manifest.json', {'name': config['name'], 'phase': 'starting',
            'started_at': time.time(), 'processes': {}, 'source_sha256': hashes})
        with (directory/'supervisor.log').open('w') as output:
            proc = subprocess.Popen([sys.executable, '-m', __package__ + '.supervisor', str(directory)],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'PYTHONPATH': control_pythonpath()})
        identity = process_identity(proc.pid, include_zombie=True)
        if identity is None:
            raise RuntimeError(f'Supervisor exited immediately; see {directory}/supervisor.log')
        state = {'name': config['name'], 'run_dir': str(directory), 'supervisor': identity}
        write_json(root/'state.json', state)
    # Stop/status remain available while models load; only registry mutations
    # are locked. A concurrent start sees the supervisor's recorded identity.
    deadline = time.monotonic() + config['startup_timeout_seconds'] * (
        len(config['replicas']) if config['sequential_start'] else 1) + 45
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Deployment startup interrupted')
    old = signal.signal(signal.SIGTERM, interrupted)
    try:
        while time.monotonic() < deadline:
            current = read(directory/'manifest.json')
            if current['phase'] == 'ready':
                return status(config)
            if current['phase'] in ('failed', 'stopped') or proc.poll() is not None:
                raise RuntimeError(f"Deployment did not start: {current.get('error', current['phase'])}; see {directory}")
            time.sleep(.1)
        raise TimeoutError(f'Deployment startup timed out; see {directory}')
    except BaseException:
        # Stop this captured run, never whichever run might replace the registry.
        with locked(root):
            stop_state(state, config['stop_timeout_seconds'])
        raise
    finally:
        signal.signal(signal.SIGTERM, old)


def supervise(directory):
    os.umask(0o077)
    config = read(directory/'config.json')
    manifest_path = directory/'manifest.json'
    manifest = read(manifest_path)
    children = {}
    stopping = False
    def requested(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    auth = {'Authorization': 'Bearer ' + config['api_key']} if config['api_key'] else {}

    def save():
        manifest['updated_at'] = time.time()
        write_json(manifest_path, manifest)

    def check_children():
        if stopping:
            raise InterruptedError('Deployment stop requested')
        for name, proc in children.items():
            if proc.poll() is not None:
                raise RuntimeError(f'{name} exited ({proc.returncode}); see {directory}/{name}.log')

    def launch(name, argv, env, cwd):
        check_children()
        with (directory/(name+'.log')).open('w') as output:
            proc = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **env}, stdout=output,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        children[name] = proc
        identity = process_identity(proc.pid, include_zombie=True)
        if identity is None:
            raise RuntimeError(f'{name} exited immediately; see {directory}/{name}.log')
        manifest['processes'][name] = {**identity, 'log': str(directory/(name+'.log'))}
        save()

    def wait_ready(url, timeout):
        deadline, last_error = time.monotonic() + timeout, 'not ready'
        while time.monotonic() < deadline:
            check_children()
            try:
                request = urllib.request.Request(url, headers=auth)
                with opener.open(request, timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.HTTPError) as error:
                last_error = str(error)
            time.sleep(.2)
        raise TimeoutError(f'Readiness timeout: {url}: {last_error}')

    try:
        for replica in config['replicas']:
            launch(replica['name'], replica['command'], replica['env'], replica['cwd'])
            if config['sequential_start']:
                wait_ready(replica['url'] + '/v1/models', config['startup_timeout_seconds'])
        if not config['sequential_start']:
            deadline = time.monotonic() + config['startup_timeout_seconds']
            for replica in config['replicas']:
                wait_ready(replica['url'] + '/v1/models', max(.01, deadline-time.monotonic()))
        launch('router', [sys.executable, '-m', __package__ + '.router', '--run-dir', str(directory)], {}, str(directory))
        host = config['router']['host']
        if host in ('0.0.0.0', ''):
            host = '127.0.0.1'
        elif host == '::':
            host = '[::1]'
        elif ':' in host:
            host = '[' + host + ']'
        wait_ready(f"http://{host}:{config['router']['port']}/health", 30)
        manifest.update(phase='ready', ready_at=time.time())
        save()
        while not stopping:
            check_children()
            time.sleep(.2)
        manifest['phase'] = 'stopping'
    except InterruptedError:
        manifest['phase'] = 'stopping'
    except BaseException as error:
        manifest.update(phase='failed', error=repr(error))
    finally:
        save()
        try:
            stop_records(list(manifest['processes'].values()), config['stop_timeout_seconds'])
            for proc in children.values():
                proc.wait(timeout=5)
            if manifest['phase'] != 'failed':
                manifest['phase'] = 'stopped'
        except BaseException as error:
            manifest.update(phase='failed', cleanup_error=repr(error))
        manifest['finished_at'] = time.time()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    supervise(parser.parse_args().run_dir)
