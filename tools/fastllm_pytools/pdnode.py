"""Owned process lifecycle for ftllm router; also runs over SSH via stdin JSON."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import urllib.request

from .deploy.process import process_identity, owned, stop_records, write_json


def operate(action, node):
    directory = Path(node['state_dir']).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / 'process.json'
        saved = json.loads(path.read_text()) if path.exists() else None
        live = bool(saved and owned(saved['process']))
        if action == 'start':
            if live:
                if (saved['command'] != node['command'] or saved['env'] != node.get('env', {})
                        or saved.get('config') != node.get('config')):
                    raise ValueError('An owned process is running with a different configuration')
            else:
                if 'config' in node:
                    write_json(directory / 'config.json', node['config'])
                with (directory / 'service.log').open('ab', buffering=0) as log:
                    child = subprocess.Popen(node['command'], cwd=node.get('cwd'),
                        env={**os.environ, **node.get('env', {})}, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                identity = process_identity(child.pid)
                if identity is None:
                    raise RuntimeError('Process exited during startup; inspect service.log')
                saved = {'process': identity, 'command': node['command'], 'env': node.get('env', {})}
                if 'config' in node:
                    saved['config'] = node['config']
                write_json(path, saved)
                live = True
        elif action == 'stop' and saved:
            stop_records([saved['process']], timeout=20)
            path.unlink(missing_ok=True)
            live = False
        elif action != 'status' and action != 'stop':
            raise ValueError('Unknown node operation')
        result = {'name': node['name'], 'running': live,
                  'pid': saved['process']['pid'] if saved else None, 'log': str(directory / 'service.log')}
        if action == 'status':
            result['ready'] = False
            if live and node.get('url'):
                try:
                    request = urllib.request.Request(node['url'] + node.get('health_path', '/v1/models'),
                                                     headers=node.get('health_headers', {}))
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(request, timeout=3) as response:
                        json.load(response)
                    result['ready'] = True
                except Exception as error:
                    result['readiness_error'] = str(error)
        return result


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'stop', 'status'))
    args = parser.parse_args()
    print(json.dumps(operate(args.action, json.load(sys.stdin))))
