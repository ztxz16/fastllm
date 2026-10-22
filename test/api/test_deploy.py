import copy
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from contextlib import redirect_stderr

TOOLS = Path(__file__).resolve().parents[2]/'tools'
sys.path.insert(0, str(TOOLS))
from fastllm_pytools import cli
from fastllm_pytools.deploy.config import resolve_config, public_plan, parse_serve_args
from fastllm_pytools.deploy.process import owned, group_members, process_identity, stop_records


def free_ports(count):
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(('127.0.0.1', 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def configuration(root, ports=None):
    ports = ports or [18080, 18081, 18082]
    return {'schema_version': 1, 'name': 'test', 'state_dir': str(root/'state'),
            'serve_args': ['/models/model with spaces', '--tp', '2', '--max_batch', '2'],
            'replicas': [{'gpus': [0, 1], 'port': ports[1]}, {'gpus': [2, 3], 'port': ports[2]}],
            'router': {'host': '127.0.0.1', 'port': ports[0]},
            'startup_timeout_seconds': 5, 'stop_timeout_seconds': 1}


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = configuration(self.root)
    def tearDown(self):
        self.temp.cleanup()
    def resolve(self):
        return resolve_config(self.data, self.root/'deployment.json')

    def test_reuses_serve_parser_without_expanding_defaults(self):
        original = cli.args_parser().parse_args(['serve', *self.data['serve_args']])
        parsed = parse_serve_args(self.data['serve_args'])
        self.assertEqual(parsed.tp, original.tp)
        self.assertEqual(parsed.max_batch, original.max_batch)
        self.assertEqual(parsed._auto_explicit, [x for x in original._auto_explicit if x != 'command'])
        config = self.resolve()
        for i, replica in enumerate(config['replicas']):
            self.assertEqual(replica['env']['CUDA_VISIBLE_DEVICES'], f'{2*i},{2*i+1}')
            command = replica['command']
            self.assertIn('/models/model with spaces', command)
            self.assertNotIn('--dtype', command)
            self.assertNotIn('--kvmem', command)
            self.assertEqual(command[command.index('--tp')+1], '2')
        self.assertNotIn('fastllm_pytools.llm', sys.modules)
        self.assertNotIn('fastllm_pytools.server', sys.modules)

    def test_worker_python_preserves_virtual_environment(self):
        import venv
        venv_dir = self.root/'venv'
        venv.EnvBuilder(with_pip=False, symlinks=True).create(venv_dir)
        executable = venv_dir/'bin/python'
        for value in (str(executable), 'venv/bin/python'):
            with self.subTest(value=value):
                self.data['worker'] = {'python': value}
                command = self.resolve()['replicas'][0]['command']
                self.assertEqual(command[0], str(executable))
                prefix = subprocess.check_output([command[0], '-c', 'import sys; print(sys.prefix)'],
                                                 text=True).strip()
                self.assertEqual(Path(prefix), venv_dir)

    def test_auto_and_explicit_disable_remain_distinct(self):
        self.data['serve_args'] = ['/models/m', '--auto', '--tp', 'auto', '--no-fast-prefill', '--cuda_slab', '0']
        self.data['router']['max_inflight_per_replica'] = 4
        config = self.resolve()
        args = parse_serve_args(config['serve_args'])
        self.assertTrue(args.auto)
        self.assertFalse(args.fast_prefill)
        self.assertEqual(args.cuda_slab, 0)
        self.assertIn('fast_prefill', args._auto_explicit)
        self.assertNotIn('max_batch', args._auto_explicit)
        self.assertNotIn('--max_batch', config['replicas'][0]['command'])
        self.data['serve_args'] = ['/models/m', '--tp', '0,1', '--max_batch', '2']
        self.data['router'].pop('max_inflight_per_replica')
        self.assertEqual(parse_serve_args(self.resolve()['serve_args']).tp, '0,1')

    def test_rejects_ambiguous_or_unsafe_deployment_configuration(self):
        cases = [lambda d: d['replicas'][1].update(gpus=[1, 3]),
                 lambda d: d['replicas'][1].update(port=18081),
                 lambda d: d['replicas'][0].update(name='router'),
                 lambda d: d.update(typo=True),
                 lambda d: d['router'].update(queue_limit_per_replica=-1),
                 lambda d: d['router'].update(max_inflight_per_replica=4),
                 lambda d: d.update(env={'CUDA_VISIBLE_DEVICES': '7'}),
                 lambda d: d['serve_args'].extend(['--port', '9999']),
                 lambda d: d['serve_args'].extend(['--unknown_option', '1']),
                 lambda d: d.update(serve_args=['model.json', '--max_batch', '2']),
                 lambda d: d.update(serve_args=['/model'])]
        for change in cases:
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                change(data)
                with self.assertRaises(ValueError), redirect_stderr(io.StringIO()):
                    resolve_config(data, self.root/'deployment.json')

    def test_render_redacts_secrets_and_does_not_create_state(self):
        self.data['serve_args'] += ['--api_key=secret-value']
        self.data['env'] = {'HF_TOKEN': 'private', 'FASTLLM_CUDA_GRAPH': '1'}
        config = self.resolve()
        output = json.dumps(public_plan(config))
        self.assertNotIn('secret-value', output)
        self.assertNotIn('private', output)
        self.assertEqual(config['api_key'], 'secret-value')
        self.assertFalse((self.root/'state').exists())

    def test_cli_subcommand_does_not_change_serve_defaults(self):
        args = cli.args_parser().parse_args(['deploy', 'render', '--config', 'deployment.json'])
        self.assertEqual(args.command, 'deploy')
        self.assertEqual(args.action, 'render')
        serve = cli.args_parser().parse_args(['serve', '/model'])
        native = parse_serve_args(['/model'])
        actual = vars(serve).copy()
        actual.pop('command')
        # The CLI's root-level version flag is unrelated to serving.
        actual.pop('version')
        actual['_auto_explicit'] = [x for x in actual['_auto_explicit'] if x != 'command']
        self.assertEqual(actual, vars(native))


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux process ownership')
class ProcessTests(unittest.TestCase):
    def test_status_reports_cleanup_failure_without_exposing_worker_arguments(self):
        from fastllm_pytools.deploy.supervisor import status
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'run'
            run.mkdir()
            (root/'state.json').write_text(json.dumps({
                'name': 'test', 'run_dir': str(run),
                'supervisor': {'pid': os.getpid(), 'start_time': 'not-current', 'boot_id': 'not-current'}}))
            (run/'manifest.json').write_text(json.dumps({
                'phase': 'failed', 'processes': {},
                'cleanup_error': "TimeoutExpired(['python', '--api_key', 'private-key'], 5)"}))
            result = status({'name': 'test', 'state_dir': tmp})
            self.assertEqual(result['phase'], 'failed')
            self.assertIn('cleanup failed', result['error'])
            self.assertIn('manifest.json', result['error'])
            self.assertNotIn('private-key', json.dumps(result))

    def test_stop_waits_for_threads_after_the_group_leader_exits(self):
        code = ('import ctypes, threading, time; '
                'threading.Thread(target=lambda: time.sleep(30)).start(); '
                'ctypes.CDLL(None).pthread_exit(None)')
        proc = subprocess.Popen([sys.executable, '-c', code], start_new_session=True)
        try:
            record = process_identity(proc.pid, include_zombie=True)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                fields = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()
                if fields[0] == 'Z':
                    break
                time.sleep(.02)
            self.assertEqual(fields[0], 'Z')
            self.assertIsNone(proc.poll())
            self.assertTrue(owned(record))
            self.assertIn(proc.pid, group_members(record))
            stop_records([record], timeout=.2)
            proc.wait(timeout=2)
            self.assertFalse(owned(record))
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)


DUMMY = r'''
import argparse,json,os,sys,time,subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
p=argparse.ArgumentParser();p.add_argument('--port',type=int);p.add_argument('--api_key',default='')
a,_=p.parse_known_args()
if os.environ.get('TEST_FAIL'):sys.exit(7)
if os.environ.get('TEST_ORPHAN'):
 child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
 fields=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
 Path(os.environ['TEST_ORPHAN']).write_text(json.dumps({'pid':child.pid,'start_time':fields[19],
  'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
 sys.exit(7)
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  if os.environ.get('TEST_NOT_READY'):
   self.send_response(503);self.end_headers();return
  if a.api_key and self.headers.get('Authorization')!='Bearer '+a.api_key:
   self.send_response(401);self.end_headers();return
  self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
  self.wfile.write(json.dumps({'data':[{'id':'model'}],'gpus':os.environ.get('CUDA_VISIBLE_DEVICES')}).encode())
 def do_POST(self):
  body=self.rfile.read(int(self.headers.get('Content-Length','0')))
  self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
  self.wfile.write(body)
ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
'''


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux process ownership')
class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = configuration(self.root, free_ports(3))
        self.worker = self.root/'dummy.py'
        self.worker.write_text(DUMMY)
        self.data['worker'] = {'script': str(self.worker)}
        self.path = self.root/'deployment.json'
        self.save()
        self.env = {**os.environ, 'PYTHONPATH': str(TOOLS)}
    def save(self):
        self.path.write_text(json.dumps(self.data))
    def command(self, action):
        return [sys.executable, '-m', 'fastllm_pytools.deploy', action, '--config', str(self.path)]
    def call(self, action):
        return subprocess.run(self.command(action), env=self.env, capture_output=True, text=True, timeout=25)
    def manifest(self):
        state = json.loads((self.root/'state/state.json').read_text())
        return json.loads((Path(state['run_dir'])/'manifest.json').read_text())
    def tearDown(self):
        self.call('stop')
        self.temp.cleanup()

    def test_start_routes_without_session_headers_rejects_duplicate_and_stops_snapshot(self):
        result = self.call('start')
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.manifest()['processes']
        self.assertEqual(len(records), 3)
        self.assertTrue(all(owned(r) for r in records.values()))
        base = f"http://127.0.0.1:{self.data['router']['port']}"
        seen = []
        for _ in range(4):
            with urllib.request.urlopen(base+'/v1/models') as response:
                seen.append(json.load(response)['gpus'])
        self.assertEqual(set(seen), {'0,1', '2,3'})
        self.assertNotEqual(self.call('start').returncode, 0)
        self.assertTrue(all(owned(r) for r in records.values()))
        # Stop can still use the registry if a model/worker path disappears and
        # editable model arguments have become invalid.
        self.worker.unlink()
        self.data['serve_args'] += ['--bad-new-option']
        self.save()
        self.assertEqual(self.call('status').returncode, 0)
        self.assertEqual(self.call('stop').returncode, 0)
        self.assertFalse(any(group_members(r) for r in records.values()))
        self.assertEqual(self.manifest()['phase'], 'stopped')

    def test_failed_second_replica_rolls_back_first(self):
        self.data['replicas'][1]['env'] = {'TEST_FAIL': '1'}
        self.save()
        result = self.call('start')
        self.assertNotEqual(result.returncode, 0)
        manifest = self.manifest()
        self.assertEqual(manifest['phase'], 'failed')
        self.assertFalse(any(group_members(r) for r in manifest['processes'].values()))

    def test_failed_worker_does_not_leave_its_descendants_running(self):
        record_path = self.root/'grandchild.json'
        self.data['replicas'][0]['env'] = {'TEST_ORPHAN': str(record_path)}
        self.save()
        self.assertNotEqual(self.call('start').returncode, 0)
        descendant = json.loads(record_path.read_text())
        self.assertFalse(owned(descendant))
        self.assertFalse(any(group_members(r) for r in self.manifest()['processes'].values()))

    def test_stop_is_available_while_start_is_waiting_for_readiness(self):
        self.data['env'] = {'TEST_NOT_READY': '1'}
        self.data['startup_timeout_seconds'] = 30
        self.save()
        with tempfile.TemporaryFile() as log:
            proc = subprocess.Popen(self.command('start'), env=self.env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic()+5
                while time.monotonic() < deadline:
                    try:
                        if self.manifest()['processes']:
                            break
                    except FileNotFoundError:
                        pass
                    time.sleep(.05)
                self.assertEqual(self.call('stop').returncode, 0)
                proc.wait(timeout=10)
                self.assertNotEqual(proc.returncode, 0)
                self.assertFalse(any(group_members(r) for r in self.manifest()['processes'].values()))
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()

    def test_worker_crash_stops_remaining_roles_and_reports_failure(self):
        result = self.call('start')
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.manifest()['processes']
        os.kill(records['replica0']['pid'], signal.SIGKILL)
        deadline = time.monotonic()+8
        while time.monotonic() < deadline and any(group_members(r) for r in records.values()):
            time.sleep(.1)
        self.assertFalse(any(group_members(r) for r in records.values()))
        self.assertEqual(self.manifest()['phase'], 'failed')

    def test_occupied_port_is_rejected_without_starting_workers(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', self.data['replicas'][0]['port']))
            sock.listen()
            self.assertNotEqual(self.call('start').returncode, 0)
            self.assertFalse((self.root/'state/state.json').exists())


if __name__ == '__main__':
    unittest.main()
