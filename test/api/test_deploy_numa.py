"""NUMA placement and failure behavior without loading CUDA or a model."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from unittest.mock import patch

from test_deploy import TOOLS, configuration, free_ports
from fastllm_pytools.deploy.config import resolve_config, public_plan
from fastllm_pytools.deploy import numa


class NumaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = configuration(self.root)
        self.inventory = [{'index': i, 'uuid': f'GPU-test-{i}', 'pci_bus_id': f'0000:{i+1:02x}:00.0',
                           'node': i // 2} for i in range(4)]
        for target, value in [('sys.platform', 'linux'), ('host_nodes', {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}),
                              ('allowed_memory_nodes', {0, 1}), ('os.sched_getaffinity', {1, 2, 5, 6}),
                              ('gpu_inventory', self.inventory), ('shutil.which', '/usr/bin/numactl')]:
            name = 'fastllm_pytools.deploy.numa.' + target
            mock = patch(name, value) if target == 'sys.platform' else patch(name, return_value=value)
            self.addCleanup(mock.stop)
            mock.start()
    def tearDown(self):
        self.temp.cleanup()
    def resolve(self):
        return resolve_config(self.data, self.root/'config.json')

    def test_default_is_auto_but_offline_render_does_not_query_hardware(self):
        with patch.object(numa, 'gpu_inventory', side_effect=AssertionError('hardware queried')):
            result = self.resolve()
            self.assertEqual(result['numa'], 'auto')
            self.assertEqual(public_plan(result)['replicas'][0]['env']['CUDA_VISIBLE_DEVICES'], '0,1')

    def test_physical_gpu_mapping_affinity_and_commands(self):
        original = self.resolve()
        result = numa.prepare_numa(original)
        self.assertEqual(original, self.resolve())  # do not mutate caller or serve defaults
        for i, replica in enumerate(result['replicas']):
            binding = replica['numa_binding']
            self.assertEqual(binding['node'], i)
            self.assertEqual(binding['cpus'], [1, 2] if i == 0 else [5, 6])
            self.assertEqual(replica['env']['CUDA_VISIBLE_DEVICES'], f'GPU-test-{i*2},GPU-test-{i*2+1}')
            self.assertEqual(replica['command'][2], f'--membind={i}')
            self.assertEqual(replica['command'][4:], original['replicas'][i]['command'])

    def test_uuid_order_and_alias_overlap(self):
        self.data['replicas'][0]['gpus'] = ['GPU-test-1', 'GPU-test-0']
        self.assertEqual(numa.prepare_numa(self.resolve())['replicas'][0]['env']['CUDA_VISIBLE_DEVICES'],
                         'GPU-test-1,GPU-test-0')
        self.data['replicas'][1]['gpus'] = [0, 3]
        with self.assertRaisesRegex(RuntimeError, 'overlap'):
            numa.prepare_numa(self.resolve())

    def test_mig_parent_topology_and_ambiguous_uuid(self):
        mig = {**self.inventory[2], 'uuid': 'MIG-child-a', 'parent_uuid': 'GPU-test-2'}
        with patch.object(numa, 'gpu_inventory', return_value=[*self.inventory, mig]):
            self.data['replicas'][1]['gpus'] = ['MIG-child']
            replica = numa.prepare_numa(self.resolve())['replicas'][1]
            self.assertEqual(replica['numa_binding']['node'], 1)
            self.assertEqual(replica['env']['CUDA_VISIBLE_DEVICES'], 'MIG-child-a')
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
            numa.resolve_gpu('GPU-test', self.inventory)

    def test_cross_node_and_unknown_topology_fail_before_launch(self):
        self.data['replicas'][0]['gpus'] = [0, 2]
        self.data['replicas'][1]['gpus'] = [1, 3]
        with self.assertRaisesRegex(RuntimeError, 'span NUMA'):
            numa.prepare_numa(self.resolve())
        self.data['replicas'][0]['gpus'] = ['GPU-missing']
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            numa.prepare_numa(self.resolve())
        self.data['replicas'][0]['gpus'] = [0]
        self.inventory[0]['node'] = None
        with self.assertRaisesRegex(RuntimeError, 'locality is unknown'):
            numa.prepare_numa(self.resolve())

    def test_disallowed_nodes_cpus_and_missing_numactl_fail(self):
        with patch.object(numa, 'allowed_memory_nodes', return_value={0}):
            with self.assertRaisesRegex(RuntimeError, 'Mems_allowed_list'):
                numa.prepare_numa(self.resolve())
        with patch.object(numa.os, 'sched_getaffinity', return_value={1}):
            with self.assertRaisesRegex(RuntimeError, 'no CPUs'):
                numa.prepare_numa(self.resolve())
        with patch.object(numa.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'requires numactl'):
                numa.prepare_numa(self.resolve())

    def test_off_uma_and_cpu_workers_do_not_require_nvidia_tools(self):
        with patch.object(numa, 'gpu_inventory', side_effect=AssertionError('must not query GPU')):
            self.data['numa'] = 'off'
            self.assertFalse(numa.prepare_numa(self.resolve())['replicas'][0]['numa_binding']['enabled'])
            self.data.pop('numa')
            with patch.object(numa, 'host_nodes', return_value={0: [1, 2]}):
                self.assertFalse(numa.prepare_numa(self.resolve())['replicas'][0]['numa_binding']['enabled'])
            for r in self.data['replicas']:
                r['gpus'] = []
            self.assertFalse(numa.prepare_numa(self.resolve())['replicas'][0]['numa_binding']['enabled'])

    def test_explicit_node_override_and_schema_validation(self):
        for r in self.data['replicas']:
            r['numa_node'] = 1
        with patch.object(numa, 'gpu_inventory', side_effect=AssertionError('explicit override')):
            result = numa.prepare_numa(self.resolve())
            self.assertTrue(all(r['numa_binding']['node'] == 1 for r in result['replicas']))
        for invalid in [-1, True, '1', [1]]:
            self.data['replicas'][0]['numa_node'] = invalid
            with self.assertRaises(ValueError):
                self.resolve()
        self.data['replicas'][0]['numa_node'] = 0
        self.data['numa'] = 'off'
        with self.assertRaises(ValueError):
            self.resolve()
        self.data['numa'] = 'preferred'
        with self.assertRaises(ValueError):
            self.resolve()

    def test_binding_probe_rejects_fallback_and_cannot_start_any_worker(self):
        config = numa.prepare_numa(self.resolve())
        with patch.object(numa, 'run_command', return_value=json.dumps(
                {'cpus': [1, 2], 'memory_policy': 1, 'memory_nodes': [0]})):
            with self.assertRaisesRegex(RuntimeError, 'verification failed'):
                numa.verify_bindings(config)
        from fastllm_pytools.deploy.supervisor import start
        with patch('fastllm_pytools.deploy.supervisor.check_ports'), \
                patch('fastllm_pytools.deploy.supervisor.configure_nofile'), \
                patch.object(numa, 'run_command', side_effect=RuntimeError('binding denied')), \
                patch('fastllm_pytools.deploy.supervisor.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'binding denied'):
                start(self.resolve())
            spawn.assert_not_called()
            self.assertFalse((self.root/'state/state.json').exists())

    def test_topology_parser_uses_cpu_numa_not_gpu_memory_numa_id(self):
        sample = '\x1b[4m\tGPU0\tGPU1\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n'
        sample += 'GPU0\tX\tSYS\tPIX\t0-7\t0\t7\nGPU1\tSYS\tX\tPHB\t8-15\t1\t8\n'
        self.assertEqual(numa.topology_nodes(sample), {0: 0, 1: 1})
        self.assertEqual(numa.id_list('1-3,6,9-10'), {1, 2, 3, 6, 9, 10})

    def test_inventory_parses_physical_devices_and_mig_parents(self):
        query = '3, GPU-aaa, 00000000:81:00.0\n7, GPU-bbb, 00000000:c1:00.0\n'
        topo = 'GPU3 GPU7 CPU Affinity NUMA Affinity GPU NUMA ID\nGPU3 X SYS 0-7 0 N/A\nGPU7 SYS X 8-15 1 N/A\n'
        listing = 'GPU 3: H20 (UUID: GPU-aaa)\nGPU 7: H20 (UUID: GPU-bbb)\n  MIG 1g Device 0: (UUID: MIG-ccc)\n'
        # Restore the real function hidden by setUp's inventory fixture.
        function = self.real_inventory
        with patch.object(numa, 'run_command', side_effect=[query, topo, listing]):
            result = function(include_mig=True)
        self.assertEqual(result[1]['pci_bus_id'], '0000:c1:00.0')
        self.assertEqual(result[2]['parent_uuid'], 'GPU-bbb')
        self.assertEqual(numa.resolve_gpu('7', result)['uuid'], 'GPU-bbb')
        self.assertEqual(numa.resolve_gpu('MIG-c', result)['node'], 1)

    def test_pci_sysfs_fallback_when_topology_query_is_unavailable(self):
        with patch.object(numa, 'run_command', side_effect=[
                '0, GPU-aaa, 00000000:81:00.0\n', RuntimeError('topo unavailable')]), \
                patch.object(numa.Path, 'exists', return_value=True), \
                patch.object(numa.Path, 'read_text', return_value='1\n'):
            self.assertEqual(self.real_inventory()[0]['node'], 1)

    def test_loaded_worker_snapshot_detects_thread_affinity_escape(self):
        task = self.root/'task/100'
        task.mkdir(parents=True)
        (self.root/'numa_maps').write_text('1000 bind:1 file=/dev/zero\\040(deleted) N1=512 kernelpagesize_kB=4\n')
        (task/'status').write_text('Cpus_allowed_list:\t5-6\n')
        with patch.object(numa, 'Path', return_value=self.root):
            result = numa.snapshot_worker(100, {'cpus': [5, 6]})
            self.assertEqual(result['dev_zero_resident_bytes_by_node'], {'1': 2 * 1024**2})
            (task/'status').write_text('Cpus_allowed_list:\t1-6\n')
            with self.assertRaisesRegex(RuntimeError, 'escaped'):
                numa.snapshot_worker(100, {'cpus': [5, 6]})

    real_inventory = staticmethod(numa.gpu_inventory)


WORKER = r'''
import argparse,ctypes,json,mmap,os,re
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
p=argparse.ArgumentParser();p.add_argument('--port',type=int);a,_=p.parse_known_args()
pool=mmap.mmap(-1,4*1024*1024,flags=mmap.MAP_SHARED|mmap.MAP_ANONYMOUS|mmap.MAP_POPULATE)
for offset in range(0,len(pool),4096):pool[offset]=1
address=ctypes.addressof(ctypes.c_char.from_buffer(pool))
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  line=next(l for l in Path('/proc/self/numa_maps').read_text().splitlines() if l.startswith(f'{address:x} '))
  body={'data':[{'id':'dummy'}],'cpus':sorted(os.sched_getaffinity(0)),
        'pages':{k:int(v) for k,v in re.findall(r'N(\d+)=(\d+)',line)},
        'gpus':os.environ.get('CUDA_VISIBLE_DEVICES')}
  self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(body).encode())
ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
'''


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('numactl'), 'Linux numactl integration')
class LiveNumaTests(unittest.TestCase):
    def test_real_topology_two_workers_and_physical_memory_placement(self):
        # Only nvidia-smi metadata queries; workers allocate 4 MiB of ordinary
        # host memory each. No GPU context, kernel, or model is created.
        nodes = numa.host_nodes()
        allowed = numa.allowed_memory_nodes()
        cpus = os.sched_getaffinity(0)
        try:
            inventory = numa.gpu_inventory()
        except RuntimeError as error:
            self.skipTest(str(error))
        devices = {}
        for gpu in inventory:
            node = gpu['node']
            if node in allowed and cpus.intersection(nodes.get(node, [])):
                devices.setdefault(node, gpu)
        if len(devices) < 2:
            self.skipTest('Need GPUs on two allowed NUMA nodes')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = configuration(root, free_ports(3))
            selected = sorted(devices)[:2]
            for r, node in zip(data['replicas'], selected):
                r['gpus'] = [devices[node]['index']]
            worker = root/'worker.py'
            worker.write_text(WORKER)
            data['worker'] = {'script': str(worker)}
            data['startup_timeout_seconds'] = 15
            path = root/'config.json'
            path.write_text(json.dumps(data))
            env = {**os.environ, 'PYTHONPATH': str(TOOLS)}
            def call(action):
                return subprocess.run([sys.executable, '-m', 'fastllm_pytools.deploy', action, '--config', str(path)],
                                      env=env, capture_output=True, text=True, timeout=45)
            try:
                started = call('start')
                self.assertEqual(started.returncode, 0, started.stderr)
                status = json.loads(started.stdout)
                commands = json.loads((Path(status['run_dir'])/'commands.json').read_text())
                for i, (r, node) in enumerate(zip(data['replicas'], selected)):
                    with urllib.request.urlopen(f"http://127.0.0.1:{r['port']}/v1/models") as response:
                        observed = json.load(response)
                    self.assertEqual(set(observed['pages']), {str(node)})
                    self.assertGreaterEqual(observed['pages'][str(node)], 1024)
                    self.assertTrue(set(observed['cpus']).issubset(nodes[node]))
                    self.assertEqual(observed['gpus'], devices[node]['uuid'])
                    binding = status['numa'][f'replica{i}']
                    self.assertEqual(binding['verified']['memory_policy'], 2)
                    self.assertGreater(binding['after_load']['dev_zero_resident_bytes_by_node'][str(node)], 0)
                    self.assertIn(f'--membind={node}', commands[f'replica{i}']['argv'])
            finally:
                stopped = call('stop')
                self.assertEqual(stopped.returncode, 0, stopped.stderr)


if __name__ == '__main__':
    unittest.main()
