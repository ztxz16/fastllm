"""Resolve host GPU locality and bind workers before Python/CUDA initialization."""
import copy
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def id_list(value):
    result = set()
    for part in value.strip().split(','):
        if not part:
            continue
        ends = part.split('-')
        first, last = int(ends[0]), int(ends[-1])
        if len(ends) > 2 or first < 0 or last < first:
            raise ValueError(f'Invalid CPU/node list: {value!r}')
        result.update(range(first, last + 1))
    return result


def host_nodes():
    return {int(p.name[4:]): sorted(id_list((p/'cpulist').read_text()))
            for p in Path('/sys/devices/system/node').glob('node[0-9]*')}


def allowed_memory_nodes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('Mems_allowed_list:'):
            return id_list(line.split(':', 1)[1])
    raise RuntimeError('Cannot read Mems_allowed_list for NUMA binding')


def run_command(argv):
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'NUMA topology/binding command {argv[0]} failed: {error}') from error
    if proc.returncode:
        raise RuntimeError(f'NUMA topology/binding command {argv[0]} failed: {proc.stderr.strip()}')
    return proc.stdout


def topology_nodes(output):
    """Read the NUMA Affinity column, not GPU NUMA ID (HBM on some systems)."""
    result, column = {}, None
    output = re.sub(r'\x1b\[[0-9;]*m', '', output)
    for line in output.splitlines():
        if 'NUMA Affinity' in line:
            header = line.replace('CPU Affinity', 'CPU_Affinity').replace(
                'NUMA Affinity', 'NUMA_Affinity').replace('GPU NUMA ID', 'GPU_NUMA_ID').split()
            column = header.index('NUMA_Affinity') + 1  # rows also have a GPU label
        elif column is not None:
            row = line.split()
            if len(row) > column and re.fullmatch(r'GPU[0-9]+', row[0]) and row[column].isdecimal():
                result[int(row[0][3:])] = int(row[column])
    return result


def gpu_inventory(include_mig=False):
    executable = shutil.which('nvidia-smi')
    if executable is None:
        raise RuntimeError('Automatic NUMA binding requires nvidia-smi; install it or set replica.numa_node explicitly')
    listing = run_command([executable, '--query-gpu=index,uuid,pci.bus_id', '--format=csv,noheader,nounits'])
    # Some containers expose PCI sysfs but disallow the topo command, or vice versa.
    try:
        topo = topology_nodes(run_command([executable, 'topo', '-m']))
    except RuntimeError:
        topo = {}
    result = []
    for row in csv.reader(listing.splitlines()):
        if len(row) != 3:
            raise RuntimeError('Unexpected nvidia-smi GPU inventory')
        index, uuid, bus = (v.strip() for v in row)
        match = re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2}\.[0-7])', bus)
        if not index.isdecimal() or not uuid.startswith('GPU-') or match is None:
            raise RuntimeError('Invalid nvidia-smi GPU identity')
        bus = f'{int(match[1], 16):04x}:{match[2]}:{match[3]}'.lower()
        node = topo.get(int(index))
        pci_node = Path('/sys/bus/pci/devices')/bus/'numa_node'
        if node is None and pci_node.exists():
            value = int(pci_node.read_text())
            node = value if value >= 0 else None
        result.append({'index': int(index), 'uuid': uuid, 'pci_bus_id': bus, 'node': node})
    if include_mig:
        parent = None
        for line in run_command([executable, '-L']).splitlines():
            match = re.match(r'GPU (\d+):.*\(UUID: (GPU-[^)]+)\)', line)
            if match:
                parent = next((g for g in result if g['uuid'] == match[2]), None)
            else:
                match = re.search(r'\(UUID: (MIG-[^)]+)\)', line)
                if match and parent:
                    result.append({**parent, 'uuid': match[1], 'parent_uuid': parent['uuid']})
    return result


def resolve_gpu(identifier, inventory):
    if identifier.isdecimal():
        candidates = [g for g in inventory if str(g['index']) == identifier and 'parent_uuid' not in g]
    else:
        candidates = [g for g in inventory if g['uuid'].startswith(identifier)]
    if len(candidates) != 1:
        raise RuntimeError(f'GPU {identifier!r} is missing or ambiguous in nvidia-smi inventory')
    return dict(candidates[0])


def prepare_numa(config):
    """Resolve a copy; render remains portable unless topology was requested."""
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Resolving NUMA topology requires Linux')
    config = copy.deepcopy(config)
    nodes = host_nodes() if config.get('numa', 'auto') != 'off' else {}
    inventory, used = None, []
    for replica in config['replicas']:
        explicit = replica.get('numa_node')
        if config.get('numa', 'auto') == 'off' or (explicit is None and (len(nodes) <= 1 or not replica['gpus'])):
            reason = 'disabled' if config.get('numa') == 'off' else ('no_gpu' if not replica['gpus'] else 'single_numa_node')
            replica['numa_binding'] = {'enabled': False, 'reason': reason}
            continue
        topology = []
        if explicit is None:
            if inventory is None:
                inventory = gpu_inventory(any(g.startswith('MIG-') for r in config['replicas'] for g in r['gpus']))
            topology = [resolve_gpu(g, inventory) for g in replica['gpus']]
            for gpu in topology:
                if any(gpu['uuid'] == prev['uuid'] or gpu.get('parent_uuid') == prev['uuid']
                       or prev.get('parent_uuid') == gpu['uuid'] for prev in used):
                    raise RuntimeError('Replica GPU assignments overlap after resolving physical identities')
                used.append(gpu)
            local_nodes = {g['node'] for g in topology}
            if None in local_nodes:
                raise RuntimeError(f"{replica['name']}: GPU NUMA locality is unknown; set replica.numa_node explicitly")
            if len(local_nodes) != 1:
                raise RuntimeError(f"{replica['name']}: GPUs span NUMA nodes {sorted(local_nodes)}; "
                                   'group GPUs on one node per replica, or explicitly select replica.numa_node')
            node = local_nodes.pop()
            # Config GPU numbers are physical IDs. UUIDs avoid CUDA enumeration
            # order selecting a different GPU than the one whose locality we read.
            replica['env']['CUDA_VISIBLE_DEVICES'] = ','.join(g['uuid'] for g in topology)
        else:
            node = explicit
        if node not in nodes or node not in allowed_memory_nodes():
            raise RuntimeError(f"{replica['name']}: NUMA node {node} is absent or excluded by Mems_allowed_list")
        cpus = sorted(set(nodes[node]) & os.sched_getaffinity(0))
        if not cpus:
            raise RuntimeError(f"{replica['name']}: NUMA node {node} has no CPUs in the allowed affinity")
        numactl = shutil.which('numactl')
        if numactl is None:
            raise RuntimeError('NUMA binding requires numactl; install numactl before starting deploy')
        prefix = [numactl, '--physcpubind=' + ','.join(map(str, cpus)), f'--membind={node}', '--']
        replica['command'] = prefix + replica['command']
        replica['numa_binding'] = {'enabled': True, 'source': 'explicit' if explicit is not None else 'gpu_topology',
                                   'node': node, 'cpus': cpus, 'memory_policy': 'bind',
                                   'gpu_topology': topology, 'prefix': prefix}
    return config


PROBE = r'''
import ctypes,json,os,sys
node=int(sys.argv[1]); bits=ctypes.sizeof(ctypes.c_ulong)*8
mask=(ctypes.c_ulong*((node+bits)//bits))(); mode=ctypes.c_int()
lib=ctypes.CDLL('libnuma.so.1',use_errno=True)
lib.get_mempolicy.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong]
if lib.get_mempolicy(ctypes.byref(mode),mask,len(mask)*bits,None,0):
    raise OSError(ctypes.get_errno(),'get_mempolicy failed')
print(json.dumps({'cpus':sorted(os.sched_getaffinity(0)),'memory_policy':mode.value,
                  'memory_nodes':[i for i in range(len(mask)*bits) if mask[i//bits] & (1<<(i%bits))]}))
'''


def verify_bindings(config):
    """Probe every worker prefix before any model starts; never downgrade policy."""
    for replica in config['replicas']:
        binding = replica['numa_binding']
        if not binding['enabled']:
            continue
        result = json.loads(run_command([*binding['prefix'], sys.executable, '-c', PROBE, str(binding['node'])]))
        if result != {'cpus': binding['cpus'], 'memory_policy': 2, 'memory_nodes': [binding['node']]}:
            raise RuntimeError(f"{replica['name']}: NUMA binding verification failed: {result}")
        binding['verified'] = result


def snapshot_worker(pid, binding):
    """Record actual resident-page locations; shared file pages may be nonlocal."""
    root = Path(f'/proc/{pid}')
    resident, dev_zero, policies = {}, {}, {}
    for line in (root/'numa_maps').read_text().splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        policies[fields[1]] = policies.get(fields[1], 0) + 1
        page_kib = next((int(v.split('=')[1]) for v in fields if v.startswith('kernelpagesize_kB=')), 4)
        for field in fields:
            match = re.fullmatch(r'N(\d+)=(\d+)', field)
            if match:
                node, size = match[1], int(match[2]) * page_kib * 1024
                resident[node] = resident.get(node, 0) + size
                if any(v.startswith('file=/dev/zero') for v in fields):
                    dev_zero[node] = dev_zero.get(node, 0) + size
    thread_count, observed = 0, set()
    for path in (root/'task').glob('*/status'):
        try:
            line = next(l for l in path.read_text().splitlines() if l.startswith('Cpus_allowed_list:'))
        except FileNotFoundError:
            continue
        cpus = id_list(line.split(':', 1)[1])
        if not cpus or not cpus.issubset(binding['cpus']):
            raise RuntimeError(f'Worker {pid} thread {path.parent.name} escaped NUMA CPU affinity')
        observed.update(cpus)
        thread_count += 1
    if not thread_count:
        raise RuntimeError(f'Worker {pid} disappeared during NUMA verification')
    return {'threads_checked': thread_count, 'cpus': sorted(observed), 'resident_bytes_by_node': resident,
            'dev_zero_resident_bytes_by_node': dev_zero, 'vma_policies': policies}
