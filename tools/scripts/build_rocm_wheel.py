#!/usr/bin/env python3
"""Build a multi-architecture ROCm wheel using an activated Python ROCm SDK."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

def device_targets(requirements):
    """Return GPU targets with a runtime package for the current platform."""
    from packaging.requirements import Requirement

    targets = set()
    for value in requirements:
        requirement = Requirement(value)
        name = re.sub(r'[-_.]+', '-', requirement.name).lower()
        match = re.fullmatch(r'rocm-sdk-device-(gfx[0-9a-f]+)', name)
        if not match:
            continue
        target = match.group(1)
        extras = ('device-all', f'device-{target}', 'device')
        if requirement.marker and not any(
                requirement.marker.evaluate({'extra': extra}) for extra in extras):
            continue
        targets.add(target)
    return targets


def select_architectures(specification, available):
    if specification.strip().lower() == 'all':
        return sorted(available, key=lambda target: int(target[3:], 16))
    targets = list(dict.fromkeys(a.strip() for a in specification.replace(',', ';').split(';')))
    if not targets or any(not re.fullmatch(r'gfx[0-9a-f]+', a) for a in targets):
        raise ValueError('Use all or a comma/semicolon-separated list of gfx targets')
    missing = set(targets) - available
    if missing:
        raise ValueError('No matching runtime package/compiler support for: ' + ', '.join(sorted(missing)))
    return targets


def run(*args, cwd=None, env=None):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def read_cache(path):
    result = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line:
            key, value = line.split('=', 1)
            result[key.split(':', 1)[0]] = value
    return result


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architectures', default='all',
                        help='all (default): every target with an SDK device package and compiler support; '
                             'or a comma/semicolon-separated gfx target list')
    parser.add_argument('--list-architectures', action='store_true',
                        help='List available SDK/compiler GPU targets and exit without compiling')
    parser.add_argument('--build-dir', type=Path, default=repo / 'build-rocm-wheel')
    parser.add_argument('--dist-dir', type=Path,
                        help='Output directory; defaults to <build-dir>/dist')
    parser.add_argument('--jobs', type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument('--skip-build', action='store_true', help='Package an existing matching build')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('This wheel builder currently supports Linux x86_64 only')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    try:
        import rocm_sdk
    except ImportError:
        parser.error('Activate a Python environment containing rocm[libraries,devel], '
                     'or use make_whl_rocm_docker.sh')
    version = rocm_sdk.__version__
    devices = device_targets(importlib.metadata.requires('rocm') or [])
    sdk = subprocess.check_output([sys.executable, '-m', 'rocm_sdk', 'path', '--root'], text=True).strip()
    for name in ('clang++', 'llvm-objdump'):
        if not (Path(sdk) / 'lib/llvm/bin' / name).is_file():
            parser.error(f'The active SDK is missing {name}; install rocm[devel]=={version}')
    processors = subprocess.check_output(
        [str(Path(sdk) / 'lib/llvm/bin/clang++'), '--target=amdgcn-amd-amdhsa', '--print-supported-cpus'],
        stderr=subprocess.STDOUT, text=True)
    compiler_targets = set(re.findall(r'^\s*(gfx[0-9a-f]+)\s*$', processors, re.MULTILINE))
    available = devices & compiler_targets
    if not available:
        parser.error(f'ROCm {version} has no matching device packages and compiler targets')
    print(f'ROCm {version}: {len(available)} available GPU targets: '
          + ';'.join(select_architectures('all', available)), flush=True)
    if devices - compiler_targets:
        print('Device packages excluded because the compiler lacks support: '
              + ', '.join(sorted(devices - compiler_targets)), flush=True)
    if args.list_architectures:
        return
    try:
        archs = select_architectures(args.architectures, available)
    except ValueError as error:
        parser.error(str(error))
    for name in ('git',) + (() if args.skip_build else ('cmake', 'ninja', 'g++')):
        if not shutil.which(name):
            parser.error(f'Missing build tool: {name}; use the ROCm builder image or install it')
    build = args.build_dir.resolve()
    dist = (args.dist_dir or build / 'dist').resolve()
    env = dict(os.environ, ROCM_PATH=sdk)
    if not args.skip_build:
        run('cmake', '-S', repo, '-B', build, '-G', 'Ninja',
            '-DUSE_ROCM=ON', '-DUSE_CUDA=OFF', '-DUSE_NUMAS=ON', '-DMAKE_WHL_X86=ON',
            '-DCMAKE_CXX_COMPILER=/usr/bin/g++', '-DROCM_ARCH=' + ';'.join(archs), env=env)
        run('cmake', '--build', build, '--target', 'fastllm_tools', '--parallel', args.jobs, env=env)
    cache = read_cache(build / 'CMakeCache.txt')
    if (cache.get('USE_ROCM') != 'ON' or cache.get('USE_CUDA') != 'OFF'
            or cache.get('MAKE_WHL_X86') != 'ON'
            or cache.get('ROCM_ARCH', '').split(';') != archs):
        raise RuntimeError('Build configuration does not match the requested ROCm wheel')
    so = build / 'tools/ftllm/libfastllm_tools.so'
    if not so.is_file():
        raise RuntimeError('Build the ROCm shared library before packaging')
    # Inspect a temporary copy: ROCm's objdump may extract code objects beside
    # its input file. Those must never end up in the Python package.
    with tempfile.TemporaryDirectory(prefix='offload-check-', dir=build) as tmp:
        probe = Path(tmp) / 'native.so'
        shutil.copy2(so, probe)
        offloading = subprocess.check_output(
            [str(Path(sdk) / 'lib/llvm/bin/llvm-objdump'), '--offloading', str(probe)], text=True)
        bundles = {}
        for bundle, target in re.findall(r'\.so\.(\d+)\.hipv4-amdgcn-amd-amdhsa--(gfx[0-9a-f]+)', offloading):
            bundles.setdefault(bundle, set()).add(target)
        if not bundles or any(not set(archs) <= targets for targets in bundles.values()):
            raise RuntimeError('The native library does not contain every requested GPU target')
        print(f'Verified {len(bundles)} HIP bundles for {len(archs)} GPU targets', flush=True)
    info = json.loads((build / 'tools/ftllm/build_info.json').read_text())
    if not info.get('USE_ROCM'):
        raise RuntimeError('Refusing to package a non-ROCm library')
    dist.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='wheel-stage-', dir=build) as tmp:
        stage = Path(tmp)
        package = stage / 'ftllm'
        shutil.copytree(repo / 'tools/fastllm_pytools', package,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(so, package / so.name)
        shutil.copy2(repo / 'tools/fastllm_triton_server.py', package / 'fastllm_triton_server.py')
        for name in ('setup_rocm.py', 'rocm_runtime.py'):
            shutil.copy2(repo / 'tools/scripts' / name, stage / name)
        (package / 'build_info.json').write_text(json.dumps(info, indent=2) + '\n')
        rocm_info = {'sdk_version': version, 'architectures': archs,
                     'build_os': platform.freedesktop_os_release().get('PRETTY_NAME', 'Linux'),
                     'build_python': platform.python_version(),
                     'build_glibc': platform.libc_ver()[1],
                     'verified_hip_bundles': len(bundles),
                     'cpu_baseline': 'AVX2, F16C, FMA',
                     'source_dirty': bool(subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()),
                     'library_sha256': hashlib.sha256(so.read_bytes()).hexdigest(),
                     'source_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()}
        (package / 'rocm_build_info.json').write_text(json.dumps(rocm_info, indent=2) + '\n')
        libraries = subprocess.check_output(['ldd', str(so)], env=env, text=True)
        numa = re.search(r'libnuma\.so\.1\s+=>\s+(/\S+)', libraries)
        if not numa:
            raise RuntimeError('Could not resolve libnuma.so.1 for bundling')
        shutil.copy2(Path(numa.group(1)).resolve(), package / 'libnuma.so.1')
        licenses = package / 'licenses'
        licenses.mkdir()
        shutil.copy2(repo / 'LICENSE', licenses / 'fastllm.txt')
        numa_license = Path('/usr/share/doc/libnuma1/copyright')
        if not numa_license.is_file():
            raise RuntimeError('Provide libnuma redistribution notices before building this wheel')
        shutil.copy2(numa_license, licenses / 'libnuma.txt')
        run(sys.executable, 'setup_rocm.py', 'bdist_wheel', '--plat-name', 'linux_x86_64',
            '--dist-dir', dist, cwd=stage)
    wheels = sorted(dist.glob('ftllm_rocm-*-py3-none-linux_x86_64.whl'), key=lambda p: p.stat().st_mtime)
    wheel = wheels[-1]
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        required = {'ftllm/libfastllm_tools.so', 'ftllm/libnuma.so.1', 'ftllm/build_info.json',
                    'ftllm/rocm_build_info.json', 'ftllm/_rocm_init.py',
                    'ftllm/openai_server/tool_parsers/__init__.py'}
        if not required <= names:
            raise RuntimeError('Wheel is missing required files: ' + str(required - names))
        metadata = archive.read(next(n for n in names if n.endswith('/METADATA'))).decode()
        if any('nvidia-' in line.lower() for line in metadata.splitlines() if line.startswith('Requires-Dist:')):
            raise RuntimeError('ROCm wheel unexpectedly depends on NVIDIA packages')
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (dist / (wheel.name + '.sha256')).write_text(f'{digest}  {wheel.name}\n')
    example_arch = 'gfx1100' if 'gfx1100' in archs else archs[0]
    (dist / 'README.txt').write_text(
        f'FastLLM ROCm wheel ({version})\n\n'
        'Requirements: Linux x86_64, AVX2/F16C/FMA CPU, Python >= 3.10, '
        'compatible AMD driver and access to /dev/kfd and /dev/dri/renderD*.\n'
        'Ubuntu 22.04 / Python 3.10 is the initial validation platform. '
        'This linux_x86_64 wheel makes no manylinux2014 compatibility claim.\n\n'
        f'Built on: {rocm_info["build_os"]}, Python {rocm_info["build_python"]}, '
        f'glibc {rocm_info["build_glibc"]} (build environment, not a computed minimum).\n'
        f'Compiled GPU targets: {", ".join(archs)}\n'
        'GPU runtime validation is separate from compilation. See the validation report.\n\n'
        'Use a separate environment from NVIDIA ftllm (both install the ftllm module).\n'
        f'Choose the extra for your GPU; example for compiled target {example_arch}:\n'
        '  python3 -m venv .venv\n'
        '  . .venv/bin/activate\n'
        f'  python -m pip install "./{wheel.name}[{example_arch}]" '
        '--extra-index-url https://stable.repo.amd.com/rocm/whl-next/\n\n'
        'For multiple GPU types, combine the corresponding compiled-target extras. '
        '[all-gpus] installs device data for every compiled target and is much larger.\n'
        'No separate ROCm SDK or compiler installation is required. '
        'The ROCm runtime packages are downloaded by pip; models are separate.\n\n'
        '  ftllm chat /path/to/model --device cuda:0 --dtype float16 --atype float16\n'
        '  ftllm server /path/to/model --device cuda:0 --dtype float16 --atype float16\n'
        'ROCM_PATH, HIP_PATH and LD_LIBRARY_PATH are not required by the installed wheel.\n'
    )
    print(f'WHEEL: {wheel}\nSIZE: {wheel.stat().st_size / 2**20:.2f} MiB\nSHA256: {digest}')


if __name__ == '__main__':
    main()
