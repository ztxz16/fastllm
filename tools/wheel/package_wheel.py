#!/usr/bin/env python3
"""Stage, audit and package an x86_64 CPU/CUDA build without mutating its outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


def output(*args):
    return subprocess.check_output(args, text=True)


def audit_elf(path, max_glibc):
    symbols = output('readelf', '--dyn-syms', '--wide', str(path))
    versions = sorted(set(re.findall(r'\bUND\b.*?@GLIBC_(\d+\.\d+(?:\.\d+)?)', symbols)),
                      key=lambda v: tuple(map(int, v.split('.'))))
    maximum = versions[-1] if versions else '0.0'
    if tuple(map(int, maximum.split('.'))) > tuple(map(int, max_glibc.split('.'))):
        raise RuntimeError('%s requires GLIBC_%s, above target %s' % (path.name, maximum, max_glibc))
    if re.search(r'\bUND\b.*?@GLIBC_PRIVATE', symbols):
        raise RuntimeError('%s imports GLIBC_PRIVATE' % path.name)
    dynamic = output('readelf', '-d', str(path))
    return {'file': path.name, 'bytes': path.stat().st_size, 'max_glibc': maximum,
            'glibc_versions': versions,
            'needed': re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic),
            'rpath': re.findall(r'\((?:RUNPATH|RPATH)\).*?\[(.*?)\]', dynamic)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source-dir', 'build-dir', 'output-dir', 'max-glibc', 'cxx', 'strip', 'do-strip']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    source, build, dest = Path(args.source_dir), Path(args.build_dir), Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    # Temporary staging prevents stale metadata and release/nightly cross-contamination.
    with tempfile.TemporaryDirectory(prefix='package-', dir=build) as temporary:
        stage = Path(temporary)
        package = stage / 'ftllm'
        # Python-only edits do not relink fastllm_tools or rerun its POST_BUILD
        # copy command. Always stage current Python sources, not that old copy.
        shutil.copytree(source / 'tools/fastllm_pytools', package,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(source / 'tools/fastllm_triton_server.py', package / 'fastllm_triton_server.py')
        shutil.copy2(source / 'tools/scripts/setup.py', stage / 'setup.py')
        shutil.copy2(build / 'cuda/build_info.json', package / 'build_info.json')
        shutil.copy2(build / 'cuda/tools/ftllm/libfastllm_tools.so', package / 'libfastllm_tools.so')
        shutil.copy2(build / 'cpu/tools/ftllm/libfastllm_tools.so', package / 'libfastllm_tools-cpu.so')
        numa = Path(output(args.cxx, '-print-file-name=libnuma.so').strip())
        if not numa.is_file():
            raise RuntimeError('libnuma.so not found; install libnuma-dev in the build environment')
        shutil.copy2(numa.resolve(), package / 'libnuma.so.1')
        libraries = []
        for path in sorted(package.rglob('*')):
            if not path.is_file():
                continue
            with path.open('rb') as f:
                is_elf = f.read(4) == b'\x7fELF'
            if is_elf:
                if args.do_strip == '1':
                    subprocess.run([args.strip, '--strip-unneeded', str(path)], check=True)
                libraries.append(audit_elf(path, args.max_glibc))
        audit = {'target_max_glibc': args.max_glibc, 'libraries': libraries,
                 'compiler': output(args.cxx, '--version').splitlines()[0],
                 'build_glibc': os.confstr('CS_GNU_LIBC_VERSION')}
        subprocess.run([sys.executable, 'setup.py', 'bdist_wheel', '--plat-name', 'linux_x86_64'],
                       cwd=stage, check=True)
        wheels = list((stage / 'dist').glob('*.whl'))
        if len(wheels) != 1:
            raise RuntimeError('Expected exactly one wheel, got %s' % wheels)
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('Wheel ZIP integrity check failed')
            expected = {'ftllm/libfastllm_tools.so', 'ftllm/libfastllm_tools-cpu.so',
                        'ftllm/libnuma.so.1'}
            if not expected <= set(archive.namelist()):
                raise RuntimeError('Wheel is missing required native libraries')
        target = dest / wheel.name
        # Publish only after the complete wheel passes the audit and archive checks.
        with tempfile.NamedTemporaryFile(prefix='.wheel-', dir=dest, delete=False) as tmp:
            temporary_wheel = Path(tmp.name)
        try:
            shutil.copyfile(wheel, temporary_wheel)
            os.replace(temporary_wheel, target)
        finally:
            temporary_wheel.unlink(missing_ok=True)
        audit.update(wheel=str(target), wheel_bytes=target.stat().st_size,
                     sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                     package_seconds=time.monotonic() - started)
        (dest / (target.stem + '.json')).write_text(json.dumps(audit, indent=2) + '\n')
        print(json.dumps({'wheel': str(target), 'bytes': audit['wheel_bytes'],
                          'max_glibc': max((x['max_glibc'] for x in libraries),
                                           key=lambda v: tuple(map(int, v.split('.')))),
                          'sha256': audit['sha256']}, indent=2))


if __name__ == '__main__':
    main()
