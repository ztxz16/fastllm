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


def stage_low_latency_licenses(source, package):
    packaged = set()
    for dep in ('naivert_projection', 'naivert_dsa', 'naivert_draft', 'naivert_flex'):
        dest = package / 'licenses' / dep
        dest.mkdir(parents=True, exist_ok=True)
        files = [(name, name) for name in ('LICENSE', 'NOTICE', 'UPSTREAM.json')]
        if dep == 'naivert_flex':
            files.append(('THIRD_PARTY_NOTICES.upstream.md', 'THIRD_PARTY_NOTICES.md'))
        for source_name, name in files:
            shutil.copy2(source / 'third_party' / dep / source_name, dest / name)
            packaged.add((dest / name).relative_to(package.parent).as_posix())
    return packaged


def stage_demo_assets(source, package):
    # Copy from current sources even when the binaries were built earlier.
    demo_files = set()
    for src, folder, names in (
        ('qwen_image', 'qwen_image_assets', ('demo.html', 'openapi.json')),
        ('lingbot_world', 'lingbot_world_assets', ('demo.html', 'server.py', 'convert_assets.py')),
        ('duo', 'duo_assets', ('live.html', 'avc-player.js', 'presets/default.txt',
                               'presets/default-256x192.bf16', 'presets/default-512x384.bf16')),
        ('minimax_h3', 'h3_assets', ('demo.html', 'openapi.json')),
        ('minimax_h3_image', 'h3_image_assets', ('demo.html', 'openapi.json')),
        ('fastllm_pytools/ui_plugins/studio', 'h3_image_assets', ('image-workspace.js', 'image-workspace.css')),
    ):
        for name in names:
            target = package / folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / 'tools' / src / name, target)
            demo_files.add(target.relative_to(package.parent).as_posix())
    return demo_files


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
        shutil.copytree(source / 'third_party/flashinfer_sm120_ep',
                        package / 'licenses/flashinfer_sm120_ep')
        shutil.copy2(source / 'tools/fastllm_triton_server.py', package / 'fastllm_triton_server.py')
        shutil.copy2(source / 'tools/scripts/setup.py', stage / 'setup.py')
        build_info = build / 'cuda/build_info.json'
        shutil.copy2(build_info, package / 'build_info.json')
        merged = json.loads(build_info.read_text()).get('merged_media', False)
        shutil.copy2(build / 'cuda/tools/ftllm/libfastllm_tools.so', package / 'libfastllm_tools.so')
        shutil.copy2(build / 'cpu/tools/ftllm/libfastllm_tools.so', package / 'libfastllm_tools-cpu.so')
        for name in ('minimax_h3_server', 'minimax_h3_generate', 'minimax_h3_image_server', 'minimax_h3_image_generate'):
            shutil.copy2(build / 'cuda/src/models/minimax_h3' / name, package / name)
        for name in ('qwen_image_server', 'qwen_image_generate'):
            shutil.copy2(build / 'cuda/src/models/qwen_image' / name, package / name)
        if merged:
            shutil.copy2(build / 'cuda/src/models/duo/live/duo_live', package / 'duo_live')
            licenses = package / 'licenses/duo'
            shutil.copytree(source / 'tools/wheel/licenses/duo', licenses)
            shutil.copy2(source / 'third_party/duo_solarx/LICENSE', licenses / 'SolarX-LICENSE')
        else:
            for name in ('libfastllm_minimax_h3.so', 'libfastllm_qwen_image.so'):
                shutil.copy2(build / 'cuda' / name, package / name)
        demo_files = stage_demo_assets(source, package)
        license_files = stage_low_latency_licenses(source, package)
        numa = Path(output(args.cxx, '-print-file-name=libnuma.so').strip())
        if not numa.is_file():
            raise RuntimeError('libnuma.so not found; install libnuma-dev in the build environment')
        shutil.copy2(numa.resolve(), package / 'libnuma.so.1')
        libraries = []
        for path in sorted(package.rglob('*')):
            # CUBINs are device ELF files whose exact bytes are pinned by the manifest.
            if not path.is_file() or path.suffix == '.cubin':
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
                        'ftllm/libnuma.so.1',
                        'ftllm/minimax_h3_server', 'ftllm/minimax_h3_generate',
                        'ftllm/minimax_h3_image_server', 'ftllm/minimax_h3_image_generate',
                        'ftllm/qwen_image_server', 'ftllm/qwen_image_generate',
                        'ftllm/nomas_demo.html', 'ftllm/ui_plugins/studio/nomas-cafe.webp'}
            if merged:
                expected.add('ftllm/duo_live')
            else:
                expected.update(('ftllm/libfastllm_minimax_h3.so', 'ftllm/libfastllm_qwen_image.so'))
            missing = (expected | demo_files | license_files) - set(archive.namelist())
            if missing:
                raise RuntimeError('Wheel is missing: ' + ', '.join(sorted(missing)))
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
