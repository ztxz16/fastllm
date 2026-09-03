#!/usr/bin/env python3
"""Collect Electron's non-glibc shared-library closure into a portable folder."""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ELF_MAGIC = b"\x7fELF"
LDD_MAPPING = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)(?:\s+\(0x[0-9a-fA-F]+\))?\s*$")
LDD_NOT_FOUND = re.compile(r"^\s*(\S+)\s+=>\s+not found\s*$")
LDD_DIRECT = re.compile(r"^\s*(/\S+)(?:\s+\(0x[0-9a-fA-F]+\))?\s*$")

# These libraries form the host glibc ABI and must stay matched to the host
# loader/kernel rather than being mixed with a different distribution's libc.
HOST_GLIBC_LIBRARIES = {
    "ld-linux-x86-64.so.2",
    "libBrokenLocale.so.1",
    "libanl.so.1",
    "libc.so.6",
    "libcrypt.so.1",
    "libdl.so.2",
    "libm.so.6",
    "libmvec.so.1",
    "libnsl.so.1",
    "libnss_compat.so.2",
    "libnss_dns.so.2",
    "libnss_files.so.2",
    "libpthread.so.0",
    "libresolv.so.2",
    "librt.so.1",
    "libthread_db.so.1",
    "libutil.so.1",
}

DRIVER_LIBRARY_PATTERNS = (
    re.compile(r"^libcuda\.so"),
    re.compile(r"^libcudadebugger\.so"),
    re.compile(r"^libnvidia-"),
    re.compile(r"^libGLX_nvidia\.so"),
    re.compile(r"^libEGL_nvidia\.so"),
    re.compile(r"^libnvcuvid\.so"),
    re.compile(r"^libnvoptix\.so"),
    re.compile(r"^libvdpau_nvidia\.so"),
)


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == ELF_MAGIC
    except OSError:
        return False


def inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def iter_initial_elfs(root: Path, output: Path, excludes: Iterable[Path]) -> Iterable[Path]:
    excluded = [item.resolve() for item in excludes]
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        if inside(candidate, output) or any(inside(candidate, item) for item in excluded):
            continue
        if is_elf(candidate):
            yield candidate


def run_ldd(path: Path, library_path: str) -> List[Tuple[str, Optional[Path]]]:
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = library_path
    result = subprocess.run(
        ["ldd", str(path)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0 and "not a dynamic executable" not in combined:
        raise RuntimeError(f"ldd failed for {path}:\n{combined.strip()}")

    dependencies: List[Tuple[str, Optional[Path]]] = []
    for line in combined.splitlines():
        missing = LDD_NOT_FOUND.match(line)
        if missing:
            dependencies.append((missing.group(1), None))
            continue
        mapping = LDD_MAPPING.match(line)
        if mapping:
            name, resolved = mapping.groups()
            dependencies.append((name, Path(resolved)))
            continue
        direct = LDD_DIRECT.match(line)
        if direct:
            resolved = Path(direct.group(1))
            dependencies.append((resolved.name, resolved))
    return dependencies


def is_driver_library(name: str) -> bool:
    return any(pattern.match(name) for pattern in DRIVER_LIBRARY_PATTERNS)


def owning_package(path: Path) -> Optional[str]:
    for candidate in (path, path.resolve()):
        result = subprocess.run(
            ["dpkg-query", "-S", str(candidate)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.split(":", 1)[0].strip()
    return None


def copy_copyright(package: str, destination: Path) -> Optional[str]:
    source = Path("/usr/share/doc") / package / "copyright"
    if not source.is_file():
        return None
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{package.replace(':', '_')}.copyright"
    if not target.exists():
        shutil.copy2(source, target)
    return target.name


def find_optional_libraries(sonames: Iterable[str]) -> Dict[str, Path]:
    requested = set(sonames)
    if not requested:
        return {}
    result = subprocess.run(
        ["ldconfig", "-p"],
        check=True,
        capture_output=True,
        text=True,
    )
    found: Dict[str, Path] = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        soname, separator, _remainder = stripped.partition(" ")
        if not separator or soname not in requested or "x86-64" not in stripped:
            continue
        _, separator, resolved = stripped.partition(" => ")
        if separator and Path(resolved).is_file():
            found.setdefault(soname, Path(resolved))
    return found


def collect(args: argparse.Namespace) -> Dict[str, object]:
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    copyright_dir = args.copyrights.resolve()
    excludes = [path.resolve() for path in args.exclude]
    library_path = ":".join(
        item for item in (str(output), str(root), os.environ.get("LD_LIBRARY_PATH", "")) if item
    )

    queue = deque(iter_initial_elfs(root, output, excludes))
    scanned = set()
    bundled: Dict[str, Dict[str, Optional[str]]] = {}
    host: Dict[str, Dict[str, Optional[str]]] = {}
    unresolved: Dict[str, List[str]] = {}
    has_dpkg_query = shutil.which("dpkg-query") is not None

    def bundle_library(soname: str, source: Path) -> None:
        source = source.resolve()
        target = output / soname
        if target.exists():
            if not filecmp.cmp(source, target, shallow=False):
                raise RuntimeError(f"Conflicting resolutions for {soname}: {source} and {target}")
        else:
            shutil.copy2(source, target)

        package = owning_package(source) if has_dpkg_query else None
        copyright_name = copy_copyright(package, copyright_dir) if package else None
        bundled[soname] = {
            "source": str(source),
            "package": package,
            "copyright": copyright_name,
        }
        queue.append(target)

    for soname, optional_path in find_optional_libraries(args.optional).items():
        bundle_library(soname, optional_path)

    while queue:
        binary = queue.popleft().resolve()
        if binary in scanned or not is_elf(binary):
            continue
        scanned.add(binary)
        for soname, source in run_ldd(binary, library_path):
            if soname in HOST_GLIBC_LIBRARIES:
                host.setdefault(
                    soname,
                    {"reason": "host-glibc", "source": str(source) if source else None},
                )
                continue
            if is_driver_library(soname):
                host.setdefault(
                    soname,
                    {
                        "reason": "nvidia-driver",
                        "source": str(source) if source else None,
                    },
                )
                continue
            if source is None:
                unresolved.setdefault(soname, []).append(str(binary))
                continue
            source = source.resolve()
            if inside(source, root):
                queue.append(source)
                continue
            bundle_library(soname, source)

    if unresolved:
        lines = ["Unresolved shared libraries:"]
        for soname, users in sorted(unresolved.items()):
            lines.append(f"  {soname}: {', '.join(sorted(set(users)))}")
        raise RuntimeError("\n".join(lines))

    copied_driver_libraries = [path.name for path in output.iterdir() if is_driver_library(path.name)]
    if copied_driver_libraries:
        raise RuntimeError(f"NVIDIA driver libraries must not be bundled: {copied_driver_libraries}")

    return {
        "format_version": 1,
        "bundled": dict(sorted(bundled.items())),
        "host": dict(sorted(host.items())),
        "scanned_elf_files": len(scanned),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Electron distribution root")
    parser.add_argument("--output", type=Path, required=True, help="Directory for copied libraries")
    parser.add_argument("--report", type=Path, required=True, help="JSON dependency report")
    parser.add_argument("--copyrights", type=Path, required=True, help="Directory for distro copyright files")
    parser.add_argument("--exclude", action="append", default=[], type=Path, help="Subtree not scanned")
    parser.add_argument("--optional", action="append", default=[], help="Runtime-loaded SONAME to include")
    args = parser.parse_args()
    try:
        report = collect(args)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Bundled {len(report['bundled'])} shared libraries; "
        f"{report['scanned_elf_files']} ELF files audited."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
