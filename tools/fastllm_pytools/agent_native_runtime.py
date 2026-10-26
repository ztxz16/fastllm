"""Reuse a private Python ELF runtime on hosts too old for bundled Node.js."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _elf(path):
    if path.is_symlink() or not path.is_file():
        return False
    with path.open("rb") as source:
        return source.read(4) == b"\x7fELF"


def prepare_node(node):
    """Adapt only the downloaded Node when the host cannot run it.

    Do not replace system libraries or export LD_LIBRARY_PATH: agent tools may
    execute host programs that still need the host's original loader/libc.
    """
    if platform.system() != "Linux" or not _elf(node):
        return None
    probe = subprocess.run([str(node), "--version"], capture_output=True, timeout=15)
    if probe.returncode == 0:
        return None
    diagnostics = probe.stderr.decode("utf-8", errors="replace")
    if not any(version in diagnostics for version in ("GLIBC_", "GLIBCXX_", "CXXABI_")):
        raise RuntimeError("Node.js could not start: " + diagnostics[-2000:])
    patcher = shutil.which("patchelf")
    if not patcher:
        raise RuntimeError("Node.js requires newer system libraries. Use a supported Linux system, "
                           "or install patchelf alongside Python's compatible private runtime.\n" + diagnostics[-2000:])
    def inspect(option):
        return subprocess.check_output([patcher, option, sys.executable], text=True, timeout=15).strip()
    loader = inspect("--print-interpreter")
    origin = str(Path(sys.executable).resolve().parent)
    paths = inspect("--print-rpath").replace("${ORIGIN}", origin).replace("$ORIGIN", origin)
    paths = ":".join(p for p in paths.split(":") if os.path.isabs(p) and Path(p).is_dir())
    if not loader or not paths:
        raise RuntimeError("Node.js requires newer system libraries; Python has no private ELF runtime to reuse.")
    # Verify compatibility before changing even the private staging files.
    subprocess.run([loader, "--library-path", paths, str(node), "--version"],
                   check=True, capture_output=True, timeout=15)
    # Preserve program-header order: sorting Node's overlapping LOAD
    # segments can make older loaders crash before reaching main().
    subprocess.run([patcher, "--no-sort", "--force-rpath", "--set-rpath", paths,
                    "--set-interpreter", loader, str(node)],
                   check=True, capture_output=True, timeout=15)
    subprocess.run([str(node), "-e", "process.exit(0)"], check=True, capture_output=True, timeout=15)
