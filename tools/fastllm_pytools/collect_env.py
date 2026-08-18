# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Collect FastLLM, PyTorch, CUDA, and host environment information.

Adapted from vLLM's ``vllm/collect_env.py``.  Keep this module importable even
when optional runtime dependencies or accelerator tools are unavailable so it
can still be used in bug reports from partially configured machines.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import locale
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable


SECRET_TERMS = ("secret", "token", "api", "access", "password", "credential")
ENV_PREFIXES = (
    "FASTLLM",
    "TORCH",
    "PYTORCH",
    "CUDA",
    "CUBLAS",
    "CUDNN",
    "NCCL",
    "NVIDIA",
    "ROCM",
    "HIP",
    "OMP_",
    "MKL_",
    "NUMA",
)
PACKAGE_PATTERNS = (
    "ftllm",
    "fastllm",
    "torch",
    "triton",
    "transformers",
    "numpy",
    "nvidia",
    "nccl",
    "cuda",
    "cudnn",
    "flashinfer",
)


def run(command: str | list[str], timeout: int = 20) -> tuple[int, str, str]:
    """Return command exit code, stdout, and stderr without raising."""
    try:
        completed = subprocess.run(
            command,
            shell=isinstance(command, str),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return 127, "", str(exc)

    encoding = "oem" if sys.platform == "win32" else locale.getpreferredencoding(False)
    return (
        completed.returncode,
        completed.stdout.decode(encoding, errors="replace").strip(),
        completed.stderr.decode(encoding, errors="replace").strip(),
    )


def command_output(command: str | list[str], fallback: str = "Could not collect") -> str:
    code, output, _ = run(command)
    return output if code == 0 and output else fallback


def package_version() -> str:
    for name in ("ftllm", "ftllm-nightly", "ftllm-rocm"):
        try:
            return f"{name} {importlib.metadata.version(name)}"
        except importlib.metadata.PackageNotFoundError:
            continue
    return "Not installed"


def package_location() -> str:
    try:
        spec = importlib.util.find_spec("ftllm")
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is None:
        return "Not installed"
    if spec.origin:
        return str(Path(spec.origin).resolve())
    if spec.submodule_search_locations:
        return str(Path(next(iter(spec.submodule_search_locations))).resolve())
    return "Could not collect"


def git_info() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    code, revision, _ = run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    if code != 0:
        return "Could not collect", "Could not collect"
    code, status, _ = run(["git", "-C", str(repo_root), "status", "--short"])
    if code != 0:
        status = "Could not collect"
    elif not status:
        status = "clean"
    return revision, status


def build_info() -> str:
    candidates: list[Path] = []
    try:
        spec = importlib.util.find_spec("ftllm")
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        candidates.extend(
            Path(location) / "build_info.json"
            for location in spec.submodule_search_locations
        )
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        (
            repo_root / "build-fastllm" / "build_info.json",
            repo_root / "tools" / "fastllm_pytools" / "build_info.json",
        )
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        return json.dumps(data, sort_keys=True)
    return "Could not collect (build_info.json not found)"


def torch_info() -> str:
    try:
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()
    except (ImportError, AttributeError, NameError, OSError, RuntimeError) as exc:
        return "PyTorch environment unavailable: {}".format(exc)


def gpu_topology() -> str:
    output = command_output("nvidia-smi topo -m", fallback="")
    if output:
        return output
    return command_output("rocm-smi --showtopo")


def relevant_packages(patterns: Iterable[str] = PACKAGE_PATTERNS) -> str:
    code, output, _ = run([sys.executable, "-m", "pip", "list", "--format=freeze"])
    if code != 0:
        return "Could not collect"
    lowered_patterns = tuple(pattern.lower() for pattern in patterns)
    selected = [
        line
        for line in output.splitlines()
        if any(pattern in line.lower() for pattern in lowered_patterns)
    ]
    return "\n".join(selected) if selected else "No relevant packages"


def environment_variables() -> str:
    selected = []
    for key, value in sorted(os.environ.items()):
        if any(term in key.lower() for term in SECRET_TERMS):
            continue
        if key.startswith(ENV_PREFIXES):
            selected.append(f"{key}={value}")
    return "\n".join(selected) if selected else "No relevant environment variables"


def fastllm_info() -> str:
    revision, status = git_info()
    return """\
==============================
        FastLLM Info
==============================
FastLLM package               : {version}
FastLLM package location      : {location}
FastLLM git revision          : {revision}
FastLLM git status            : {status}
FastLLM build flags           : {build}
Python executable             : {python}
Platform                      : {platform}
GCC version                   : {gcc}
Clang version                 : {clang}
CMake version                 : {cmake}
NVCC version                  : {nvcc}

==============================
         GPU Topology
==============================
{topology}

==============================
    Relevant Python Packages
==============================
{packages}

==============================
     Environment Variables
==============================
{env_vars}""".format(
        version=package_version(),
        location=package_location(),
        revision=revision,
        status=status,
        build=build_info(),
        python=sys.executable,
        platform=platform.platform(),
        gcc=command_output("gcc --version").splitlines()[0],
        clang=command_output("clang --version").splitlines()[0],
        cmake=command_output("cmake --version").splitlines()[0],
        nvcc=command_output("nvcc --version"),
        topology=gpu_topology(),
        packages=relevant_packages(),
        env_vars=environment_variables(),
    )


def get_pretty_env_info() -> str:
    return torch_info().rstrip() + "\n\n" + fastllm_info()


def main() -> None:
    print("Collecting environment information...")
    print(get_pretty_env_info())


if __name__ == "__main__":
    main()
