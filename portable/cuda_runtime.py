#!/usr/bin/env python3
"""Check bundled CUDA runtime files without relying on the build host's libraries."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path


CUDA_RUNTIME_FILES = {
    "nvidia-cuda-runtime-cu12": ("cuda_runtime", ("libcudart.so.12",)),
    "nvidia-cublas-cu12": ("cublas", ("libcublas.so.12", "libcublasLt.so.12")),
    "nvidia-nccl-cu12": ("nccl", ("libnccl.so.2",)),
}


def check_cuda_runtime(runtime: Path) -> dict[str, str]:
    runtime = runtime.resolve()
    libraries = {}
    for package, (module, sonames) in CUDA_RUNTIME_FILES.items():
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Missing bundled CUDA package: {package}") from error
        for soname in sonames:
            path = Path(distribution.locate_file(f"nvidia/{module}/lib/{soname}"))
            if not path.is_file() or not path.resolve().is_relative_to(runtime):
                raise RuntimeError(f"Missing bundled CUDA library: {soname} ({path})")
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    raise RuntimeError(f"Invalid bundled CUDA library: {path}")
            libraries[soname] = str(path.resolve().relative_to(runtime))
    return libraries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        for soname, path in check_cuda_runtime(args.runtime_root).items():
            print(f"Bundled CUDA library: {soname} ({path})")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"CUDA runtime check failed: {error}\n")
