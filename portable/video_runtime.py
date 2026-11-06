#!/usr/bin/env python3
"""Validate the video backend and its package-local media tools."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess


def check_video_runtime(bundle: Path, run: bool = False) -> dict[str, str]:
    bundle = bundle.resolve()
    try:
        distribution = importlib.metadata.distribution("ftllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Missing bundled ftllm package") from error
    files = {}
    for name in ("minimax_h3_server", "minimax_h3_generate", "libfastllm_minimax_h3.so"):
        files[name] = Path(distribution.locate_file("ftllm/" + name))
    for name in ("ffmpeg", "ffprobe"):
        files[name] = bundle / "media/bin" / name
    for name, path in files.items():
        parent = bundle / ("media" if name in ("ffmpeg", "ffprobe") else "runtime")
        if not path.is_file() or not path.resolve().is_relative_to(parent):
            raise RuntimeError(f"Missing bundled video runtime: {name} ({path})")
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise RuntimeError(f"Invalid bundled video ELF: {path}")
        if not name.endswith(".so") and not os.access(path, os.X_OK):
            raise RuntimeError(f"Bundled video tool is not executable: {path}")
    if run:
        for name in ("minimax_h3_server", "minimax_h3_generate", "ffmpeg", "ffprobe"):
            argument = "-version" if name in ("ffmpeg", "ffprobe") else "--help"
            result = subprocess.run([str(files[name]), argument], capture_output=True,
                                    text=True, timeout=30)
            if result.returncode:
                raise RuntimeError(f"Bundled {name} failed: {result.stderr.strip()}")
    return {name: str(path.resolve().relative_to(bundle)) for name, path in files.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        for name, path in check_video_runtime(args.bundle_root).items():
            print(f"Bundled video runtime: {name} ({path})")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"Video runtime check failed: {error}\n")
