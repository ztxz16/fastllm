#!/usr/bin/env python3
"""Validate the native image/video backends and package-local media tools."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess


NATIVE_FILES = (
    "minimax_h3_server", "minimax_h3_generate", "libfastllm_minimax_h3.so",
    "minimax_h3_image_server", "minimax_h3_image_generate",
    "qwen_image_server", "qwen_image_generate", "libfastllm_qwen_image.so",
)
ASSETS = (
    "h3_assets/demo.html", "h3_assets/openapi.json",
    "h3_image_assets/demo.html", "h3_image_assets/openapi.json",
    "h3_image_assets/image-workspace.js", "h3_image_assets/image-workspace.css",
    "qwen_image_assets/demo.html", "qwen_image_assets/openapi.json",
)


def check_video_runtime(bundle: Path, run: bool = False) -> dict[str, str]:
    bundle = bundle.resolve()
    try:
        distribution = importlib.metadata.distribution("ftllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Missing bundled ftllm package") from error
    files = {}
    for name in NATIVE_FILES + ASSETS:
        files[name] = Path(distribution.locate_file("ftllm/" + name))
    for name in ("ffmpeg", "ffprobe"):
        files[name] = bundle / "media/bin" / name
    for name, path in files.items():
        parent = bundle / ("media" if name in ("ffmpeg", "ffprobe") else "runtime")
        if not path.is_file() or not path.resolve().is_relative_to(parent):
            raise RuntimeError(f"Missing bundled media runtime: {name} ({path})")
        if name in ASSETS:
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise RuntimeError(f"Invalid bundled media ELF: {path}")
        if not name.endswith(".so") and not os.access(path, os.X_OK):
            raise RuntimeError(f"Bundled media tool is not executable: {path}")
    if run:
        for name in (*[name for name in NATIVE_FILES if not name.endswith(".so")], "ffmpeg", "ffprobe"):
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
            print(f"Bundled media runtime: {name} ({path})")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"Media runtime check failed: {error}\n")
