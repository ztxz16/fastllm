#!/usr/bin/env python3
"""Fetch the pinned search tools and licenses included in the runtime wheel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

from fetch_pi import atomic_write, cached_archive, extract_member


TOOLS = {
    "rg": {
        "version": "14.1.1",
        "url": "https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz",
        "sha256": "4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e",
        "licenses": ["COPYING", "LICENSE-MIT", "UNLICENSE"],
    },
    "fd": {
        "version": "10.2.0",
        "url": "https://github.com/sharkdp/fd/releases/download/v10.2.0/fd-v10.2.0-x86_64-unknown-linux-musl.tar.gz",
        "sha256": "d9bfa25ec28624545c222992e1b00673b7c9ca5eb15393c40369f10b28f9c932",
        "licenses": ["LICENSE-APACHE", "LICENSE-MIT"],
    },
}


def install_tools(binary_dir: Path, license_dir: Path, cache_dir: Path | None, offline: bool) -> None:
    for name, info in TOOLS.items():
        filename = info["url"].rsplit("/", 1)[-1]
        cached = cache_dir / filename if cache_dir is not None else None
        archive = cached_archive(info["url"], info["sha256"], cached, offline)
        prefix = filename.removesuffix(".tar.gz")
        binary = extract_member(archive, f"{prefix}/{name}")
        atomic_write(binary_dir / name, binary, 0o755)
        for license_name in info["licenses"]:
            atomic_write(license_dir / name / license_name,
                         extract_member(archive, f"{prefix}/{license_name}"), 0o644)
        print(f"{name} {info['version']}: {binary_dir / name} ({len(binary)} bytes)")
    atomic_write(license_dir / "manifest.json", (json.dumps(TOOLS, indent=2) + "\n").encode(), 0o644)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="Cache verified release archives")
    parser.add_argument("--offline", action="store_true", help="Use verified local archives only")
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        parser.error("the runtime wheel supports Linux x86-64 only")
    package = Path(__file__).resolve().parents[1] / "src/ftllm_agent_runtime"
    cache = args.cache_dir.expanduser() if args.cache_dir is not None else None
    install_tools(package / "bin", package / "licenses/agent-tools", cache, args.offline)


if __name__ == "__main__":
    main()
