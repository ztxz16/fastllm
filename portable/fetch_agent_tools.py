#!/usr/bin/env python3
"""Install pinned standalone search tools required by Pi's offline tool mode."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/ftllm_agent_runtime/scripts"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    metadata = args.runtime_dir / "share/ftllm-agent-tools"
    for name, info in TOOLS.items():
        filename = info["url"].rsplit("/", 1)[-1]
        archive = cached_archive(info["url"], info["sha256"], args.cache_dir / filename, args.offline)
        prefix = filename.removesuffix(".tar.gz")
        atomic_write(args.runtime_dir / "bin" / name, extract_member(archive, f"{prefix}/{name}"), 0o755)
        for license_name in info["licenses"]:
            atomic_write(metadata / name / license_name, extract_member(archive, f"{prefix}/{license_name}"), 0o644)
    atomic_write(metadata / "manifest.json", (json.dumps(TOOLS, indent=2) + "\n").encode(), 0o644)


if __name__ == "__main__":
    main()
