#!/usr/bin/env python3
"""Install pinned standalone search tools required by Pi's offline tool mode."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/ftllm_agent_runtime/scripts"))
from fetch_tools import TOOLS, install_tools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    metadata = args.runtime_dir / "share/ftllm-agent-tools"
    install_tools(args.runtime_dir / "bin", metadata, args.cache_dir, args.offline)


if __name__ == "__main__":
    main()
