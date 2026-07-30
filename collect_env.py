#!/usr/bin/env python3
"""Repository entry point for FastLLM environment collection."""

import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from fastllm_pytools.collect_env import main  # noqa: E402


if __name__ == "__main__":
    main()
