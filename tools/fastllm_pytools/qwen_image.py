"""Run the bundled native Qwen-Image-2.1 programs."""
import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"server": "qwen_image_server", "generate": "qwen_image_generate"}
    parser = argparse.ArgumentParser(
        prog="ftllm qwen-image",
        description="Native Qwen-Image-2.1 generation and HTTP service")
    parser.add_argument("command", choices=commands)
    if not argv or argv[0] not in commands:
        parser.parse_args(argv)
        return
    package = Path(__file__).resolve().parent
    binary = package / commands[argv[0]]
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"Missing native program: {binary}. Install a wheel with Qwen Image support.")
    options = argv[1:]
    if argv[0] == "server" and "--demo" not in options:
        options += ["--demo", str(package / "qwen_image_assets/demo.html")]
    os.execv(str(binary), [str(binary), *options])


if __name__ == "__main__":
    main()
