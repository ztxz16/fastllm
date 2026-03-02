#!/usr/bin/env python3
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Fastllm Studio")
    parser.add_argument("--style", type=str, default="default",
                        help="QML style folder name (default: default)")
    args, remaining = parser.parse_known_args()

    app, engine, repo_vm, market_vm, chat_vm = create_app(
        argv=[sys.argv[0]] + remaining,
        style=args.style,
    )

    ret = app.exec()
    repo_vm.shutdown()
    sys.exit(ret)


if __name__ == "__main__":
    main()
