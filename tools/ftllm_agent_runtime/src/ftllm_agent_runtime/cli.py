"""Command-line smoke tests for the packaged runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import PiAgentRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftllm-agent-runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="show packaged Pi runtime information")

    probe = subparsers.add_parser("probe", help="run one read-only agent probe")
    probe.add_argument("prompt")
    probe.add_argument("--api-base", required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--api-key", default="")
    probe.add_argument("--file", action="append", default=[])
    probe.add_argument(
        "--directory",
        help="run with writable Pi coding tools from this working directory",
    )
    probe.add_argument("--timeout", type=float, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runtime = PiAgentRuntime(
        api_base=getattr(args, "api_base", "http://127.0.0.1:8080/v1"),
        model=getattr(args, "model", "runtime-info"),
        api_key=getattr(args, "api_key", ""),
        timeout=getattr(args, "timeout", 300),
    )
    if args.command == "info":
        print(json.dumps(runtime.info(), ensure_ascii=False, indent=2))
        return 0

    files = []
    for value in args.file:
        path = Path(value).expanduser().resolve()
        files.append(
            {
                "name": path.name,
                "text": path.read_text(encoding="utf-8"),
                "size": path.stat().st_size,
            }
        )
    if not files and not args.directory:
        files = [{"name": "runtime.txt", "text": "FastLLM Pi bridge smoke test."}]
    system_prompt = (
        "You are a coding agent. Inspect the selected working directory, complete "
        "the requested task, and verify your work."
        if args.directory else
        "You are a read-only project assistant. Always call runtime_info and "
        "read_project_file before answering. Never claim to execute project code."
    )
    for event in runtime.stream(
        args.prompt,
        files,
        system_prompt,
        thinking_level="off",
        working_directory=args.directory,
    ):
        print(json.dumps(event, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
