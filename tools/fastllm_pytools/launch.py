#!/usr/bin/env python3
"""Run the ftllm CLI with one new, never reused log folder per run.

    python -m ftllm.launch --log_dir DIR [--mode user] [--user-language zh] server MODEL ...

Each run creates DIR/runs/<UTC time>-<pid>/ with exclusive creation (fails if
the name exists, so no file is ever appended to or overwritten across runs):
  server.log      complete plain stream of the server (stdout and stderr), written once
  user.log        plain user view (only with --mode user)
  diagnostics/    FASTLLM_DIAGNOSTIC_LOG_DIR of this run
  run-meta.json   build, library hash, command, environment, start time
  run-end.json    end time and exit status
  recorded/       copies of files given with --record-file (for example the deploy script)
Nothing is ever deleted. The old --log/--raw-log pipe through terminal_colorizer
keeps working unchanged; this launcher is opt-in.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as _dt
import hashlib
import io
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

try:
    from . import terminal_colorizer as _tc
except ImportError:  # run as a plain script from the tools directory
    import terminal_colorizer as _tc  # type: ignore

_PACKAGE_DIR = Path(__file__).resolve().parent
_ENV_PREFIXES = ("FASTLLM_", "FT_", "CUDA_", "NCCL_", "PYTHON", "TRITON_")
_ENV_NAMES = ("LD_LIBRARY_PATH", "TMPDIR", "XDG_CACHE_HOME", "PATH")
_FORWARDED = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_PR_SET_PDEATHSIG = 1


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def default_run_id(now: Optional[_dt.datetime] = None, pid: Optional[int] = None) -> str:
    now = now or _utc_now()
    return now.strftime("%Y%m%dT%H%M%S") + f".{now.microsecond:06d}Z-{pid if pid is not None else os.getpid()}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json_exclusive(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _tree_bytes(root: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def _build_info() -> dict:
    info: dict = {"package_dir": str(_PACKAGE_DIR)}
    build_info = _PACKAGE_DIR / "build_info.json"
    if build_info.is_file():
        try:
            info["build_info"] = json.loads(build_info.read_text(encoding="utf-8"))
        except ValueError as e:
            info["build_info_error"] = str(e)
    for lib in ("libfastllm_tools.so", "fastllm_tools.dll"):
        p = _PACKAGE_DIR / lib
        if p.is_file():
            info["library"] = {"path": str(p), "bytes": p.stat().st_size, "sha256": _sha256(p)}
            break
    return info


def _selected_env(env: dict) -> dict:
    return {k: v for k, v in sorted(env.items()) if k.startswith(_ENV_PREFIXES) or k in _ENV_NAMES}


def create_run_dir(log_dir: Path, run_id: str) -> Path:
    """Create log_dir/runs/run_id exclusively; raise FileExistsError if it exists."""
    runs = log_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = runs / run_id
    os.mkdir(run_dir)  # no exist_ok: a name is never reused
    os.mkdir(run_dir / "diagnostics")
    return run_dir


def _child_preexec() -> None:
    # Own process group: a terminal Ctrl-C reaches only the launcher, which forwards it
    # exactly once (a double SIGINT would force-exit the server and skip its shutdown).
    os.setpgid(0, 0)
    # If the launcher dies without forwarding, the server still gets SIGTERM.
    try:
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def run(
    log_dir: Path,
    child_argv: Sequence[str],
    mode: Optional[str] = None,
    user_language: str = "en",
    tty: Optional[str] = "/dev/tty",
    record_files: Sequence[Path] = (),
    launcher_argv: Optional[Sequence[str]] = None,
    run_id_fn: Callable[[], str] = default_run_id,
) -> int:
    log_dir = Path(log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    bytes_before = _tree_bytes(log_dir)
    run_dir = create_run_dir(log_dir, run_id_fn())

    env = dict(os.environ)
    previous_diag = env.get("FASTLLM_DIAGNOSTIC_LOG_DIR")
    env["FASTLLM_DIAGNOSTIC_LOG_DIR"] = str(run_dir / "diagnostics")
    env["FASTLLM_RUN_DIR"] = str(run_dir)
    env["PYTHONUNBUFFERED"] = "1"

    recorded = []
    if record_files:
        (run_dir / "recorded").mkdir()
    for src in record_files:
        src = Path(src).expanduser().resolve()
        dst = run_dir / "recorded" / src.name
        data = src.read_bytes()
        with dst.open("xb") as f:
            f.write(data)
        recorded.append({"source": str(src), "copy": str(dst), "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()})

    start = _utc_now()
    meta = {
        "schema": "ftllm.launch/1",
        "run_dir": str(run_dir),
        "start_utc": start.isoformat(),
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "launcher_pid": os.getpid(),
        "launcher_argv": list(launcher_argv if launcher_argv is not None else sys.argv),
        "python": sys.executable,
        "child_argv": list(child_argv),
        "env": _selected_env(env),
        "diagnostic_dir_overridden_from": previous_diag,
        "recorded_files": recorded,
        "log_dir_bytes_before_run": bytes_before,
        **_build_info(),
    }

    terminal = None
    if tty:
        try:
            terminal = open(tty, "w", encoding="utf-8", buffering=1)
        except OSError:
            terminal = None
    log = (run_dir / "server.log").open("x", encoding="utf-8", buffering=1)
    user_log = (run_dir / "user.log").open("x", encoding="utf-8", buffering=1) \
        if _tc.observability_mode(mode) == "user" else None
    banner = (f"[ftllm.launch] run_dir={run_dir} log_dir_bytes_before_run={bytes_before} "
              f"(nothing is deleted automatically)\n")

    child = subprocess.Popen(list(child_argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=env, preexec_fn=_child_preexec)
    meta["child_pid"] = child.pid
    _write_json_exclusive(run_dir / "run-meta.json", meta)

    received: List[int] = []

    def _forward(signum, _frame):
        received.append(signum)
        if child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    previous = {s: signal.signal(s, _forward) for s in _FORWARDED}
    lines = 0
    stream = None
    try:
        log.write(banner)
        if terminal is not None:
            terminal.write(banner)
        stream = io.TextIOWrapper(child.stdout, encoding="utf-8", errors="replace")

        def _counted():
            nonlocal lines
            for line in stream:
                lines += 1
                yield line

        _tc.process_lines(_counted(), log, terminal, mode, None, user_log, user_language)
        code = child.wait()
    finally:
        for s, h in previous.items():
            signal.signal(s, h)
        if stream is not None:
            stream.close()
        log.close()
        if user_log is not None:
            user_log.close()
        if terminal is not None:
            terminal.close()
    end = _utc_now()
    _write_json_exclusive(run_dir / "run-end.json", {
        "end_utc": end.isoformat(),
        "seconds": round((end - start).total_seconds(), 3),
        "child_returncode": code,
        "child_killed_by_signal": -code if code < 0 else None,
        "signals_forwarded": received,
        "server_log_lines": lines,
        "server_log_bytes": (run_dir / "server.log").stat().st_size,
    })
    return 128 - code if code < 0 else code


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", "--log-dir", required=True,
                        help="root folder; each run gets a new folder under <log_dir>/runs/")
    parser.add_argument("--mode", choices=("off", "user", "debug"), default=None,
                        help="terminal presentation (default: FASTLLM_OBSERVABILITY_MODE or off); server.log is always complete")
    parser.add_argument("--user-language", "--user_language", choices=("en", "zh"), default="en")
    parser.add_argument("--tty", default="/dev/tty", help="terminal device; empty string for none")
    parser.add_argument("--record-file", "--record_file", action="append", default=[],
                        help="copy this file into the run folder (repeatable), e.g. the deploy script")
    parser.add_argument("cli_args", nargs=argparse.REMAINDER,
                        help="arguments for ftllm.cli, e.g. server MODEL --port 8080 ...")
    args = parser.parse_args(argv)
    cli_args = args.cli_args[1:] if args.cli_args[:1] == ["--"] else args.cli_args
    if not cli_args:
        parser.error("missing ftllm.cli arguments (for example: server MODEL ...)")
    child = [sys.executable, "-m", "ftllm.cli", *cli_args]
    return run(Path(args.log_dir), child, args.mode, args.user_language, args.tty or None,
               [Path(p) for p in args.record_file], launcher_argv=sys.argv)


if __name__ == "__main__":
    raise SystemExit(_main())
