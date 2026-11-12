#!/usr/bin/env python3
"""运行已有进程底座产物并生成报告；本脚本从不编译或安装任何组件。"""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test/nvfp4"))
from xlsx_report import write_xlsx
from linux_process import check_pidfd_support, open_pidfd

sys.path.insert(0, str(ROOT / "tools"))
from tp_process_utils import enable_child_subreaper, wait_case, cleanup_process_group, isolated_process


def capture(command, timeout=10):
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def save_report(output, rows, metadata):
    expected = metadata.get("expected_case_count")
    complete = expected is None or len(rows) == expected
    status = "PASS" if complete and rows and all(row["status"] == "PASS" for row in rows) else "FAIL"
    summary = {"status": status, "metadata": metadata, "cases": rows}
    if expected is not None:
        summary.update(complete=complete, completed_case_count=len(rows), expected_case_count=expected)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["case", "backend", "status", "max_abs_error", "mean_abs_error", "failed_elements", "reason"]
    for key in ('custom_calls', 'nccl_calls', 'prefix_hits', 'reused_tokens', 'evictions', 'prefix_bytes',
                'token_mismatches', 'compared_steps', 'first_divergence_token_1based',
                'requests', 'generated_tokens', 'elapsed_s', 'http_requests', 'sse_requests', 'reaped_processes',
                'operation_timeout_ms', 'failure_return_ms', 'failure_phase', 'fault_rank',
                'workers_reaped', 'observed_error'):
        if any(key in row for row in rows):
            fields.append(key)

    def cell(value):
        if isinstance(value, float):
            return f"{value:.3f}"
        return "n/a" if value is None else str(value)

    display = [[cell(row.get(key)) for key in fields] for row in rows]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(display)
    title = metadata.get("report_title", "Process TP infrastructure")
    description = metadata.get("report_description", "无数值输出的生命周期检查，误差字段为 n/a。所有原始结果见 summary.json。")
    if expected is not None:
        description += f" 已完成 {len(rows)}/{expected} 个用例。"
    lines = [f"# {title}: {status}", "",
             "功能检查；通过范围仅限本报告实际执行的配置与用例，不代表性能结果。",
             description, "",
             "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    lines.extend("| " + " | ".join(item.replace("|", "\\|").replace("\n", " ") for item in row) + " |"
                 for row in display)
    lines.extend(["", "```json", json.dumps(metadata, ensure_ascii=False, indent=2), "```"])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_xlsx(output / "summary.xlsx", [
        ("Cases", fields, display),
        ("Metadata", ["key", "value"], [[key, json.dumps(value, ensure_ascii=False)] for key, value in metadata.items()]),
    ])
    return status


@contextmanager
def case_process(command, **kwargs):
    with isolated_process(command, cwd=ROOT, **kwargs) as process:
        yield process


def parse_report(text):
    offset = 0
    # json11 输出多行 JSON；NCCL 等库也可能先向 stdout 写日志。
    # 第一个行首对象必须完整占据剩余输出；解析失败后不能跳到嵌套对象误报 PASS。
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("{"):
            return json.loads(text[offset + len(line) - len(stripped):])
        offset += len(line)
    raise ValueError("stdout has no complete final JSON object")


def read_startup_report(stream, timeout, log_path):
    deadline = time.monotonic() + timeout
    output = bytearray()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
                raise RuntimeError("parent did not report worker identities in time")
            piece = os.read(stream.fileno(), 65536)
            if not piece:
                raise RuntimeError("invalid parent startup output: EOF before complete JSON")
            output.extend(piece)
            if len(output) > 1024 * 1024:
                raise RuntimeError("parent startup output exceeds 1 MiB")
            if output.endswith(b"\n"):
                try:
                    return parse_report(output.decode("utf-8"))
                except ValueError:
                    pass
    finally:
        log_path.write_bytes(output)


def run_case(name, command, output, timeout, parse_json=True, environment=None):
    row = {"case": name, "status": "FAIL", "command": shlex.join(command)}
    stdout_path = output / f"{name}.stdout.log"
    stderr_path = output / f"{name}.stderr.log"
    try:
        # 直接写日志文件：超时或中断也保留输出，不等待后代关闭继承的 pipe。
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            with case_process(command, stdout=stdout, stderr=stderr, env=environment) as process:
                returncode = wait_case(process, timeout)
        result = stdout_path.read_text(encoding="utf-8", errors="replace")
        errors = stderr_path.read_text(encoding="utf-8", errors="replace")
        if parse_json:
            data = parse_report(result)
            row.update(data)
            row["case"] = name
            if data.get("status") != "PASS":
                row["status"] = "FAIL"
        else:
            row.update(backend="python/options", status="PASS" if returncode == 0 else "FAIL")
        if returncode != 0:
            row.update(status="FAIL", reason=row.get("reason", errors.strip() or f"exit {returncode}"))
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError, IndexError) as error:
        row.update(status="FAIL", reason=str(error))
    return row


def parent_death_case(command, timeout, output):
    row = {"case": "parent_death", "status": "FAIL", "backend": "process/control-only",
           "command": shlex.join(command)}
    pidfds = []
    try:
        with (output / "parent_death.stderr.log").open("w", encoding="utf-8") as errors:
            with case_process(command, stdout=subprocess.PIPE, stderr=errors) as process:
                with process.stdout:
                    data = read_startup_report(process.stdout, timeout, output / "parent_death.stdout.log")
                    row["workers"] = data["workers"]
                    for worker in data["workers"]:
                        pidfds.append(open_pidfd(worker["pid"]))
                    if not pidfds:
                        raise RuntimeError("parent reported no workers")

                    # 先只杀协调器，观察 worker 自行退出；整组兜底清理必须在观察之后。
                    os.kill(process.pid, signal.SIGKILL)
                    wait_case(process, 5)
                    deadline = time.monotonic() + 5
                    for fd in pidfds:
                        if not select.select([fd], [], [], max(0, deadline - time.monotonic()))[0]:
                            raise RuntimeError("worker survived parent death")
                    row["status"] = "PASS"
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired, AttributeError) as error:
        row["status"] = "FAIL"
        row["reason"] = str(error)
    finally:
        for fd in pidfds:
            os.close(fd)
    return row


def run_cleanup_checks(output):
    from test_process_tp_runner import run_tests

    # 自测会创建多组独立 session，因此直接在验收器内执行。
    # 若再套一层可被 SIGKILL 的子进程，其清理逻辑会被跳过，内层 session 也不在外层组内。
    with (output / "runner_cleanup.stdout.log").open("w", encoding="utf-8") as stdout, \
            (output / "runner_cleanup.stderr.log").open("w", encoding="utf-8") as stderr:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            row = run_tests()
    row.update(case="runner_cleanup", execution="in-process Python test suite")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="已编译的 fastllm_tp_check")
    parser.add_argument("--worker", type=Path, help="已编译的 fastllm_tp_worker")
    parser.add_argument("--devices", default="0,1", help="CUDA_VISIBLE_DEVICES 之后的显式逻辑设备列表")
    parser.add_argument("--cuda", action="store_true", help="额外验证真实 CUDA/NCCL，必须有 GPU")
    parser.add_argument("--custom-allreduce", action="store_true", help="CUDA 进程使用 TP2 IPC 规约并验证分支边界")
    parser.add_argument("--frontend-only", action="store_true", help="仅执行 Python 参数与进程清理检查，不运行 C++ 产物")
    parser.add_argument("--pid-namespace", action="store_true", help="额外用 unshare 验证 PID 1 协调器，须允许创建 user/PID namespace")
    parser.add_argument("--build-cache", type=Path, help="服务器构建的 CMakeCache.txt，保存架构配置")
    parser.add_argument("--output-root", type=Path, default=ROOT / "test/distributed/results")
    args = parser.parse_args()
    if args.custom_allreduce and not args.cuda:
        parser.error("--custom-allreduce requires --cuda")
    if args.frontend_only and (args.cuda or args.pid_namespace):
        parser.error("--frontend-only cannot be combined with --cuda or --pid-namespace")
    if not args.frontend_only and (args.check is None or args.worker is None):
        parser.error("--check and --worker are required unless --frontend-only")

    output = args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "command": shlex.join([sys.executable, *sys.argv]), "git": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]), "python": sys.version,
        "python_executable": sys.executable,
        "pidfd_open_api": "python" if callable(getattr(os, "pidfd_open", None)) else "linux syscall",
        "pidfd_signal_api": "python" if callable(getattr(signal, "pidfd_send_signal", None)) else "linux syscall",
        "test_kind": "functional infrastructure checks; not an inference benchmark",
        "seed": 0, "warmup": 0, "iterations": 1, "repeats": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "gpu": "not queried (frontend/control-only)",
        "cuda_runtime_version": "worker metadata when --cuda; otherwise n/a",
        "compile_architecture": "see build cache; CUDA worker contains production kernels",
        "custom_allreduce": args.custom_allreduce,
        "cmake_cuda_architecture": "not supplied", "frontend_only": args.frontend_only,
    }
    rows = []
    try:
        check_pidfd_support()
        if args.cuda:
            metadata["gpu"] = capture(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"])
        if args.build_cache:
            cache = args.build_cache.read_text(encoding="utf-8")
            (output / "CMakeCache.txt").write_text(cache, encoding="utf-8")
            metadata["cmake_cuda_architecture"] = [line for line in cache.splitlines()
                if line.startswith(("CUDA_ARCH:", "CMAKE_CUDA_ARCHITECTURES:", "CMAKE_CUDA_COMPILER:"))]
        rows.append(run_case("frontend_options", [sys.executable, str(ROOT / "test/distributed/test_process_tp_options.py")],
                             output, 30, parse_json=False))
        rows.append(run_cleanup_checks(output))
        fixture_check = run_case("exit_fixture_protocol",
                                [sys.executable, str(ROOT / "test/distributed/test_control_exit_worker.py")],
                                output, 60, parse_json=False)
        fixture_check["backend"] = "python/control-exit-fixture"
        rows.append(fixture_check)
        if not args.frontend_only:
            base = [str(args.check.resolve()), "--worker", str(args.worker.resolve()), "--devices", args.devices]
            for name in ("smoke", "restart", "crash", "stall", "invalid_devices", "missing_worker", "startup_exit"):
                command = base + ["--control-only", "--case", name, "--operation-timeout-ms", "1000"]
                rows.append(run_case("control_" + name, command, output, 90))
            fixture = output / "startup_timeout_worker"
            source = (ROOT / "test/distributed/startup_timeout_worker.py").read_text(encoding="utf-8")
            fixture.write_text(source, encoding="utf-8")
            fixture.chmod(0o700)
            for name in ("startup_timeout", "protocol_version", "protocol_session"):
                command = [str(args.check.resolve()), "--worker", str(fixture.resolve()), "--devices", args.devices,
                           "--control-only", "--case", name, "--startup-timeout-ms", "2000"]
                environment = dict(os.environ, FASTLLM_TEST_STARTUP_FAULT="timeout" if name == "startup_timeout" else name)
                rows.append(run_case("control_" + name, command, output, 20, environment=environment))
            exit_fixture = output / "control_exit_worker"
            exit_fixture.write_text((ROOT / "test/distributed/control_exit_worker.py").read_text(encoding="utf-8"), encoding="utf-8")
            exit_fixture.chmod(0o700)
            for mode in ("wait_peer_exit", "partial_header_peer_exit", "partial_payload_peer_exit",
                         "buffered_error_exit", "current_error_exit", "partial_error_exit", "stale_error_exit",
                         "startup_peer_exit", "reply_then_exit", "stall", "shutdown_order"):
                name = "watch_" + mode
                barrier_dir = output / name
                barrier_dir.mkdir()
                command = [str(args.check.resolve()), "--worker", str(exit_fixture.resolve()),
                           "--devices", "0,1", "--control-only", "--case", name]
                environment = dict(os.environ, FASTLLM_TEST_EXIT_MODE=mode,
                                   FASTLLM_TEST_EXIT_DIR=str(barrier_dir.resolve()))
                rows.append(run_case("control_" + name, command, output, 45, environment=environment))
            rows.append(parent_death_case(base + ["--control-only", "--case", "hold"], 90, output))
            if args.pid_namespace:
                command = ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--kill-child=SIGKILL"]
                rows.append(run_case("control_pid1", command + base + ["--control-only", "--case", "pid1"], output, 90))
            if args.cuda:
                cuda_base = base + (["--custom-allreduce", "1"] if args.custom_allreduce else [])
                for name in ("smoke", "restart", "crash", "stall", "allreduce_boundaries", "collective_crash"):
                    rows.append(run_case("cuda_" + name, cuda_base + ["--case", name], output, 240))
                if args.custom_allreduce:
                    rows.append(run_case("custom_invalid_world", cuda_base + ["--case", "custom_invalid_world"], output, 30))
    except (Exception, KeyboardInterrupt) as error:
        rows.append({"case": "runner", "status": "FAIL", "reason": str(error) or type(error).__name__})
    status = save_report(output, rows, metadata)
    print(f"{status}: {output}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
