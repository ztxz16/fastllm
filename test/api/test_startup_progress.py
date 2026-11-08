import argparse
import ast
import contextlib
import io
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.startup_progress import (
    PROGRESS_PREFIX,
    StartupProgressReporter,
    StartupStatusHeartbeat,
    _startup_memory_snapshot,
)
from fastllm_pytools.util import add_server_args


def read_events(stream):
    events = []
    for line in stream.getvalue().splitlines():
        assert line.startswith(PROGRESS_PREFIX)
        events.append(json.loads(line[len(PROGRESS_PREFIX):]))
    return events


def test_ndjson_progress_prefers_byte_counts_and_finishes_ready():
    stream = io.StringIO()
    reporter = StartupProgressReporter("ndjson", stream = stream)

    reporter.progress(
        "weights_load",
        current = 5,
        total = 10,
        completed_bytes = 10,
        total_bytes = 100,
    )
    reporter.ready()

    events = read_events(stream)
    assert events[0]["type"] == "startup.progress"
    assert events[0]["stage"] == "weights_load"
    assert events[0]["stage_percent"] == 10.0
    assert events[0]["percent"] == 22.0
    assert events[0]["completed_bytes"] == 10
    assert events[-1]["type"] == "startup.ready"
    assert events[-1]["percent"] == 100.0


def test_progress_is_monotonic_and_regressions_are_suppressed():
    stream = io.StringIO()
    reporter = StartupProgressReporter("ndjson", stream = stream)

    reporter.progress("weights_load", 50, 100)
    reporter.progress("weights_load", 10, 100)
    reporter.progress("weights_load", 75, 100)

    events = read_events(stream)
    assert len(events) == 2
    assert [event["percent"] for event in events] == sorted(
        event["percent"] for event in events
    )
    assert events[-1]["stage_percent"] == 75.0


def test_warmup_emits_indeterminate_heartbeat():
    stream = io.StringIO()
    reporter = StartupProgressReporter(
        "ndjson",
        stream = stream,
        heartbeat_interval = 0.01,
    )

    reporter.model_load_progress("warmup", 0, 1)
    deadline = time.time() + 0.5
    while '"indeterminate":true' not in stream.getvalue() and time.time() < deadline:
        time.sleep(0.01)
    reporter.model_load_progress("warmup", 1, 1)
    reporter.close()

    events = read_events(stream)
    assert any(event.get("indeterminate") is True for event in events)
    assert events[-1]["stage"] == "warmup"
    assert events[-1]["stage_percent"] == 100.0


def test_off_mode_writes_nothing():
    stream = io.StringIO()
    reporter = StartupProgressReporter("off", stream = stream)
    reporter.progress("weights_load", 1, 2)
    reporter.ready()
    reporter.close()
    assert stream.getvalue() == ""


def test_server_parser_accepts_startup_progress_mode():
    parser = argparse.ArgumentParser()
    add_server_args(parser)
    args = parser.parse_args(["--startup-progress", "ndjson"])
    assert args.startup_progress == "ndjson"


def test_observability_startup_default_one_second_without_callbacks():
    stream = io.StringIO()
    reporter = StartupProgressReporter("off", stream=stream, observation_mode="user")
    try:
        reporter.model_load_progress("warmup:sampling_prefill_max_batch", 0, 0)
        assert stream.getvalue() == "", "native callback must only update a snapshot"
        deadline = time.monotonic() + 2.5
        while "event=startup_status" not in stream.getvalue() and time.monotonic() < deadline:
            time.sleep(0.01)
        line = stream.getvalue()
        assert "event=startup_status" in line
        assert "stage=warmup:sampling_prefill_max_batch" in line
        assert "detail=sampling_prefill_max_batch" in line
        assert "process_rss_bytes=" in line
        assert "system_mem_available_bytes=" in line
        assert "percent=" not in line
        elapsed = float(line.split("phase_elapsed_ms=", 1)[1].split()[0])
        assert elapsed >= 900
    finally:
        reporter.close()


def test_startup_user_and_debug_share_exact_raw_schema_and_values():
    streams = []
    for mode in ("user", "debug"):
        stream = io.StringIO()
        now = [0.0]
        reporter = StartupProgressReporter(
            "off", stream=stream, observation_mode=mode, clock=lambda: now[0],
            resource_reader=lambda: {"process_rss_bytes": "123456", "system_mem_available_bytes": "987654"})
        try:
            reporter.model_load_progress("weights_load", 3, 8, 1024, 4096)
            now[0] = 1.0
            reporter._status.emit_snapshot()
            line = stream.getvalue()
            assert "current=3 total=8 completed_bytes=1024 total_bytes=4096" in line
            assert "process_rss_bytes=123456 system_mem_available_bytes=987654" in line
            assert "phase_elapsed_ms=1000.000 idle_ms=1000.000" in line
            streams.append(re.sub(r"(?:seq|timestamp_monotonic_ns|timestamp_utc)=\S+", "", line))
        finally:
            reporter.close()
    assert streams[0] == streams[1]


def test_startup_raw_unknown_resources_are_never_estimated():
    stream = io.StringIO()
    reporter = StartupProgressReporter("off", stream=stream, observation_mode="debug", resource_reader=lambda: {})
    try:
        reporter._status.emit_snapshot()
        line = stream.getvalue()
        assert "process_rss_bytes=unknown" in line
        assert "system_mem_available_bytes=unknown" in line
        assert "percent=" not in line
    finally:
        reporter.close()


def test_startup_reads_actual_proc_resource_units_and_missing_values():
    def opened(path, *args, **kwargs):
        return io.StringIO("VmRSS:\t123 kB\n" if path == "/proc/self/status" else "MemAvailable: 456 kB\n")
    with patch("builtins.open", side_effect=opened):
        assert _startup_memory_snapshot() == {"process_rss_bytes": str(123 * 1024), "system_mem_available_bytes": str(456 * 1024)}
    with patch("builtins.open", side_effect=OSError("unavailable")):
        assert _startup_memory_snapshot() == {"process_rss_bytes": "unknown", "system_mem_available_bytes": "unknown"}


def test_detailed_warmup_does_not_get_overwritten_by_legacy_heartbeat():
    ndjson, raw = io.StringIO(), io.StringIO()
    reporter = StartupProgressReporter(
        "ndjson", stream=ndjson, status_stream=raw, observation_mode="user", heartbeat_interval=0.01)
    try:
        reporter.model_load_progress("warmup", 0, 1)
        reporter.model_load_progress("warmup:sampling_prefill_max_batch", 0, 0)
        deadline = time.monotonic() + 0.5
        while "startup_status" not in raw.getvalue() and time.monotonic() < deadline:
            time.sleep(0.005)
        reporter._status.emit_snapshot()
        assert "stage=warmup:sampling_prefill_max_batch" in raw.getvalue().splitlines()[-1]
        assert all(event["stage"] == "warmup" for event in read_events(ndjson))
        assert "percent=" not in raw.getvalue()
    finally:
        reporter.close()


def test_startup_ready_fail_and_close_end_future_raw_snapshots():
    for action in ("ready", "fail", "close"):
        stream = io.StringIO()
        reporter = StartupProgressReporter("off", stream=stream, observation_mode="user")
        reporter.model_load_progress("warmup:sampling_prefill_max_batch", 0, 0)
        if action == "fail":
            reporter.fail(RuntimeError("warmup failed"))
        else:
            getattr(reporter, action)()
        reporter._status.emit_snapshot()
        assert stream.getvalue() == ""
        assert reporter._status.stop_event.is_set()
        reporter.close()


def test_startup_blocked_output_does_not_block_callback_or_cleanup():
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()

    class BlockingStream:
        def write(self, value):
            entered.set()
            release.wait(2.0)

        def flush(self):
            pass

    reporter = StartupProgressReporter("off", status_stream=BlockingStream(), observation_mode="user")
    writer = threading.Thread(target=reporter._status.emit_snapshot)
    writer.start()
    assert entered.wait(1.0)

    def update_and_close():
        reporter.model_load_progress("warmup:sampling_prefill_max_batch", 0, 0)
        reporter.close()
        stopped.set()

    closer = threading.Thread(target=update_and_close)
    closer.start()
    try:
        assert stopped.wait(0.5), "startup callback/cleanup must not wait for heartbeat output"
    finally:
        release.set()
        writer.join(2.0)
        closer.join(2.0)
        reporter.close()
    assert not writer.is_alive()
    assert not closer.is_alive()


def test_startup_off_creates_no_status_thread_or_output():
    stream = io.StringIO()
    reporter = StartupProgressReporter("off", stream=stream, observation_mode="off")
    reporter.model_load_progress("warmup:sampling_prefill_max_batch", 0, 0)
    reporter._status.emit_snapshot()
    reporter.close()
    assert reporter._status.thread is None
    assert stream.getvalue() == ""


def test_inference_observability_does_not_enable_server_startup_heartbeat():
    # Execute the real entrypoint with a CPU fake body, without importing the
    # serving stack or loading a model/native library. This catches accidental
    # coupling of inference user/debug mode to deployment-stage observation.
    source = Path(__file__).resolve().parents[2] / "tools/fastllm_pytools/server.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    entrypoint = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "fastllm_server")
    for mode in ("off", "user", "debug"):
        stdout, stderr = io.StringIO(), io.StringIO()
        seen = []

        def load(args, progress):
            seen.append(progress)
            assert progress._status.thread is None
            print("[Model] Loading weights")
            progress.model_load_progress("warmup:sampling_prefill_max_batch", 0, 0)
            progress._status.emit_snapshot()
            assert progress._status.thread is None
            print("[Model] Warmup complete")
            return "model-ready"

        namespace = {"StartupProgressReporter": StartupProgressReporter,
                     "_fastllm_server": load, "os": os}
        exec(compile(ast.Module(body=[entrypoint], type_ignores=[]), str(source), "exec"), namespace)
        with patch.dict(os.environ, {"FASTLLM_OBSERVABILITY_MODE": mode}), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            assert namespace["fastllm_server"](SimpleNamespace(startup_progress="off")) == "model-ready"
        assert stdout.getvalue() == "[Model] Loading weights\n[Model] Warmup complete\n"
        assert stderr.getvalue() == ""
        assert len(seen) == 1
        assert seen[0]._status.thread is None
