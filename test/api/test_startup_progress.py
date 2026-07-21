import argparse
import io
import json
import os
import sys
import time


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.startup_progress import (
    PROGRESS_PREFIX,
    StartupProgressReporter,
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
