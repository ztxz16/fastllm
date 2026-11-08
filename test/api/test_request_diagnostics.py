"""Independent stall evidence CPU acceptance. No model/native/GPU is loaded."""
import ast
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/fastllm_pytools"))
import request_diagnostics as diagnostics
import request_progress as progress
from terminal_colorizer import process_lines

LOG_ROOT = Path(os.environ.get("FASTLLM_DIAGNOSTIC_LOG_ROOT", "~/.cache/fastllm/diagnostics")).expanduser()
RUN = Path(os.environ.get("FASTLLM_DIAGNOSTIC_ACCEPTANCE_RUN",
    str(LOG_ROOT / "debug-evidence-stall-20261021-r1/python")))
READY = "INFO:     Uvicorn running on http://0.0.0.0:8080\n"


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("bounded wait did not observe expected evidence")


def tiny_capture(stop):
    return {"python_threads": [], "proc_tasks": [], "fixture": "explicit CPU collector substitute"}


class RequestDiagnosticTest(unittest.TestCase):
    def setUp(self):
        RUN.resolve().relative_to(LOG_ROOT.resolve())
        self.sink = RUN / (self._testMethodName + "-" + str(time.time_ns()))
        self.sink.mkdir(parents=True)
        self.env = patch.dict(os.environ, {
            "FASTLLM_DIAGNOSTIC_LOG_DIR": str(self.sink),
            "FASTLLM_DIAGNOSTIC_STALL_SECONDS": "0.12"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.watchers, self.events = [], []

    def watcher(self, request_id="req-watch", **kwargs):
        kwargs.setdefault("threshold_s", 0.12)
        kwargs.setdefault("max_captures", 1)
        kwargs.setdefault("budget", diagnostics.CaptureBudget())
        kwargs.setdefault("emit", lambda event, fields: self.events.append((event, fields)))
        watcher = diagnostics.RequestDiagnosticWatchdog(
            request_id, 7, mode="debug", **kwargs)
        self.watchers.append(watcher)
        return watcher

    def tearDown(self):
        for watcher in self.watchers:
            watcher.stop()
        for watcher in self.watchers:
            if watcher._thread is not None:
                watcher._thread.join(1)
                self.assertFalse(watcher._thread.is_alive())

    def files(self):
        # Readers consume only the direct fsynced pointer, never an in-flight
        # filename that has been created but not fully written yet.
        return [Path(row["snapshot_path"]) for row in self.index()
                if row["event"] == "diagnostic_snapshot"]

    def index(self):
        path = self.sink / "events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
                if line.endswith("\n")] if path.exists() else []

    def test_constructor_and_off_mode_never_capture_startup(self):
        debug = self.watcher()
        off = diagnostics.RequestDiagnosticWatchdog("off", 7, mode="off",
            native_provider=lambda: self.fail("off queried native diagnostics"),
            emit=lambda *args: self.fail("off emitted"))
        off.start()
        self.assertIsNone(debug._thread)
        self.assertIsNone(off._thread)
        self.assertEqual(self.files(), [])
        self.assertFalse((self.sink / "events.jsonl").exists())

    def test_blocked_status_and_finish_wait_still_capture_real_thread_evidence(self):
        gate, entered = threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        def blocked_status_provider():
            entered.set()
            gate.wait(3)
            return {"phase": "prefill", "phase_revision": "1",
                    "prefill_completed_tokens": "0", "native_generation": "91"}
        start = time.monotonic_ns()
        operation = {"operation": "forward_lock_wait", "resource": "test_mutex",
            "state": "running", "start_monotonic_ns": start,
            "thread_id": threading.get_native_id(),
            "stack_scope": "first_operation_entry_same_label_not_current",
            "entry_stack_capture_monotonic_ns": start,
            "entry_stack_frames": [{"address": "0x1234", "module": "CPU_fixture"}]}
        provider = lambda: {"native_diag_state": "ok", "native_diag_generation": "7",
            "native_diag_operations_json": json.dumps([operation], separators=(",", ":"))}
        events = []
        reporter = progress.RequestProgressReporter("blocked-request", native_handle=7,
            mode="debug", emit=events.append, interval_s=0.025,
            status_provider=blocked_status_provider, diagnostics_provider=provider)
        with patch.object(progress, "resource_snapshot", return_value={}):
            reporter.received(path="/v1/chat/completions")
            self.watchers.append(reporter._diagnostics)
            wait_for(entered.is_set)
            finishing = threading.Thread(target=lambda: reporter.finish(status="cancelled"))
            finishing.start()
            self.addCleanup(lambda: (gate.set(), finishing.join(1)))
            wait_for(lambda: self.files())
            evidence = json.loads(self.files()[0].read_text())
            self.assertTrue(finishing.is_alive(), "finish actually waits behind blocked reporter query")
            self.assertTrue(reporter._diagnostics._thread.is_alive())
            self.assertIn("finish_waiting_for_completion", evidence["python_outstanding_calls"])
            functions = [frame["function"] for row in evidence["thread_evidence"]["python_threads"]
                         for frame in row["frames"]]
            self.assertIn("blocked_status_provider", functions)
            self.assertIn("finish", functions)
            self.assertEqual(evidence["native_operations"][0]["operation"], "forward_lock_wait")
            self.assertEqual(evidence["native_operations"][0]["stack_scope"],
                             "first_operation_entry_same_label_not_current")
            self.assertFalse(evidence["deadlock_proven"])
            self.assertTrue(evidence["thread_evidence"]["proc_tasks"])
            self.assertTrue(all("state" in row["wchan"] and "state" in row["kernel_stack"]
                                for row in evidence["thread_evidence"]["proc_tasks"]))
            gate.set()
            finishing.join(1)
            reporter._pump.join(1)
        self.assertFalse(finishing.is_alive())
        self.assertTrue(reporter._finished)
        self.assertTrue(reporter._diagnostics._stop.is_set())
        terminal = next(i for i, event in enumerate(events) if "event=request_finished " in event)
        self.assertFalse(any("category=progress " in event for event in events[terminal + 1:]))
        count = len(self.files())
        time.sleep(0.15)
        self.assertEqual(len(self.files()), count)

    def test_close_requested_is_retained_while_reporter_lock_is_busy(self):
        gate, entered = threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        def blocked():
            entered.set()
            gate.wait(3)
            return {"phase_revision": "1", "phase": "prefill"}
        reporter = progress.RequestProgressReporter("close-wait", mode="debug",
            emit=lambda _: None, interval_s=0.02, status_provider=blocked)
        with patch.object(progress, "resource_snapshot", return_value={}):
            reporter.received(path="/")
            self.watchers.append(reporter._diagnostics)
            wait_for(entered.is_set)
            closing = threading.Thread(target=lambda: reporter.close_native_observation("test-release"))
            closing.start()
            self.addCleanup(lambda: (gate.set(), closing.join(1)))
            wait_for(lambda: self.files())
            evidence = json.loads(self.files()[0].read_text())
            self.assertIn("native_close_waiting_for_completion", evidence["python_outstanding_calls"])
            self.assertTrue(closing.is_alive())
            gate.set()
            closing.join(1)
            reporter.finish(status="cancelled")
            reporter._pump.join(1)
        self.assertFalse(closing.is_alive())
        self.assertTrue(reporter._diagnostics._stop.is_set())

    def test_real_increment_resets_stall_and_combined_sources_do_not_reset_it_again(self):
        value = {"completed": 0}
        def native():
            return {"native_diag_state": "ok", "native_diag_generation": "3",
                    "native_diag_progress_revision": str(value["completed"]),
                    "native_diag_progress_end_ns": str(value["completed"])}
        watcher = self.watcher(native_provider=native, capture=tiny_capture, threshold_s=0.20)
        watcher.observe_status({"native_generation": "101", "phase_revision": "1",
                                "prefill_completed_tokens": "0"})
        watcher.start()
        time.sleep(0.12)
        value["completed"] = 1
        watcher.observe_status({"native_generation": "101", "phase_revision": "1",
                                "prefill_completed_tokens": "2048"})
        time.sleep(0.13)
        self.assertEqual(self.files(), [])
        wait_for(lambda: self.files())
        document = json.loads(self.files()[0].read_text())
        self.assertEqual(document["progress_owner_generations"], {"status": "101", "diagnostic": "3"})
        self.assertEqual(document["last_confirmed_progress_fields"]["prefill_completed_tokens"], "2048")
        self.assertGreaterEqual(document["quiet_elapsed_ns"], 0.20 * 1e9)

    def test_sample_times_and_lock_poll_counts_never_disguise_no_progress(self):
        count = [0]
        def native():
            count[0] += 1
            return {"native_diag_state": "ok", "native_diag_sample_ns": str(time.monotonic_ns()),
                    "native_diag_revision": str(count[0]), "native_diag_completed_count": str(count[0]),
                    "native_diag_operations_json": "[]"}
        watcher = self.watcher(native_provider=native, capture=tiny_capture)
        watcher.start()
        wait_for(lambda: self.files())
        document = json.loads(self.files()[0].read_text())
        self.assertGreater(count[0], 1)
        self.assertEqual(document["reason"], "no_confirmed_progress")

    def test_progressing_long_compute_scope_does_not_consume_stall_budget(self):
        count = [0]
        start = time.monotonic_ns() - 10_000_000_000
        def native():
            count[0] += 1
            return {"native_diag_state": "ok", "native_diag_generation": "3",
                "native_diag_progress_revision": str(count[0]),
                "native_diag_progress_end_ns": str(time.monotonic_ns()),
                "native_diag_operations_json": json.dumps([{"operation": "native_forward",
                    "state": "running", "start_monotonic_ns": start}], separators=(",", ":"))}
        watcher = self.watcher(native_provider=native, capture=tiny_capture)
        watcher.start()
        time.sleep(0.36)
        watcher.stop()
        self.assertGreater(count[0], 2)
        self.assertEqual(self.files(), [])

    def test_episode_cap_recovers_after_actual_progress(self):
        watcher = self.watcher(capture=tiny_capture, max_captures=1, threshold_s=0.07)
        watcher.observe_status({"native_generation": "1", "prefill_completed_tokens": "1"})
        watcher.start()
        wait_for(lambda: len(self.files()) == 1)
        wait_for(lambda: any(row.get("scope") == "no_progress_episode" for row in self.index()))
        time.sleep(0.1)
        self.assertEqual(len(self.files()), 1)
        watcher.observe_status({"native_generation": "1", "prefill_completed_tokens": "2"})
        wait_for(lambda: len(self.files()) == 2)
        self.assertTrue(watcher._thread.is_alive())

    def test_rolling_process_budget_recovers_without_deleting_evidence(self):
        clock = [0]
        budget = diagnostics.CaptureBudget(captures=1, bytes_limit=100, window_s=10,
                                          clock_ns=lambda: clock[0])
        first = budget.reserve(80)
        self.assertIsNotNone(first)
        budget.release_unused_bytes(first, 60)
        self.assertIsNone(budget.reserve(20))
        self.assertEqual(budget.state()["next_recovery_monotonic_ns"], 10_000_000_000)
        clock[0] = 10_000_000_001
        self.assertIsNotNone(budget.reserve(80))
        self.assertEqual(budget.state()["captures"], 1)
        self.assertEqual(budget.state()["bytes"], 80)

    def test_sink_failure_is_explicit_and_does_not_fallback_to_tmp_or_build(self):
        with patch.dict(os.environ, {"FASTLLM_DIAGNOSTIC_LOG_DIR": str(REPO.parent / "build/forbidden-diagnostics")}):
            watcher = self.watcher(capture=tiny_capture)
            watcher.start()
            wait_for(lambda: any(event == "diagnostic_sink_unavailable" for event, _ in self.events))
        self.assertEqual(self.files(), [])
        self.assertTrue(watcher._stop.is_set())

    def test_capture_failure_is_explicit_and_stops_predictably_failed_path(self):
        def fail(_):
            raise RuntimeError("CPU fixture capture failed")
        watcher = self.watcher(capture=fail)
        watcher.start()
        wait_for(lambda: any(event == "diagnostic_capture_failed" for event, _ in self.events))
        self.assertTrue(watcher._stop.is_set())
        self.assertEqual(self.files(), [])
        row = next(row for row in self.index() if row["event"] == "diagnostic_capture_failed")
        self.assertEqual(json.loads(row["error_json"])["message"], "CPU fixture capture failed")

    def test_native_query_failure_preserves_python_capture_and_error(self):
        def fail():
            raise RuntimeError("independent ABI unavailable")
        watcher = self.watcher(native_provider=fail, capture=tiny_capture)
        watcher.start()
        wait_for(lambda: self.files())
        document = json.loads(self.files()[0].read_text())
        self.assertEqual(document["native_fields"]["native_diag_state"], "query_failed")
        self.assertTrue(any(row["event"] == "diagnostic_native_query_failed" for row in self.index()))

    def test_logger_block_does_not_block_independent_fsync_captures(self):
        gate, entered = threading.Event(), threading.Event()
        def blocked_logger(event, fields):
            entered.set()
            gate.wait(3)
        self.addCleanup(gate.set)
        dispatcher = diagnostics._NoticeDispatcher()
        with patch.object(diagnostics, "_NOTICES", dispatcher):
            watcher = self.watcher(emit=blocked_logger, capture=tiny_capture,
                                   threshold_s=0.06, max_captures=2)
            watcher.start()
            wait_for(entered.is_set)
            wait_for(lambda: len(self.files()) == 2)
            snapshots = [json.loads(p.read_text()) for p in self.files()]
            self.assertTrue(any(doc["main_raw_notice_delivery"]["inflight_since_ns"] is not None
                                for doc in snapshots))
            pointers = [row for row in self.index() if row["event"] == "diagnostic_snapshot"]
            self.assertEqual(len(pointers), 2)
            for row in pointers:
                self.assertEqual(hashlib.sha256(Path(row["snapshot_path"]).read_bytes()).hexdigest(),
                                 row["snapshot_sha256"])
            watcher.stop()
            gate.set()

    def test_concurrent_requests_are_distinct_and_cancel_stops_only_its_owner(self):
        first = self.watcher("first", capture=tiny_capture, threshold_s=0.08)
        second = self.watcher("second", capture=tiny_capture, threshold_s=0.08)
        first.start()
        second.start()
        first.stop()
        wait_for(lambda: self.files())
        documents = [json.loads(path.read_text()) for path in self.files()]
        self.assertEqual({doc["request_id"] for doc in documents}, {"second"})
        self.assertTrue(second._thread.is_alive())

    def test_foreign_generation_cannot_reset_original_request_progress(self):
        watcher = self.watcher(capture=tiny_capture)
        watcher.observe_status({"native_generation": "old", "prefill_completed_tokens": "4"})
        stamp = watcher._progress_clock_ns
        watcher.observe_status({"native_generation": "new", "prefill_completed_tokens": "900"})
        self.assertEqual(watcher._progress_clock_ns, stamp)
        self.assertEqual(watcher._progress_fields["prefill_completed_tokens"], "4")
        # Independent native diagnostics generations are a different namespace.
        watcher.observe_status({"native_diag_generation": "independent-9",
                                "native_diag_progress_revision": "2"}, namespace="diagnostic")
        self.assertEqual(watcher._progress_owners,
                         {"status": "old", "diagnostic": "independent-9"})

    def test_debug_notice_is_absent_from_user_projection_and_raw_remains_exact(self):
        events = []
        reporter = progress.RequestProgressReporter("projection", mode="debug", emit=events.append)
        reporter._diagnostic_notice("diagnostic_snapshot", {"snapshot_path": str(self.sink / "capture.json"),
            "snapshot_sha256": "a" * 64, "classification": "observation_not_deadlock_proof"})
        row = events[0] + "\n"
        raw, server, terminal = io.StringIO(), io.StringIO(), io.StringIO()
        process_lines([READY, row], server, terminal, "user", raw)
        self.assertEqual(raw.getvalue(), READY + row)
        self.assertEqual(server.getvalue(), raw.getvalue())
        self.assertNotIn("capture.json", terminal.getvalue())

    def test_directory_byte_limit_is_explicit_and_preserves_existing_evidence(self):
        existing = self.sink / "existing-original.json"
        existing.write_text('{"original":true}\n')
        directory = diagnostics._directory_budget(self.sink.resolve())
        directory.bytes_limit = directory.bytes
        watcher = self.watcher(capture=tiny_capture)
        watcher.start()
        wait_for(lambda: any(event == "diagnostic_capture_limit" and
            fields.get("scope") == "diagnostic_directory_bytes" for event, fields in self.events))
        self.assertEqual(list(self.sink.glob("request-*.json")), [])
        self.assertEqual(existing.read_text(), '{"original":true}\n')
        self.assertTrue(watcher._stop.is_set())

    def test_native_waiter_tid_is_prioritized_beyond_first_128_proc_tasks(self):
        tasks = [Path("/proc/self/task") / str(1000 + index) for index in range(150)]
        tasks.append(Path("/proc/self/task/9999"))
        def read(path, limit):
            return {"state": "ok", "path": str(path), "truncated": False, "text": "CPU fixture"}
        with patch.object(Path, "iterdir", return_value=iter(tasks)), \
             patch.object(diagnostics.sys, "_current_frames", return_value={}), \
             patch.object(diagnostics.threading, "enumerate", return_value=[]), \
             patch.object(diagnostics, "_read_proc", side_effect=read):
            evidence = diagnostics.capture_thread_evidence(threading.Event(), priority_tids=[9999])
        self.assertEqual(len(evidence["proc_tasks"]), 128)
        self.assertEqual(evidence["proc_tasks"][0]["native_thread_id"], 9999)
        self.assertEqual(evidence["priority_native_tids_selected"], [9999])
        self.assertEqual(evidence["priority_native_tids_omitted_by_limit"], [])
        self.assertEqual(evidence["process_maps"]["path"], "/proc/self/maps")
        self.assertEqual(evidence["process_maps_limit_bytes"], 256 * 1024)

    def test_ctypes_diagnostics_getter_is_separate_bounded_and_has_no_status_fallback(self):
        module = ast.parse((REPO / "tools/fastllm_pytools/llm.py").read_text())
        method = next(node for owner in module.body if isinstance(owner, ast.ClassDef) and owner.name == "model"
            for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "get_request_diagnostics")
        calls = []
        def abi(model, handle, public_id, buffer, capacity):
            calls.append((model, handle, public_id, capacity))
            buffer.value = b'native_diag_state=busy native_diag_generation=7 native_diag_operations_json=[]'
            return True
        scope = {"ctypes": ctypes, "_get_request_diagnostics": abi}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     "<actual-ctypes-getter>", "exec"), scope)
        value = scope["get_request_diagnostics"](SimpleNamespace(model=12), 7, "public-7")
        self.assertEqual(value["native_diag_state"], "busy")
        self.assertEqual(calls, [(12, 7, b"public-7", 65536)])
        scope["_get_request_diagnostics"] = None
        self.assertIsNone(scope["get_request_diagnostics"](SimpleNamespace(model=12), 7, "public-7"))


if __name__ == "__main__":
    unittest.main()
