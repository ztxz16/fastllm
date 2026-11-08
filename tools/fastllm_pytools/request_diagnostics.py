"""Bounded request stall evidence, independent of reporter/model/cache locks.

Python stacks are current capture-time frame metadata without locals. Native
stacks, when supplied by the ABI, are explicitly operation-entry observations.
No snapshot establishes a deadlock or pauses the service.
"""
from __future__ import annotations

from datetime import datetime, timezone
from collections import deque
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from typing import Callable, Mapping

# Diagnostic sinks must resolve under this root (override with FASTLLM_DIAGNOSTIC_LOG_ROOT).
_LOG_ROOT = Path(os.environ.get("FASTLLM_DIAGNOSTIC_LOG_ROOT", "~/.cache/fastllm/diagnostics")).expanduser()
_MAX_SNAPSHOT_BYTES = 1024 * 1024
_PROGRESS_KEYS = ("phase_revision", "prefill_completed_tokens",
                  "native_output_tokens", "native_interval_end_ns",
                  "native_diag_progress_revision", "native_diag_progress_end_ns")
_EVENT_SEQUENCE = itertools.count(1)
_WAIT_OPERATIONS = frozenset({
    "scheduler_dict_lock_wait", "forward_lock_wait", "scheduler_publish_dict_lock_wait",
    "scheduler_batch_collection_wait", "scheduler_condition_wait", "scheduler_kv_release_wait",
    "api_output_condition_wait", "cuda_event_completion_wait", "ssd_restore_forward_lock_try",
    "ssd_import_page_capacity_wait", "ssd_h2d_completion_wait", "ssd_capture_mutex_wait",
    "ssd_coordinator_admission_wait", "ssd_export_pin_budget_wait", "ssd_read_consumption_wait",
    "l1_publication_promote_lock_wait", "l2_cuda_stream_completion_wait"})


def _clock() -> dict:
    unix_ns = time.time_ns()
    return {"monotonic_ns": time.monotonic_ns(), "unix_ns": unix_ns,
            "utc": datetime.fromtimestamp(unix_ns / 1e9, timezone.utc).isoformat(),
            "wall_clock_source": "OS_SYSTEM_CLOCK", "utc_sync": "not_verified"}


def _error(error: BaseException) -> dict:
    return {"type": type(error).__name__, "message": str(error)[:4096],
            "errno": getattr(error, "errno", None)}


class CaptureBudget:
    """Rolling process budget; reservations contain no capture/I/O work."""
    def __init__(self, captures=16, bytes_limit=16 * 1024 * 1024,
                 window_s=600, clock_ns=time.monotonic_ns):
        self.max_captures, self.bytes_limit = captures, bytes_limit
        self.window_ns, self.clock_ns = int(window_s * 1e9), clock_ns
        self.captures = self.bytes = 0
        self._lock = threading.Lock()
        self._order, self._entries = deque(), {}
        self._sequence = itertools.count(1)

    def _expire(self, now):
        while self._order and now - self._entries[self._order[0]][0] >= self.window_ns:
            token = self._order.popleft()
            self.bytes -= self._entries.pop(token)[1]
            self.captures -= 1

    def reserve(self, size: int):
        with self._lock:
            now = self.clock_ns()
            self._expire(now)
            if self.captures >= self.max_captures or self.bytes + size > self.bytes_limit:
                return None
            token = next(self._sequence)
            self._order.append(token)
            self._entries[token] = [now, size]
            self.captures += 1
            self.bytes += size
            return token

    def release_unused_bytes(self, token: int, actual: int):
        with self._lock:
            entry = self._entries.get(token)
            if entry is not None:
                unused = max(0, entry[1] - actual)
                entry[1] -= unused
                self.bytes -= unused

    def state(self):
        with self._lock:
            now = self.clock_ns()
            self._expire(now)
            return {"captures": self.captures, "bytes": self.bytes,
                "capture_limit": self.max_captures, "byte_limit": self.bytes_limit,
                "window_ns": self.window_ns,
                "next_recovery_monotonic_ns": self._entries[self._order[0]][0] +
                    self.window_ns if self._order else now}


class _DirectoryBudget:
    """Hard bound for this process plus files already present at first open."""
    def __init__(self, directory):
        files = list(directory.iterdir())
        if len(files) > 2048:
            raise ValueError("diagnostic directory has more than 2048 existing entries")
        self.bytes_limit = 128 * 1024 * 1024
        self.bytes = sum(p.stat().st_size for p in files if p.is_file())
        self._lock = threading.Lock()

    def reserve(self, amount):
        with self._lock:
            if self.bytes + amount > self.bytes_limit:
                return False
            self.bytes += amount
            return True

    def release_unused_bytes(self, reserved, actual):
        with self._lock:
            self.bytes -= max(0, reserved - actual)


_DIRECTORY_BUDGETS = {}
_DIRECTORY_BUDGETS_LOCK = threading.Lock()


def _directory_budget(path):
    with _DIRECTORY_BUDGETS_LOCK:
        if path not in _DIRECTORY_BUDGETS:
            _DIRECTORY_BUDGETS[path] = _DirectoryBudget(path)
        return _DIRECTORY_BUDGETS[path]


_PROCESS_BUDGET = CaptureBudget()


class _NoticeDispatcher:
    """One bounded process-wide delivery worker; never block a watchdog on logging."""
    def __init__(self):
        self._queue = queue.Queue(maxsize=64)
        self._start_lock = threading.Lock()
        self._thread = None
        self._inflight = None
        self._last_failure = None

    def submit(self, callback, event, fields) -> str:
        if callback is None:
            return "no_main_raw_callback"
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run,
                    name="fastllm-diagnostic-notices", daemon=True)
                self._thread.start()
        try:
            self._queue.put_nowait((callback, event, dict(fields)))
            return "queued"
        except queue.Full:
            return "queue_full_sidecar_retained"

    def state(self) -> dict:
        inflight = self._inflight
        return {"scope": "process_main_raw_delivery_callback",
                "queued": self._queue.qsize(),
                "inflight_since_ns": inflight,
                "inflight_age_ms": None if inflight is None else
                    (time.monotonic_ns() - inflight) / 1e6,
                "last_delivery_failure": self._last_failure}

    def _run(self):
        while True:
            callback, event, fields = self._queue.get()
            self._inflight = time.monotonic_ns()
            try:
                callback(event, fields)
            except Exception as error:
                # Capture files/index are already durable before delivery.
                # The failure is retained and included in the next sidecar.
                self._last_failure = {"clock": _clock(), "error": _error(error)}
            finally:
                self._inflight = None
                self._queue.task_done()


_NOTICES = _NoticeDispatcher()


def _read_proc(path: Path, limit: int) -> dict:
    try:
        with path.open("rb", buffering=0) as source:
            data = source.read(limit + 1)
        return {"state": "ok", "path": str(path), "truncated": len(data) > limit,
                "text": data[:limit].decode("utf-8", errors="replace")}
    except (OSError, ValueError) as error:
        return {"state": "unavailable", "path": str(path), "error": _error(error)}


def capture_thread_evidence(stop: threading.Event, priority_tids=()) -> dict:
    """No locals, source-line evaluation, signals, ptrace, GPU or app locks."""
    started = _clock()
    frames = sys._current_frames()
    threads = {t.ident: t for t in threading.enumerate()}
    python_rows = []
    requested_priority = []
    for value in priority_tids:
        try:
            tid = int(value)
            if tid > 0 and tid not in requested_priority:
                requested_priority.append(tid)
        except (TypeError, ValueError):
            continue
    frame_items = list(frames.items())
    frame_items.sort(key=lambda item: getattr(threads.get(item[0]), "native_id", None)
                     not in requested_priority)
    for ident, frame in frame_items[:128]:
        if stop.is_set():
            break
        stack, seen = [], 0
        while frame is not None and seen < 48:
            stack.append({"file": frame.f_code.co_filename,
                          "line": frame.f_lineno, "function": frame.f_code.co_name})
            frame, seen = frame.f_back, seen + 1
        thread = threads.get(ident)
        python_rows.append({"python_thread_ident": ident,
            "native_thread_id": getattr(thread, "native_id", None),
            "name": getattr(thread, "name", None), "frames": stack,
            "truncated": frame is not None, "stack_scope": "python_current_frames_no_locals"})
    del frames
    proc_rows, task_error = [], None
    try:
        task_paths = sorted(Path("/proc/self/task").iterdir(), key=lambda p: int(p.name))
    except OSError as error:
        task_paths, task_error = [], _error(error)
    by_tid = {int(path.name): path for path in task_paths}
    python_tids = [thread.native_id for thread in threads.values()
                   if thread.native_id is not None]
    ordered_tids = list(dict.fromkeys(requested_priority + python_tids + list(by_tid)))
    selected = [by_tid[tid] for tid in ordered_tids if tid in by_tid][:128]
    # CPU delta per thread over a short window: this is what tells a spinning
    # native worker (CPU 100%, GPU idle) apart from a thread blocked on a lock.
    cpu_hz = 100
    try:
        cpu_hz = int(os.sysconf("SC_CLK_TCK"))
    except (ValueError, OSError, AttributeError):
        cpu_hz = 100
    cpu_window_s = 0.1

    def _cpu_ticks(task_path):
        try:
            text = (task_path / "stat").read_text(errors="replace")
            tail = text[text.rfind(")") + 2:].split()
            return int(tail[11]) + int(tail[12])
        except (OSError, ValueError, IndexError):
            return None

    cpu_before = {int(task.name): _cpu_ticks(task) for task in selected}
    time.sleep(cpu_window_s)
    cpu_after = {int(task.name): _cpu_ticks(task) for task in selected}
    cpu_pct = {}
    for tid, before in cpu_before.items():
        after = cpu_after.get(tid)
        if before is None or after is None:
            continue
        cpu_pct[tid] = round((after - before) * 100.0 / (cpu_hz * cpu_window_s), 1)
    top_cpu = sorted(({"native_thread_id": tid, "cpu_pct_delta": pct}
                      for tid, pct in cpu_pct.items() if pct >= 1.0),
                     key=lambda row: row["cpu_pct_delta"], reverse=True)[:5]
    for task in selected:
        if stop.is_set():
            break
        proc_rows.append({"native_thread_id": int(task.name),
            "sample_clock": _clock(), "wchan": _read_proc(task / "wchan", 512),
            "syscall": _read_proc(task / "syscall", 2048),
            "kernel_stack": _read_proc(task / "stack", 4096),
            "status": _read_proc(task / "status", 4096),
            "cpu_pct_delta": cpu_pct.get(int(task.name)),
            "stack_scope": "proc_kernel_stack_not_current_userspace_native_stack"})
    return {"capture_begin": started, "capture_end": _clock(),
        "python_threads": python_rows, "python_threads_observed": len(threads),
        "python_thread_limit": 128, "proc_tasks": proc_rows,
        "proc_tasks_observed": len(task_paths), "proc_task_limit": 128,
        "priority_native_tids": requested_priority,
        "priority_native_tids_selected": [int(task.name) for task in selected
                                         if int(task.name) in requested_priority],
        "priority_native_tids_missing": [tid for tid in requested_priority if tid not in by_tid],
        "priority_native_tids_omitted_by_limit": [tid for tid in requested_priority
            if tid in by_tid and tid not in {int(task.name) for task in selected}],
        "proc_task_selection": "running_native_operation_tids_then_python_threads_then_other_tids",
        "proc_tasks_error": task_error,
        "top_cpu_threads": top_cpu,
        "cpu_sample_window_s": cpu_window_s,
        "process_maps": _read_proc(Path("/proc/self/maps"), 256 * 1024),
        "process_maps_limit_bytes": 256 * 1024,
        "resources": {"process_status": _read_proc(Path("/proc/self/status"), 8192),
                      "host_meminfo": _read_proc(Path("/proc/meminfo"), 16384),
                      "host_loadavg": _read_proc(Path("/proc/loadavg"), 1024)},
        "capture_interrupted_by_request_stop": stop.is_set()}


class RequestDiagnosticWatchdog:
    """Request-owned watcher. Constructing it or running in off/user starts no thread."""
    def __init__(self, request_id: str, native_handle=None, *, native_provider=None,
                 emit=None, mode="off", threshold_s=None, poll_s=None,
                 max_captures=3, max_bytes=16 * _MAX_SNAPSHOT_BYTES,
                 max_total_captures=16, budget=None, capture=None):
        self.request_id, self.native_handle = request_id, native_handle
        self.native_provider, self.emit, self.mode = native_provider, emit, mode
        self._threshold_input = threshold_s
        self.threshold_s = 15.0
        self.poll_s = poll_s
        self.max_captures, self.max_bytes = max_captures, max_bytes
        self.max_total_captures = max_total_captures
        self.budget = _PROCESS_BUDGET if budget is None else budget
        self._default_capture = capture is None
        self.capture = capture_thread_evidence if capture is None else capture
        self._stop = threading.Event()
        self._thread = None
        self._sink = None
        self._fingerprint = None
        self._progress_clock_ns = time.monotonic_ns()
        self._progress_fields = {}
        self._progress_owners = {}
        self._progress_epoch = self._episode_epoch = 0
        self._episode_attempts = 0
        self._episode_limit_reported = False
        self._budget_retry_ns = 0
        self._directory = None
        self._lifecycle = ()
        self._python_waits = {}
        self._attempts = self._bytes = 0
        self._last_capture_ns = 0
        self._native_query_failure_reported = False
        self._events_written = 0

    def start(self):
        if self.mode != "debug" or self._thread is not None or self._stop.is_set():
            return
        self._progress_clock_ns = time.monotonic_ns()
        self._thread = threading.Thread(target=self._run,
            name="fastllm-request-diagnostics", daemon=True)
        self._thread.start()

    def stop(self):
        # Never wait for filesystem capture or reporter/native locks here.
        self._stop.set()

    def note_lifecycle(self, event: str, **fields):
        clock = _clock()
        self._lifecycle = (self._lifecycle + ({"event": event, "clock": clock,
            "fields": dict(fields)},))[-16:]
        starts = {"native_status_query_begin": "native_status_query",
                  "finish_requested": "finish_waiting_for_completion",
                  "native_close_requested": "native_close_waiting_for_completion",
                  "abort_requested": "abort_call"}
        ends = {"native_status_query_end": "native_status_query",
                "finish_completed": "finish_waiting_for_completion",
                "native_close_completed": "native_close_waiting_for_completion",
                "conversation_handle_released": "abort_call"}
        if event in starts:
            self._python_waits = {**self._python_waits, starts[event]: clock}
        elif event in ends:
            self._python_waits = {key: value for key, value in self._python_waits.items()
                                  if key != ends[event]}

    def observe_status(self, fields: Mapping, *, namespace="status"):
        # Sampling timestamps, op IDs/counts and diagnostic registry revisions
        # are not forward progress. Immutable assignments require no reporter lock.
        observed = {k: fields[k] for k in _PROGRESS_KEYS if k in fields}
        if not observed:
            return
        owner = fields.get("native_generation") if namespace == "status" else fields.get("native_diag_generation")
        prior_owner = self._progress_owners.get(namespace)
        if prior_owner is not None and owner is not None and str(owner) != str(prior_owner):
            return  # Each ABI has its own generation namespace.
        merged = {**self._progress_fields, **observed}
        fingerprint = tuple((k, str(merged[k])) for k in _PROGRESS_KEYS if k in merged)
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self._progress_clock_ns = time.monotonic_ns()
            self._progress_fields = merged
            self._progress_epoch += 1
            if owner is not None:
                self._progress_owners = {**self._progress_owners, namespace: owner}

    def _event_record(self, event: str, fields: Mapping) -> dict:
        return {"seq": f"diagnostic-{os.getpid()}-{next(_EVENT_SEQUENCE)}",
            "sequence_scope": "independent_diagnostic_sidecar_process",
            "clock": _clock(), "source": "python_watchdog", "category": "diagnostic",
            "request_id": self.request_id, "native_handle": self.native_handle,
            "event": event, **dict(fields)}

    def _publish(self, event: str, **fields):
        record = self._event_record(event, fields)
        # Direct durable event is authoritative even if main logging is blocked.
        if self._sink is not None and self._events_written < 64:
            encoded = (json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
            if self._directory.reserve(len(encoded)):
                with (self._sink / "events.jsonl").open("ab") as out:
                    out.write(encoded)
                    out.flush()
                    os.fsync(out.fileno())
                self._events_written += 1
            else:
                fields = {**fields, "sidecar_index_state": "directory_byte_limit_reached"}
        if self._events_written >= 64:
            fields = {**fields, "sidecar_index_state": "request_event_count_limit_reached",
                      "sidecar_index_recovery": "next_request_event_budget"}
        _NOTICES.submit(self.emit, event, {**fields,
            "diagnostic_event_clock_json": json.dumps(record["clock"], separators=(",", ":")),
            "diagnostic_sequence": record["seq"],
            "notice_delivery": "bounded_async_main_raw_callback",
            "sidecar_events_path": str(self._sink / "events.jsonl") if self._sink else "unavailable"})

    def _native_snapshot(self):
        if self.native_provider is None:
            return {"native_diag_state": "unsupported"}
        try:
            fields = self.native_provider()
            if not fields:
                return {"native_diag_state": "query_unavailable"}
            return dict(fields)
        except Exception as error:
            if not self._native_query_failure_reported:
                self._native_query_failure_reported = True
                self._publish("diagnostic_native_query_failed", error_json=json.dumps(_error(error)))
            return {"native_diag_state": "query_failed", "error": _error(error)}

    def _configure(self):
        value = self._threshold_input
        if value is None:
            value = os.environ.get("FASTLLM_DIAGNOSTIC_STALL_SECONDS", "15")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0 or parsed > 3600:
            raise ValueError("FASTLLM_DIAGNOSTIC_STALL_SECONDS must be finite, >0 and <=3600")
        self.threshold_s = parsed
        self.poll_s = max(0.01, min(1.0, parsed / 4)) if self.poll_s is None else self.poll_s
        text = os.environ.get("FASTLLM_DIAGNOSTIC_LOG_DIR")
        if not text:
            raise ValueError(f"FASTLLM_DIAGNOSTIC_LOG_DIR is required under {_LOG_ROOT}")
        sink = Path(text).expanduser().resolve()
        try:
            sink.relative_to(_LOG_ROOT.resolve())
        except ValueError:
            raise ValueError(f"FASTLLM_DIAGNOSTIC_LOG_DIR must resolve under {_LOG_ROOT}")
        sink.mkdir(parents=True, exist_ok=True)
        self._directory = _directory_budget(sink)
        self._sink = sink

    def _capture(self, native: dict, reason: str, quiet_ns: int):
        reserve = _MAX_SNAPSHOT_BYTES + 8192
        if self._attempts >= self.max_total_captures or self._bytes + _MAX_SNAPSHOT_BYTES > self.max_bytes:
            self._publish("diagnostic_capture_limit", scope="request_lifetime",
                captures=self._attempts, bytes=self._bytes, recovery="next_request",
                capture_limit=self.max_total_captures, byte_limit=self.max_bytes)
            self.stop()
            return
        reservation = self.budget.reserve(reserve)
        if reservation is None:
            state = self.budget.state()
            self._budget_retry_ns = state["next_recovery_monotonic_ns"]
            self._publish("diagnostic_capture_limit", scope="process_rolling_window",
                budget_json=json.dumps(state, separators=(",", ":")),
                recovery="rolling_window_expiry")
            return
        if not self._directory.reserve(_MAX_SNAPSHOT_BYTES):
            self.budget.release_unused_bytes(reservation, 0)
            self._publish("diagnostic_capture_limit", scope="diagnostic_directory_bytes",
                byte_limit=self._directory.bytes_limit,
                recovery="user_selects_new_diagnostic_directory_no_automatic_deletion")
            self.stop()
            return
        self._attempts += 1
        self._episode_attempts += 1
        actual = 0
        try:
            operations = native.get("native_diag_operations_json")
            if isinstance(operations, str):
                try:
                    operations = json.loads(operations)
                except (TypeError, ValueError) as error:
                    operations = {"parse_state": "failed", "error": _error(error), "raw": operations}
            priority = [row.get("thread_id") for row in operations
                        if isinstance(row, Mapping) and row.get("state") == "running"] \
                        if isinstance(operations, list) else []
            evidence = self.capture(self._stop, priority_tids=priority) if self._default_capture \
                       else self.capture(self._stop)
            if self._stop.is_set():
                return
            document = {"schema_version": 1, "request_id": self.request_id,
                "native_handle": self.native_handle, "capture_clock": _clock(),
                "reason": reason, "classification": "long_operation_or_no_progress_observation",
                "deadlock_proven": False, "quiet_elapsed_ns": quiet_ns,
                "last_confirmed_progress_clock_ns": self._progress_clock_ns,
                "last_confirmed_progress_fields": self._progress_fields,
                "progress_owner_generations": self._progress_owners,
                "threshold_s": self.threshold_s, "episode_captures": self._episode_attempts,
                "request_lifetime_capture_limit": self.max_total_captures,
                "request_lifetime_byte_limit": self.max_bytes,
                "process_budget": self.budget.state(),
                "directory_byte_limit": self._directory.bytes_limit,
                "directory_budget_scope": "this_process_and_existing_files_at_open",
                "python_lifecycle": self._lifecycle, "python_outstanding_calls": self._python_waits,
                "native_fields": native,
                "native_operations": operations, "thread_evidence": evidence,
                "main_raw_notice_delivery": _NOTICES.state()}
            payload = json.dumps(document, ensure_ascii=True, separators=(",", ":")).encode()
            # A bounded diagnostic must explicitly identify omitted thread rows.
            while len(payload) > _MAX_SNAPSHOT_BYTES and (
                    evidence.get("proc_tasks") or evidence.get("python_threads")):
                document["thread_rows_truncated_for_byte_limit"] = True
                for key in ("proc_tasks", "python_threads"):
                    rows = evidence.get(key, [])
                    if rows:
                        del rows[len(rows) // 2:]
                payload = json.dumps(document, ensure_ascii=True, separators=(",", ":")).encode()
            if len(payload) > _MAX_SNAPSHOT_BYTES:
                raise ValueError("diagnostic snapshot exceeds bounded byte limit")
            if self._stop.is_set():
                return
            digest = hashlib.sha256(payload).hexdigest()
            tag = hashlib.sha256(self.request_id.encode()).hexdigest()[:16]
            path = self._sink / f"request-{tag}-{time.time_ns()}-{self._attempts}.json"
            with path.open("xb") as out:
                out.write(payload)
                out.flush()
                os.fsync(out.fileno())
            actual = len(payload)
            self._bytes += actual
            self._directory.release_unused_bytes(_MAX_SNAPSHOT_BYTES, actual)
            self._publish("diagnostic_snapshot", snapshot_path=str(path),
                snapshot_sha256=digest, snapshot_bytes=actual, capture_reason=reason,
                classification="observation_not_deadlock_proof",
                quiet_elapsed_ns=quiet_ns, capture_number=self._attempts)
        except Exception as error:
            # Sink-write failures cannot be made durable on the failed sink.
            # The main raw event explicitly reports this instead of pretending
            # the capture exists; it is delivered from a different bounded worker.
            try:
                self._publish("diagnostic_capture_failed", error_json=json.dumps(_error(error)))
            except Exception as sink_error:
                _NOTICES.submit(self.emit, "diagnostic_sink_unavailable", {
                    "capture_error_json": json.dumps(_error(error)),
                    "sink_error_json": json.dumps(_error(sink_error))})
            self.stop()  # Do not repeatedly hit a predictably failed sink/capture.
        finally:
            if actual == 0:
                self._directory.release_unused_bytes(_MAX_SNAPSHOT_BYTES, 0)
            self.budget.release_unused_bytes(reservation, actual + 8192)

    def _run(self):
        try:
            self._configure()
        except Exception as error:
            _NOTICES.submit(self.emit, "diagnostic_sink_unavailable", {
                "error_json": json.dumps(_error(error)),
                "snapshot_written": False, "request_id": self.request_id})
            self.stop()
            return
        while not self._stop.wait(self.poll_s):
            try:
                native = self._native_snapshot()
                state = native.get("native_diag_state")
                if state not in {"owner_mismatch", "missing", "query_failed", "query_unavailable"}:
                    self.observe_status(native, namespace="diagnostic")
                now = time.monotonic_ns()
                if self._progress_epoch != self._episode_epoch:
                    self._episode_epoch = self._progress_epoch
                    self._episode_attempts = 0
                    self._episode_limit_reported = False
                    self._last_capture_ns = 0
                if self._episode_attempts >= self.max_captures:
                    if not self._episode_limit_reported:
                        self._episode_limit_reported = True
                        self._publish("diagnostic_capture_limit", scope="no_progress_episode",
                            captures=self._episode_attempts,
                            recovery="next_confirmed_phase_or_forward_progress")
                    continue
                if now < self._budget_retry_ns:
                    continue
                quiet = now - self._progress_clock_ns
                raw_operations = native.get("native_diag_operations_json", "[]")
                try:
                    operations = json.loads(raw_operations) if isinstance(raw_operations, str) else raw_operations
                    long_op = any(row.get("state") == "running" and
                        row.get("operation") in _WAIT_OPERATIONS and
                        now - int(row.get("start_monotonic_ns", now)) >= self.threshold_s * 1e9
                        for row in operations if isinstance(row, Mapping))
                except (TypeError, ValueError):
                    long_op = False
                long_op = long_op or any(now - int(value["monotonic_ns"]) >= self.threshold_s * 1e9
                    for value in self._python_waits.values())
                if (long_op or quiet >= self.threshold_s * 1e9) and (
                    not self._last_capture_ns or now - self._last_capture_ns >= self.threshold_s * 1e9):
                    self._last_capture_ns = now
                    self._capture(native, "long_operation" if long_op else "no_confirmed_progress", quiet)
            except Exception as error:
                try:
                    self._publish("diagnostic_capture_failed", error_json=json.dumps(_error(error)))
                except Exception as sink_error:
                    _NOTICES.submit(self.emit, "diagnostic_sink_unavailable",
                        {"error_json": json.dumps(_error(sink_error))})
                self.stop()
