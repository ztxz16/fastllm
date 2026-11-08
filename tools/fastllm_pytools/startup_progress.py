import json
import os
import sys
import threading
import time

from .request_progress import RequestProgressReporter, observability_mode


PROGRESS_PREFIX = "FTLLM_PROGRESS "


def _startup_memory_snapshot():
    """Read current host facts; unavailable data remains explicitly unknown."""
    result = {"process_rss_bytes": "unknown", "system_mem_available_bytes": "unknown"}
    for path, prefix, field in (
            ("/proc/self/status", "VmRSS:", "process_rss_bytes"),
            ("/proc/meminfo", "MemAvailable:", "system_mem_available_bytes")):
        try:
            with open(path, "r", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith(prefix):
                        parts = line.split()
                        if len(parts) >= 3 and parts[2] == "kB":
                            result[field] = str(int(parts[1]) * 1024)
                        break
        except (OSError, ValueError):
            pass
    return result


class StartupStatusHeartbeat:
    """Snapshot explicit load stages without executing logging in callbacks."""

    def __init__(self, mode="off", stream=None, clock=None,
                 heartbeat_interval=1.0, resource_reader=None):
        self.mode = observability_mode(mode)
        self.stream = sys.stderr if stream is None else stream
        self.clock = time.monotonic if clock is None else clock
        self.resource_reader = resource_reader or _startup_memory_snapshot
        self.interval = max(0.01, float(heartbeat_interval))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.started_at = self.phase_started = self.updated_at = self.clock()
        self.state = ("initializing", 0, 0, 0, 0)
        self.closed = False
        self.sequence_provider = None
        self.thread = None
        self.reporter = RequestProgressReporter(
            "startup", mode=self.mode, emit=self._write_raw, clock=self.clock)
        if self.mode != "off":
            self.thread = threading.Thread(
                target=self._run, name="ftllm-startup-status", daemon=True)
            self.thread.start()

    def _write_raw(self, line):
        self.stream.write(line + "\n")
        self.stream.flush()

    def set_sequence_provider(self, provider):
        with self.lock:
            self.sequence_provider = provider

    def update(self, stage, current=0, total=0, completed_bytes=0, total_bytes=0):
        if self.mode == "off":
            return
        state = (str(stage), int(current), int(total), int(completed_bytes), int(total_bytes))
        with self.lock:
            if self.closed or state == self.state:
                return
            now = self.clock()
            if state[0] != self.state[0]:
                self.phase_started = now
            self.updated_at = now
            self.state = state

    def emit_snapshot(self):
        with self.lock:
            if self.closed or self.mode == "off":
                return
            stage, current, total, completed_bytes, total_bytes = self.state
            now = self.clock()
            elapsed_ms = f"{max(0.0, now - self.started_at) * 1000:.3f}"
            phase_elapsed_ms = f"{max(0.0, now - self.phase_started) * 1000:.3f}"
            idle_ms = f"{max(0.0, now - self.updated_at) * 1000:.3f}"
            provider = self.sequence_provider
        # Resource reads and output happen outside the snapshot lock. Even a
        # blocked stderr cannot hold model-load callbacks or close()/ready().
        memory = self.resource_reader()
        self.reporter.sequence_provider = provider
        self.reporter._event(
            "startup", "startup_status", stage=stage,
            detail=stage.partition(":")[2] or stage,
            elapsed_ms=elapsed_ms, phase_elapsed_ms=phase_elapsed_ms, idle_ms=idle_ms,
            current=current, total=total, completed_bytes=completed_bytes, total_bytes=total_bytes,
            process_pid=os.getpid(),
            process_rss_bytes=memory.get("process_rss_bytes", "unknown"),
            system_mem_available_bytes=memory.get("system_mem_available_bytes", "unknown"))

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.emit_snapshot()
            except Exception:
                # Startup observation must not fail the actual model load.
                pass

    def close(self):
        with self.lock:
            self.closed = True
        self.stop_event.set()


class StartupProgressReporter:
    _STAGE_RANGES = {
        "initializing": (0.0, 2.0),
        "tokenizer": (2.0, 8.0),
        "weights_prepare": (8.0, 15.0),
        "weights_load": (15.0, 85.0),
        "weights_finalize": (85.0, 90.0),
        "warmup": (90.0, 98.0),
        "server_starting": (98.0, 99.0),
        "ready": (100.0, 100.0),
    }

    _STAGE_MESSAGES = {
        "initializing": "Initializing model",
        "tokenizer": "Loading tokenizer",
        "weights_prepare": "Preparing model weights",
        "weights_load": "Loading model weights",
        "weights_finalize": "Finalizing model weights",
        "warmup": "Warming up model",
        "server_starting": "Starting API server",
        "ready": "API server is ready",
    }

    def __init__(self, mode = "off", stream = None, clock = None,
                 heartbeat_interval = 1.0, observation_mode = "off",
                 status_stream = None, resource_reader = None):
        if mode not in ["off", "ndjson"]:
            raise ValueError("startup progress mode must be 'off' or 'ndjson'")
        self.mode = mode
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock if clock is not None else time.monotonic
        self.heartbeat_interval = max(0.01, float(heartbeat_interval))
        self.started_at = self.clock()
        self._lock = threading.RLock()
        self._sequence = 0
        self._global_percent = 0.0
        self._stage_percents = {}
        self._last_signature = None
        self._last_stage = "initializing"
        self._heartbeat_stop = None
        self._heartbeat_thread = None
        self._ready = False
        self._closed = False
        self._status = StartupStatusHeartbeat(
            observability_mode(observation_mode),
            stream=self.stream if status_stream is None else status_stream,
            clock=self.clock, heartbeat_interval=self.heartbeat_interval,
            resource_reader=resource_reader)

    def set_sequence_provider(self, provider):
        self._status.set_sequence_provider(provider)

    def observe_stage(self, stage):
        self._status.update(stage)

    @property
    def enabled(self):
        return self.mode == "ndjson"

    def _calculate_stage_percent(self, current, total,
                                 completed_bytes, total_bytes):
        if total_bytes > 0:
            return min(100.0, max(0.0, completed_bytes * 100.0 / total_bytes))
        if total > 0:
            return min(100.0, max(0.0, current * 100.0 / total))
        return None

    def _write_event_locked(self, event):
        if not self.enabled:
            return
        self._sequence += 1
        event.update({
            "version": 1,
            "sequence": self._sequence,
            "elapsed_ms": max(0, int((self.clock() - self.started_at) * 1000)),
        })
        try:
            self.stream.write(
                PROGRESS_PREFIX +
                json.dumps(event, ensure_ascii = False, separators = (",", ":")) +
                "\n"
            )
            self.stream.flush()
        except Exception:
            # Progress reporting must never make model startup fail.
            pass

    def progress(self, stage, current = 0, total = 0,
                 completed_bytes = 0, total_bytes = 0,
                 message = None, indeterminate = False, force = False,
                 _observe_status = True):
        if _observe_status:
            self._status.update(stage, current, total, completed_bytes, total_bytes)
        if not self.enabled:
            return
        current = max(0, int(current))
        total = max(0, int(total))
        completed_bytes = max(0, int(completed_bytes))
        total_bytes = max(0, int(total_bytes))

        with self._lock:
            if self._closed or self._ready:
                return
            stage_percent = self._calculate_stage_percent(
                current, total, completed_bytes, total_bytes
            )
            previous_stage_percent = self._stage_percents.get(stage)
            if (stage_percent is not None and previous_stage_percent is not None
                    and stage_percent < previous_stage_percent and not indeterminate):
                return
            if stage_percent is not None:
                stage_percent = max(stage_percent, previous_stage_percent or 0.0)
                self._stage_percents[stage] = stage_percent

            stage_start, stage_end = self._STAGE_RANGES.get(
                stage, (self._global_percent, self._global_percent)
            )
            if stage_percent is None:
                percent = stage_start
            else:
                percent = stage_start + (stage_end - stage_start) * stage_percent / 100.0
            percent = max(self._global_percent, min(100.0, percent))
            self._global_percent = percent
            self._last_stage = stage

            rounded_percent = round(percent, 2)
            rounded_stage_percent = (
                round(stage_percent, 2) if stage_percent is not None else None
            )
            message = message or self._STAGE_MESSAGES.get(stage, stage)
            signature = (
                "startup.progress",
                stage,
                rounded_percent,
                rounded_stage_percent,
                bool(indeterminate),
                message,
            )
            if not force and signature == self._last_signature:
                return
            self._last_signature = signature

            event = {
                "type": "startup.progress",
                "stage": stage,
                "percent": rounded_percent,
                "current": current,
                "total": total,
                "message": message,
            }
            if rounded_stage_percent is not None:
                event["stage_percent"] = rounded_stage_percent
            if total_bytes > 0 or completed_bytes > 0:
                event["completed_bytes"] = completed_bytes
                event["total_bytes"] = total_bytes
            if indeterminate:
                event["indeterminate"] = True
            self._write_event_locked(event)

    def model_load_progress(self, stage, current = 0, total = 0,
                            completed_bytes = 0, total_bytes = 0):
        if stage.startswith("warmup:"):
            self._status.update(stage, current, total, completed_bytes, total_bytes)
            self.progress("warmup", current, total, completed_bytes, total_bytes,
                          indeterminate=True, _observe_status=False)
            return
        if stage == "warmup":
            complete = total > 0 and current >= total
            if complete:
                self._stop_heartbeat()
                self.progress(stage, current, total, completed_bytes, total_bytes)
            else:
                self.progress(stage, current, total, completed_bytes, total_bytes)
                self._start_heartbeat(current, total, completed_bytes, total_bytes)
            return
        self.progress(stage, current, total, completed_bytes, total_bytes)

    def _start_heartbeat(self, current, total, completed_bytes, total_bytes):
        if not self.enabled:
            return
        with self._lock:
            if self._heartbeat_thread is not None:
                return
            stop_event = threading.Event()
            self._heartbeat_stop = stop_event

            def heartbeat():
                while not stop_event.wait(self.heartbeat_interval):
                    self.progress(
                        "warmup",
                        current,
                        total,
                        completed_bytes,
                        total_bytes,
                        indeterminate = True,
                        force = True,
                        _observe_status = False,
                    )

            thread = threading.Thread(
                target = heartbeat,
                name = "ftllm-startup-progress",
                daemon = True,
            )
            self._heartbeat_thread = thread
            thread.start()

    def _stop_heartbeat(self):
        with self._lock:
            stop_event = self._heartbeat_stop
            thread = self._heartbeat_thread
            self._heartbeat_stop = None
            self._heartbeat_thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout = min(1.0, self.heartbeat_interval * 2))

    def ready(self):
        self._status.close()
        self._stop_heartbeat()
        if not self.enabled:
            self._ready = True
            return
        with self._lock:
            if self._closed or self._ready:
                return
            self._ready = True
            self._global_percent = 100.0
            self._last_stage = "ready"
            self._write_event_locked({
                "type": "startup.ready",
                "stage": "ready",
                "percent": 100.0,
                "stage_percent": 100.0,
                "current": 1,
                "total": 1,
                "message": self._STAGE_MESSAGES["ready"],
            })

    def fail(self, error):
        self._status.close()
        self._stop_heartbeat()
        if not self.enabled:
            return
        with self._lock:
            if self._closed or self._ready:
                return
            message = str(error) or error.__class__.__name__
            self._write_event_locked({
                "type": "startup.error",
                "stage": self._last_stage,
                "percent": round(self._global_percent, 2),
                "message": message,
                "error_type": error.__class__.__name__,
            })

    def close(self):
        self._status.close()
        self._stop_heartbeat()
        with self._lock:
            self._closed = True
