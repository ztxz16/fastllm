import json
import sys
import threading
import time


PROGRESS_PREFIX = "FTLLM_PROGRESS "


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
                 heartbeat_interval = 1.0):
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
                 message = None, indeterminate = False, force = False):
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
        self._stop_heartbeat()
        with self._lock:
            self._closed = True
