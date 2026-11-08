"""A single, mode-gated request observation channel for the Python server."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import itertools
import functools
import json
import sys
from pathlib import Path
from urllib.parse import quote
import traceback
import logging
import os
import subprocess
import threading
import time
from typing import Callable, Mapping

try:
    from .request_diagnostics import RequestDiagnosticWatchdog
except ImportError:
    from request_diagnostics import RequestDiagnosticWatchdog


_SEQUENCE = itertools.count(1)
_SEQUENCE_LOCK = threading.Lock()


def observability_mode(value: str | None = None) -> str:
    value = os.environ.get("FASTLLM_OBSERVABILITY_MODE", "off") if value is None else value
    value = str(value).strip().lower()
    return value if value in {"off", "user", "debug"} else "off"


def next_observation_sequence(provider: Callable[[], int] | None = None) -> int:
    if provider is not None:
        try:
            value = provider()
            if value is not None:
                return int(value)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    with _SEQUENCE_LOCK:
        return next(_SEQUENCE)


def _positive(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _json_field(value: object) -> str:
    return quote(json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str), safe="")

@functools.lru_cache(maxsize=1)
def diagnostic_run_context() -> dict[str, object]:
    # Allow-list configuration only. No full argv/environment/prompts/credentials.
    flags = {"--tp", "--gpu_mem_ratio", "--kv_cache_dtype", "--prefix_cache_dir",
             "--prefix_cache_restore_policy", "--prefix_cache_l2", "--prefix_cache_l2_gb",
             "--max_batch", "--model_name", "--image_embedding_cache",
             "--speculative_draft_model_path"}
    args = sys.argv[1:]
    config = {name: args[i + 1] for i, name in enumerate(args[:-1]) if name in flags}
    env_keys = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "FASTLLM_OBSERVABILITY_MODE",
                "FASTLLM_PREFIX_CACHE_DIRECT_IO", "FASTLLM_PREFIX_CACHE_LOG_LEVEL",
                "FASTLLM_PREFIX_CACHE_SPILL_STATS",
                "FASTLLM_DFLASH_DRAFT_DEVICE", "FASTLLM_CUDA_DFLASH_TP_BACKBONE",
                "FASTLLM_PREFIX_CACHE_GPU_PIN_LIMIT_MB", "FASTLLM_PREFIX_CACHE_HOST_LIMIT_MB",
                "FASTLLM_PREFIX_CACHE_CHECKPOINT_INTERVAL")
    result: dict[str, object] = {
        "pid": os.getpid(), "python": sys.executable,
        "module_path": str(Path(__file__).resolve()),
        "cli_config": config,
        "environment": {key: os.environ[key] for key in env_keys if key in os.environ},
        "clock": {"wall": "OS_SYSTEM_CLOCK_UTC", "duration": "CLOCK_MONOTONIC",
                  "utc_sync": "not_verified", "timestamp_scope": "event_capture"},
    }
    for name in ("build_info.json", "observation_artifact.json"):
        path = Path(__file__).with_name(name)
        try:
            result[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            result[name] = {"state": "unavailable", "error_type": type(error).__name__}
    return result

class _ResourceSampler:
    """Shared process/host/service-GPU observations; no per-request GPU query."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self._interval = interval_s
        self._lock = threading.Lock()
        self._values: dict[str, str] = {}
        self._sampled_at: float | None = None
        self._started = False
        self._stop = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        threading.Thread(target=self._loop, name="fastllm-resource-sampler", daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._sample()
            except Exception as error:
                with self._lock:
                    self._values = {"resource_sample_state": "failed", "resource_sample_error": type(error).__name__}
                    self._sampled_at = None
            self._stop.wait(max(0.0, self._interval - (time.monotonic() - started)))

    @staticmethod
    def _service_gpus(rows: list[list[str]]) -> list[list[str]] | None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            # No resolved service device scope: do not silently report all GPUs.
            return None
        ids = [value.strip() for value in visible.split(",") if value.strip()]
        if not ids or ids == ["-1"]:
            return []
        ordered = sorted(rows, key=lambda row: row[2])
        selected = []
        for value in ids:
            if value.startswith("GPU-"):
                matches = [row for row in rows if row[1].startswith(value)]
            elif value.isdigit() and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID":
                index = int(value)
                matches = ordered[index:index + 1]
            else:
                return None  # FASTEST_FIRST numeric ordinals are not NVML indices.
            if len(matches) != 1:
                return None
            if matches[0] not in selected:
                selected.append(matches[0])
        return selected

    def _sample(self) -> None:
        begun = time.monotonic()
        values: dict[str, str] = {"resource_sample_begin_ns": str(time.monotonic_ns()),
                                  "resource_sample_begin_utc": datetime.now(timezone.utc).isoformat(),
                                  "resource_scope": "process_host_and_service_device_samples"}
        try:
            with open("/proc/meminfo", encoding="ascii") as stream:
                for line in stream:
                    if line.startswith("MemAvailable:"):
                        values["host_mem_available_mib"] = str(int(line.split()[1]) // 1024)
                        break
        except (OSError, ValueError, IndexError):
            pass
        try:
            with open("/proc/self/statm", encoding="ascii") as stream:
                pages = int(stream.read().split()[1])
            values["proc_rss_mib"] = str(pages * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024))
        except (OSError, ValueError, IndexError):
            pass
        try:
            with open("/proc/loadavg", encoding="ascii") as stream:
                values["cpu_load1"] = stream.read().split()[0]
        except (OSError, IndexError):
            pass
        try:
            wanted = {"VmHWM": "proc_rss_peak_mib", "VmSwap": "proc_swap_mib",
                      "RssAnon": "proc_anon_mib", "RssFile": "proc_file_mib"}
            with open("/proc/self/status", encoding="ascii") as stream:
                for line in stream:
                    label, _, value = line.partition(":")
                    if label in wanted:
                        values[wanted[label]] = str(int(value.split()[0]) // 1024)
        except (OSError, ValueError, IndexError) as error:
            values["proc_memory_detail_state"] = type(error).__name__
        values["gpu_sample_state"] = "unavailable"
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=False)
            values["gpu_query_exit_code"] = str(completed.returncode)
            values["gpu_sample_source"] = "nvidia_smi_device_utilization_window"
            if completed.returncode == 0:
                rows = [[part.strip() for part in line.split(",")]
                        for line in completed.stdout.strip().splitlines()]
                rows = [row for row in rows if len(row) == 5]
                selected = self._service_gpus(rows)
                if selected is None:
                    values["gpu_sample_state"] = "scope_unknown"
                elif not selected:
                    values["gpu_sample_state"] = "disabled"
                else:
                    utils = [int(row[3]) for row in selected]
                    values["gpu_util_max_pct"] = str(max(utils))
                    values["gpu_mem_used_mib"] = str(sum(int(row[4]) for row in selected))
                    values["gpu_device_indices"] = ",".join(row[0] for row in selected)
                    values["gpu_scope"] = "service_visible_devices"
                    values["gpu_devices_json"] = _json_field([{"index": r[0], "uuid": r[1], "pci_bus": r[2],
                        "util_pct": r[3], "memory_used_mib": r[4]} for r in selected])
                    values["gpu_sample_state"] = "ok"
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            values["gpu_sample_error_type"] = type(error).__name__
        values["resource_sample_end_ns"] = str(time.monotonic_ns())
        values["resource_sample_end_utc"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._values = values
            self._sampled_at = begun

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            values = dict(self._values)
            sampled_at = self._sampled_at
        if sampled_at is None:
            values["resource_sample_state"] = "pending"
        else:
            age = max(0.0, time.monotonic() - sampled_at)
            values["resource_sample_age_ms"] = f"{age * 1000:.0f}"
            values["resource_sample_state"] = "fresh" if age <= 2.5 else "stale"
        return values


_SAMPLER = _ResourceSampler()


def resource_snapshot() -> dict[str, str]:
    """Start the shared sampler on first use and return its latest values."""
    _SAMPLER.start()
    return _SAMPLER.snapshot()


@dataclass
class RequestProgressReporter:
    """Emit one correlated lifecycle/progress stream for a request."""

    request_id: str
    emit: Callable[[str], None] | None = None
    clock: Callable[[], float] = time.monotonic
    interval_s: float = 1.0
    native_handle: int | None = None
    mode: str | None = None
    sequence_provider: Callable[[], int] | None = None
    emit_cache_events: bool = True
    status_provider: Callable[[], Mapping[str, object] | None] | None = None
    diagnostics_provider: Callable[[], Mapping[str, object] | None] | None = None
    request_context: Mapping[str, object] | None = None
    _native_closed: bool = field(default=False, init=False)
    _last_native_generation: object = field(default=None, init=False)
    _started_at: float = field(init=False)
    _last_report: float = field(init=False)
    _last_output_tokens: int = field(default=0, init=False)
    _output_tokens: int = field(default=0, init=False)
    _prefill_tokens: int = field(default=0, init=False)
    _callback_events: int = field(default=0, init=False)
    _last_callback_events: int = field(default=0, init=False)
    _output_count_known: bool = field(default=False, init=False)
    _cached_input_tokens: int = field(default=0, init=False)
    _missed_tokens: int = field(default=0, init=False)
    _effective_reused_tokens: int = field(default=0, init=False)
    _cache_segments: list[dict[str, object]] = field(default_factory=list, init=False)
    _first_token_at: float | None = field(default=None, init=False)
    _finished: bool = field(default=False, init=False)
    _cache_reported: bool = field(default=False, init=False)
    _cache_lookup_ms: float | None = field(default=None, init=False)
    _restore_elapsed_ms: float | None = field(default=None, init=False)
    _ssd_read_mib_s: float | None = field(default=None, init=False)
    _ssd_restore_tokens_s: float | None = field(default=None, init=False)
    _vision_ssd_mib_s: float | None = field(default=None, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _pump: threading.Thread | None = field(default=None, init=False, repr=False)
    _pump_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _finish_requested: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _diagnostics: RequestDiagnosticWatchdog | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self._started_at = now
        self._last_report = now
        # A custom emitter is explicit and useful for contract tests. The
        # production logger path follows the documented off-by-default mode.
        if self.mode is None:
            self.mode = observability_mode()
            if self.emit is not None and "FASTLLM_OBSERVABILITY_MODE" not in os.environ:
                # Preserve the prototype's diagnostic test hook without
                # changing the production logger's documented off default.
                self.mode = "debug"
        else:
            self.mode = observability_mode(self.mode)
        if self.emit is None:
            self.emit = logging.getLogger(__name__).info

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def _event(self, category: str, event: str, **fields: object) -> None:
        if not self.enabled:
            return
        if category == "progress" and (self._finish_requested.is_set() or self._pump_stop.is_set()):
            return  # A pending/late status query cannot revive terminal progress.
        parts = [
            "[fastllm-event]",
            f"seq={next_observation_sequence(self.sequence_provider)}",
            f"timestamp_monotonic_ns={time.monotonic_ns()}",
            f"timestamp_utc={datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')}",
            "source=python",
            f"category={category}",
            f"request={self.request_id}",
            f"request_id={self.request_id}",
            f"native_handle={self.native_handle if self.native_handle is not None else 'unknown'}",
            f"event={event}",
        ]
        parts.extend(f"{key}={value}" for key, value in fields.items())
        assert self.emit is not None
        self.emit(" ".join(parts))

    def _diagnostic_notice(self, event: str, fields: Mapping) -> None:
        # Invoked only by the bounded process delivery worker, never by the
        # capture thread; it deliberately does not acquire Reporter._lock.
        values = {key: quote(value, safe="") if isinstance(value, str) and
                  any(c.isspace() for c in value) else value
                  for key, value in fields.items()}
        self._event("diagnostic", event, **values)

    def _start_diagnostics(self) -> None:
        if self.mode != "debug" or self._diagnostics is not None or self._finished:
            return
        self._diagnostics = RequestDiagnosticWatchdog(
            self.request_id, self.native_handle, native_provider=self.diagnostics_provider,
            emit=self._diagnostic_notice, mode=self.mode)
        self._diagnostics.start()

    def received(self, *, path: str) -> None:
        # Start before logging so a blocked main logger cannot suppress the
        # independent capture source for an already active request.
        self._start_diagnostics()
        self._event("startup", "request_received", path=path,
                    request_context_json=_json_field(self.request_context or {}))
        if self.enabled:
            self._event("diagnostic", "request_context", context_json=_json_field(diagnostic_run_context()),
                        timestamp_semantics="event_capture_wall_clock",
                        phase_semantics="last_confirmed_native_boundary",
                        sequence_semantics="unique_allocation_id_not_cross_thread_emission_order")
        self.start()

    def start(self) -> None:
        """Emit one progress line per interval even when the request is idle."""
        if not self.enabled or self._pump is not None or self._finished:
            return
        self._start_diagnostics()
        self._pump = threading.Thread(
            target=self._pump_loop, name="fastllm-request-progress", daemon=True)
        self._pump.start()

    def _pump_loop(self) -> None:
        while not self._pump_stop.wait(self.interval_s):
            try:
                self.maybe()
            except Exception as error:
                if self._diagnostics is not None:
                    self._diagnostics.note_lifecycle("progress_pump_failed", error_type=type(error).__name__)
                self.error(error, stage="progress_pump")
                self._pump_stop.set()
                # A broken progress pump is a reason to retain its independent
                # watcher until the actual request lifecycle completes.
                return

    def stop(self) -> None:
        self._pump_stop.set()
        if self._diagnostics is not None:
            self._diagnostics.stop()

    def tool_parser_selected(self, parser_name: str) -> None:
        self._event("tool", "tool_parser_selected", parser=parser_name)

    def cache_lookup(self) -> None:
        if self.enabled:
            self._event("cache", "cache_lookup", phase="begin")

    def vision_lookup(self) -> None:
        if self.enabled:
            self._event("vision", "vision_cache_lookup", phase="begin")

    def _emit_cache_segments(self) -> None:
        if self._cache_reported or not self._cache_segments:
            return
        if not self.emit_cache_events:
            self._cache_reported = True
            return
        for segment in self._cache_segments:
            self._event("cache", "cache_result", **segment)
        self._cache_reported = True

    def observe_cache(self, statistics: Mapping[str, object] | None) -> None:
        if not statistics:
            return
        self._cached_input_tokens = max(
            self._cached_input_tokens,
            int(statistics.get("cached_input_tokens", 0) or 0),
        )
        self._missed_tokens = max(
            self._missed_tokens,
            int(statistics.get("missed_input_tokens", 0) or 0),
        )
        raw_segments = statistics.get("cache_segments")
        if isinstance(raw_segments, (list, tuple)):
            normalized: list[dict[str, object]] = []
            for raw in raw_segments:
                if not isinstance(raw, Mapping):
                    continue
                item: dict[str, object] = {
                    "source_tier": str(raw.get("source_tier", "unknown")),
                    "promoted_to": str(raw.get("promoted_to", "NONE")),
                    "source_tokens": max(0, int(raw.get("source_tokens", 0) or 0)),
                    "effective_reused_tokens": max(
                        0, int(raw.get("effective_reused_tokens", 0) or 0)),
                }
                for key in ("lookup_ms", "restore_ms", "read_mib_s", "restore_tokens_s"):
                    value = _positive(raw.get(key))
                    if value is not None:
                        item[key] = f"{value:.3f}"
                normalized.append(item)
            if normalized:
                self._cache_segments = normalized
                self._effective_reused_tokens = min(
                    max(0, self._cached_input_tokens),
                    sum(int(item["effective_reused_tokens"]) for item in normalized),
                )
        else:
            self._effective_reused_tokens = max(
                self._effective_reused_tokens,
                int(statistics.get("effective_reused_tokens", 0) or 0),
            )
        self._emit_cache_segments()

    def callback(self, statistics: Mapping[str, object] | None = None) -> None:
        """Observe callback delivery; an empty heartbeat is not a generated token."""
        with self._lock:
            self._callback_events += 1
        now = self.clock()
        count = statistics.get("output_tokens") if statistics else None
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            self._output_count_known = True
            delta = count - self._output_tokens
            if delta > 0:
                self.output(delta, now=now)
                return
        self.maybe(now=now)

    def output(self, count: int = 1, *, now: float | None = None) -> None:
        """Observe an explicitly counted backend-token delta, never a text chunk."""
        if count <= 0:
            return
        self._output_count_known = True
        with self._lock:
            self._output_tokens += count
        current = self.clock() if now is None else now
        if self._first_token_at is None:
            self._first_token_at = current
            self._event(
                "output", "first_token", measurement="api_observed_native_tokens",
                ttft_ms=f"{(current - self._started_at) * 1000:.1f}",
            )
        self.maybe(now=current)

    def prefill(self, tokens: int) -> None:
        if tokens > 0:
            with self._lock:
                self._prefill_tokens += tokens

    def tool_call(self, count: int = 1) -> None:
        self._event("tool", "tool_call", count=max(1, int(count)))

    def set_cache_timings(
        self,
        *,
        lookup_ms: float | None = None,
        restore_ms: float | None = None,
        ssd_read_mib_s: float | None = None,
        ssd_restore_tokens_s: float | None = None,
        vision_ssd_mib_s: float | None = None,
    ) -> None:
        for name, value in (
            ("_cache_lookup_ms", lookup_ms),
            ("_restore_elapsed_ms", restore_ms),
            ("_ssd_read_mib_s", ssd_read_mib_s),
            ("_ssd_restore_tokens_s", ssd_restore_tokens_s),
            ("_vision_ssd_mib_s", vision_ssd_mib_s),
        ):
            parsed = _positive(value)
            if parsed is not None:
                current = getattr(self, name)
                setattr(self, name, parsed if current is None else max(current, parsed))

    def maybe(self, *, now: float | None = None, force: bool = False) -> bool:
        with self._lock:
            return self._maybe_locked(now=now, force=force)

    def _maybe_locked(self, *, now: float | None = None, force: bool = False) -> bool:
        if not self.enabled or self._finished:
            return False
        now = self.clock() if now is None else now
        elapsed = now - self._last_report
        if not force and elapsed < self.interval_s:
            return False
        elapsed = max(elapsed, 1e-9)
        delta_output = self._output_tokens - self._last_output_tokens
        delta_callbacks = self._callback_events - self._last_callback_events
        fields: dict[str, object] = {
            "measurement": "python_callback_interval",
            "elapsed_s": f"{elapsed:.3f}",
            "callback_events": delta_callbacks,
            "python_callback_events_s": f"{delta_callbacks / elapsed:.2f}",
            "reported_prefill_tokens": self._prefill_tokens,
            "cached_input_tokens": self._cached_input_tokens,
            "missed_input_tokens": self._missed_tokens,
            "effective_reused_tokens": self._effective_reused_tokens,
        }
        if self._output_count_known:
            fields["observed_output_tokens"] = self._output_tokens
            fields["python_callback_tokens_s"] = f"{delta_output / elapsed:.2f}"
        for key, value in (
            ("prefix_lookup_ms", self._cache_lookup_ms),
            ("prefix_restore_ms", self._restore_elapsed_ms),
            ("ssd_read_mib_s", self._ssd_read_mib_s),
            ("ssd_restore_tokens_s", self._ssd_restore_tokens_s),
            ("vision_ssd_load_mib_s", self._vision_ssd_mib_s),
        ):
            if value is not None:
                fields[key] = f"{value:.3f}"
        snapshot = resource_snapshot()
        rss = snapshot.get("proc_rss_mib")
        if rss is not None:
            fields["proc_rss_mib"] = rss
        host = snapshot.get("host_mem_available_mib")
        if host is not None:
            fields["host_mem_available_mib"] = host
        load = snapshot.get("cpu_load1")
        if load is not None:
            fields["cpu_load1"] = load
        gpu_util = snapshot.get("gpu_util_max_pct")
        if gpu_util is not None:
            fields["gpu_util_max_pct"] = gpu_util
        gpu_mem = snapshot.get("gpu_mem_used_mib")
        if gpu_mem is not None:
            fields["gpu_mem_used_mib"] = gpu_mem
        self._event("progress", "progress", **fields)
        if self._native_closed:
            self._event("lifecycle", "api_finalizing", native_observer_state="closed")
        else:
            self._sample_native_status(snapshot)
        self._prefill_tokens = 0
        self._last_output_tokens = self._output_tokens
        self._last_report = now
        self._last_callback_events = self._callback_events
        return True

    def _sample_native_status(self, resources=None, *, sample_kind="periodic") -> None:
        status: dict[str, object] = {
            "measurement": "native_status_snapshot",
            "snapshot_kind": sample_kind,
            "native_status_state": "unsupported",
        }
        begin = time.monotonic_ns()
        if self.status_provider is not None:
            if self._diagnostics is not None:
                self._diagnostics.note_lifecycle("native_status_query_begin")
            try:
                native = self.status_provider()
                if native:
                    status["native_status_state"] = "observed"
                    status.update(native)
                    self._last_native_generation = native.get("native_generation")
                    if self._diagnostics is not None:
                        self._diagnostics.observe_status(native)
                else:
                    status["native_status_state"] = "observer_missing_or_owner_changed"
            except Exception as error:
                status["native_status_state"] = "query_failed"
                self.error(error, stage="native_status_query")
            finally:
                if self._diagnostics is not None:
                    self._diagnostics.note_lifecycle("native_status_query_end")
        status["python_query_begin_ns"] = begin
        status["python_query_end_ns"] = time.monotonic_ns()
        if resources:
            status.update(resources)
        self._event("progress", "request_status", **status)

    def close_native_observation(self, reason: str) -> None:
        """Capture a final real sample before unregister, then stop querying it."""
        if self._diagnostics is not None:
            self._diagnostics.note_lifecycle("native_close_requested", reason=reason)
        with self._lock:
            if self._native_closed or not self.enabled:
                return
            self._sample_native_status(sample_kind="native_release")
            self._native_closed = True
            self.status_provider = None
            if self._diagnostics is not None:
                self._diagnostics.note_lifecycle("native_close_completed", reason=reason)
            self._event("lifecycle", "native_observer_closed", reason=reason,
                        native_generation=self._last_native_generation or "unknown",
                        capacity_scope="last_sample_not_cache_failure")

    def error(self, error: BaseException, *, stage: str) -> None:
        self._event("error", "request_error", stage=stage,
                    error_type=type(error).__name__,
                    error_json=_json_field({"message": str(error),
                        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))}))

    def lifecycle(self, event: str, **fields: object) -> None:
        if self._diagnostics is not None:
            self._diagnostics.note_lifecycle(event, **fields)
        self._event("lifecycle", event, **fields)

    def finish(
        self, *, status: str = "ok", output_tokens: int | None = None,
    ) -> None:
        self._finish_requested.set()
        if self._diagnostics is not None:
            self._diagnostics.note_lifecycle("finish_requested", status=status)
        with self._lock:
            if self._finished:
                return
            try:
                if output_tokens is not None and output_tokens >= 0:
                    self._output_tokens = int(output_tokens)
                    self._output_count_known = True
                self._emit_cache_segments()
                self._maybe_locked(force=True)
                elapsed_ms = (self.clock() - self._started_at) * 1000
                fields: dict[str, object] = {
                    "status": status, "output_count_state": "known" if self._output_count_known else "unknown",
                    "cached_input_tokens": self._cached_input_tokens,
                    "effective_reused_tokens": self._effective_reused_tokens,
                    "missed_input_tokens": self._missed_tokens,
                    "elapsed_ms": f"{elapsed_ms:.1f}",
                    "native_observer_state": "closed" if self._native_closed else "attached",
                }
                if self._output_count_known:
                    fields["output_tokens"] = self._output_tokens
                if self._output_count_known and self._output_tokens > 0 and elapsed_ms > 0:
                    fields["request_total_average_tokens_s"] = f"{self._output_tokens / (elapsed_ms / 1000):.2f}"
                self._event("output", "request_finished", **fields)
            finally:
                self._finished = True
                if self._diagnostics is not None:
                    self._diagnostics.note_lifecycle("finish_completed", status=status)
                self.stop()
