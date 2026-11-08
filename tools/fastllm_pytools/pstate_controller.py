"""Optional NVIDIA P-State control for the FastLLM API server.

The controller deliberately lives outside the CUDA kernels. It loads the host
NVIDIA NVAPI library and changes the performance state while FastLLM keeps its
model and CUDA context resident. The feature is disabled unless
``FASTLLM_PSTATE_ENABLE`` is explicitly enabled.
"""

from __future__ import annotations

import contextvars
import dataclasses
import ctypes
import logging
import os
import secrets
import subprocess
import threading
import time
import urllib.request
from typing import Callable, Mapping

from .request_progress import next_observation_sequence

logger = logging.getLogger(__name__)


def _observability_mode() -> str:
    value = os.environ.get("FASTLLM_OBSERVABILITY_MODE", "off").strip().lower()
    return value if value in {"off", "user", "debug"} else "off"

_PSTATE_POLL_HEADER = "X-Fastllm-Pstate-Poll"
# A per-process marker distinguishes our poll from other localhost clients.
_PSTATE_POLL_TOKEN = secrets.token_hex(32)
_quiet_pstate_access = contextvars.ContextVar("quiet_pstate_access", default=False)


class _PStateAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _quiet_pstate_access.get()


def install_pstate_access_log_filter() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _PStateAccessLogFilter) for f in access_logger.filters):
        access_logger.addFilter(_PStateAccessLogFilter())


class PStateAccessLogMiddleware:
    """Silence only successful self-polls, at Uvicorn's response-start send."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        client = scope.get("client")
        own_poll = (
            scope["type"] == "http"
            and scope.get("method") == "GET"
            and scope.get("path") == "/metrics"
            and client is not None
            and client[0] in ("127.0.0.1", "::1")
            and (_PSTATE_POLL_HEADER.lower().encode("ascii"),
                 _PSTATE_POLL_TOKEN.encode("ascii")) in scope.get("headers", ())
        )
        if not own_poll:
            return await self.app(scope, receive, send)

        async def quiet_send(message):
            if message["type"] == "http.response.start" and message["status"] == 200:
                token = _quiet_pstate_access.set(True)
                try:
                    await send(message)
                finally:
                    _quiet_pstate_access.reset(token)
            else:
                await send(message)

        await self.app(scope, receive, quiet_send)


_TRUE_VALUES = frozenset(("1", "true", "yes", "on", "pstate"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off", ""))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
    return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
    if value < minimum:
        logger.warning("Ignoring %s=%r below %.3f; using %s", name, raw, minimum, default)
        return default
    return value


def _default_gpu_ids() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return visible or "0"


@dataclasses.dataclass(frozen=True)
class PStateConfig:
    """Resolved controller configuration."""

    gpu_ids: str
    idle_timeout_s: float = 5.0
    confirm_s: float = 5.0
    poll_s: float = 5.0
    util_threshold: float = 5.0
    low: int = 8
    high: int = 16
    metrics_url: str = "http://127.0.0.1:8080/metrics"
    required: bool = False

    @classmethod
    def from_env(cls, metrics_url: str | None = None) -> "PStateConfig | None":
        if not _env_bool("FASTLLM_PSTATE_ENABLE"):
            return None

        gpu_ids = os.environ.get("FASTLLM_PSTATE_GPUS", _default_gpu_ids()).strip()
        if not gpu_ids:
            raise ValueError("FASTLLM_PSTATE_GPUS must not be empty when P-State control is enabled")

        low = int(_env_float("FASTLLM_PSTATE_LOW", 8.0, 0.0))
        high = int(_env_float("FASTLLM_PSTATE_HIGH", 16.0, 0.0))
        if high <= 0 or low > high or (low == 0 and high == 0):
            raise ValueError(
                "FASTLLM_PSTATE_LOW/HIGH must satisfy 0 <= LOW <= HIGH and HIGH > 0"
            )

        return cls(
            gpu_ids=gpu_ids,
            idle_timeout_s=_env_float("FASTLLM_PSTATE_IDLE_TIMEOUT_S", 5.0, 0.001),
            confirm_s=_env_float("FASTLLM_PSTATE_CONFIRM_S", 5.0, 0.0),
            poll_s=_env_float("FASTLLM_PSTATE_POLL_S", 5.0, 0.1),
            util_threshold=_env_float("FASTLLM_PSTATE_UTIL_THRESHOLD", 5.0, 0.0),
            low=low,
            high=high,
            metrics_url=(
                os.environ.get("FASTLLM_PSTATE_METRICS_URL", "").strip()
                or metrics_url
                or "http://127.0.0.1:8080/metrics"
            ),
            required=_env_bool("FASTLLM_PSTATE_REQUIRED"),
        )


def parse_metrics(text: str) -> dict[str, float]:
    """Parse the small Prometheus subset emitted by FastLLM."""

    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or " " not in line:
            continue
        name, raw_value = line.rsplit(None, 1)
        name = name.split("{", 1)[0]
        try:
            result[name] = float(raw_value)
        except ValueError:
            continue
    return result


class RequestActivity:
    """Thread-safe request counters used by the P-State supervisor."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._active = 0
        self._last_request_unix_s = 0.0

    def begin(self) -> None:
        with self._lock:
            self._total += 1
            self._active += 1
            self._last_request_unix_s = time.time()

    def end(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    def snapshot(self) -> Mapping[str, float]:
        with self._lock:
            return {
                "fastllm_requests_total": float(self._total),
                "fastllm_active_requests": float(self._active),
                "fastllm_last_request_unix_seconds": self._last_request_unix_s,
            }

    def render_prometheus(self) -> str:
        values = self.snapshot()
        return "".join(
            f"# TYPE {name} gauge\n{name} {value:g}\n"
            for name, value in values.items()
        )


class NvApiPStateBackend:
    """Small direct NVAPI bridge; no nvidia-pstated process is required."""

    _MAX_GPUS = 64
    _QUERY_INTERFACE = {
        "enum": 0xE5AC921F,
        "get_bus": 0x1BE0B8E5,
        "set_pstate": 0x025BFB10,
        "error_message": 0x6C2D048C,
        "initialize": 0x0150E828,
        "unload": 0xD22BDD7E,
    }

    def __init__(self, gpu_ids: str) -> None:
        library = os.environ.get("FASTLLM_PSTATE_NVAPI_LIB", "libnvidia-api.so.1")
        self._lib = ctypes.CDLL(library)
        query_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)
        query_address = getattr(self._lib, "nvapi_QueryInterface")
        query_address.argtypes = [ctypes.c_uint32]
        query_address.restype = ctypes.c_void_p

        def function(name, result, arguments):
            address = query_address(self._QUERY_INTERFACE[name])
            if not address:
                raise RuntimeError(f"NVAPI function {name} is unavailable")
            return ctypes.CFUNCTYPE(result, *arguments)(address)

        self._initialize = function("initialize", ctypes.c_int, [])
        self._unload = function("unload", ctypes.c_int, [])
        self._enum = function(
            "enum",
            ctypes.c_int,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32)],
        )
        self._get_bus = function(
            "get_bus", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        )
        self._set_pstate = function(
            "set_pstate", ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        )
        self._error_message = function(
            "error_message", ctypes.c_int, [ctypes.c_int, ctypes.c_char_p]
        )
        self._initialized = False
        status = self._initialize()
        if status != 0:
            raise RuntimeError(f"NvAPI_Initialize failed with status {status}")
        self._initialized = True
        try:
            self._handles = self._resolve_handles(gpu_ids)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _gpu_bus_ids(gpu_ids: str) -> list[int]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id",
            "--format=csv,noheader,nounits",
            "-i",
            gpu_ids,
        ]
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=3.0
        )
        if result.returncode != 0:
            raise RuntimeError("nvidia-smi could not map the requested GPUs")
        buses: list[int] = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 2:
                continue
            pci = fields[1]
            try:
                buses.append(int(pci.split(":")[1], 16))
            except (IndexError, ValueError):
                continue
        if not buses:
            raise RuntimeError("nvidia-smi returned no PCI bus IDs")
        return buses

    def _resolve_handles(self, gpu_ids: str) -> list[ctypes.c_void_p]:
        handles = (ctypes.c_void_p * self._MAX_GPUS)()
        count = ctypes.c_uint32()
        status = self._enum(handles, ctypes.byref(count))
        if status != 0:
            raise RuntimeError(f"NvAPI_EnumPhysicalGPUs failed with status {status}")
        by_bus: dict[int, ctypes.c_void_p] = {}
        for index in range(min(count.value, self._MAX_GPUS)):
            bus = ctypes.c_uint32()
            status = self._get_bus(handles[index], ctypes.byref(bus))
            if status == 0:
                by_bus[bus.value] = handles[index]
        selected = []
        for bus in self._gpu_bus_ids(gpu_ids):
            if bus not in by_bus:
                raise RuntimeError(f"NVAPI has no handle for PCI bus 0x{bus:02x}")
            selected.append(by_bus[bus])
        return selected

    def set_pstate(self, pstate: int) -> None:
        for handle in self._handles:
            status = self._set_pstate(handle, int(pstate), 0)
            if status != 0:
                message = ctypes.create_string_buffer(64)
                try:
                    self._error_message(status, message)
                    detail = message.value.decode("ascii", "replace")
                except Exception:
                    detail = ""
                raise RuntimeError(
                    f"NvAPI_GPU_SetForcePstate({pstate}) failed: {status} {detail}"
                )

    def close(self) -> None:
        if self._initialized:
            try:
                self._unload()
            finally:
                self._initialized = False


class PStateController:
    """Switch between driver-auto and an idle P8-capable state range."""

    def __init__(
        self,
        config: PStateConfig,
        sequence_provider: Callable[[], int | None] | None = None,
    ) -> None:
        self.config = config
        self._sequence_provider = sequence_provider
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._backend: NvApiPStateBackend | None = None
        self._mode: str | None = None
        self._mode_lock = threading.Lock()
        self._activity_lock = threading.Lock()
        self._last_activity = time.monotonic()
        self._activity_generation = 0
        self._idle_since: float | None = None
        self._last_total: float | None = None
        self._warned_metrics = False
        self._warned_util = False
        self._started = False
        self._ready = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def mode(self) -> str | None:
        return self._mode

    @property
    def ready(self) -> bool:
        return self._ready

    def mark_ready(self) -> None:
        """Allow structured P-State events after Uvicorn announces readiness."""

        self._ready = True

    def _structured_event(self, event: str, fields: str) -> str:
        seq = next_observation_sequence(self._sequence_provider)
        return (
            f"[fastllm-event] seq={seq} "
            f"timestamp_monotonic_ns={time.monotonic_ns()} "
            "source=python category=pstate request=controller native_handle=none "
            f"event={event} {fields}"
        )

    def pstate_for_mode(self, mode: str) -> int:
        if mode not in ("active", "idle"):
            raise ValueError(f"unknown P-State mode: {mode}")
        return self.config.high if mode == "active" else self.config.low

    def start(self) -> bool:
        if self._started:
            return True
        try:
            self._backend = NvApiPStateBackend(self.config.gpu_ids)
            self._set_mode("active")
        except Exception as exc:
            self._backend = None
            message = f"P-State disabled: direct NVAPI initialization failed: {exc}"
            if self.config.required:
                raise RuntimeError(message) from exc
            logger.warning(message)
            return False
        if self._mode != "active":
            if self.config.required:
                raise RuntimeError("P-State controller could not enter active mode")
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="fastllm-pstate",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        if not self._ready or _observability_mode() == "off":
            logger.info(
                "FastLLM P-State controller enabled: gpus=%s idle=%ss confirm=%ss util<%s%%",
                self.config.gpu_ids,
                self.config.idle_timeout_s,
                self.config.confirm_s,
                self.config.util_threshold,
            )
        else:
            logger.info(self._structured_event(
                "pstate_enabled",
                "gpus=%s idle=%ss confirm=%ss util<%s%%" % (
                    self.config.gpu_ids,
                    self.config.idle_timeout_s,
                    self.config.confirm_s,
                    self.config.util_threshold,
                )))
        return True

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=max(2.0, self.config.poll_s + 1.0))
        with self._mode_lock:
            backend, self._backend = self._backend, None
            if backend is not None:
                try:
                    backend.set_pstate(self.config.high)
                finally:
                    backend.close()
            self._mode = None
        self._started = False
        self._ready = False

    def notify_request(self) -> None:
        """Reset idle state and synchronously restore driver-auto mode."""

        with self._activity_lock:
            self._last_activity = time.monotonic()
            self._activity_generation += 1
            self._idle_since = None
        # Taking the mode lock waits out an idle switch already in flight, which
        # would otherwise leave the GPUs in P8 under this request.
        with self._mode_lock:
            if self._mode == "idle":
                self._apply_mode("active")

    def _set_mode(self, mode: str) -> None:
        with self._mode_lock:
            self._apply_mode(mode)

    def _apply_mode(self, mode: str) -> None:
        """Caller holds _mode_lock."""

        if self._mode == mode:
            return
        if self._backend is None:
            raise RuntimeError("P-State backend is not initialized")
        pstate = self.pstate_for_mode(mode)
        self._backend.set_pstate(pstate)
        self._mode = mode
        if not self._ready or _observability_mode() == "off":
            logger.info("FastLLM P-State mode=%s pstate=%s", mode, pstate)
        else:
            logger.info(self._structured_event(
                "pstate_mode", "mode=%s pstate=%s" % (mode, pstate)))

    def _run(self) -> None:
        while not self._stop.wait(self.config.poll_s):
            self._sample()

    def _sample(self) -> None:
        metrics = self._fetch_metrics()
        if metrics is None:
            if not self._warned_metrics:
                self._warned_metrics = True
                logger.warning("FastLLM P-State waiting for %s", self.config.metrics_url)
            return

        total = metrics.get("fastllm_requests_total")
        active = metrics.get("fastllm_active_requests", 0.0)
        if total is None:
            return
        now = time.monotonic()
        with self._activity_lock:
            if self._last_total is not None and total != self._last_total:
                self._last_activity = now
                self._activity_generation += 1
                self._idle_since = None
            self._last_total = total
            if active > 0:
                self._last_activity = now
                self._activity_generation += 1
                self._idle_since = None
            last_activity = self._last_activity
            idle_since = self._idle_since
            activity_generation = self._activity_generation
        # Do not depend on the utilization read below: it can fail under load.
        if active > 0 and self._mode == "idle":
            self._set_mode("active")

        util = self._query_gpu_utilization()
        if util is None:
            if not self._warned_util:
                self._warned_util = True
                logger.warning("FastLLM P-State cannot read GPU utilization; treating GPU as busy")
            return
        if util >= self.config.util_threshold:
            with self._activity_lock:
                self._last_activity = now
                self._activity_generation += 1
                self._idle_since = None
            if self._mode != "active":
                self._set_mode("active")
            return

        if now - last_activity < self.config.idle_timeout_s:
            return
        if idle_since is None:
            with self._activity_lock:
                if self._idle_since is None:
                    self._idle_since = now
            return
        if now - idle_since >= self.config.confirm_s:
            # A request can arrive while the GPU query is running. Recheck the
            # activity generation before changing the mode so an old poll
            # cannot undo a just-completed idle->active notification. The mode
            # lock spans the recheck and the switch, as notify_request reads
            # the mode under it after publishing its activity.
            with self._mode_lock:
                with self._activity_lock:
                    if (
                        activity_generation != self._activity_generation
                        or self._idle_since is None
                        or time.monotonic() - self._last_activity < self.config.idle_timeout_s
                    ):
                        return
                self._apply_mode("idle")

    def _fetch_metrics(self) -> dict[str, float] | None:
        try:
            request = urllib.request.Request(
                self.config.metrics_url,
                headers={_PSTATE_POLL_HEADER: _PSTATE_POLL_TOKEN},
            )
            with urllib.request.urlopen(request, timeout=2.0) as response:
                return parse_metrics(response.read().decode("utf-8", "replace"))
        except (OSError, ValueError):
            return None

    def _query_gpu_utilization(self) -> int | None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            self.config.gpu_ids,
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        values: list[int] = []
        for line in result.stdout.splitlines():
            try:
                values.append(int(line.strip().split()[0]))
            except (IndexError, ValueError):
                continue
        return max(values) if values else None
