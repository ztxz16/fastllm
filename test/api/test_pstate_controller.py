import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from fastllm_pytools.pstate_controller import (  # noqa: E402
    PStateConfig,
    PStateController,
    RequestActivity,
    parse_metrics,
)


class PStateControllerTest(unittest.TestCase):
    def test_parse_metrics_ignores_comments_and_labels(self):
        metrics = parse_metrics(
            "# HELP fastllm_requests_total requests\n"
            "fastllm_requests_total 7\n"
            "fastllm_active_requests{source=\"api\"} 2\n"
            "invalid value\n"
        )
        self.assertEqual(metrics["fastllm_requests_total"], 7.0)
        self.assertEqual(metrics["fastllm_active_requests"], 2.0)
        self.assertNotIn("invalid", metrics)

    def test_request_activity_metrics(self):
        activity = RequestActivity()
        activity.begin()
        activity.begin()
        activity.end()
        snapshot = activity.snapshot()
        self.assertEqual(snapshot["fastllm_requests_total"], 2.0)
        self.assertEqual(snapshot["fastllm_active_requests"], 1.0)
        rendered = activity.render_prometheus()
        self.assertIn("fastllm_requests_total 2", rendered)
        self.assertIn("fastllm_active_requests 1", rendered)

    def test_config_is_opt_in_and_resolves_visible_gpus(self):
        with patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "2,3",
                "FASTLLM_PSTATE_ENABLE": "1",
                "FASTLLM_PSTATE_IDLE_TIMEOUT_S": "12",
                "FASTLLM_PSTATE_CONFIRM_S": "4",
                "FASTLLM_PSTATE_POLL_S": "3",
                "FASTLLM_PSTATE_UTIL_THRESHOLD": "6",
            },
            clear=False,
        ):
            os.environ.pop("FASTLLM_PSTATE_GPUS", None)
            config = PStateConfig.from_env("http://127.0.0.1:18080/metrics")
        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.gpu_ids, "2,3")
        self.assertEqual(config.idle_timeout_s, 12.0)
        self.assertEqual(config.confirm_s, 4.0)
        self.assertEqual(config.poll_s, 3.0)
        self.assertEqual(config.util_threshold, 6.0)
        self.assertEqual(config.metrics_url, "http://127.0.0.1:18080/metrics")

    def test_approved_defaults_without_timing_overrides(self):
        direct = PStateConfig(gpu_ids="0")
        with patch.dict(os.environ, {
                "FASTLLM_PSTATE_ENABLE": "1",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4"}, clear=True):
            config = PStateConfig.from_env()
        self.assertIsNotNone(config)
        for actual in (direct, config):
            self.assertEqual(
                (actual.idle_timeout_s, actual.confirm_s,
                 actual.poll_s, actual.util_threshold),
                (5.0, 5.0, 5.0, 5.0))
        self.assertEqual(config.gpu_ids, "0,1,2,3,4")

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(PStateConfig.from_env())

    def test_mode_commands(self):
        config = PStateConfig(gpu_ids="0,1", low=8, high=16)
        controller = PStateController(config)
        self.assertEqual(controller.pstate_for_mode("idle"), 8)
        self.assertEqual(controller.pstate_for_mode("active"), 16)

    def test_debug_pstate_events_start_after_ready(self):
        config = PStateConfig(gpu_ids="0", low=8, high=16)
        controller = PStateController(config)
        backend = type("FakeBackend", (), {
            "set_pstate": lambda self, pstate: None,
            "close": lambda self: None,
        })()
        with patch.dict(os.environ, {"FASTLLM_OBSERVABILITY_MODE": "debug"}, clear=False), \
             patch("fastllm_pytools.pstate_controller.NvApiPStateBackend", return_value=backend):
            with self.assertLogs("fastllm_pytools.pstate_controller", level="INFO") as startup_logs:
                self.assertTrue(controller.start())
            self.assertTrue(any("FastLLM P-State controller enabled" in line for line in startup_logs.output))
            controller.mark_ready()
            with self.assertLogs("fastllm_pytools.pstate_controller", level="INFO") as ready_logs:
                controller._set_mode("idle")
            self.assertTrue(any("[fastllm-event] seq=" in line for line in ready_logs.output))
            self.assertTrue(any("timestamp_monotonic_ns=" in line for line in ready_logs.output))
            self.assertTrue(any("native_handle=none" in line for line in ready_logs.output))
        controller.stop()

    def test_idle_mode_returns_to_active_on_request(self):
        class FakeBackend:
            def __init__(self):
                self.pstates = []

            def set_pstate(self, pstate):
                self.pstates.append(pstate)

            def close(self):
                return None

        config = PStateConfig(
            gpu_ids="0",
            idle_timeout_s=0.01,
            confirm_s=0.0,
            util_threshold=5.0,
        )
        controller = PStateController(config)
        backend = FakeBackend()
        with patch("fastllm_pytools.pstate_controller.NvApiPStateBackend", return_value=backend), \
             patch.object(controller, "_query_gpu_utilization", return_value=0), \
             patch.object(controller, "_fetch_metrics", return_value={
                 "fastllm_requests_total": 0.0,
                 "fastllm_active_requests": 0.0,
             }):
            self.assertTrue(controller.start())
            controller._last_activity = time.monotonic() - 1.0
            controller._idle_since = time.monotonic() - 1.0
            controller._sample()
            self.assertEqual(controller.mode, "idle")
            controller.notify_request()
            self.assertEqual(controller.mode, "active")
        controller.stop()

    def test_missing_nvapi_is_fail_safe_by_default(self):
        controller = PStateController(PStateConfig(gpu_ids="0"))
        with patch(
            "fastllm_pytools.pstate_controller.NvApiPStateBackend",
            side_effect=RuntimeError("libnvidia-api.so.1 missing"),
        ):
            self.assertFalse(controller.start())
        self.assertFalse(controller.started)


if __name__ == "__main__":
    unittest.main()
