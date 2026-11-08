"""Production reporter and API lifecycle regressions; CPU only, no native library."""
import asyncio
import ast
import ctypes
import io
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools/fastllm_pytools"))
from request_progress import RequestProgressReporter
from terminal_colorizer import UserTerminalView, process_lines, strip_ansi
from test_qwen35_reasoning import FakeQwen35Model, RawRequest, completion, request
from tools.fastllm_pytools import request_progress as api_progress

READY = "INFO:     Uvicorn running on http://0.0.0.0:8080\n"
NATIVE = {"phase": "prefill", "native_status_state": "observed",
          "native_generation": "91", "native_rate_state": "pending",
          "prefill_total_tokens": "4096", "l1_capacity_state": "ok",
          "l1_used_tokens": "128", "l1_available_tokens": "896",
          "l1_total_tokens": "1024", "l2_capacity_state": "ok",
          "l2_available_gib": "16.000", "l2_limit_gib": "16.000",
          "ssd_capacity_state": "ready", "ssd_available_gib": "239.780",
          "ssd_limit_gib": "256.000"}

class ReporterEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.resource_patch = patch("request_progress.resource_snapshot", return_value={})
        self.resource_patch.start()
        self.addCleanup(self.resource_patch.stop)

    def test_release_then_finish_cannot_invent_cache_failure_or_query_reused_handle(self):
        events, calls = [], []
        def provider():
            calls.append(1)
            return dict(NATIVE)
        reporter = RequestProgressReporter("req-release", emit=events.append, mode="debug",
                                           status_provider=provider)
        reporter.close_native_observation("stream_exhausted")
        reporter.finish(status="ok", output_tokens=5)
        reporter.maybe(force=True)
        self.assertEqual(len(calls), 1)
        status = [e for e in events if "event=request_status " in e]
        self.assertEqual(len(status), 1)
        self.assertIn("snapshot_kind=native_release", status[0])
        self.assertFalse(any("capacity_state=unavailable" in e for e in events))
        self.assertTrue(reporter._pump_stop.is_set())
        self.assertEqual(sum("event=request_finished " in e for e in events), 1)

    def test_missing_observer_is_not_a_cache_failure(self):
        events = []
        reporter = RequestProgressReporter("req-missing", emit=events.append, mode="debug",
                                           status_provider=lambda: None)
        reporter.maybe(force=True)
        row = next(e for e in events if "event=request_status " in e)
        self.assertIn("native_status_state=observer_missing_or_owner_changed", row)
        self.assertNotIn("capacity_state", row)
        self.assertNotIn("phase=unknown", row)
        self.assertIsNone(UserTerminalView().line(row))

    def test_query_exception_preserves_error_without_fabricated_capacity(self):
        def fail():
            raise RuntimeError("status query failed")
        events = []
        reporter = RequestProgressReporter("req-error", emit=events.append, mode="debug",
                                           status_provider=fail)
        reporter.maybe(force=True)
        row = next(e for e in events if "event=request_status " in e)
        self.assertIn("native_status_state=query_failed", row)
        self.assertNotIn("capacity_state", row)
        error = next(e for e in events if "event=request_error " in e)
        self.assertIn("error_type=RuntimeError", error)
        self.assertIn("status query failed", unquote(error))

    def test_concurrent_finish_has_one_terminal_event(self):
        events = []
        reporter = RequestProgressReporter("req-race", emit=events.append, mode="debug")
        barrier = threading.Barrier(3)
        def finish():
            barrier.wait()
            reporter.finish(status="cancelled")
        threads = [threading.Thread(target=finish) for _ in range(2)]
        for t in threads:
            t.start()
        barrier.wait()
        for t in threads:
            t.join(2)
            self.assertFalse(t.is_alive())
        self.assertEqual(sum("event=request_finished " in e for e in events), 1)
        self.assertTrue(reporter._pump_stop.is_set())

    def test_pump_exception_records_failure_and_stops(self):
        events = []
        reporter = RequestProgressReporter("req-pump", emit=events.append, mode="debug",
                                           interval_s=0.01)
        with patch.object(reporter, "maybe", side_effect=RuntimeError("pump failure")):
            reporter.start()
            reporter._pump.join(1)
        self.assertFalse(reporter._pump.is_alive())
        self.assertTrue(reporter._pump_stop.is_set())
        self.assertTrue(any("stage=progress_pump" in e for e in events))
        self.assertTrue(any("pump failure" in unquote(e) for e in events))

    def test_user_timestamp_copies_native_snapshot_not_logger_emit_time(self):
        row = ("2026-10-20 22:53:09,875 logger INFO: [fastllm-event] "
               "seq=1 timestamp_monotonic_ns=5 timestamp_utc=2026-10-20T14:53:09.874Z "
               "source=python category=progress request_id=req-stamp native_handle=0 "
               "event=request_status phase=prefill native_status_state=observed "
               "snapshot_timestamp_utc=2026-10-20T14:53:09.870000000Z native_rate_state=pending\n")
        visible = UserTerminalView().line(row)
        self.assertIn("[2026-10-20T14:53:09.870000000Z]", visible)
        self.assertNotIn("22:53:09,875", visible)
        log, raw, tty = io.StringIO(), io.StringIO(), io.StringIO()
        process_lines([READY, row], log, tty, "user", raw)
        self.assertEqual(raw.getvalue(), READY + row)
        self.assertEqual(log.getvalue(), raw.getvalue())

    def test_final_capacity_uses_final_real_sample_and_ssd_matching_window(self):
        events = []
        reporter = RequestProgressReporter("req-final", emit=events.append, mode="debug",
                                           status_provider=lambda: dict(NATIVE))
        reporter.close_native_observation("done")
        row = next(e for e in events if "event=request_status " in e)
        view = UserTerminalView()
        self.assertIn("Cache capacity", view.line(row))
        self.assertIn("Cache capacity", view.line(row))
        row = ("[fastllm-event] seq=2 timestamp_utc=2026-10-20T13:00:00Z "
               "source=native category=cache request_id=req-final native_handle=0 "
               "event=cache_result source_tier=L3 promoted_to=L1 effective_reused_tokens=131072 "
               "restore_ms=3896.203 import_ms=3896.203 restore_total_ms=6273.189 "
               "read_mib_s=1783.084 restore_tokens_s=20893.999\n")
        visible = view.line(row)
        self.assertIn("total restore time: 6273.189 ms", visible)
        self.assertNotIn("3896.203", visible)

class LifecycleModel(FakeQwen35Model):
    def __init__(self, output="answer"):
        super().__init__(output)
        self.aborts, self.unregisters = [], []
    def abort_handle(self, handle):
        self.aborts.append(handle)
    def unregister_request_observation_scoped(self, handle, public_id):
        self.unregisters.append((handle, public_id))
    def get_request_status(self, handle, public_id):
        return dict(NATIVE)

class CompletionLifecycleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events, self.reporters = [], []
        self.mode = patch.dict(os.environ, {"FASTLLM_OBSERVABILITY_MODE": "debug"})
        self.resources = patch.object(api_progress, "resource_snapshot", return_value={})
        self.mode.start()
        self.resources.start()
        self.addCleanup(self.mode.stop)
        self.addCleanup(self.resources.stop)

    def instance(self, model=None):
        instance = completion(model or LifecycleModel())
        original = instance._new_request_reporter
        def factory(request_id, handle):
            reporter = original(request_id, handle)
            reporter.emit = self.events.append
            self.reporters.append(reporter)
            return reporter
        instance._new_request_reporter = factory
        return instance

    async def assert_stopped(self, instance):
        self.assertEqual(getattr(instance, "_request_reporters", {}), {})
        for reporter in self.reporters:
            self.assertTrue(reporter._pump_stop.is_set())
            if reporter._pump:
                reporter._pump.join(1)
                self.assertFalse(reporter._pump.is_alive())
        self.assertFalse(any("capacity_state=unavailable" in e for e in self.events))

    async def test_success_full_releases_observer_and_stops_reporter(self):
        instance = self.instance()
        await instance.create_chat_completion(request(stream=False), RawRequest())
        await self.assert_stopped(instance)
        self.assertEqual(len(instance.model.unregisters), 1)
        self.assertEqual(instance.model.aborts, [])
        self.assertEqual(sum("event=request_finished " in e for e in self.events), 1)

    async def test_stream_generator_close_aborts_once_and_stops_reporter(self):
        instance = self.instance()
        generator, background = await instance.create_chat_completion(request(stream=True), RawRequest())
        await anext(generator)
        await generator.aclose()
        await background()
        await self.assert_stopped(instance)
        self.assertEqual(instance.model.aborts, [101])
        self.assertEqual(len(instance.model.unregisters), 1)
        done = [e for e in self.events if "event=request_finished " in e]
        self.assertEqual(len(done), 1)
        self.assertIn("status=cancelled", done[0])

    async def test_full_disconnect_has_terminal_disconnected_and_no_reporter_leak(self):
        class Disconnected(RawRequest):
            async def is_disconnected(self):
                return True
        instance = self.instance()
        result = await instance.create_chat_completion(request(stream=False), Disconnected())
        await self.assert_stopped(instance)
        self.assertEqual(result.message, "Client disconnected")
        self.assertEqual(instance.model.aborts, [101])
        self.assertTrue(any("status=disconnected" in e for e in self.events))

    async def test_general_exception_has_original_error_terminal_and_cleanup(self):
        class Broken(LifecycleModel):
            def stream_response_handle_async(self, handle):
                async def generate():
                    yield "part"
                    raise RuntimeError("backend exploded")
                return generate()
        instance = self.instance(Broken())
        with self.assertRaisesRegex(RuntimeError, "backend exploded"):
            await instance.create_chat_completion(request(stream=False), RawRequest())
        await self.assert_stopped(instance)
        self.assertEqual(instance.model.aborts, [101])
        self.assertTrue(any("error_type=RuntimeError" in e for e in self.events))
        self.assertTrue(any("event=request_finished " in e and "status=error" in e for e in self.events))

    async def test_actual_native_terminal_tail_cancel_releases_owner_before_yield(self):
        # Extract the actual production method, replacing only native calls and
        # tokenizer. This reproduces the post--1 buffered tail, not a mock of
        # the API cleanup policy.
        tree = ast.parse((REPO / "tools/fastllm_pytools/llm.py").read_text())
        method = next(n for owner in tree.body if isinstance(owner, ast.ClassDef) and owner.name == "model"
                      for n in owner.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "stream_response_handle_async")
        class Native:
            calls = 0
            owner = None
            def fetch_response_tokens_batch_llm_model(self, model_id, handle, buffer, capacity):
                self.calls += 1
                if self.calls == 1:
                    buffer[0] = 123
                    return 1
                self.owner.native_released = True
                self.owner.instance.conversation_handles["new-request"] = handle
                return -1
        native = Native()
        namespace = {"asyncio": asyncio, "ctypes": ctypes, "fastllm_lib": native}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     "actual-stream-method", "exec"), namespace)
        class Tokenizer:
            def decode(self, tokens):
                return "\ufffd"
        class TailModel(LifecycleModel):
            stream_response_handle_async = namespace["stream_response_handle_async"]
            def __init__(self):
                super().__init__()
                self.model = 123
                self.save_history = False
                self.hf_tokenizer = Tokenizer()
                self.native_released = False
                self.stats_after_release = 0
            def _can_apply_hf_chat_template(self):
                return True
            def _uses_hf_deepseek_v4_tokenizer(self):
                return False
            def get_response_statistics(self, handle):
                if self.native_released:
                    self.stats_after_release += 1
                return {"output_tokens": 1, "cached_input_tokens": 0, "missed_input_tokens": 17}
        model = TailModel()
        native.owner = model
        instance = self.instance(model)
        instance.enable_thinking = False
        model.instance = instance
        generator, background = await instance.create_chat_completion(request(stream=True), RawRequest())
        tail_seen = False
        for _ in range(5):
            chunk = await anext(generator)
            if "\ufffd" in chunk:
                tail_seen = True
                break
        self.assertTrue(tail_seen)
        self.assertTrue(model.native_released)
        self.assertEqual(instance.conversation_handles, {"new-request": 101})
        await generator.aclose()
        await background()
        self.assertEqual(model.aborts, [])
        self.assertEqual(model.stats_after_release, 0)
        self.assertEqual(instance.conversation_handles, {"new-request": 101})
        await self.assert_stopped(instance)

    async def test_native_capacity_error_does_not_abort_reused_handle_in_either_api(self):
        from tools.fastllm_pytools.generation_errors import CapacityUnavailableError
        from tools.fastllm_pytools.openai_server.protocal.anthropic_protocol import AnthropicMessageRequest
        class ReusedCapacity(LifecycleModel):
            instance = None
            def stream_response_handle_async(self, handle):
                async def generate():
                    self.instance.conversation_handles["new-request"] = handle
                    raise CapacityUnavailableError()
                    yield ""
                return generate()
        for api in ("openai", "anthropic"):
            for streaming in (False, True):
                with self.subTest(api=api, stream=streaming):
                    self.events.clear()
                    self.reporters.clear()
                    model = ReusedCapacity()
                    instance = self.instance(model)
                    model.instance = instance
                    if api == "openai":
                        response = await instance.create_chat_completion(request(stream=streaming), RawRequest())
                    else:
                        response = await instance.create_anthropic_message(
                            AnthropicMessageRequest(model="qwen3.5", max_tokens=128,
                                stream=streaming, messages=[{"role":"user","content":"answer"}]), RawRequest())
                    if streaming:
                        generator, background = response
                        chunks = [chunk async for chunk in generator]
                        await background()
                        decoded_frames = []
                        for chunk in chunks:
                            for line in chunk.splitlines():
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    decoded_frames.append(json.loads(line[6:]))
                        self.assertIn("缓存容量不足", json.dumps(decoded_frames, ensure_ascii=False))
                    else:
                        self.assertIn("缓存容量不足", response.message)
                    self.assertEqual(model.aborts, [])
                    self.assertEqual(instance.conversation_handles, {"new-request": 101})
                    await self.assert_stopped(instance)

    async def test_stream_value_error_finishes_error_without_success_fallthrough(self):
        class Broken(LifecycleModel):
            def stream_response_handle_async(self, handle):
                async def generate():
                    yield "part"
                    raise ValueError("bad native request")
                return generate()
        instance = self.instance(Broken())
        generator, background = await instance.create_chat_completion(request(stream=True), RawRequest())
        chunks = [chunk async for chunk in generator]
        await background()
        await self.assert_stopped(instance)
        self.assertTrue(any("bad native request" in c for c in chunks))
        done = [e for e in self.events if "event=request_finished " in e]
        self.assertEqual(len(done), 1)
        self.assertIn("status=error", done[0])
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

if __name__ == "__main__":
    unittest.main()
