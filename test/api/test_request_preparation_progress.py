"""CPU-only preparation/registration ownership checks through production APIs."""
import asyncio
import ast
from pathlib import Path
import types
import unittest

from test_qwen35_reasoning import FakeQwen35Model, RawRequest, completion, request
from tools.fastllm_pytools.request_progress import RequestPhaseReporter, _PreparationHeartbeatMonitor
from tools.fastllm_pytools.openai_server.protocal.anthropic_protocol import AnthropicMessageRequest


class PreparationProgressIntegrationTest(unittest.IsolatedAsyncioTestCase):
    def instrument(self, instance):
        now = [0.0]
        events = []
        reporters = []
        monitor = _PreparationHeartbeatMonitor(clock=lambda: now[0], start_thread=False)

        def make_reporter(request_id):
            reporter = RequestPhaseReporter(
                request_id, emit=events.append, mode="user", clock=lambda: now[0], monitor=monitor)
            reporters.append(reporter)
            return reporter

        instance._new_preparation_reporter = make_reporter
        return now, events, reporters, monitor

    async def test_image_loading_has_public_id_and_handover_stops_python_timer(self):
        instance = completion(FakeQwen35Model())
        now, events, reporters, monitor = self.instrument(instance)
        registered = []

        def decode(image_bytes):
            self.assertEqual(image_bytes, b"x")
            current = RequestPhaseReporter.current()
            self.assertIs(current, reporters[0])
            self.assertTrue(current.reporter.request_id.startswith("fastllm-qwen3.5-"))
            self.assertEqual(current.phase, "image_preprocessing")
            now[0] = 1.0
            monitor.poll()
            self.assertIn("detail=loading_and_decoding_image", events[-1])
            return object()

        def register(handle, request_id):
            registered.append((handle, request_id))
            self.assertEqual(monitor.active, {})
            self.assertFalse(reporters[0].active)
            count = len(events)
            now[0] += 1.0
            monitor.poll()
            self.assertEqual(len(events), count)

        instance._load_image_from_bytes = decode
        instance.model.register_request_observation = register
        result = await instance.create_chat_completion(request(stream=True, messages=[{
            "role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
                {"type": "text", "text": "describe"},
            ],
        }]), RawRequest())
        self.assertEqual(registered, [(101, reporters[0].reporter.request_id)])
        self.assertEqual(monitor.active, {})
        self.assertIsNone(RequestPhaseReporter.current())
        generator, _ = result
        await generator.aclose()

    async def test_release_uses_public_id_and_preserves_recycled_handle(self):
        instance = completion()
        calls = []
        native_ids = {101: "new-request"}

        def unregister_by_id(handle, request_id):
            calls.append((handle, request_id))
            if native_ids.get(handle) == request_id:
                native_ids.pop(handle)

        def legacy_unregister(handle):
            self.fail("guarded-capable model used unsafe legacy unregister")

        instance.model.unregister_request_observation_by_id = unregister_by_id
        instance.model.unregister_request_observation = legacy_unregister
        instance.conversation_handles = {"old-request": 101, "new-request": 101}
        self.assertTrue(instance._release_conversation_handle("old-request", 101))
        self.assertEqual(native_ids, {101: "new-request"})
        self.assertEqual(instance.conversation_handles, {"new-request": 101})
        self.assertFalse(instance._release_conversation_handle("old-request", 101))
        self.assertTrue(instance._release_conversation_handle("new-request", 101))
        self.assertEqual(native_ids, {})
        self.assertEqual(calls, [(101, "old-request"), (101, "new-request")])

    async def test_release_preserves_legacy_fake_model_api(self):
        instance = completion()
        calls = []
        instance.model.unregister_request_observation = calls.append
        instance.conversation_handles = {"request": 101}
        self.assertTrue(instance._release_conversation_handle("request", 101))
        self.assertEqual(calls, [101])

    async def test_real_model_wrapper_calls_guarded_abi_without_unsafe_fallback(self):
        # Extract these pure Python wrapper methods without loading lib/CUDA.
        source = Path(__file__).resolve().parents[2] / "tools/fastllm_pytools/llm.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        names = {"unregister_request_observation", "unregister_request_observation_by_id"}
        methods = [node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in methods}, names)
        calls = []
        namespace = {
            "_unregister_request_observation_by_id": lambda *args: calls.append(("guarded", args)),
            "_unregister_request_observation": lambda *args: calls.append(("legacy", args)),
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
        model = types.SimpleNamespace(model=7)
        model.unregister_request_observation_by_id = types.MethodType(
            namespace["unregister_request_observation_by_id"], model)
        unregister = types.MethodType(namespace["unregister_request_observation"], model)
        unregister(101, "public-request")
        self.assertEqual(calls, [("guarded", (7, 101, b"public-request"))])
        calls.clear()
        namespace["_unregister_request_observation_by_id"] = None
        unregister(101, "public-request")
        self.assertEqual(calls, [])
        unregister(101)
        self.assertEqual(calls, [("legacy", (7, 101))])

    async def test_input_preparation_failure_cleans_timer_and_preserves_error(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                instance = completion()
                _, events, reporters, monitor = self.instrument(instance)

                def fail(*args, **kwargs):
                    raise ValueError("invalid image preprocessing")

                instance._compute_multimodal_input_token_len = fail
                if api == "openai":
                    response = await instance.create_chat_completion(request(), RawRequest())
                    self.assertTrue(reporters[0].reporter.request_id.startswith("fastllm-qwen3.5-"))
                else:
                    response = await instance.create_anthropic_message(AnthropicMessageRequest(
                        model="qwen3.5", max_tokens=128,
                        messages=[{"role": "user", "content": "describe"}]), RawRequest())
                    self.assertTrue(reporters[0].reporter.request_id.startswith("msg_"))
                self.assertIn("invalid image preprocessing", response.message)
                self.assertEqual(monitor.active, {})
                self.assertFalse(reporters[0].active)
                self.assertIsNone(RequestPhaseReporter.current())
                self.assertFalse(instance.conversation_handles)
                self.assertTrue(events)

    async def test_cancelled_preparation_cleans_timer_before_native_launch(self):
        instance = completion()
        _, _, reporters, monitor = self.instrument(instance)
        entered = asyncio.Event()

        async def pending_model_check(_):
            entered.set()
            await asyncio.Event().wait()

        instance._check_model = pending_model_check
        task = asyncio.create_task(instance.create_chat_completion(request(), RawRequest()))
        await entered.wait()
        self.assertEqual(len(monitor.active), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(monitor.active, {})
        self.assertFalse(reporters[0].active)
        self.assertFalse(instance.conversation_handles)
        self.assertIsNone(RequestPhaseReporter.current())


if __name__ == "__main__":
    unittest.main()
