"""Input admission must not block HTTP or share pending per-request state."""
import asyncio
import contextvars
import copy
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from test_qwen35_reasoning import FakeQwen35Model, RawRequest, completion, request
from tools.fastllm_pytools.openai_server import fastllm_completion as module
from tools.fastllm_pytools.openai_server.protocal.anthropic_protocol import (
    AnthropicMessageRequest,
)


class InputPreparationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.patch = patch.object(module, "_input_preparation_executor", self.executor)
        self.patch.start()
        self.release = threading.Event()

    async def asyncTearDown(self):
        self.release.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.patch.stop()

    async def wait_event(self, event):
        async def wait():
            while not event.is_set():
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait(), 2)

    async def test_text_tools_prepare_off_loop_with_request_local_state(self):
        loop_thread = threading.get_ident()
        local = threading.local()
        launched = []
        case = self
        trace = contextvars.ContextVar("request_trace")

        class Model(FakeQwen35Model):
            def get_input_token_len(self, messages, **kwargs):
                case.assertNotEqual(threading.get_ident(), loop_thread)
                local.input = copy.deepcopy((messages, kwargs))
                return 17

            def launch_stream_response(self, messages, **kwargs):
                counted_messages, counted = local.input
                case.assertEqual(messages, counted_messages)
                for key in ("tools", "enable_thinking", "chat_template_kwargs"):
                    case.assertEqual(counted.get(key), kwargs.get(key))
                case.assertEqual(trace.get(), messages[-1]["content"])
                constraint = kwargs["tool_call_constraint"]
                case.assertEqual(constraint, {"request": messages[-1]["content"]})
                launched.append(messages[-1]["content"])
                return len(launched)

        instance = completion(Model())
        # Keep the native descriptor request-specific to detect cross-request
        # swaps, without requiring a native library in this API regression.
        def attach(kwargs, req):
            kwargs["tool_call_constraint"] = {"request": req.messages[-1]["content"]}
        instance._attach_tool_call_constraint_if_supported = attach

        async def send(index, anthropic):
            text = str(index)
            token = trace.set(text)
            tool = {"name": "tool_" + text, "parameters": {"type": "object"}}
            try:
                if anthropic:
                    req = AnthropicMessageRequest(
                        model="qwen3.5", max_tokens=1, stream=True,
                        messages=[{"role": "user", "content": text}],
                        tools=[{"name": tool["name"], "input_schema": tool["parameters"]}],
                        thinking={"type": "enabled" if (index // 2) % 2 else "disabled"})
                    result = await instance.create_anthropic_message(req, RawRequest())
                else:
                    req = request(messages=[{"role": "user", "content": text}],
                                  stream=True, tools=[{"type": "function", "function": tool}])
                    result = await instance.create_chat_completion(req, RawRequest())
                self.assertIsInstance(result, tuple)
                await result[0].aclose()
            finally:
                trace.reset(token)

        await asyncio.gather(*(send(i, i % 2 == 0) for i in range(32)))
        self.assertCountEqual(launched, [str(i) for i in range(32)])
        self.assertEqual(len(instance.conversation_handles), 32)

    async def test_input_preparation_does_not_wait_for_token_fetch_executor(self):
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        fetching = threading.Event()

        def fetch():
            fetching.set()
            self.release.wait(3)

        blocked_fetch = loop.run_in_executor(None, fetch)
        await self.wait_event(fetching)
        try:
            result = await asyncio.wait_for(
                completion()._prepare_text_request(lambda: (17, 101)), .5)
            self.assertEqual(result, (17, 101))
        finally:
            self.release.set()
            await blocked_fetch

    async def test_cancel_running_preparation_aborts_late_handle(self):
        started, aborted = threading.Event(), threading.Event()
        instance = completion()
        handles = []

        def prepare():
            started.set()
            self.release.wait(3)
            return 17, 102

        def abort(handle):
            handles.append(handle)
            aborted.set()

        instance.model.abort_handle = abort
        task = asyncio.create_task(instance._prepare_text_request(prepare))
        await self.wait_event(started)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(aborted.is_set())
        self.release.set()
        await self.wait_event(aborted)
        self.assertEqual(handles, [102])
        self.assertEqual(instance.conversation_handles, {})

    async def test_cancel_queued_preparation_does_not_launch(self):
        occupied = [threading.Event(), threading.Event()]

        def occupy(event):
            event.set()
            self.release.wait(3)

        blockers = [self.executor.submit(occupy, event) for event in occupied]
        for event in occupied:
            await self.wait_event(event)
        called = threading.Event()
        task = asyncio.create_task(completion()._prepare_text_request(called.set))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.release.set()
        for blocker in blockers:
            blocker.result()
        self.assertFalse(called.is_set())

    async def test_preparation_errors_propagate(self):
        def invalid():
            raise ValueError("invalid template")
        with self.assertRaisesRegex(ValueError, "invalid template"):
            await completion()._prepare_text_request(invalid)


if __name__ == "__main__":
    unittest.main()
