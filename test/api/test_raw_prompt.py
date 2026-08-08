import argparse
import asyncio
import ctypes
import importlib
import os
import sys
import threading
import unittest
from unittest import mock


TEST_API_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path = [path for path in sys.path
            if os.path.abspath(path or os.getcwd()) != TEST_API_DIR]
TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.openai_server.fastllm_completion import FastLLmCompletion
from fastllm_pytools.openai_server.protocal.openai_protocol import (
    ChatCompletionRequest,
    ErrorResponse,
)


class FakeRawModel:
    def __init__(self):
        self.encode_calls = []
        self.launch_calls = []
        self.template_calls = 0

    def encode(self, prompt):
        self.encode_calls.append(prompt)
        return [11, 22, 33]

    def launch_stream_response(self, prompt, **kwargs):
        self.launch_calls.append((prompt, kwargs))
        return 7

    def apply_chat_template(self, messages):
        self.template_calls += 1
        raise AssertionError("raw prompt must not apply a chat template")


class RawPromptTest(unittest.TestCase):
    def make_completion(self):
        completion = object.__new__(FastLLmCompletion)
        completion.model = FakeRawModel()
        completion.model_name = "test-model"
        completion.conversation_handles = {}
        completion._ensure_handle_tracking()
        return completion

    def test_raw_prompt_requires_prompt_and_no_messages(self):
        completion = self.make_completion()
        request = ChatCompletionRequest(
            model="test-model", raw_prompt=True, prompt="rendered")
        self.assertEqual(completion._raw_prompt_text(request), "rendered")

        for invalid in [
            ChatCompletionRequest(
                model="test-model", raw_prompt=True, prompt=""),
            ChatCompletionRequest(
                model="test-model", raw_prompt=True, prompt="rendered",
                messages=[{"role": "user", "content": "hello"}]),
        ]:
            with self.assertRaises(ValueError):
                completion._raw_prompt_text(invalid)

    def test_raw_prompt_encodes_once_and_reuses_exact_tokens(self):
        completion = self.make_completion()
        input_len, handle = completion._launch_raw_prompt(
            "req-raw", "<rendered>", {"max_length": 64})

        self.assertEqual(input_len, 3)
        self.assertEqual(handle, 7)
        self.assertEqual(completion.model.encode_calls, ["<rendered>"])
        self.assertEqual(completion.model.template_calls, 0)
        self.assertEqual(len(completion.model.launch_calls), 1)
        prompt, kwargs = completion.model.launch_calls[0]
        self.assertEqual(prompt, "<rendered>")
        self.assertTrue(kwargs["raw_prompt"])
        self.assertEqual(kwargs["raw_prompt_tokens"], [11, 22, 33])
        self.assertEqual(kwargs["max_length"], 64)
        self.assertEqual(completion.conversation_handles["req-raw"], 7)

    def test_native_raw_prompt_initializes_stop_token_arguments(self):
        class FakeFunction:
            def __init__(self, name):
                self.name = name
                self.calls = []
                self.argtypes = None
                self.restype = None

            def __call__(self, *args):
                self.calls.append(args)
                return 73 if self.name == "launch_raw_prompt_llm_model" else 0

        class FakeLibrary:
            def __init__(self):
                self.functions = {}

            def __getattr__(self, name):
                if name not in self.functions:
                    self.functions[name] = FakeFunction(name)
                return self.functions[name]

        module_name = "fastllm_pytools.llm"
        previous = sys.modules.pop(module_name, None)
        fake_library = FakeLibrary()
        try:
            with mock.patch.object(ctypes, "CDLL", return_value=fake_library), \
                    mock.patch.object(
                        ctypes.cdll, "LoadLibrary", return_value=fake_library):
                llm_module = importlib.import_module(module_name)
            model = object.__new__(llm_module.model)
            model.model = 9
            model.enable_thinking = True
            model.thread_local_obj = threading.local()
            handle = model.launch_stream_response(
                "rendered", raw_prompt=True,
                raw_prompt_tokens=[11, 22, 33],
                stop_token_ids=[44, 55], max_length=64,
                do_sample=False)
            args = fake_library.functions[
                "launch_raw_prompt_llm_model"].calls[-1]
            self.assertEqual(handle, 73)
            self.assertEqual(args[1], 3)
            self.assertEqual(tuple(args[2]), (11, 22, 33))
            self.assertEqual(args[-2].value, 2)
            self.assertEqual(tuple(args[-1]), (44, 55))
        finally:
            sys.modules.pop(module_name, None)
            if previous is not None:
                sys.modules[module_name] = previous

    def test_handler_rejects_invalid_raw_prompt_before_launch(self):
        completion = self.make_completion()
        completion.default_max_tokens = 16384
        completion.model.default_generation_config = {
            "top_p": 0.8,
            "top_k": 1,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
        }

        class RawRequest:
            async def is_disconnected(self):
                return False

        request = ChatCompletionRequest(
            model="test-model", raw_prompt=True, prompt="rendered",
            messages=[{"role": "user", "content": "hello"}])
        response = asyncio.run(
            completion.create_chat_completion(request, RawRequest()))
        self.assertIsInstance(response, ErrorResponse)
        self.assertEqual(response.code, 400)
        self.assertEqual(completion.model.encode_calls, [])
        self.assertEqual(completion.model.launch_calls, [])


if __name__ == "__main__":
    unittest.main()
