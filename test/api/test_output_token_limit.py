import argparse
import os
import sys
import unittest

from pydantic import ValidationError


TEST_API_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path = [path for path in sys.path
            if os.path.abspath(path or os.getcwd()) != TEST_API_DIR]
TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.openai_server.fastllm_completion import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    FastLLmCompletion,
)
from fastllm_pytools.openai_server.protocal.anthropic_protocol import (
    AnthropicMessageRequest,
)
from fastllm_pytools.openai_server.protocal.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ResponsesRequest,
)
from fastllm_pytools.util import add_server_args


def make_completion(default_max_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    completion = object.__new__(FastLLmCompletion)
    completion.default_max_tokens = default_max_tokens
    return completion


class OutputTokenLimitTest(unittest.TestCase):
    def test_default_and_explicit_output_limits_are_selected_once(self):
        completion = make_completion(16384)
        missing = ChatCompletionRequest(model="test", messages=[])
        explicit = ChatCompletionRequest(
            model="test", messages=[], max_tokens=4096)

        self.assertEqual(
            completion._effective_max_tokens(missing.max_tokens), 16384)
        self.assertEqual(
            completion._effective_max_tokens(explicit.max_tokens), 4096)

        effective_missing = completion._with_effective_max_tokens(
            missing, 16384)
        effective_explicit = completion._with_effective_max_tokens(
            explicit, 4096)
        self.assertEqual(effective_missing.max_tokens, 16384)
        self.assertEqual(effective_explicit.max_tokens, 4096)
        self.assertIsNone(missing.max_tokens)
        self.assertEqual(explicit.max_tokens, 4096)

    def test_effective_limit_keeps_length_finish_reason_truthful(self):
        completion = make_completion(16384)
        self.assertEqual(completion._chat_finish_reason(16383, 16384), "stop")
        self.assertEqual(
            completion._chat_finish_reason(16384, 16384), "length")
        self.assertEqual(
            completion._chat_finish_reason(
                16384, 16384, stopped_by_stop_string=True),
            "stop")

    def test_chat_rejects_invalid_explicit_max_tokens(self):
        for value in [0, -1, 1.5, "16", True, False]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ChatCompletionRequest(
                    model="test", messages=[], max_tokens=value)

    def test_responses_rejects_invalid_explicit_limits(self):
        for field in ["max_tokens", "max_output_tokens"]:
            for value in [0, -1, 1.5, "16", True, False]:
                with self.subTest(field=field, value=value), \
                        self.assertRaises(ValidationError):
                    ResponsesRequest(
                        model="test", input="hello", **{field: value})

    def test_completion_rejects_invalid_explicit_max_tokens(self):
        for value in [0, -1, 1.5, "16", True, False]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                CompletionRequest(
                    model="test", prompt="hello", max_tokens=value)

    def test_anthropic_rejects_invalid_explicit_max_tokens(self):
        for value in [0, -1, 1.5, "16", True, False]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AnthropicMessageRequest(
                    model="test", messages=[], max_tokens=value)

    def test_server_cli_default_and_override(self):
        parser = argparse.ArgumentParser()
        add_server_args(parser)
        self.assertEqual(parser.parse_args([]).default_max_tokens, 16384)
        self.assertEqual(
            parser.parse_args(
                ["--default-max-tokens", "4096"]).default_max_tokens,
            4096)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--default-max-tokens", "0"])


if __name__ == "__main__":
    unittest.main()
