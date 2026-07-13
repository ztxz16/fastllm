import asyncio
import json
import os
import sys


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
    ResponsesRequest,
)
from fastllm_pytools.openai_server.protocal.anthropic_protocol import (
    AnthropicInputMessage,
    AnthropicMessageRequest,
)


class FakeModel:
    def get_response_statistics(self, handle):
        return {
            "cached_input_tokens": 8,
            "missed_input_tokens": 4,
            "output_tokens": 2,
        }

    def stream_response_handle_async(self, handle, response_statistics=None):
        async def generator():
            if response_statistics is not None:
                response_statistics.update(
                    self.get_response_statistics(handle))
            async for delta in model_deltas():
                yield delta

        return generator()

    def abort_handle(self, handle):
        pass


class LegacyStreamModel(FakeModel):
    def stream_response_handle_async(self, handle):
        return model_deltas()


class RawRequest:
    async def is_disconnected(self):
        return False


async def model_deltas():
    yield "hello"
    yield " world"


def make_completion():
    completion = object.__new__(FastLLmCompletion)
    completion.model = FakeModel()
    completion.model_name = "test-model"
    completion.conversation_handles = {}
    return completion


def assert_openai_usage(usage):
    assert usage["prompt_tokens"] == 12
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 14
    assert usage["prompt_tokens_details"] == {
        "cached_tokens": 8,
        "missed_tokens": 4,
    }


def test_usage_protocol_mappings():
    completion = make_completion()
    chat_usage = completion._chat_usage(1, {}, 99, 99)
    assert_openai_usage(chat_usage.model_dump())

    responses_usage = completion._responses_usage_from_chat_usage(
        chat_usage).model_dump()
    assert responses_usage["input_tokens"] == 12
    assert responses_usage["output_tokens"] == 2
    assert responses_usage["total_tokens"] == 14
    assert responses_usage["input_tokens_details"] == {
        "cached_tokens": 8,
        "missed_tokens": 4,
    }

    anthropic_usage = completion._anthropic_usage(
        1, {}, 99, 99).model_dump()
    assert anthropic_usage == {
        "input_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 8,
        "output_tokens": 2,
    }


def test_stream_adapter_supports_new_and_legacy_models():
    async def collect(completion, response_statistics):
        result = []
        async for delta in completion._stream_response_handle_with_statistics(
                1, response_statistics):
            result.append(delta)
        return result

    completion = make_completion()
    response_statistics = {}
    assert asyncio.run(collect(completion, response_statistics)) == [
        "hello", " world"]
    assert response_statistics == {
        "cached_input_tokens": 8,
        "missed_input_tokens": 4,
        "output_tokens": 2,
    }

    completion.model = LegacyStreamModel()
    response_statistics = {}
    assert asyncio.run(collect(completion, response_statistics)) == [
        "hello", " world"]
    assert response_statistics == {}


def test_stream_and_nonstream_include_native_usage():
    async def run_test():
        completion = make_completion()
        raw_request = RawRequest()

        nonstream_request = ChatCompletionRequest(
            model="test-model", messages=[], stream=False, max_tokens=16)
        completion.conversation_handles["req-full"] = 1
        response = await completion.chat_completion_full_generator(
            nonstream_request, raw_request, 1, model_deltas(),
            "req-full", 12, response_statistics={})
        assert_openai_usage(response.usage.model_dump())

        stream_request = ChatCompletionRequest(
            model="test-model", messages=[], stream=True, max_tokens=16)
        completion.conversation_handles["req-stream"] = 2
        final_usage = None
        async for event in completion.chat_completion_stream_generator(
                stream_request, raw_request, model_deltas(),
                "req-stream", 12, False, handle=2,
                response_statistics={}):
            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    continue
                data = json.loads(payload)
                if "usage" in data:
                    final_usage = data["usage"]

        assert final_usage is not None
        assert_openai_usage(final_usage)
        stream_responses_usage = completion._responses_usage_from_dict(
            final_usage).model_dump()
        assert stream_responses_usage["input_tokens_details"] == {
            "cached_tokens": 8,
            "missed_tokens": 4,
        }
        assert stream_responses_usage["output_tokens"] == 2

        completion.conversation_handles["req-responses"] = 3
        responses_request = ResponsesRequest(
            model="test-model", input="hello", stream=True,
            max_output_tokens=16)
        chat_stream = completion.chat_completion_stream_generator(
            stream_request, raw_request, model_deltas(),
            "req-responses", 12, False, handle=3,
            response_statistics={})
        final_responses_usage = None
        async for event in completion.responses_stream_generator(
                responses_request, chat_stream):
            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue
                data = json.loads(line[len("data:"):].strip())
                if data.get("type") == "response.completed":
                    final_responses_usage = data["response"]["usage"]

        assert final_responses_usage is not None
        assert final_responses_usage["input_tokens_details"] == {
            "cached_tokens": 8,
            "missed_tokens": 4,
        }
        assert final_responses_usage["output_tokens"] == 2

        completion.conversation_handles["req-anthropic"] = 4
        anthropic_request = AnthropicMessageRequest(
            model="test-model",
            messages=[AnthropicInputMessage(role="user", content="hello")],
            max_tokens=16,
            stream=True,
        )
        final_anthropic_usage = None
        async for event in completion.anthropic_message_stream_generator(
                anthropic_request, raw_request, model_deltas(),
                "req-anthropic", 12, None, handle=4,
                response_statistics={}):
            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue
                data = json.loads(line[len("data:"):].strip())
                if data.get("type") == "message_delta":
                    final_anthropic_usage = data["usage"]

        assert final_anthropic_usage == {
            "input_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 8,
            "output_tokens": 2,
        }

    asyncio.run(run_test())


if __name__ == "__main__":
    test_usage_protocol_mappings()
    test_stream_adapter_supports_new_and_legacy_models()
    test_stream_and_nonstream_include_native_usage()
    print("usage accounting tests passed")
