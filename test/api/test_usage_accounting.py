import asyncio
import ast
import ctypes
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


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
    def __init__(self):
        self.aborted_handles = []

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
        self.aborted_handles.append(handle)


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


def test_captured_terminal_usage_wins_over_reused_handle():
    completion = make_completion()
    terminal_statistics = {
        "cached_input_tokens": 0,
        "missed_input_tokens": 512,
        "output_tokens": 128,
    }

    # FakeModel.get_response_statistics() represents a different request that
    # has already reused the same integer handle.
    usage = completion._chat_usage(1, terminal_statistics, 512, 128)
    assert usage.prompt_tokens == 512
    assert usage.completion_tokens == 128
    assert usage.total_tokens == 640


def test_stream_background_cleanup_does_not_abort_reused_handle():
    async def run_test():
        completion = make_completion()
        raw_request = RawRequest()

        # The old request completed and removed its request_id.  A new request
        # now owns the same recycled integer handle.
        completion.conversation_handles["new-request"] = 7
        await completion.check_disconnect(raw_request, "old-request", 7)
        assert completion.model.aborted_handles == []
        assert completion.conversation_handles == {"new-request": 7}

        # A disconnected request is still tracked and must be aborted exactly
        # once, then removed from the active map.
        completion.conversation_handles["active-request"] = 8
        await completion.check_disconnect(raw_request, "active-request", 8)
        assert completion.model.aborted_handles == [8]
        assert "active-request" not in completion.conversation_handles

    asyncio.run(run_test())


def test_terminal_stream_event_releases_handle_before_yield():
    async def run_test():
        completion = make_completion()
        raw_request = RawRequest()
        stream_request = ChatCompletionRequest(
            model="test-model", messages=[], stream=True, max_tokens=16)

        completion.conversation_handles["old-request"] = 7
        stream = completion.chat_completion_stream_generator(
            stream_request, raw_request, model_deltas(),
            "old-request", 12, False, handle=7,
            response_statistics={})
        async for event in stream:
            if '"usage"' not in event:
                continue
            assert "old-request" not in completion.conversation_handles

            # The terminal backend context has already released handle 7.  A
            # new request may reuse it before the old client consumes [DONE].
            completion.conversation_handles["new-request"] = 7
            await completion.check_disconnect(
                raw_request, "old-request", 7)
            assert completion.model.aborted_handles == []
            assert completion.conversation_handles == {"new-request": 7}
            await stream.aclose()
            break

        anthropic_request = AnthropicMessageRequest(
            model="test-model",
            messages=[AnthropicInputMessage(role="user", content="hello")],
            max_tokens=16,
            stream=True,
        )
        completion.conversation_handles["old-anthropic"] = 8
        anthropic_stream = completion.anthropic_message_stream_generator(
            anthropic_request, raw_request, model_deltas(),
            "old-anthropic", 12, None, handle=8,
            response_statistics={})
        async for event in anthropic_stream:
            if '"type":"message_delta"' not in event:
                continue
            assert "old-anthropic" not in completion.conversation_handles
            completion.conversation_handles["new-anthropic"] = 8
            await completion.check_disconnect(
                raw_request, "old-anthropic", 8)
            assert completion.model.aborted_handles == []
            assert completion.conversation_handles == {
                "new-request": 7,
                "new-anthropic": 8,
            }
            await anthropic_stream.aclose()
            break

    asyncio.run(run_test())


def test_early_stream_stop_aborts_owned_handle_before_final_yield():
    async def stopped_deltas():
        yield "visible<stop>hidden"
        yield "must not be consumed"

    async def run_test():
        completion = make_completion()
        raw_request = RawRequest()
        request = ChatCompletionRequest(
            model="test-model", messages=[], stream=True, max_tokens=16,
            stop="<stop>")
        completion.conversation_handles["stopped-request"] = 9
        stream = completion.chat_completion_stream_generator(
            request, raw_request, stopped_deltas(),
            "stopped-request", 12, False, handle=9,
            response_statistics={})

        async for event in stream:
            if '"usage"' not in event:
                continue
            assert completion.model.aborted_handles == [9]
            assert "stopped-request" not in completion.conversation_handles
            await stream.aclose()
            break

    asyncio.run(run_test())


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


class ReusingFetchLibrary:
    def __init__(self, payload, terminal=-1, missing_terminal=False):
        self.payload = payload
        self.offset = 0
        self.calls = 0
        self.reused = False
        self.terminal = terminal
        self.missing_terminal = missing_terminal

    def fetch_response_tokens_batch_with_statistics_llm_model(
            self, model, handle, output, capacity, statistics):
        self.calls += 1
        statistics[:] = [8000, 2000, len(self.payload)]
        if self.calls == 1:
            return 0  # A prefill keepalive must not become an output token.
        count = min(capacity, len(self.payload) - self.offset)
        if count:
            output[:count] = self.payload[self.offset:self.offset + count]
            self.offset += count
            return count
        # Simulate another request claiming the handle before the C call
        # returns to Python. The copied terminal statistics remain our own.
        self.reused = True
        if self.missing_terminal:
            statistics[:] = [-1, -1, -1]
        return self.terminal

    def fetch_response_tokens_batch_llm_model(self, model, handle, output, capacity):
        # Also supports running this regression against the old implementation.
        return self.fetch_response_tokens_batch_with_statistics_llm_model(
            model, handle, output, capacity, (ctypes.c_int * 3)())


def make_native_stream_model(use_hf, payload, terminal=-1, missing_terminal=False):
    # Load the actual streaming method without loading a platform-specific .so.
    # The C++ companion test covers the real scheduler lock and context removal.
    source = Path(TOOLS_DIR) / "fastllm_pytools/llm.py"
    tree = ast.parse(source.read_text())
    model_class = next(n for n in tree.body
                       if isinstance(n, ast.ClassDef) and n.name == "model")
    method = next(n for n in model_class.body
                  if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "stream_response_handle_async")
    library = ReusingFetchLibrary(payload, terminal, missing_terminal)
    scope = {"asyncio": asyncio, "ctypes": ctypes, "fastllm_lib": library}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)

    class StreamModel(FakeModel):
        model = 0
        save_history = False
        hf_tokenizer = SimpleNamespace(decode=lambda ids: bytes(ids).decode(errors="replace"))
        stream_response_handle_async = scope["stream_response_handle_async"]

        def _can_apply_hf_chat_template(self):
            return use_hf

        def _uses_hf_deepseek_v4_tokenizer(self):
            return False

        def _decode_fastllm_token(self, token):
            return bytes([token])

        def _raise_prompt_too_long(self, handle):
            raise ValueError("prompt too long")

        def get_response_statistics(self, handle):
            if library.reused:
                return {"cached_input_tokens": 0, "missed_input_tokens": 17000,
                        "output_tokens": 0}
            return {"cached_input_tokens": 8000, "missed_input_tokens": 2000,
                    "output_tokens": len(payload)}

    return StreamModel(), library


def test_atomic_fetch_usage_and_handle_reuse():
    async def run_test():
        # The UTF-8 character straddles the 64-token fetch boundary. Neither
        # buffering nor coalescing text may change the exact token count.
        payload = b"a" * 63 + "中".encode() + b"b" * 64
        for use_hf in (False, True):
            for content, terminal, missing in (
                    (payload, -1, False), (b"", -1, False),
                    (payload, -1, True), (b"", -2, False)):
                model, library = make_native_stream_model(use_hf, content, terminal, missing)
                statistics = {}
                chunks = []
                try:
                    async for chunk in model.stream_response_handle_async(7, statistics):
                        chunks.append(chunk)
                    assert terminal == -1
                except ValueError as error:
                    assert terminal == -2 and str(error) == "prompt too long"
                assert "".join(chunks) == content.decode()
                assert library.reused
                assert statistics == {"cached_input_tokens": 8000,
                                      "missed_input_tokens": 2000,
                                      "output_tokens": len(content)}

            # A cancelled consumer must not perform a terminal lookup later.
            model, library = make_native_stream_model(use_hf, payload)
            statistics = {}
            stream = model.stream_response_handle_async(7, statistics)
            await stream.__anext__()
            calls = library.calls
            await stream.aclose()
            library.reused = True
            assert library.calls == calls and statistics["output_tokens"] == len(payload)

            for protocol in ("chat", "anthropic", "responses"):
                for streaming in (False, True):
                    completion = make_completion()
                    completion.model, library = make_native_stream_model(use_hf, payload)
                    completion.conversation_handles["atomic"] = 7
                    statistics = {}
                    stream = completion._stream_response_handle_with_statistics(7, statistics)
                    if protocol == "anthropic":
                        request = AnthropicMessageRequest(
                            model="test-model", messages=[], max_tokens=len(payload), stream=streaming)
                        if streaming:
                            events = completion.anthropic_message_stream_generator(
                                request, RawRequest(), stream, "atomic", 10000, None,
                                handle=7, response_statistics=statistics)
                        else:
                            response = await completion.anthropic_message_full_generator(
                                request, RawRequest(), 7, stream, "atomic", 10000, None,
                                response_statistics=statistics)
                            assert response.stop_reason == "max_tokens"
                            usage = response.usage.model_dump()
                    else:
                        request = ChatCompletionRequest(
                            model="test-model", messages=[], max_tokens=len(payload), stream=streaming)
                        if streaming:
                            events = completion.chat_completion_stream_generator(
                                request, RawRequest(), stream, "atomic", 10000, False,
                                handle=7, response_statistics=statistics)
                            if protocol == "responses":
                                events = completion.responses_stream_generator(
                                    ResponsesRequest(model="test-model", input="hello", stream=True), events)
                        else:
                            response = await completion.chat_completion_full_generator(
                                request, RawRequest(), 7, stream, "atomic", 10000,
                                response_statistics=statistics)
                            assert response.choices[0].finish_reason == "length"
                            usage = response.usage.model_dump()
                            if protocol == "responses":
                                usage = completion._responses_usage_from_chat_usage(response.usage).model_dump()
                    if streaming:
                        usage = None
                        async for event in events:
                            for line in event.splitlines():
                                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                                    continue
                                data = json.loads(line[5:])
                                if protocol == "responses" and data.get("type") in (
                                        "response.completed", "response.incomplete"):
                                    assert data["type"] == "response.incomplete"
                                    usage = data["response"]["usage"]
                                elif data.get("usage") is not None:
                                    usage = data["usage"]
                                if protocol == "anthropic" and data.get("type") == "message_delta":
                                    assert data["delta"]["stop_reason"] == "max_tokens"
                                if protocol == "chat":
                                    for choice in data.get("choices", []):
                                        if choice.get("finish_reason") is not None:
                                            assert choice["finish_reason"] == "length"
                    assert library.reused and usage is not None, (use_hf, protocol, streaming)
                    if protocol == "chat":
                        assert usage["prompt_tokens"] == 10000
                        assert usage["completion_tokens"] == len(payload)
                        assert usage["prompt_tokens_details"]["cached_tokens"] == 8000
                    elif protocol == "anthropic":
                        assert usage == {"input_tokens": 2000, "cache_creation_input_tokens": 0,
                                         "cache_read_input_tokens": 8000, "output_tokens": len(payload)}
                    else:
                        assert usage["input_tokens"] == 10000 and usage["output_tokens"] == len(payload)
                        assert usage["input_tokens_details"]["cached_tokens"] == 8000

    asyncio.run(run_test())


if __name__ == "__main__":
    test_usage_protocol_mappings()
    test_captured_terminal_usage_wins_over_reused_handle()
    test_stream_background_cleanup_does_not_abort_reused_handle()
    test_terminal_stream_event_releases_handle_before_yield()
    test_early_stream_stop_aborts_owned_handle_before_final_yield()
    test_stream_adapter_supports_new_and_legacy_models()
    test_stream_and_nonstream_include_native_usage()
    test_atomic_fetch_usage_and_handle_reuse()
    print("usage accounting tests passed")
