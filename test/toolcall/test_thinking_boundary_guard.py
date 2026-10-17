"""Thinking-boundary guards: quoted markers, calls written before the boundary.

Regression tests for the failure agents hit with Qwen thinking models:

* the thinking block ends at the first literal ``</think>`` even when the model
  merely *quotes* that marker inside a code span while reasoning about the
  protocol - the rest of the reasoning is then misclassified as content;
* a tool call emitted before the boundary never reaches the tool parser, so the
  turn ends as ``finish_reason=stop`` with no call and no diagnostic, and an
  agent client stops its loop without acting;
* a turn that thinks and then never writes the end marker produces neither an
  answer nor a call - it has to be reported instead of returned as a silent stop;
* a context saturated with the literal marker primes the model to write it back,
  and live turns stop mid-thought right before it; the boundary guard can only
  report that as a retryable error, so the occurrences that arrive in message
  text are rewritten to ``<ft:/think>`` before the prompt is rendered.
"""
import json
import os
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fastllm_pytools.openai_server.fastllm_completion import (  # noqa: E402
    FastLLmCompletion,
    _input_think_marker_replacement,
)
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (  # noqa: E402
    ChatCompletionRequest,
    ErrorResponse,
)


class _Tokenizer:
    chat_template = "<tool_call><function=<parameter="

    def get_vocab(self):
        return {"<tool_call>": 1, "</tool_call>": 2}


class _Model:
    force_chat_template = False
    tool_call_parser = "qwen3_coder"
    hf_tokenizer = _Tokenizer()

    def get_type(self):
        return "qwen3_5"


class _RawRequest:
    async def is_disconnected(self):
        return False


_TOOL = {"type": "function", "function": {
    "name": "read_file", "strict": True, "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}}
_FUNCTION = (
    "<tool_call><function=read_file>"
    "<parameter=path>README.md</parameter></function>"
)


class ThinkingBoundaryTest(unittest.IsolatedAsyncioTestCase):
    async def _response(self, text, *, stream, max_tokens=4096,
                        chunk_size=7, thinking=True, output_tokens=24,
                        tools=True):
        completion = FastLLmCompletion.__new__(FastLLmCompletion)
        completion.model = _Model()
        completion.model_name = "dummy"
        completion.conversation_handles = {}
        request = ChatCompletionRequest(
            model="dummy", messages=[], tools=[_TOOL] if tools else None,
            stream=stream, max_tokens=max_tokens, tool_choice="auto",
        )

        async def generated():
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]

        kwargs = dict(
            request=request, raw_request=_RawRequest(),
            result_generator=generated(), request_id="chatcmpl-thinking",
            input_token_len=3, emit_reasoning_content=thinking,
            response_statistics={"output_tokens": output_tokens},
        )
        if not stream:
            response = await completion.chat_completion_full_generator(
                **kwargs, handle=0)
            if isinstance(response, ErrorResponse):
                return {"errors": [response.message], "calls": [],
                        "reasoning": "", "content": "", "finish": None}
            choice = response.choices[0]
            return {
                "errors": [], "finish": choice.finish_reason,
                "calls": [c.model_dump() for c in choice.message.tool_calls or []],
                "reasoning": choice.message.reasoning_content or "",
                "content": choice.message.content or "",
            }
        events = []
        done = 0
        async for event in completion.chat_completion_stream_generator(
                **kwargs, think=False):
            for line in event.splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    done += 1
                else:
                    events.append(json.loads(payload))
        self.assertEqual(done, 1)
        choices = [c for e in events for c in e.get("choices", [])]
        finishes = [c["finish_reason"] for c in choices if c.get("finish_reason")]
        return {
            "errors": [e["error"]["message"] for e in events if "error" in e],
            "finish": finishes[0] if finishes else None,
            "calls": [t for c in choices for t in c["delta"].get("tool_calls", [])],
            "reasoning": "".join(c["delta"].get("reasoning_content") or ""
                                 for c in choices),
            "content": "".join(c["delta"].get("content") or "" for c in choices),
        }

    async def test_quoted_marker_stays_in_reasoning(self):
        text = ("We quote `</think>` while explaining the protocol.\n"
                "Reasoning continues here.\n</think>\nHere is the answer.")
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertEqual(result["errors"], [])
                self.assertIn("</think>", result["reasoning"])
                self.assertIn("Reasoning continues here.", result["reasoning"])
                self.assertEqual(result["content"].strip(), "Here is the answer.")
                self.assertEqual(result["finish"], "stop")

    async def test_fenced_quoted_marker_stays_in_reasoning(self):
        text = ("The template renders:\n```\n</think>\n```\n"
                "and then the answer follows.\n</think>\nDone.")
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertEqual(result["errors"], [])
                self.assertIn("and then the answer follows.", result["reasoning"])
                self.assertEqual(result["content"].strip(), "Done.")

    async def test_call_before_boundary_is_recovered(self):
        text = "I will read the file now.\n" + _FUNCTION
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["finish"], "tool_calls")
                self.assertEqual(len(result["calls"]), 1)
                self.assertEqual(result["calls"][0]["function"]["name"],
                                 "read_file")

    async def test_quoted_call_is_not_recovered(self):
        text = "The call would look like `" + _FUNCTION + "` in the template."
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertEqual(result["calls"], [])
                self.assertTrue(result["errors"])

    async def test_call_followed_by_answer_is_not_recovered(self):
        text = ("Draft: " + _FUNCTION + "\n</think>\n"
                "Here is the answer instead.")
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertEqual(result["calls"], [])
                self.assertEqual(result["errors"], [])
                self.assertIn("Here is the answer", result["content"])

    async def test_missing_boundary_is_reported(self):
        text = "Thinking about the problem without ever writing the end marker."
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertTrue(result["errors"])
                self.assertIn("no answer and no tool call", result["errors"][0].lower())
                self.assertIn("never closed", result["errors"][0].lower())

    async def test_quoted_marker_only_is_reported(self):
        text = "Only a quoted `</think>` marker and nothing else."
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertTrue(result["errors"])

    async def test_repeated_mentions_then_real_boundary(self):
        # Shape observed live: the model mentions the marker twice while echoing
        # the task (both inside backticks), goes on reasoning, then emits the
        # real marker on its own line and finally the answer.  An earlier
        # version of this fix accepted the second *mention* as the boundary and
        # leaked the rest of the reasoning into the answer.
        text = ("We need one short sentence. The note says the block is closed by "
                "the marker `</think>` and the tool block is `<tool_call>`.\n"
                "Task: reply with exactly: The thinking block closes with the marker "
                "`</think>` and then the answer follows.\n"
                "Now decide what to output.\n"
                "</think>\n"
                "The thinking block closes with the marker `</think>` and then the "
                "answer follows.")
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["reasoning"].count("</think>"), 2)
                self.assertIn("Now decide what to output.", result["reasoning"])
                self.assertNotIn("Now decide what to output", result["content"])
                self.assertTrue(result["content"].strip().startswith(
                    "The thinking block closes with the marker"))
                self.assertEqual(result["calls"], [])

    async def test_backtick_prefixed_marker_is_not_a_boundary(self):
        # A mention without a closing backtick is still a mention: the block only
        # ends at a marker the model did not write as code.
        text = "Quote `</think> and keep going without closing the span."
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertTrue(result["errors"])
                self.assertEqual(result["calls"], [])

    async def test_boundary_without_answer_is_reported(self):
        # The model ends the block and then says nothing.  An agent client
        # reads that as a finished turn, so it is reported as well.
        text = "Thinking that ends properly.\n</think>\n"
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertTrue(result["errors"])
                self.assertEqual(result["calls"], [])

    async def test_length_truncation_stays_silent(self):
        text = "Thinking about the problem without ever writing the end marker."
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream,
                                              max_tokens=24, output_tokens=24)
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["finish"], "length")

    async def test_bare_midsentence_marker_stays_in_reasoning(self):
        # Shape observed live (Qwen3.8-27B agent session): the reasoning itself
        # discusses the marker and writes it bare, in the middle of a sentence.
        # Scanning for the first textual marker split there, sent the rest of
        # the reasoning as the answer and left the real marker inside the
        # content - all of it silently.
        text = ("... if the model emits EOS mid-thought, the output has no "
                "</think> at all → \"phase didn't close\" → the fix should "
                "raise invalid_thinking_block! Hmm, unless there's some content "
                "after it... no, if EOS ends generation, there's no </think>, "
                "and the content is empty.\n"
                "Let me write the script and run it.\n"
                "</think>\n\n"
                "这个问题不能用记忆回答——先审计本会话自己的 jsonl。")
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["finish"], "stop")
                # Both bare mentions stay in the reasoning...
                self.assertEqual(result["reasoning"].count("</think>"), 2)
                self.assertIn("Let me write the script and run it.",
                              result["reasoning"])
                # ...and the answer is the real answer, with no leaked tail.
                self.assertNotIn("</think>", result["content"])
                self.assertNotIn("Let me write the script", result["content"])
                self.assertTrue(result["content"].strip().startswith(
                    "这个问题不能用记忆回答"))

    async def test_bare_midsentence_marker_before_call_is_recovered(self):
        # A mid-sentence mention must not hide the tool call written at the end
        # of the reasoning, which only the recovery path can see.
        text = ("The output has no </think> so far, so I will read the file now.\n"
                + _FUNCTION)
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["finish"], "tool_calls")
                self.assertEqual(len(result["calls"]), 1)
                self.assertEqual(result["calls"][0]["function"]["name"],
                                 "read_file")

    async def test_midsentence_marker_only_is_reported(self):
        # A marker the model kept talking across is not accepted as the end of
        # the block, so with no line-delimited marker at all the turn is reported
        # (retryable) instead of being split mid-sentence into a silent leak.
        text = "The output has no </think> at all in this draft."
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertTrue(result["errors"])
                self.assertIn("did not start a line", result["errors"][0])
                self.assertEqual(result["calls"], [])

    async def test_answer_mentioning_marker_after_boundary_is_kept(self):
        # Once the block is closed a marker in the answer is plain text: the
        # scan stops at the first line-delimited boundary, so an answer that
        # writes the marker bare is not mistaken for another end marker.
        text = ("We thought about it.\n</think>\n"
                "回答里直接提到 </think> 这个标记也不算边界。")
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertEqual(result["errors"], [])
                self.assertNotIn("We thought about it.", result["content"])
                self.assertIn("</think>", result["content"])
                self.assertIn("这个标记也不算边界", result["content"])

    async def test_marker_at_end_of_stream_is_reported_as_closed(self):
        text = "Thinking that ends exactly at the marker.</think>"
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream)
                self.assertTrue(result["errors"])
                self.assertIn("closed without an answer", result["errors"][0])
                self.assertEqual(result["calls"], [])

    async def test_no_tools_stream_keeps_the_answer(self):
        # Regression: the empty-turn guard used to look only at the text the
        # tool parser had seen, and that stays empty for a plain streaming chat
        # without tools - so a perfectly good answer was reported as
        # invalid_thinking_block.
        text = "We thought about it.\n</think>\n\n答案在这里。"
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream, tools=False)
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["finish"], "stop")
                self.assertEqual(result["content"].strip(), "答案在这里。")

    async def test_no_tools_stream_without_answer_is_still_reported(self):
        # The same guard must still fire when the block never ended, with or
        # without tools in the request.
        text = "Thinking about the problem without ever writing the end marker."
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = await self._response(text, stream=stream, tools=False)
                self.assertTrue(result["errors"])


class InputThinkMarkerNeutralizationTest(unittest.TestCase):
    """Input-side rewrite of literal thinking-close markers.

    A context saturated with the marker primes the model to write it back, and
    live turns have ended mid-thought right before it; the boundary guard can
    only report that as ``invalid_thinking_block``.  Rewriting the occurrences
    that arrive in message text to ``<ft:/think>`` removes the trigger while
    leaving the chat template's own end marker alone.
    """

    MARKER = "</think" + ">"
    PLACEHOLDER = "<ft:/think>"
    ENV = "FT_INPUT_THINK_MARKER_POLICY"

    def setUp(self):
        self._saved = os.environ.pop(self.ENV, None)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            os.environ.pop(self.ENV, None)
        else:
            os.environ[self.ENV] = self._saved

    def _messages(self):
        return [
            {"role": "user", "content": "用户消息里的 " + self.MARKER},
            {"role": "assistant", "content": [
                {"type": "text", "text": "多模态片段 " + self.MARKER},
                {"type": "image_url", "image_url": {"url": "x"}},
            ]},
            {"role": "assistant",
             "reasoning_content": self.MARKER + " 与 " + self.MARKER,
             "content": "答案"},
            {"role": "tool", "content": "工具返回，无标记",
             "tool_call_id": "call_1"},
        ]

    def _neutralize(self, messages):
        completion = FastLLmCompletion.__new__(FastLLmCompletion)
        return completion._neutralize_input_think_close(messages)

    def test_default_is_enabled(self):
        self.assertIsNone(os.environ.get(self.ENV))
        self.assertEqual(_input_think_marker_replacement(), self.PLACEHOLDER)

    def test_off_switch(self):
        for value in ("off", "0", "false", "no", "disable"):
            with self.subTest(value=value):
                os.environ[self.ENV] = value
                self.assertIsNone(_input_think_marker_replacement())

    def test_custom_replacement_is_used_verbatim(self):
        os.environ[self.ENV] = "[[ft:/think]]"
        self.assertEqual(_input_think_marker_replacement(), "[[ft:/think]]")

    def test_rewrites_message_text_and_keeps_structure(self):
        messages = self._messages()
        self._neutralize(messages)
        self.assertEqual(messages[0]["content"], "用户消息里的 " + self.PLACEHOLDER)
        self.assertEqual(messages[1]["content"][0]["text"],
                         "多模态片段 " + self.PLACEHOLDER)
        self.assertEqual(messages[1]["content"][1]["type"], "image_url")
        self.assertEqual(messages[1]["content"][1]["image_url"], {"url": "x"})
        self.assertNotIn(self.MARKER, messages[2]["reasoning_content"])
        self.assertEqual(messages[2]["reasoning_content"].count(self.PLACEHOLDER), 2)
        self.assertEqual(messages[3]["content"], "工具返回，无标记")

    def test_idempotent(self):
        messages = self._messages()
        self._neutralize(messages)
        once = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        self._neutralize(messages)
        self.assertEqual(
            json.dumps(messages, ensure_ascii=False, sort_keys=True), once)

    def test_disabled_leaves_everything_untouched(self):
        messages = self._messages()
        before = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        os.environ[self.ENV] = "off"
        self._neutralize(messages)
        self.assertEqual(
            json.dumps(messages, ensure_ascii=False, sort_keys=True), before)

    def test_no_marker_is_a_no_op(self):
        messages = [{"role": "user", "content": "普通消息"}]
        before = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        self._neutralize(messages)
        self.assertEqual(
            json.dumps(messages, ensure_ascii=False, sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
