"""Naive tool responses must separate reasoning before parsing XML calls."""
import json
import unittest

from test_qwen4_exp_reasoning import (
    DummyQwenTokenizer, FakeQwen4ExpModel, RawRequest, completion, request,
)


TOOL = {"type": "function", "function": {
    "name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}}
CALL = ("<tool_call>\n<function=get_weather>\n"
        "<parameter=city>北京</parameter>\n</function>\n</tool_call>")


class NaiveModel(FakeQwen4ExpModel):
    hf_tokenizer = DummyQwenTokenizer()

    def get_type(self):
        return "naive_n05_flash"

    def stream_response_handle_async(self, handle):
        async def generator():
            # Split every XML and reasoning delimiter across decoder chunks.
            for char in self.output:
                yield char
        return generator()


class NaiveToolResponseTest(unittest.IsolatedAsyncioTestCase):
    def test_reasoning_switch_and_custom_template(self):
        instance = completion(NaiveModel())
        self.assertTrue(instance._uses_tagged_reasoning_response(True))
        self.assertFalse(instance._uses_tagged_reasoning_response(False))
        instance.model.force_chat_template = True
        self.assertFalse(instance._uses_tagged_reasoning_response(True))

    async def test_thinking_tool_response_full_and_stream(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                model = NaiveModel("<think>需要查询实时天气。</think>" + CALL)
                response = await completion(model).create_chat_completion(
                    request(tools=[TOOL], stream=stream, max_tokens=512,
                            chat_template_kwargs={"enable_thinking": True}),
                    RawRequest())
                if not stream:
                    choice = response.choices[0]
                    reasoning = choice.message.reasoning_content
                    content = choice.message.content or ""
                    calls = choice.message.tool_calls
                    self.assertEqual(choice.finish_reason, "tool_calls")
                    self.assertEqual(len(calls), 1)
                    name = calls[0].function.name
                    arguments = calls[0].function.arguments
                else:
                    events = []
                    async for event in response[0]:
                        for line in event.splitlines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                events.append(json.loads(line[6:]))
                    choices = [c for event in events for c in event.get("choices", [])]
                    self.assertEqual(
                        [c["finish_reason"] for c in choices if c.get("finish_reason")],
                        ["tool_calls"])
                    deltas = [c["delta"] for c in choices]
                    reasoning = "".join(d.get("reasoning_content") or "" for d in deltas)
                    content = "".join(d.get("content") or "" for d in deltas)
                    calls = [t for d in deltas for t in d.get("tool_calls", [])]
                    self.assertEqual({c["index"] for c in calls}, {0})
                    name = "".join(c.get("function", {}).get("name") or "" for c in calls)
                    arguments = "".join(c.get("function", {}).get("arguments") or "" for c in calls)
                self.assertEqual(reasoning, "需要查询实时天气。")
                self.assertFalse(content.strip())
                self.assertEqual(name, "get_weather")
                self.assertEqual(json.loads(arguments), {"city": "北京"})

    async def test_truncated_thought_stays_in_reasoning(self):
        response = await completion(NaiveModel("<think>还在思考")).create_chat_completion(
            request(tools=[TOOL], max_tokens=1), RawRequest())
        choice = response.choices[0]
        self.assertEqual(choice.message.reasoning_content, "还在思考")
        self.assertFalse(choice.message.content)
        self.assertFalse(choice.message.tool_calls)
        self.assertEqual(choice.finish_reason, "length")


if __name__ == "__main__":
    unittest.main()
