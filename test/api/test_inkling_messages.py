import unittest

from tools.fastllm_pytools.openai_server.inkling_messages import consume


class InklingMessagesTest(unittest.TestCase):
    def test_text_and_reasoning_across_chunk_boundaries(self):
        raw = ("<|message_model|><|content_thinking|>check 1 < 2<|end_message|>"
               "<|message_model|><|content_text|>yes<|end_message|><|content_model_end_sampling|>")
        for chunk in (1, 2, 7, len(raw)):
            state = {"phase": "content", "buffer": ""}
            parts = []
            for start in range(0, len(raw), chunk):
                parts.extend(consume(raw[start:start + chunk], state))
            self.assertEqual(state["buffer"], "")
            self.assertEqual("".join(s for k, s in parts if k == "content"), "yes")
            self.assertEqual("".join(s for k, s in parts if k == "reasoning"), "check 1 < 2")

    def test_tool_channels_and_openai_calls(self):
        import json
        from tools.fastllm_pytools.openai_server.tool_parsers.inkling_tool_parser import InklingToolParser
        from tools.fastllm_pytools.openai_server.inkling_messages import END, INVOKE

        args = {"city": "苏州", "nested": {"values": [1, "<limit", 'a\"' + END + INVOKE]}}
        payload = json.dumps({"name": "weather", "args": args}, ensure_ascii=False)
        raw = ("<|message_model|><|content_thinking|>plan" + END +
               "<|message_model|><|content_text|>checking" + END +
               ("<|message_model|>weather" + INVOKE + payload + END) * 2)
        for size in (1, 7, len(raw)):
            state = {"phase": "content", "buffer": ""}
            parser = InklingToolParser(None)
            content, calls, wire = [], [], []
            for start in range(0, len(raw), size):
                text = "".join(part for channel, part in consume(raw[start:start+size], state)
                               if channel in ("content", "tool", "tool_json"))
                wire.append(text)
                delta = parser.extract_tool_calls_streaming("", "", text, [], [], [], None)
                if delta:
                    content.append(delta.content or "")
                    calls.extend(delta.tool_calls or [])
            self.assertEqual("".join(content), "checking")
            self.assertEqual([call.index for call in calls], [0, 1])
            self.assertEqual(len({call.id for call in calls}), 2)
            full = parser.extract_tool_calls("".join(wire), None)
            self.assertEqual(full.content, "checking")
            self.assertEqual(len(full.tool_calls), 2)
            for call in [*calls, *full.tool_calls]:
                self.assertEqual(call.function.name, "weather")
                self.assertEqual(json.loads(call.function.arguments), args)

    def test_truncated_or_invalid_invocation(self):
        from tools.fastllm_pytools.openai_server.inkling_messages import END, INVOKE
        from tools.fastllm_pytools.openai_server.tool_parsers.inkling_tool_parser import InklingToolParser

        parser = InklingToolParser(None)
        for body in ('{"name":', '[]' + END, '{"name":"weather","args":[]}' + END):
            result = parser.extract_tool_calls(INVOKE + body, None)
            self.assertFalse(result.tools_called)
            self.assertFalse(result.content)
            self.assertEqual(parser.incomplete_tool_call, not body.endswith(END))
