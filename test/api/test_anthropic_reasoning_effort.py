import unittest

from test_qwen35_reasoning import completion, RawRequest
from tools.fastllm_pytools.openai_server.protocal.anthropic_protocol import AnthropicMessageRequest
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import ErrorResponse


class AnthropicReasoningEffortTest(unittest.IsolatedAsyncioTestCase):
    async def test_effort_reaches_token_counting_and_generation(self):
        for effort in ("low", "medium", "xhigh"):
            for stream in (False, True):
                with self.subTest(effort=effort, stream=stream):
                    instance = completion(); instance.enable_thinking = False
                    result = await instance.create_anthropic_message(AnthropicMessageRequest(
                        model="qwen3.5", messages=[{"role":"user", "content":"answer"}], max_tokens=128,
                        stream=stream, thinking={"type":"adaptive"}, output_config={"effort":effort}), RawRequest())
                    self.assertNotIsInstance(result, ErrorResponse)
                    for kwargs in (instance.model.input_kwargs, instance.model.launch_kwargs):
                        self.assertEqual(kwargs["chat_template_kwargs"]["reasoning_effort"], effort)
                        self.assertTrue(kwargs["enable_thinking"])
                    if stream:
                        generator, background = result
                        async for chunk in generator: pass
                        await background()

    async def test_disabled_thinking_and_invalid_effort(self):
        for thinking, effort, enabled, error in ((None, None, False, False),
                ({"type":"disabled"}, "low", False, False),
                ({"type":"adaptive"}, "high", None, True)):
            with self.subTest(thinking=thinking, effort=effort):
                instance = completion(); instance.enable_thinking = False
                result = await instance.create_anthropic_message(AnthropicMessageRequest(
                    model="qwen3.5", messages=[{"role":"user", "content":"answer"}], max_tokens=128,
                    thinking=thinking, output_config={"effort":effort}), RawRequest())
                if error:
                    self.assertIsInstance(result, ErrorResponse)
                    self.assertIsNone(instance.model.launch_kwargs)
                else:
                    self.assertEqual(instance.model.launch_kwargs["enable_thinking"], enabled)
