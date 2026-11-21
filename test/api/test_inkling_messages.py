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
