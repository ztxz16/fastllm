import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Keep test/api/openai.py from shadowing the OpenAI SDK during discovery.
with patch.object(sys, "path", [str(Path(__file__).resolve().parents[2])] +
                  [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).resolve().parent]):
    from tools.fastllm_pytools.openai_server.fastllm_completion import FastLLmCompletion


class Hy4ReasoningTest(unittest.TestCase):
    def completion(self, namespace, custom=False):
        start, end = f"<think:{namespace}>", f"</think:{namespace}>"
        tokenizer = Mock()
        tokenizer.get_added_vocab.return_value = {start: 1, end: 2}
        server = FastLLmCompletion.__new__(FastLLmCompletion)
        server.model = SimpleNamespace(get_type=lambda: "hy_v4_flash", hf_tokenizer=tokenizer,
                                       force_chat_template=custom)
        return server, start, end

    def test_stream_and_full_responses_use_checkpoint_tags(self):
        for namespace in ("opensource", "6124c78e"):
            server, start, end = self.completion(namespace)
            for prefix in ("", start + "\n"):
                wire = prefix + "算一下" + end + "2"
                with self.subTest(namespace=namespace, prefix=prefix):
                    self.assertEqual(server._split_reasoning_response(wire, True), ("2", "算一下"))
                    state = server._new_reasoning_state(True)
                    thoughts, content = [], []
                    for char in wire:
                        messages, delta = server._consume_tagged_reasoning_delta(char, state)
                        thoughts.extend(msg.reasoning_content for msg in messages)
                        content.append(delta)
                    self.assertEqual(("".join(content), "".join(thoughts)), ("2", "算一下"))
            self.assertEqual(server._split_reasoning_response("算一下", True), ("", "算一下"))

    def test_reasoning_mode_matches_the_generation_prompt(self):
        server, _, _ = self.completion("opensource")
        for effort, options, enabled, expected in (
            (None, None, False, "no_think"),
            ("none", None, False, "no_think"),
            ("max", None, True, "max"),
            (None, {"reasoning_effort": "no_think"}, True, "no_think"),
            (None, None, True, None),
        ):
            request = SimpleNamespace(reasoning_effort=effort, chat_template_kwargs=options)
            kwargs = server._resolve_chat_template_kwargs(request, None, enable_thinking=enabled)
            self.assertEqual(kwargs.get("reasoning_effort"), expected)
            self.assertEqual(server._uses_tagged_reasoning_response(enabled, kwargs), expected != "no_think")
        self.assertEqual(server._split_reasoning_response("2", False), ("2", None))
        server, _, _ = self.completion("opensource", custom=True)
        self.assertFalse(server._uses_tagged_reasoning_response(True))
