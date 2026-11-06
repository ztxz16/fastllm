#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.fastllm_pytools.openai_server.fastllm_completion import FastLLmCompletion, ConversationMessage


class ToolResultOrderTest(unittest.TestCase):
    def setUp(self):
        self.completion = object.__new__(FastLLmCompletion)

    def assistant(self, ids=("redis", "mongo")):
        return ConversationMessage(role="assistant", content="", tool_calls=[
            {"id": call_id, "type": "function", "function": {
                "name": "Read", "arguments": {"file_path": call_id + ".js"}}}
            for call_id in ids])

    def result(self, call_id, content=None):
        return ConversationMessage(role="tool", tool_call_id=call_id,
                                   content=content or call_id)

    def order(self, messages):
        return self.completion._order_tool_results(messages)

    def test_parallel_results_follow_call_ids_without_mutation(self):
        call = self.assistant()
        mongo = self.result("mongo", "dropDatabase()")
        redis = self.result("redis", "send_command('flushdb')")
        source = [call, mongo, redis]
        self.assertEqual(self.order(source), [call, redis, mongo])
        self.assertEqual(source, [call, mongo, redis])

    def test_anthropic_blocks_keep_contents_with_matching_calls(self):
        blocks = [{"type": "tool_result", "tool_use_id": "mongo",
                   "content": "dropDatabase()"},
                  {"type": "tool_result", "tool_use_id": "redis",
                   "content": "send_command('flushdb')"}]
        results, _ = self.completion._parse_anthropic_message_content("user", blocks)
        ordered = self.order([self.assistant(), *results])
        self.assertEqual([x.tool_call_id for x in ordered[1:]], ["redis", "mongo"])
        self.assertEqual([x.content for x in ordered[1:]],
                         ["send_command('flushdb')", "dropDatabase()"])

    def test_ambiguous_and_incomplete_groups_are_unchanged(self):
        for ids, results in [
            (("redis", "mongo"), [self.result("mongo")]),
            (("redis", "mongo"), [self.result("unknown"), self.result("redis")]),
            (("redis", "mongo"), [self.result("mongo"), self.result("mongo")]),
            (("redis", "redis"), [self.result("redis"), self.result("redis")]),
            (("", "mongo"), [self.result("mongo"), self.result("")]),
        ]:
            with self.subTest(ids=ids, results=results):
                source = [self.assistant(ids), *results]
                self.assertEqual(self.order(source), source)

    def test_does_not_move_results_across_user_text_or_media(self):
        for content in ["additional instruction", [{"type": "image"}]]:
            source = [self.assistant(), self.result("mongo"),
                      ConversationMessage(role="user", content=content),
                      self.result("redis")]
            self.assertEqual(self.order(source), source)

    def test_multiple_turns_and_already_ordered_results(self):
        first = [self.assistant(), self.result("mongo"), self.result("redis")]
        middle = ConversationMessage(role="user", content="continue")
        second = [self.assistant(("a", "b")), self.result("a"), self.result("b")]
        self.assertEqual(self.order([*first, middle, *second]),
                         [first[0], first[2], first[1], middle, *second])

    def test_plain_conversation_and_orphan_tools_are_preserved(self):
        source = [ConversationMessage(role="system", content="system"),
                  self.result("orphan"), ConversationMessage(role="user", content="user")]
        self.assertEqual(self.order(source), source)


if __name__ == "__main__":
    unittest.main()
