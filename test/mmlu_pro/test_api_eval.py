#!/usr/bin/env python3
import unittest

from api_eval import extract_answer


class ExtractAnswerTest(unittest.TestCase):
    def test_plain_answer(self) -> None:
        self.assertEqual(extract_answer("Answer: C", "ABCD"), "C")

    def test_uses_answer_after_thinking(self) -> None:
        output = "<think>The tentative answer is C.</think>\nFinal answer: B"
        self.assertEqual(extract_answer(output, "ABCD"), "B")

    def test_explicit_thinking_length_is_incomplete(self) -> None:
        self.assertIsNone(
            extract_answer(
                "The tentative answer is C.",
                "ABCD",
                thinking=True,
                finish_reason="length",
            )
        )

    def test_unclosed_thinking_is_incomplete_without_request_flag(self) -> None:
        self.assertIsNone(
            extract_answer(
                "<think>The tentative answer is C.",
                "ABCD",
                finish_reason="length",
            )
        )

    def test_unclosed_thinking_never_uses_reasoning_answer(self) -> None:
        self.assertIsNone(
            extract_answer("<THINK>The tentative answer is D.", "ABCD")
        )


if __name__ == "__main__":
    unittest.main()
