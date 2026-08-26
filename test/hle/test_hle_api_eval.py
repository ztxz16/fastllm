#!/usr/bin/env python3
import argparse
import unittest

from hle_api_eval import (
    answer_region,
    build_payload,
    filter_examples,
    normalize_choice,
    score_answer,
    selection_sha256,
    summarize_group,
)


class HLEEvalTest(unittest.TestCase):
    def test_zero_max_tokens_omits_request_limit(self) -> None:
        args = argparse.Namespace(
            model="model",
            system_prompt="",
            no_system_prompt=True,
            temperature=0.0,
            max_tokens=0,
            top_p=None,
            top_k=None,
        )
        example = {"question": "Question", "image": ""}
        self.assertNotIn("max_tokens", build_payload(args, example, {}))
        self.assertEqual(
            build_payload(args, example, {}, max_tokens=64)["max_tokens"], 64
        )

    def test_official_answer_format(self) -> None:
        output = "Explanation: Seven is prime.\nAnswer: C\nConfidence: 97%"
        self.assertEqual(answer_region(output), "C")
        self.assertEqual(score_answer(output, "C", "multipleChoice")[:2], ("C", True))

    def test_answer_after_closed_thinking(self) -> None:
        output = "<think>Maybe A.</think>\nAnswer: Option D\nConfidence: 60%"
        self.assertEqual(normalize_choice(answer_region(output)), "D")

    def test_choice_with_markdown_and_option_text(self) -> None:
        output = "Explanation: done\nAnswer: **C. Seven**\nConfidence: 90%"
        self.assertEqual(normalize_choice(answer_region(output)), "C")

    def test_unclosed_thinking_is_not_scored(self) -> None:
        self.assertIsNone(answer_region("<think>The answer might be B."))

    def test_exact_match_is_explicitly_separate(self) -> None:
        output = "Explanation: done\nAnswer:  Alpha   Beta.\nConfidence: 80%"
        prediction, correct, method = score_answer(output, "alpha beta", "exactMatch")
        self.assertEqual(prediction, "alpha beta")
        self.assertTrue(correct)
        self.assertEqual(method, "normalized_exact")

    def test_seeded_filter_is_reproducible(self) -> None:
        examples = [
            {
                "case_id": f"test:{idx}",
                "image": "",
                "answer_type": "multipleChoice",
                "category": "Math",
            }
            for idx in range(10)
        ]
        args = argparse.Namespace(
            text_only=True,
            answer_type=["multipleChoice"],
            category=None,
            shuffle=True,
            seed=42,
            start=0,
            limit=5,
        )
        first = filter_examples(examples, args)
        second = filter_examples(examples, args)
        self.assertEqual(selection_sha256(first), selection_sha256(second))
        self.assertEqual([item["case_id"] for item in first], ["test:7", "test:3", "test:2", "test:8", "test:5"])

    def test_summary_uses_request_time_for_single_concurrency_tps(self) -> None:
        records = [
            {
                "correct": True,
                "prediction": "A",
                "error": None,
                "input_tokens": 10,
                "output_tokens": 100,
                "latency_ms": 500.0,
            },
            {
                "correct": False,
                "prediction": "B",
                "error": None,
                "input_tokens": 20,
                "output_tokens": 200,
                "latency_ms": 1000.0,
            },
        ]
        summary = summarize_group(records)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["output_tokens_per_second"], 200.0)


if __name__ == "__main__":
    unittest.main()
