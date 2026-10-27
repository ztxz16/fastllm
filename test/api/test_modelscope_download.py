import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.modelscope_download import (
    AggregateProgress,
    PROGRESS_PREFIX,
    _filter_file_sizes,
    download,
)


class ModelScopeProgressAdapterTest(unittest.TestCase):
    def test_unmatched_filter_is_not_reported_as_a_successful_download(self):
        snapshot = Mock()
        args = SimpleNamespace(model="test/model", revision="master", local_dir="/tmp/model",
                               max_workers=2, include=["missing/*"])
        with patch.dict(sys.modules, {
            "modelscope.hub.snapshot_download": SimpleNamespace(snapshot_download=snapshot),
        }), patch("fastllm_pytools.modelscope_download._download_plan",
                  return_value=({}, "master", None)):
            with self.assertRaisesRegex(ValueError, "No repository files match"):
                download(args, None)
        snapshot.assert_not_called()

    def test_file_patterns_select_nested_quantization_directory(self):
        files = {
            "Q2_0/model-00001.gguf": 100,
            "Q2_0/model-00002.gguf": 200,
            "IQ3_S/model-00001.gguf": 300,
            "mmproj-Qwen3.8-Flash-Next-BF16.gguf": 40,
        }
        self.assertEqual(
            _filter_file_sizes(files, ["Q2_0/*", "mmproj-*.gguf"]),
            {
                "Q2_0/model-00001.gguf": 100,
                "Q2_0/model-00002.gguf": 200,
                "mmproj-Qwen3.8-Flash-Next-BF16.gguf": 40,
            },
        )

    def test_aggregate_progress_is_byte_weighted(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            progress = AggregateProgress({"a": 100, "b": 900})
            callback = progress.callback_type()("a", 100)
            callback.update(50)
            callback.end()

        events = [
            json.loads(line[len(PROGRESS_PREFIX):])
            for line in stream.getvalue().splitlines()
        ]
        self.assertEqual(events[0]["totalBytes"], 1000)
        self.assertTrue(any(event["downloadedBytes"] == 50 for event in events))
        self.assertEqual(events[-1]["downloadedBytes"], 100)
        self.assertEqual(events[-1]["completedFiles"], 1)

    def test_success_counts_cached_files_without_callbacks(self):
        def snapshot_download(**kwargs):
            callback = kwargs["progress_callbacks"][0]("new", 100)
            callback.update(100)
            callback.end()
            return kwargs["local_dir"]

        args = SimpleNamespace(model="test/model", revision="master", local_dir="/tmp/model", max_workers=4, include=[])
        stream = io.StringIO()
        with patch.dict(sys.modules, {
            "modelscope.hub.snapshot_download": SimpleNamespace(snapshot_download=snapshot_download),
        }), patch("fastllm_pytools.modelscope_download._download_plan",
                  return_value=({"cached": 900, "new": 100}, "master", None)), redirect_stdout(stream):
            self.assertEqual(download(args, None), "/tmp/model")
        events = [json.loads(line[len(PROGRESS_PREFIX):]) for line in stream.getvalue().splitlines()]
        self.assertEqual(events[-2]["downloadedBytes"], 100)
        self.assertEqual(events[-1]["downloadedBytes"], 1000)
        self.assertEqual(events[-1]["completedFiles"], 2)

    def test_include_patterns_are_forwarded_to_modelscope(self):
        captured = {}

        def snapshot_download(**kwargs):
            captured.update(kwargs)
            return kwargs["local_dir"]

        args = SimpleNamespace(
            model="test/model",
            revision="master",
            local_dir="/tmp/model",
            max_workers=2,
            include=["Q2_0/*", "mmproj-*.gguf"],
        )
        with patch.dict(sys.modules, {
            "modelscope.hub.snapshot_download": SimpleNamespace(snapshot_download=snapshot_download),
        }), patch("fastllm_pytools.modelscope_download._download_plan",
                  return_value=({"Q2_0/model.gguf": 100}, "master", None)):
            self.assertEqual(download(args, None), "/tmp/model")
        self.assertEqual(captured["allow_patterns"], ["Q2_0/*", "mmproj-*.gguf"])


if __name__ == "__main__":
    unittest.main()
