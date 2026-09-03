import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.modelscope_download import AggregateProgress, PROGRESS_PREFIX


class ModelScopeProgressAdapterTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
