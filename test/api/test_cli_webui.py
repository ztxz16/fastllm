import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools import cli


class CliWebuiTest(unittest.TestCase):
    def test_webui_uses_argument_array_for_model_path(self):
        with tempfile.TemporaryDirectory(prefix="model path ") as model_path:
            argv = [
                "ftllm",
                "webui",
                model_path,
                "--port",
                "17777",
                "--device",
                "cpu",
                "--max_token",
                "2048",
                "--think",
                "true",
            ]
            with (
                patch(
                    "fastllm_pytools.cli.subprocess.call",
                    return_value=0,
                ) as call,
                redirect_stdout(io.StringIO()),
            ):
                result = cli.main(argv[1:])

        command = call.call_args.args[0]
        self.assertEqual(result, 0)
        self.assertEqual(command[:2], ["streamlit", "run"])
        self.assertIn("--server.port", command)
        self.assertIn("--browser.gatherUsageStats", command)
        self.assertIn("--", command)
        self.assertIn(model_path, command)
        self.assertNotIn("--draft_tokens", command)


if __name__ == "__main__":
    unittest.main()
