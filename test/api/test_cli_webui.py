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
    def test_webui_preserves_model_path_and_api_arguments(self):
        with tempfile.TemporaryDirectory(prefix="model path ") as model_path:
            argv = [
                "ftllm",
                "webui",
                model_path,
                "--port",
                "17777",
                "--api_base",
                "http://127.0.0.1:8081/v1",
                "--max_token",
                "2048",
            ]
            with (
                patch(
                    "fastllm_pytools.webui_server.serve_webui",
                    return_value=0,
                ) as call,
                redirect_stdout(io.StringIO()),
            ):
                result = cli.main(argv[1:])

        args = call.call_args.args[0]
        self.assertEqual(result, 0)
        self.assertEqual(args.model, model_path)
        self.assertEqual(args.api_base, "http://127.0.0.1:8081/v1")
        self.assertEqual(args.port, 17777)
        self.assertEqual(args.max_token, 2048)
        self.assertFalse(hasattr(args, "draft_tokens"))


if __name__ == "__main__":
    unittest.main()
