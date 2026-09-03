import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools import cli


class CliLaunchTest(unittest.TestCase):
    def test_launcher_default_port_does_not_conflict_with_webui(self):
        args = cli.args_parser().parse_args(["launch"])

        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8000)
        self.assertFalse(args.no_browser)

    def test_no_arguments_start_launcher_and_open_browser_by_default(self):
        with patch(
            "fastllm_pytools.launcher.fastllm_launcher",
            return_value=0,
        ) as launcher:
            result = cli.main([])

        args = launcher.call_args.args[0]
        self.assertEqual(result, 0)
        self.assertEqual(args.command, "launch")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8000)
        self.assertFalse(args.no_browser)

    def test_launcher_can_listen_on_all_interfaces(self):
        args = cli.args_parser().parse_args(
            ["launch", "--host", "0.0.0.0"]
        )

        self.assertEqual(args.host, "0.0.0.0")

    def test_launcher_browser_can_be_disabled_explicitly(self):
        args = cli.args_parser().parse_args(["launch", "--no-browser"])

        self.assertTrue(args.no_browser)

    def test_launcher_help_is_english(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(output):
            cli.args_parser().parse_args(["launch", "--help"])

        self.assertIn("Launcher listen address", output.getvalue())
        self.assertNotRegex(output.getvalue(), r"[一-龥]")

if __name__ == "__main__":
    unittest.main()
