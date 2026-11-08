#!/usr/bin/env python3
import io
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "fastllm_pytools"))
from terminal_colorizer import (
    ANSI_RESET,
    CACHE_CAPACITY_COLOR,
    CACHE_COLORS,
    CACHE_RESTORE_COLOR,
    COLORS,
    DECODE_AVERAGE_COLOR,
    colorize_line,
    process_lines,
    strip_ansi,
    user_line_allowed,
)


READY = "INFO:     Uvicorn running on http://0.0.0.0:8080\n"


class TerminalColorizerTest(unittest.TestCase):
    def test_event_line_is_colored_without_changing_plain_text(self):
        line = "[fastllm-event] category=decode request=7 event=progress decode_tokens_s=3\n"
        colored = colorize_line(line)
        self.assertIn("\033[32m", colored)
        self.assertEqual(strip_ansi(colored), line)

    def test_prefill_and_native_decode_status_use_distinct_contract_colors(self):
        prefill = ("[fastllm-event] seq=1 timestamp_monotonic_ns=2 category=progress "
                   "request=req-1 native_handle=7 event=request_status phase=prefill "
                   "native_rate_state=fresh native_interval_tokens_s=886.324 "
                   "prefill_completed_tokens=1024 prefill_total_tokens=1280\n")
        decode = ("[fastllm-event] seq=2 timestamp_monotonic_ns=3 category=progress "
                  "request=req-1 native_handle=7 event=request_status phase=decode "
                  "native_rate_state=fresh native_interval_tokens_s=68.570 "
                  "native_output_tokens=475\n")
        log, terminal = io.StringIO(), io.StringIO()
        process_lines([READY, prefill, decode], log, terminal, "user")
        tty = terminal.getvalue()
        prefill_line = next(line for line in tty.splitlines() if "Prefill" in line)
        decode_line = next(line for line in tty.splitlines() if "Native decode" in line)
        self.assertTrue(prefill_line.startswith(COLORS["prefill"]))
        self.assertTrue(decode_line.startswith(COLORS["decode"]))
        self.assertNotIn(COLORS["progress"], tty)
        self.assertNotEqual(COLORS["prefill"], COLORS["decode"])
        self.assertEqual(log.getvalue(), READY + prefill + decode)
        self.assertNotIn("\033[", log.getvalue())

    def test_cache_restore_status_uses_cache_color_not_decode(self):
        restore = ("[fastllm-event] seq=3 timestamp_monotonic_ns=4 category=progress "
                   "request=req-1 native_handle=7 event=request_status phase=cache_restore "
                   "native_rate_state=pending\n")
        log, terminal = io.StringIO(), io.StringIO()
        process_lines([READY, restore], log, terminal, "user")
        line = next(item for item in terminal.getvalue().splitlines()
                    if "Restoring prefix cache" in item)
        self.assertTrue(line.startswith(CACHE_RESTORE_COLOR))
        self.assertNotIn(COLORS["decode"], terminal.getvalue())
        self.assertNotEqual(CACHE_RESTORE_COLOR, COLORS["decode"])
        self.assertEqual(log.getvalue(), READY + restore)

    def test_capacity_and_decode_average_differ_from_decode(self):
        status = ("[fastllm-event] seq=5 timestamp_monotonic_ns=6 category=progress "
                  "request=req-1 native_handle=7 event=request_status phase=decode "
                  "native_rate_state=fresh native_interval_tokens_s=70.540 "
                  "native_output_tokens=125 l1_capacity_state=ok l1_used_tokens=99456 "
                  "l1_available_tokens=719232 l1_total_tokens=818688\n")
        average = ("[fastllm-event] seq=6 timestamp_monotonic_ns=7 category=output "
                   "request=req-1 native_handle=7 event=native_generation_summary "
                   "native_rate_available=1 native_average_tokens_s=83.529\n")
        log, terminal = io.StringIO(), io.StringIO()
        process_lines([READY, status, average], log, terminal, "user")
        tty = terminal.getvalue()
        decode_line = next(item for item in tty.splitlines() if "Native decode |" in item)
        capacity_line = next(item for item in tty.splitlines() if "Cache capacity" in item)
        average_line = next(item for item in tty.splitlines() if "Native decode average" in item)
        self.assertTrue(decode_line.startswith(COLORS["decode"]))
        self.assertTrue(capacity_line.startswith(CACHE_CAPACITY_COLOR))
        self.assertTrue(average_line.startswith(DECODE_AVERAGE_COLOR))
        self.assertNotEqual(CACHE_CAPACITY_COLOR, COLORS["decode"])
        self.assertNotEqual(DECODE_AVERAGE_COLOR, COLORS["decode"])
        self.assertEqual(log.getvalue(), READY + status + average)

    def test_cache_family_is_yellow_and_tiers_are_distinct(self):
        candidates = ("[fastllm-event] seq=7 timestamp_monotonic_ns=8 category=cache "
                      "request=req-1 native_handle=7 event=cache_candidates "
                      "l1_resident_tokens=14976 l2_host_tokens=0 l2_query_status=ok\n")
        ssd = ("[fastllm-event] seq=8 timestamp_monotonic_ns=9 category=cache "
               "request=req-1 native_handle=7 event=cache_result source_tier=L3 "
               "effective_reused_tokens=96896 read_mib_s=2011.9\n")
        log, terminal = io.StringIO(), io.StringIO()
        process_lines([READY, candidates, ssd], log, terminal, "user")
        tty = terminal.getvalue()
        l1 = next(item for item in tty.splitlines() if "L1 prefix candidate" in item)
        l2 = next(item for item in tty.splitlines() if "L2 prefix candidate" in item)
        l3 = next(item for item in tty.splitlines() if "SSD cache: hit" in item)
        self.assertTrue(l1.startswith(CACHE_COLORS["L1"]))
        self.assertTrue(l2.startswith(CACHE_COLORS["L2"]))
        self.assertTrue(l3.startswith(CACHE_COLORS["L3"]))
        self.assertEqual([CACHE_COLORS[t] for t in ("L1", "L2", "L3")],
                         ["\033[38;5;226m", "\033[38;5;178m", "\033[38;5;136m"])
        self.assertEqual(len(set(CACHE_COLORS.values())), 4)
        self.assertEqual(log.getvalue(), READY + candidates + ssd)

    def test_first_token_uses_prefill_blue(self):
        first = ("[fastllm-event] seq=9 timestamp_monotonic_ns=10 category=output "
                 "request=req-1 native_handle=7 event=native_first_token ttft_ms=6674.545\n")
        log, terminal = io.StringIO(), io.StringIO()
        process_lines([READY, first], log, terminal, "user")
        line = next(item for item in terminal.getvalue().splitlines() if "First token" in item)
        self.assertTrue(line.startswith(COLORS["prefill"]))
        self.assertNotIn(COLORS["decode"], terminal.getvalue())
        self.assertEqual(log.getvalue(), READY + first)

    def test_unknown_line_stays_plain(self):
        line = "ordinary server message\n"
        self.assertEqual(colorize_line(line), line)

    def test_log_stream_never_receives_ansi(self):
        source = [
            READY,
            "[fastllm-event] category=error request=9 event=failed\n",
            "plain\n",
        ]
        log = io.StringIO()
        terminal = io.StringIO()
        process_lines(source, log, terminal, "user")
        self.assertNotIn("\033[", log.getvalue())
        self.assertIn("\033[31m", terminal.getvalue())
        self.assertEqual(log.getvalue(), "".join(source))
        self.assertTrue(terminal.getvalue().startswith(READY))
        self.assertIn("Error: failed", strip_ansi(terminal.getvalue()))
        self.assertNotIn("plain\n", terminal.getvalue())

    def test_off_mode_is_plain_passthrough(self):
        source = [
            "[fastllm-event] category=error request=9 event=failed\n",
            "plain\n",
        ]
        log = io.StringIO()
        terminal = io.StringIO()
        process_lines(source, log, terminal, "off")
        self.assertEqual(log.getvalue(), "".join(source))
        self.assertEqual(terminal.getvalue(), "".join(source))

    def test_debug_keeps_startup_plain_until_uvicorn_ready(self):
        source = [
            "[Fastllm] loading model\n",
            "INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)\n",
            "[fastllm-event] category=error request=9 event=failed\n",
        ]
        log = io.StringIO()
        terminal = io.StringIO()
        process_lines(source, log, terminal, "debug")
        self.assertEqual(strip_ansi(terminal.getvalue()), log.getvalue())
        self.assertNotIn("\033[", terminal.getvalue().split(source[1])[0])
        self.assertIn("\033[", terminal.getvalue().split(source[1], 1)[1])

    def test_raw_stream_preserves_every_plain_line(self):
        source = [
            "startup\033[31m-red\033[0m\n",
            "INFO:     Uvicorn running on http://0.0.0.0:8080\n",
            "[fastllm-event] source=native category=output request=1 event=request_finished\n",
        ]
        log = io.StringIO()
        raw = io.StringIO()
        process_lines(source, log, None, "debug", raw)
        self.assertEqual(raw.getvalue(), strip_ansi("".join(source)))
        self.assertNotIn("\033[", log.getvalue())

    def test_user_mode_filters_internal_lines(self):
        self.assertTrue(user_line_allowed(
            "[fastllm-event] category=cache request=1 event=cache_result source_tier=L1\n"))
        self.assertTrue(user_line_allowed(
            "[fastllm-event] seq=1 timestamp_monotonic_ns=2 "
            "category=output request=1 event=first_token\n"))
        self.assertTrue(user_line_allowed(
            "INFO:     Uvicorn running on http://0.0.0.0:8080\n"))
        self.assertFalse(user_line_allowed("[Qwen3.5 DFlash2] pos_accept_rate=[80%]\n"))
        self.assertFalse(user_line_allowed("[Decode] alive = 1, pending = 0\n"))

    def test_user_mode_projects_only_terminal(self):
        source = [
            READY,
            "[Qwen3.5 DFlash2] pos_accept_rate=[80%]\n",
            "[fastllm-event] source=native category=output request=1 event=native_first_token ttft_ms=10\n",
        ]
        log = io.StringIO()
        terminal = io.StringIO()
        process_lines(source, log, terminal, "user")
        self.assertIn("pos_accept_rate", log.getvalue())
        self.assertNotIn("pos_accept_rate", terminal.getvalue())
        self.assertIn("event=native_first_token", log.getvalue())
        self.assertIn("\033[", terminal.getvalue())


if __name__ == "__main__":
    unittest.main()
