"""User request-terminal lifecycle projection; all native/debug evidence retained."""
import hashlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/fastllm_pytools"))
import terminal_colorizer as colorizer

# Verbatim causal tail from the bounded real T1, not a fabricated model run.
# Full original runtime/raw.log SHA is recorded alongside the source fixture.
REAL_SOURCE_SHA256 = 'e7712fbf6a414c6a1fa5cead4b1962f85847be03daa2d848e7f38a972b4a6722'
REAL_CANCELLED_TAIL = 'INFO:     Uvicorn running on http://127.0.0.1:18081 (Press CTRL+C to quit)\n2026-10-21 08:02:08,704 56080 request_progress.py[line:318] INFO: [fastllm-event] seq=88 timestamp_monotonic_ns=2583101951435 timestamp_utc=2026-10-21T00:02:08.704Z source=python category=output request=fastllm-Qwen3.8-27B-FP8-1549302812d34bb9a8d8787233fa7b1a request_id=fastllm-Qwen3.8-27B-FP8-1549302812d34bb9a8d8787233fa7b1a native_handle=0 event=request_finished status=cancelled output_count_state=unknown cached_input_tokens=0 effective_reused_tokens=0 missed_input_tokens=0 elapsed_ms=1015.9 native_observer_state=closed\n[fastllm-event] seq=89 timestamp_monotonic_ns=2584135438426 timestamp_utc=2026-10-21T00:02:09.737Z source=native category=prefill request=0 native_handle=0 public_request=fastllm-Qwen3.8-27B-FP8-1549302812d34bb9a8d8787233fa7b1a native_generation=2 event=prefill tokens=1720\n[fastllm-event] seq=90 timestamp_monotonic_ns=2584135487362 timestamp_utc=2026-10-21T00:02:09.738Z source=native category=prefill request=0 native_handle=0 public_request=fastllm-Qwen3.8-27B-FP8-1549302812d34bb9a8d8787233fa7b1a native_generation=2 event=prefill_summary measurement=native_forward_excluding_vision input_tokens=1720 cached_input_tokens=0 missed_input_tokens=1720 prefill_tokens=1720 elapsed_ns=1129305498 excluded_vision_ns=912886976\n[fastllm-event] seq=91 timestamp_monotonic_ns=2584135505519 timestamp_utc=2026-10-21T00:02:09.738Z source=native category=prefill request=0 native_handle=0 public_request=fastllm-Qwen3.8-27B-FP8-1549302812d34bb9a8d8787233fa7b1a native_generation=2 event=prefill_speed measurement=native_forward_excluding_vision prefill_tokens=1720 elapsed_ns=1129305498 elapsed_ms=1129.305 prefill_tokens_s=1523.060\n[fastllm-event] seq=92 timestamp_monotonic_ns=2584135521056 timestamp_utc=2026-10-21T00:02:09.738058107Z source=native category=lifecycle request=0 native_handle=0 native_generation=2 event=request_phase phase=decode previous_phase=prefill phase_revision=6 phase_boundary=native_decode_ready_or_returned public_request=fastllm-Qwen3.8-27B-FP8-1549302812d34bb9a8d8787233fa7b1a\n[fastllm-event] seq=93 timestamp_monotonic_ns=2584135535131 timestamp_utc=2026-10-21T00:02:09.738Z source=native category=output request=0 native_handle=0 public_request=fastllm-Qwen3.8-27B-FP8-1549302812d34bb9a8d8787233fa7b1a native_generation=2 event=native_first_token measurement=native_committed_token window_start_ns=2582084563368 window_end_ns=2584135476983 ttft_ms=2050.914\n[fastllm-event] seq=94 timestamp_monotonic_ns=2584135550449 timestamp_utc=2026-10-21T00:02:09.738Z source=native category=progress request=0 native_handle=0 public_request=fastllm-Qwen3.8-27B-FP8-1549302812d34bb9a8d8787233fa7b1a native_generation=2 event=progress phase=mixed elapsed_s=2.051 prefill_interval_tokens_s=838.62 committed_interval_tokens_s=0.49 l1_reused_tokens=0 l2_reused_tokens=0 l3_reused_tokens=0 effective_reused_tokens=0 prefix_lookup_ms=0.0 prefix_restore_ms=0.0 ssd_read_mib_s=0.00 ssd_restore_tokens_s=0.00 vision_ssd_load_mib_s=0.00\n[fastllm-event] seq=95 timestamp_monotonic_ns=2584138587201 timestamp_utc=2026-10-21T00:02:09.741Z source=native category=lifecycle request=0 native_handle=0 public_request=fastllm-Qwen3.8-27B-FP8-1549302812d34bb9a8d8787233fa7b1a native_generation=2 event=native_context_cleanup_begin ending=0 abort_requested=1\n[fastllm-event] seq=96 timestamp_monotonic_ns=2584143561641 timestamp_utc=2026-10-21T00:02:09.746098726Z source=native category=lifecycle request=0 native_handle=0 native_generation=2 event=request_phase phase=native_closed previous_phase=decode phase_revision=7 phase_boundary=native_context_cleanup_completed public_request=fastllm-Qwen3.8-27B-FP8-1549302812d34bb9a8d8787233fa7b1a\n'


def event(public, name, category="output", source="native", handle="0", extra=""):
    return (f"[fastllm-event] seq=1 timestamp_utc=2026-10-21T00:02:09.738Z "
            f"source={source} category={category} native_handle={handle} "
            f"public_request={public} event={name} {extra}\n")


class UserTerminalCompletionTest(unittest.TestCase):
    def test_real_cancelled_request_never_revives_in_either_language_and_debug_is_lossless(self):
        path = os.environ.get("FASTLLM_TERMINAL_REAL_RAW")
        payload = Path(path).read_bytes() if path else REAL_CANCELLED_TAIL.encode()
        text = payload.decode("utf-8")
        lines = list(io.StringIO(text))
        self.assertTrue(any("status=cancelled" in line for line in lines))
        for language in ("en", "zh"):
            with self.subTest(language=language):
                view = colorizer.UserTerminalView(language)
                terminal = set()
                for line in lines:
                    fields = {m[1]: m[2] for m in colorizer._FIELD_RE.finditer(line)}
                    public = fields.get("public_request") or fields.get("request_id")
                    visible = view.line(line)
                    if public in terminal and fields.get("category") not in {"error", "warning"}:
                        self.assertIsNone(visible, (fields.get("event"), visible))
                    if fields.get("event") == "request_finished" and fields.get("source") == "python":
                        terminal.add(public)
                raw, server, terminal_stream, user = (io.StringIO() for _ in range(4))
                colorizer.process_lines(lines, server, terminal_stream, "user", raw, user, language)
                self.assertEqual(raw.getvalue().encode(), payload)
                self.assertEqual(server.getvalue().encode(), payload)
                self.assertNotIn("1523.060", user.getvalue())
                self.assertNotIn("2050.914", user.getvalue())
        for mode in ("debug", "off"):
            raw, server, terminal = io.StringIO(), io.StringIO(), io.StringIO()
            colorizer.process_lines(lines, server, terminal, mode, raw)
            self.assertEqual(raw.getvalue().encode(), payload)
            self.assertEqual(server.getvalue().encode(), payload)
            self.assertEqual(colorizer.strip_ansi(terminal.getvalue()).encode(), payload)

    def test_all_terminal_outcomes_suppress_duplicate_completion_and_late_activity(self):
        late = (
            event("request-a", "prefill_summary", "prefill", extra="missed_input_tokens=1720"),
            event("request-a", "prefill_speed", "prefill", extra="prefill_tokens=1720 prefill_tokens_s=1523.060"),
            event("request-a", "native_first_token", extra="ttft_ms=2050.914"),
            event("request-a", "cache_result", "cache", extra="source_tier=MISS"),
            event("request-a", "vision_encode_begin", "vision"),
            event("request-a", "request_status", "progress", "python",
                  extra="phase=prefill native_status_state=observed"),
            event("request-a", "request_received", "startup", "python"))
        for language in ("en", "zh"):
            for outcome in ("ok", "cancelled", "error", "disconnected"):
                with self.subTest(language=language, outcome=outcome):
                    view = colorizer.UserTerminalView(language)
                    done = event("request-a", "request_finished", source="python", extra="status=" + outcome)
                    self.assertIsNotNone(view.line(done))
                    self.assertIsNone(view.line(done))
                    for line in late:
                        self.assertIsNone(view.line(line))
                    self.assertEqual(view._requests, {})

    def test_late_error_stays_visible_and_cannot_steal_a_reused_handle(self):
        for language in ("en", "zh"):
            with self.subTest(language=language):
                view = colorizer.UserTerminalView(language)
                view.line(event("old", "request_received", "startup", "python"))
                view.line(event("old", "request_finished", source="python", extra="status=cancelled"))
                # The raw native owner-binding event precedes the API receipt.
                self.assertIsNone(view.line(event("new", "native_observer_bound", "lifecycle")))
                anonymous = ("[fastllm-event] source=python category=progress request=0 native_handle=0 "
                             "event=request_status phase=prefill native_status_state=observed\n")
                self.assertIsNotNone(view.line(anonymous))
                self.assertIsNotNone(view.line(event("new", "request_received", "startup", "python")))
                self.assertIsNone(view.line(event("old", "native_first_token", extra="ttft_ms=2050.914")))
                warning = view.line(event("old", "cleanup_warning", "warning", extra="message=cleanup_warning"))
                error = view.line(event("old", "cleanup_error", "error", extra="message=cleanup_error"))
                self.assertIn("cleanup_warning", warning)
                self.assertIn("cleanup_error", error)
                self.assertEqual(view._handles["0"], "new")
                self.assertIsNotNone(view.line(event("new", "prefill_speed", "prefill",
                    extra="prefill_tokens=256 prefill_tokens_s=123.456")))
                self.assertNotIn("old", view._requests)

    def test_anonymous_late_records_stay_hidden_until_a_new_owner_is_observed(self):
        view = colorizer.UserTerminalView()
        view.line(event("old", "request_received", "startup", "python"))
        view.line(event("old", "request_finished", source="python", extra="status=cancelled"))
        anonymous = ("[fastllm-event] source=python category=progress request=0 native_handle=0 "
                     "event=request_status phase=prefill native_status_state=observed\n")
        self.assertIsNone(view.line(anonymous))
        view.line(event("new", "native_observer_bound", "lifecycle"))
        self.assertIsNotNone(view.line(anonymous))

    def test_terminal_history_and_retained_handle_owners_are_bounded(self):
        with patch.object(colorizer, "_USER_TERMINAL_HISTORY_LIMIT", 2):
            view = colorizer.UserTerminalView()
            for number in range(8):
                public = "owner-" + str(number)
                view.line(event(public, "request_received", "startup", "python", handle=str(number)))
                view.line(event(public, "request_finished", source="python",
                                handle=str(number), extra="status=cancelled"))
                self.assertLessEqual(len(view._terminal_requests), 2)
                self.assertLessEqual(len(view._handles), 2)
            self.assertEqual(set(view._terminal_requests), {"owner-6", "owner-7"})
            self.assertIsNone(view.line(event("owner-7", "prefill_summary", "prefill",
                                             handle="7", extra="missed_input_tokens=4")))
            self.assertIsNotNone(view.line(event("fresh-owner", "request_received", "startup", "python",
                                                handle="7")))


if __name__ == "__main__":
    unittest.main()
