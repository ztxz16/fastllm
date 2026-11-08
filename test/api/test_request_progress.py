#!/usr/bin/env python3
import asyncio
import sys
import threading
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/fastllm_pytools"))
from request_progress import RequestPhaseReporter, RequestProgressReporter, _PreparationHeartbeatMonitor

class RequestProgressTest(unittest.TestCase):
    def reporter(self):
        self.now = [0.0]; self.events = []
        return RequestProgressReporter("req-1", emit=self.events.append, mode="debug", clock=lambda: self.now[0])

    def test_callback_rates_are_labeled_and_not_native_decode(self):
        r=self.reporter();r.received(path="/v1/chat/completions");self.now[0]=1.0
        r.callback({"output_tokens":4})
        progress=next(x for x in self.events if "event=progress" in x)
        self.assertIn("measurement=python_callback_interval",progress)
        self.assertIn("python_callback_tokens_s=4.00",progress)
        self.assertIn("python_callback_events_s=1.00",progress)
        self.assertNotIn(" decode_tokens_s=",progress)
        self.assertNotIn(" output_tokens_s=",progress)
        self.now[0]=2.0;r.finish(output_tokens=5)
        done=next(x for x in self.events if "event=request_finished" in x)
        self.assertIn("output_tokens=5",done)
        self.assertIn("request_total_average_tokens_s=2.50",done)

    def test_empty_heartbeat_cannot_create_output_or_first_token(self):
        r=self.reporter();self.now[0]=5.0;r.callback({})
        self.assertFalse(any("event=first_token" in x for x in self.events))
        self.assertEqual(r._output_tokens,0)
        self.assertNotIn("python_callback_tokens_s",self.events[-1])
        self.now[0]=6.0;r.callback({"output_tokens":3})
        first=next(x for x in self.events if "event=first_token" in x)
        self.assertIn("ttft_ms=6000.0",first)
        self.assertIn("measurement=api_observed_native_tokens",first)
        self.assertEqual(r._output_tokens,3)

    def test_coalesced_or_empty_text_uses_native_count(self):
        r=self.reporter();self.now[0]=1.0;r.callback({"output_tokens":8})
        self.now[0]=2.0;r.callback({"output_tokens":8})
        self.assertEqual(r._output_tokens,8)
        self.assertEqual(sum("event=first_token" in x for x in self.events),1)

    def test_terminal_authoritative_count_replaces_previous_observation(self):
        r=self.reporter();self.now[0]=1.0;r.output(4);self.now[0]=2.0;r.finish(output_tokens=3)
        self.assertEqual(r._output_tokens,3)
        self.assertIn("output_tokens=3",self.events[-1])

    def test_off_remains_silent(self):
        events=[];r=RequestProgressReporter("req",emit=events.append,mode="off")
        r.received(path="/");r.callback({"output_tokens":4});r.finish(output_tokens=4)
        self.assertEqual(events,[])


class RequestPhaseHeartbeatTest(unittest.TestCase):
    def test_default_one_second_timer_survives_no_callbacks(self):
        events = []
        ticked = threading.Event()
        monitor = _PreparationHeartbeatMonitor()

        def emit(line):
            events.append(line)
            if len(events) >= 2:
                ticked.set()

        with RequestPhaseReporter("blocked-request", emit=emit, mode="user", monitor=monitor):
            # Simulate a synchronous preprocessing call: neither callback nor
            # event-loop iteration drives the reporter during this wait.
            self.assertTrue(ticked.wait(2.5), "independent one-second heartbeat missing")
        self.assertIn("event=phase_status", events[-1])
        elapsed = float(events[-1].split("phase_elapsed_ms=", 1)[1].split()[0])
        self.assertGreaterEqual(elapsed, 900)
        self.assertNotIn("tokens_s=", events[-1])
        self.assertNotIn("completed_tokens=", events[-1])
        self.assertEqual(monitor.active, {})

    def test_parallel_requests_keep_separate_state_and_stop_at_handover(self):
        now = [0.0]
        clock = lambda: now[0]
        monitor = _PreparationHeartbeatMonitor(clock=clock, start_thread=False)
        events = []
        first = RequestPhaseReporter("first", emit=events.append, mode="debug", clock=clock, monitor=monitor)
        second = RequestPhaseReporter("second", emit=events.append, mode="debug", clock=clock, monitor=monitor)
        with first, second:
            first.set_phase("image_preprocessing", "loading_and_decoding_image")
            second.set_phase("tokenization", "preparing_model_input")
            now[0] = 1.0
            monitor.poll()
            self.assertIn("request=first", events[-2])
            self.assertIn("phase=image_preprocessing", events[-2])
            self.assertIn("request=second", events[-1])
            self.assertIn("phase=tokenization", events[-1])
            first.close()
            now[0] = 2.0
            before = len(events)
            monitor.poll()
            self.assertEqual(len(events), before + 1)
            self.assertIn("request=second", events[-1])
        now[0] = 3.0
        before = len(events)
        monitor.poll()
        self.assertEqual(len(events), before)
        self.assertIsNone(RequestPhaseReporter.current())

    def test_phase_entry_resets_age_without_fabricating_progress(self):
        now = [0.0]
        monitor = _PreparationHeartbeatMonitor(clock=lambda: now[0], start_thread=False)
        events = []
        with RequestPhaseReporter("one", emit=events.append, mode="user", clock=lambda: now[0], monitor=monitor) as reporter:
            now[0] = 1.0
            monitor.poll()
            self.assertIn("idle_ms=1000.000", events[-1])
            reporter.set_phase("native_launch", "preparing_and_submitting_native_request")
            self.assertIn("phase_elapsed_ms=0.000", events[-1])
            now[0] = 2.0
            monitor.poll()
            self.assertIn("phase=native_launch", events[-1])
            self.assertIn("idle_ms=1000.000", events[-1])
            self.assertNotIn("total_tokens=", events[-1])

    def test_error_and_cancellation_remove_timer_and_context(self):
        monitor = _PreparationHeartbeatMonitor(start_thread=False)
        with self.assertRaisesRegex(ValueError, "invalid media"):
            with RequestPhaseReporter("error", mode="user", emit=lambda _: None, monitor=monitor):
                raise ValueError("invalid media")
        self.assertEqual(monitor.active, {})
        self.assertIsNone(RequestPhaseReporter.current())

        async def run():
            entered = asyncio.Event()

            async def pending():
                with RequestPhaseReporter("cancel", mode="user", emit=lambda _: None, monitor=monitor):
                    entered.set()
                    await asyncio.Event().wait()

            task = asyncio.create_task(pending())
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        self.assertEqual(monitor.active, {})
        self.assertIsNone(RequestPhaseReporter.current())

    def test_off_does_not_start_timer_or_emit(self):
        monitor = _PreparationHeartbeatMonitor()
        events = []
        with RequestPhaseReporter("off", emit=events.append, mode="off", monitor=monitor) as reporter:
            reporter.set_phase("image_preprocessing", "loading_and_decoding_image")
            self.assertEqual(monitor.active, {})
            self.assertIsNone(monitor.thread)
        self.assertEqual(events, [])

    def test_blocked_emitter_does_not_block_close_or_schedule_after_handover(self):
        now = [0.0]
        monitor = _PreparationHeartbeatMonitor(clock=lambda: now[0], start_thread=False)
        writing = threading.Event()
        release = threading.Event()
        closed = threading.Event()
        events = []

        def emit(line):
            if "phase_elapsed_ms=1000.000" in line:
                writing.set()
                release.wait(2.0)
            events.append(line)

        with RequestPhaseReporter("original-request", emit=emit, mode="user", clock=lambda: now[0], monitor=monitor) as reporter:
            now[0] = 1.0
            writer = threading.Thread(target=monitor.poll)
            writer.start()
            self.assertTrue(writing.wait(1.0))

            def close():
                reporter.close()
                closed.set()

            closer = threading.Thread(target=close)
            closer.start()
            try:
                self.assertTrue(closed.wait(0.5), "logging held the registry lock during close")
            finally:
                release.set()
                writer.join(2.0)
                closer.join(2.0)
            self.assertFalse(writer.is_alive())
            self.assertFalse(closer.is_alive())
            self.assertIn("request=original-request", events[-1])
            # The already captured write is allowed to finish; removal must
            # prevent every subsequent timer scheduling decision.
            count = len(events)
            now[0] = 2.0
            monitor.poll()
            self.assertEqual(len(events), count)

if __name__ == "__main__":unittest.main()
