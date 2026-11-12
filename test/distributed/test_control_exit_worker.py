#!/usr/bin/env python3
"""验证退出故障夹具的真实进程和协议行为；不编译、不代替 C++ 底座验收。"""

from contextlib import contextmanager
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest

import control_exit_worker as fixture


@contextmanager
def workers(mode):
    with tempfile.TemporaryDirectory(prefix="fastllm-exit-fixture-") as directory:
        connections, processes = [], []
        try:
            for rank in range(2):
                parent, child = socket.socketpair()
                connections.append(parent)
                parent.settimeout(3)
                args = [sys.executable, str(Path(fixture.__file__).resolve()),
                        "--control-fd", str(child.fileno()), "--rank", str(rank),
                        "--world-size", "2", "--device", str(rank), "--session", "123",
                        "--control-only", "1"]
                environment = dict(os.environ, FASTLLM_TEST_EXIT_MODE=mode,
                                   FASTLLM_TEST_EXIT_DIR=directory)
                try:
                    processes.append(subprocess.Popen(args, pass_fds=(child.fileno(),), env=environment,
                                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
                finally:
                    child.close()
            yield connections, processes
        finally:
            for connection in connections:
                connection.close()
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                process.stderr.close()


class ExitFixtureTest(unittest.TestCase):
    def initialize(self, connections, processes):
        for rank, connection in enumerate(connections):
            kind, session, sequence, payload = fixture.receive(connection)
            self.assertEqual((kind, session, sequence), (1, 123, 0))
            self.assertEqual(struct.unpack("<IIII", payload[:16]), (rank, rank, processes[rank].pid, 0))
            connection.sendall(fixture.frame(4, 123, 1))
        for connection in connections:
            self.assertEqual(fixture.receive(connection), (5, 123, 1, b""))

    def ping(self, connections):
        for connection in connections:
            connection.sendall(fixture.frame(6, 123, 2))

    def test_peer_exits_after_other_worker_received_command(self):
        for mode, prefix in (("wait_peer_exit", b""),
                             ("partial_header_peer_exit", fixture.frame(7, 123, 2)[:8]),
                             ("partial_payload_peer_exit", fixture.frame(7, 123, 2, b"12345678")[:-4])):
            with self.subTest(mode=mode), workers(mode) as (connections, processes):
                self.initialize(connections, processes)
                self.ping(connections)
                self.assertEqual(processes[1].wait(timeout=3), 0)
                self.assertIsNone(processes[0].poll())
                if prefix:
                    data = bytearray()
                    while len(data) < len(prefix):
                        chunk = connections[0].recv(len(prefix) - len(data))
                        self.assertTrue(chunk, "fixture closed before sending the expected partial frame")
                        data.extend(chunk)
                    self.assertEqual(bytes(data), prefix)
                connections[0].settimeout(0.05)
                with self.assertRaises(socket.timeout):
                    connections[0].recv(1)

    def test_error_diagnostics_are_complete_partial_or_stale(self):
        for mode in ("buffered_error_exit", "partial_error_exit", "stale_error_exit"):
            with self.subTest(mode=mode), workers(mode) as (connections, processes):
                self.initialize(connections, processes)
                self.ping(connections)
                self.assertEqual(processes[1].wait(timeout=3), 0)
                data = bytearray()
                while True:
                    chunk = connections[1].recv(4096)
                    if not chunk:
                        break
                    data.extend(chunk)
                expected = fixture.frame(10, 123, 3 if mode == "stale_error_exit" else 2, fixture.ERROR_TEXT)
                self.assertEqual(bytes(data), expected[:-4] if mode == "partial_error_exit" else expected)
                self.assertIsNone(processes[0].poll())

    def test_current_worker_sends_error_before_exit(self):
        with workers("current_error_exit") as (connections, processes):
            self.initialize(connections, processes)
            self.ping(connections)
            self.assertEqual(fixture.receive(connections[0]), (10, 123, 2, fixture.ERROR_TEXT))
            self.assertEqual(processes[0].wait(timeout=3), 0)

    def test_startup_peer_exit_precedes_hello(self):
        with workers("startup_peer_exit") as (connections, processes):
            self.assertEqual(processes[1].wait(timeout=3), 0)
            self.assertIsNone(processes[0].poll())
            self.assertEqual(connections[1].recv(1), b"")

    def test_reply_then_exit_precedes_other_reply(self):
        with workers("reply_then_exit") as (connections, processes):
            self.initialize(connections, processes)
            self.ping(connections)
            self.assertEqual(fixture.receive(connections[1]), (7, 123, 2, b""))
            self.assertEqual(processes[0].wait(timeout=3), 0)

    def test_live_stall_has_no_exit(self):
        with workers("stall") as (connections, processes):
            self.initialize(connections, processes)
            self.ping(connections)
            self.assertEqual(fixture.receive(connections[1]), (7, 123, 2, b""))
            connections[0].settimeout(0.05)
            with self.assertRaises(socket.timeout):
                connections[0].recv(1)
            self.assertTrue(all(process.poll() is None for process in processes))

    def test_shutdown_other_worker_exits_before_first_reply(self):
        with workers("shutdown_order") as (connections, processes):
            self.initialize(connections, processes)
            self.ping(connections)
            for connection in connections:
                self.assertEqual(fixture.receive(connection), (7, 123, 2, b""))
                connection.sendall(fixture.frame(8, 123, 3))
            self.assertEqual(fixture.receive(connections[0]), (9, 123, 3, b""))
            self.assertEqual(processes[1].wait(timeout=3), 0)
            self.assertEqual(fixture.receive(connections[1]), (9, 123, 3, b""))
            self.assertEqual(processes[0].wait(timeout=3), 0)


if __name__ == "__main__":
    unittest.main()
