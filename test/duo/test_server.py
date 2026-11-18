"""Exercise session ownership, malformed payloads and cancellation on a tiny GPU model."""

import argparse
import json
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from client import DuoClient


def expect(code, fn):
    try:
        fn()
    except HTTPError as error:
        assert error.code == code, (error.code, error.read())
        return
    raise AssertionError(f"expected HTTP {code}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = args.fixture
    process = subprocess.Popen(
        [args.binary, str(root), "4", "4", str(port), str(root / "adapter")]
    )
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(200):
            try:
                client = DuoClient(url, timeout=20)
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError("server exited")
                time.sleep(0.05)
        else:
            raise TimeoutError("server startup")
        shared = np.fromfile(root / "shared.f32", dtype="float32").reshape(3, 48)
        reference = (
            np.fromfile(root / "reference.f32", dtype="float32")
            .reshape(1, 2, 2, 24, 2, 2)
            .transpose(3, 0, 1, 4, 2, 5)
            .reshape(24, 1, 4, 4)
        )
        intent = np.fromfile(root / "intent-0.f32", dtype="float32").reshape(4, 48)
        client.reset(reference, shared)
        old = client.session
        expect(400, lambda: urlopen(Request(url + "/v1/session", b"bad"), timeout=5))
        first = client.chunk(0, intent, 42)
        assert first["video"].shape == (24, 5, 4, 4) and first["audio"].shape == (
            2,
            32,
            29,
        )
        assert np.isfinite(first["video"]).all()
        expect(400, lambda: client.chunk(0, intent, 42))
        expect(409, lambda: client.chunk(1, intent, 42))
        client.reset(reference, shared)
        expect(409, lambda: client._json("/v1/cancel", {"session": old}))
        client.cancel()
        expect(409, lambda: client.chunk(0, intent))
        client.reset(reference, shared)
        repeated = client.chunk(0, intent, 42)
        assert np.array_equal(first["video"], repeated["video"])
        # Force a longer request to exercise a cancel arriving while a GPU forward is running.
        client.reset(reference, shared)
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(client.chunk, 0, np.tile(intent, (256, 1)), 42)
            for _ in range(200):
                if client._json("/health")["busy"]:
                    break
                if future.done():
                    raise AssertionError(
                        "cancellation request finished before becoming busy"
                    )
                time.sleep(0.001)
            else:
                raise AssertionError("worker did not become busy")
            expect(409, lambda: client.chunk(1, intent))
            client.cancel()
            expect(400, future.result)
        expect(409, lambda: client.chunk(1, intent))
        client.reset(reference, shared)
        client.chunk(0, intent, 42)
        print(
            json.dumps(
                {
                    "session_reset": True,
                    "stale_session": True,
                    "out_of_order": True,
                    "inflight_cancel": True,
                    "reproducible_seed": True,
                    "malformed_packet": True,
                }
            )
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
