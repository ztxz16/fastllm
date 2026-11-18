"""Exercise the single-GPU live service with a short video containing speech."""

import argparse
import json
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen


def check_avc_stream(stream, chunks):
    offset = 0
    for chunk in chunks:
        size, block, frames, samples = struct.unpack_from("<4I", stream, offset)
        assert (block, frames, samples) == (
            chunk["index"],
            chunk["frames"],
            chunk["samples"],
        )
        lengths = struct.unpack_from(f"<{frames}I", stream, offset + 16)
        pos = offset + 16 + 4 * frames + 4 * samples
        for length in lengths:
            length &= 0x7FFFFFFF
            assert length > 0 and stream.startswith((b"\0\0\1", b"\0\0\0\1"), pos)
            pos += length
        offset += 4 + size
        assert pos == offset <= len(stream)
    assert offset == len(stream)


def smoke(url, video, playback):
    def fetch(path, method="GET", body=None):
        request = Request(url.rstrip("/") + path, data=body, method=method)
        with urlopen(request, timeout=40) as response:
            return response.read()

    selected = json.loads(fetch("/resolutions"))[0]
    expected_size = tuple(map(int, selected.split("x")))
    session = (
        "/sessions/" + json.loads(fetch("/sessions?playback=" + playback, "POST"))["id"]
    )
    cursor = 0

    def until(kind, timeout=300):
        nonlocal cursor
        seen = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            batch = json.loads(fetch(f"{session}/events?after={cursor}"))
            cursor = batch["cursor"]
            seen.extend(batch["events"])
            for item in batch["events"]:
                if item["type"] == kind:
                    return seen
                if item["type"] in {"error", "input_error", "closed"}:
                    raise RuntimeError(item)
            if batch["closed"]:
                raise RuntimeError("Worker exited")
        raise TimeoutError(kind)

    recording = video.read_bytes()
    closed = False
    try:
        rejected = json.loads(fetch(session + "/turns", "POST", b"invalid video"))
        until("input_error")
        accepted = json.loads(fetch(session + "/turns", "POST", recording))
        first = until("done")
        chunks = [e for e in first if e["type"] == "chunk"]
        plans = [e for e in first if e["type"] == "plan"]
        done = next(e for e in first if e["type"] == "done")
        assert any(e["type"] == "transcript" and e["text"].strip() for e in first)
        assert len(chunks) == len(plans) == done["chunks"] > 1
        assert [e["index"] for e in chunks] == list(range(len(chunks)))
        assert all(e["frames"] == 17 for e in chunks)
        assert len(fetch(chunks[0]["url"])) > 0
        if playback == "avc":
            assert fetch(rejected["stream"]) == b""
            check_avc_stream(fetch(accepted["stream"]), chunks)
        with tempfile.TemporaryDirectory() as temp:
            mp4 = Path(temp) / "reply.mp4"
            mp4.write_bytes(fetch(done["url"]))
            probe = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(mp4)]
            )
        streams = json.loads(probe)["streams"]
        picture = next(s for s in streams if s["codec_type"] == "video")
        sound = next(s for s in streams if s["codec_type"] == "audio")
        frames = sum(e["frames"] for e in chunks)
        samples = sum(e["samples"] for e in chunks)
        assert samples == (frames * 32000 + 23) // 24
        assert picture["codec_name"] == "h264" and int(picture["nb_frames"]) == frames
        assert (picture["width"], picture["height"]) == expected_size
        assert sound["codec_name"] == "aac" and int(sound["sample_rate"]) == 32000
        assert sound["channels"] == 2
        assert abs(float(sound["duration"]) - samples / 32000) < 0.002

        fetch(session + "/turns", "POST", recording)
        second = until("chunk")
        first_plan = next(e for e in second if e["type"] == "plan")
        assert first_plan["index"] == plans[-1]["index"] + 1
        fetch(session, "DELETE")
        closed = True
        assert json.loads(fetch(session + f"/events?after={cursor}"))["closed"]
        fresh = "/sessions/" + json.loads(fetch("/sessions", "POST"))["id"]
        fetch(fresh, "DELETE")
        print(
            json.dumps(
                {
                    "session": session,
                    "first_turn_chunks": len(chunks),
                    "frames": frames,
                    "samples_per_channel": samples,
                    "second_turn_first_global_index": first_plan["index"],
                    "invalid_input_retry": True,
                    "cancel_and_reopen": True,
                    "playback": playback,
                },
                indent=2,
            )
        )
    finally:
        if not closed:
            fetch(session, "DELETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8190")
    parser.add_argument("--playback", choices=("mp4", "avc"), default="mp4")
    arguments = parser.parse_args()
    smoke(arguments.url, arguments.video, arguments.playback)
