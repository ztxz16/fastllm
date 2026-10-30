"""Exercise the real HTTP service, including cancellation and recovery after it.

The server must be launched with --api-key test-h3 (or supply --key), and a single
permitted GPU. This test produces a small 124-frame MP4 with a two-point schedule.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:8009")
p.add_argument("--key", default="test-h3")
p.add_argument("--output", required=True)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)


def request(method, path, data=None, auth=True, raw=None, content_type=None):
    headers = {}
    if auth:
        headers["Authorization"] = "Bearer " + a.key
    if data is not None:
        raw = json.dumps(data).encode()
        content_type = "application/json"
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(a.base + path, data=raw, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


def obj(method, path, data=None, expected=200, **kwargs):
    code, body, _ = request(method, path, data, **kwargs)
    assert code == expected, (method, path, code, body)
    return json.loads(body)


for _ in range(60):
    try:
        if request("GET", "/health")[0] == 200:
            break
    except OSError:
        pass
    time.sleep(1)
else:
    raise RuntimeError("server did not become ready")
assert request("GET", "/demo", auth=False)[0] == 200
assert obj("GET", "/openapi.json", auth=False)["openapi"] == "3.1.0"
obj("GET", "/v1/models", auth=False, expected=401)
assert obj("GET", "/v1/models")["data"][0]["id"] == "minimax-h3"
valid = {
    "model": "minimax-h3",
    "prompt": "A cat plays piano.",
    "size": "64x64",
    "seconds": "5",
    "seed": 42,
    "num_inference_steps": 2,
}
for change in [
    {"size": "65x64"},
    {"size": "64x8192"},
    {"seconds": "15"},
    {"prompt": ""},
    {"model": "wrong"},
    {"num_inference_steps": 1},
    {"frames": 124},
    {"reference_image": "x"},
    {"seconds": "nan"},
    {"seconds": True},
]:
    obj("POST", "/v1/videos", {**valid, **change}, expected=400)
obj("POST", "/v1/videos", raw=b"{", content_type="application/json", expected=400)
obj("GET", "/v1/videos/video_0_0", expected=404)
jobs = [obj("POST", "/v1/videos", {**valid, "num_inference_steps": 4}, expected=202)]
jobs += [obj("POST", "/v1/videos", valid, expected=202) for _ in range(3)]
obj("POST", "/v1/videos", valid, expected=429)
obj("DELETE", "/v1/videos/" + jobs[0]["id"], expected=409)
# Cancel after the first full DiT pass, when optional packed weights are cached.
# The subsequent multipart generation must recover from this populated state.
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    running = obj("GET", "/v1/videos/" + jobs[0]["id"])
    if running["stage"] == "denoise" and running["current"] >= 1:
        break
    assert running["status"] not in ("completed", "failed", "cancelled"), running
    time.sleep(0.1)
else:
    raise AssertionError(running)
for job in jobs:
    obj("POST", "/v1/videos/" + job["id"] + "/cancel")
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    current = [obj("GET", "/v1/videos/" + job["id"]) for job in jobs]
    if all(job["status"] == "cancelled" for job in current):
        break
    time.sleep(1)
else:
    raise AssertionError(current)
for job in jobs:
    obj("GET", "/v1/videos/" + job["id"] + "/content", expected=409)
    obj("DELETE", "/v1/videos/" + job["id"])
    obj("GET", "/v1/videos/" + job["id"], expected=404)
# Multipart follows the same contract as JSON; unknown file fields are rejected.
boundary = "fastllm-h3-test"
body = (
    "".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
        for key, value in valid.items()
    )
    + f"--{boundary}--\r\n"
)
job = obj(
    "POST",
    "/v1/videos",
    raw=body.encode(),
    content_type=f"multipart/form-data; boundary={boundary}",
    expected=202,
)
assert job["frames"] == 124 and job["fps"] == 24
start = time.monotonic()
deadline = start + 600
while time.monotonic() < deadline:
    job = obj("GET", "/v1/videos/" + job["id"])
    if job["status"] == "completed":
        break
    assert job["status"] not in ("failed", "cancelled"), job
    time.sleep(2)
else:
    raise AssertionError(job)
code, video, mime = request("GET", "/v1/videos/" + job["id"] + "/content")
assert code == 200 and mime.startswith("video/mp4") and video[4:8] == b"ftyp"
(out / "video.mp4").write_bytes(video)
assert obj("GET", "/v1/videos")["data"][0]["id"] == job["id"]
obj("DELETE", "/v1/videos/" + job["id"])
assert not obj("GET", "/v1/videos")["data"]
report = {
    "completed_job": job,
    "seconds": time.monotonic() - start,
    "video_bytes": len(video),
    "checks": [
        "authentication",
        "validation",
        "queue_limit",
        "cancel",
        "cancel_after_denoise",
        "delete",
        "multipart",
        "recovery",
        "generation",
        "download",
        "list",
    ],
}
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
