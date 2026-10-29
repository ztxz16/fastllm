"""Exercise the real native HTTP server, including cancellation and multipart edits.
Run only when the server owns the authorized GPU. Uses Python's standard library.
"""

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8008")
p.add_argument("--key", default="")
p.add_argument("--image", required=True)
a = p.parse_args()


def request(path, body=None, content_type="application/json", auth=True):
    headers = {"Content-Type": content_type}
    if auth and a.key:
        headers["Authorization"] = "Bearer " + a.key
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    req = urllib.request.Request(a.url + path, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def data(path, body=None, expected=200, **kwargs):
    code, raw = request(path, body, **kwargs)
    assert code == expected, (path, code, raw[:1000])
    return json.loads(raw)


def wait(job):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        result = data("/v1/images/jobs/" + job["id"])
        if result["status"] in ["completed", "failed", "cancelled"]:
            return result
        time.sleep(0.5)
    raise TimeoutError(job)


assert data("/health")["status"] == "ready"
assert data("/v1/models")["data"][0]["id"] == "qwen-image-2.1"
assert data("/openapi.json")["openapi"] == "3.0.3"
if a.key:
    assert request("/v1/models", auth=False)[0] == 401
base = dict(model="qwen-image-2.1", prompt="A red apple.", size="128x128", num_inference_steps=2, seed=42)
for invalid in [
    dict(size="129x128"),
    dict(n=2),
    dict(seed=-1),
    dict(prompt=""),
    dict(mask="unsupported"),
    dict(num_inference_steps=1),
    dict(model="other"),
    dict(prompt="a " * 5000),
]:
    assert (
        data("/v1/images/generations", {**base, **invalid}, 400)["error"]["type"] == "invalid_request_error"
    )
assert request("/v1/images/generations", b"bad JSON")[0] == 400
assert request("/v1/images/edits", base)[0] == 400
first = data("/v1/images/generations", {**base, "response_format": "b64_json"})
png = base64.b64decode(first["data"][0]["b64_json"])
assert png.startswith(b"\x89PNG\r\n\x1a\n")
job = wait(data("/v1/images/jobs", base, 202))
assert job["status"] == "completed", job
assert request(job["url"])[1] == png, "same seed must reproduce identical native output"
# Active cancellation and queue capacity must leave the worker usable.
queued = [data("/v1/images/jobs", {**base, "num_inference_steps": 40}, 202) for _ in range(4)]
assert request("/v1/images/jobs", base)[0] == 429
for job in queued:
    data("/v1/images/jobs/" + job["id"] + "/cancel", {})
for job in queued:
    assert wait(job)["status"] == "cancelled"
boundary = "FastLLMTestBoundary42"
parts = []
for key, value in {**base, "reference_resolution": 256, "response_format": "b64_json"}.items():
    parts.append(
        (
            "--"
            + boundary
            + '\r\nContent-Disposition: form-data; name="'
            + key
            + '"\r\n\r\n'
            + str(value)
            + "\r\n"
        ).encode()
    )
parts.append(
    (
        "--"
        + boundary
        + '\r\nContent-Disposition: form-data; name="image[]"; filename="reference.png"\r\nContent-Type: image/png\r\n\r\n'
    ).encode()
    + Path(a.image).read_bytes()
    + b"\r\n"
)
parts.append(("--" + boundary + "--\r\n").encode())
result = data("/v1/images/edits", b"".join(parts), content_type="multipart/form-data; boundary=" + boundary)
assert base64.b64decode(result["data"][0]["b64_json"]).startswith(b"\x89PNG")
assert request("/v1/images/jobs/image_0_0")[0] == 404
print(
    "PASS: schemas, authentication, validation, deterministic PNG, async jobs, queue bound, cancellation recovery, multipart edits"
)
