"""Multipart conditioning integration: all modes, validation and downloadable media.

Use a dedicated GPU-7 server, not a service with unrelated queued work.
"""

import argparse
import io
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:8019")
p.add_argument("--key", default="test-h3")
p.add_argument("--output", required=True)
p.add_argument("--verify-restart", action="store_true")
p.add_argument("--server-output", type=Path, help="Shared server output directory for sidecar checks")
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
headers = {"Authorization": "Bearer " + a.key}


def request(method, path, body=None, content_type=None):
    h = dict(headers)
    if content_type:
        h["Content-Type"] = content_type
    req = urllib.request.Request(a.base + path, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def json_request(method, path, **kw):
    code, body = request(method, path, **kw)
    assert code == 200, (code, body)
    return json.loads(body)


def multipart(fields, files):
    boundary = "fastllm-h3-test-conditions"
    body = bytearray()
    for key, value in fields.items():
        body.extend(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        )
    for key, name, mime, data in files:
        body.extend(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{name}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
        )
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def submit(files, expected=202, **extra):
    fields = {
        "prompt": "A colorful object moves slowly. Soft music plays.",
        "size": "64x64",
        "frames": 124,
        "seed": 42,
        "num_inference_steps": 2,
        "reference_image_short_edge": 64,
        "reference_video_short_edge": 64,
        **extra,
    }
    body, ct = multipart(fields, files)
    code, data = request("POST", "/v1/videos", body, ct)
    assert code == expected, (code, data)
    return json.loads(data)


if a.verify_restart:
    previous = json.loads((out / "report.json").read_text())
    jobs = {job["id"]: job for job in json_request("GET", "/v1/videos")["data"]}
    for name, saved in previous.items():
        job = jobs[saved["id"]]
        assert job["status"] == "completed" and job["mode"] == saved["mode"], job
        assert job["num_inference_steps"] == (4 if name == "reference_audio" else 2), (
            job
        )
        code, content = request("GET", "/v1/videos/" + job["id"] + "/content")
        assert code == 200 and content == (out / (name + ".mp4")).read_bytes()
    print("PASS completed jobs and media survive restart")
    raise SystemExit(0)

health = json_request("GET", "/health")
assert health["conditioning"] and health["reference_model_available"], health
png = io.BytesIO()
Image.new("RGB", (64, 64), (145, 86, 43)).save(png, format="PNG")
image = png.getvalue()
first = ("input_reference", "first.png", "image/png", image)
jpeg = io.BytesIO()
Image.new("RGB", (64, 64), (145, 86, 43)).save(jpeg, format="JPEG")
last = ("last_frame", "last.jpg", "image/jpeg", jpeg.getvalue())
ref = ("reference", "subject.png", "image/png", image)
submit([first, first], 400)
submit([first, ref], 400)
submit([("input_reference", "bad.png", "image/png", b"not a PNG")], 400)
submit([("reference", "unknown.bin", "application/octet-stream", image)], 400)
submit([ref] * 10, 400)
submit([first], 400, reference_video_short_edge=65)
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=orange:s=64x64:r=24:d=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=220:sample_rate=32000:duration=2",
        "-ac",
        "2",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-c:a",
        "aac",
        "-shortest",
        str(out / "motion.mp4"),
    ],
    check=True,
)
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=330:sample_rate=44100:duration=2",
        "-ac",
        "2",
        str(out / "voice.wav"),
    ],
    check=True,
)
audio = ("reference", "voice.wav", "audio/wav", (out / "voice.wav").read_bytes())
video = ("reference", "motion.mp4", "video/mp4", (out / "motion.mp4").read_bytes())
submit([audio], 400)
reports = {}
for name, files, mode in [
    ("first", [first], "first_frame"),
    ("last", [last], "last_frame"),
    ("both", [first, last], "first_last"),
    ("reference_image", [ref], "ref2va"),
    ("reference_audio", [ref, audio], "ref2va"),
    ("reference_video", [video], "ref2va"),
]:
    begin = time.monotonic()
    job = submit(files, num_inference_steps=4 if name == "reference_audio" else 2)
    assert job["mode"] == mode, job
    deadline = begin + 600
    while job["status"] not in ("completed", "failed", "cancelled"):
        assert time.monotonic() < deadline, job
        time.sleep(1)
        job = json_request("GET", "/v1/videos/" + job["id"])
    assert job["status"] == "completed", job
    if a.server_output:
        saved = json.loads((a.server_output / (job["id"] + ".json")).read_text())
        assert saved == job, (saved, job)
        assert not (a.server_output / (job["id"] + ".json.tmp")).exists()
    code, content = request("GET", "/v1/videos/" + job["id"] + "/content")
    assert code == 200 and content[4:8] == b"ftyp"
    path = out / (name + ".mp4")
    path.write_bytes(content)
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
        )
    )
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    au = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert (v["width"], v["height"], int(v["nb_frames"])) == (64, 64, 124), v
    assert au["channels"] == 2, au
    reports[name] = {
        "id": job["id"],
        "mode": mode,
        "seconds": time.monotonic() - begin,
        "bytes": len(content),
    }
    (out / "report.json").write_text(json.dumps(reports, indent=2))
    print(name, reports[name], flush=True)
print("PASS all multipart conditioning modes and validation")
