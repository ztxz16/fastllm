"""HY-Nomas HTTP transport. All model inference and sampling run in C++/CUDA.

Uses fastllm's existing FastAPI/uvicorn dependencies and Python's standard
library. Frames are PNG, so neither FFmpeg nor a video codec package is needed.
"""
import argparse
import asyncio
import ctypes as C
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import queue
import shutil
import struct
import sys
import threading
import time
from typing import Literal, Optional
import uuid
import zlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
import uvicorn


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user"]
    content: str = Field(min_length=1, max_length=16384)


class PromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: Optional[str] = Field(default=None, min_length=1, max_length=16384)
    messages: Optional[list[Message]] = Field(default=None, min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_prompt(self):
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("provide exactly one of prompt or messages")
        if len(self.caption().encode()) > 16384 or not self.caption().strip():
            raise ValueError("prompt must be nonempty and at most 16384 UTF-8 bytes")
        return self

    def caption(self):
        return self.prompt if self.prompt is not None else "\n".join(m.content for m in self.messages)


class GenerateRequest(PromptRequest):
    model: Literal["hy-nomas"] = "hy-nomas"
    width: int = Field(default=128, ge=64, le=512, multiple_of=32, strict=True)
    height: int = Field(default=128, ge=64, le=512, multiple_of=32, strict=True)
    num_frames: int = Field(default=16, ge=4, le=40, multiple_of=4, strict=True)
    seed: int = Field(default=42, ge=0, le=2**64-1, strict=True)
    steps: int = Field(default=8, ge=1, le=32, strict=True)
    response_format: Literal["png_frames"] = "png_frames"


Action = Literal["stay", "forward", "backward", "left", "right", "look_left", "look_right", "look_up", "look_down", "follow_prompt"]
ACTIONS = ("stay", "forward", "backward", "left", "right", "look_left", "look_right", "look_up", "look_down", "follow_prompt")


class WorldRequest(GenerateRequest):
    width: int = Field(default=384, ge=64, le=512, multiple_of=32, strict=True)
    height: int = Field(default=224, ge=64, le=512, multiple_of=32, strict=True)
    num_frames: Literal[20] = 20

    @model_validator(mode="after")
    def validate_area(self):
        if self.width*self.height > 512*288:
            raise ValueError("world resolution is limited to 147456 pixels (for example 512x288)")
        return self


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Action


class WorldGenerationRequest(WorldRequest):
    world_id: str
    action: Action


class NativeBackend:
    callback_type = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p)

    def __init__(self, library, model, text_encoder, device=0):
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.fastllm_nomas_error.restype = C.c_char_p
        self.lib.fastllm_nomas_pipeline_create.argtypes = [C.c_char_p, C.c_char_p, C.c_int]
        self.lib.fastllm_nomas_pipeline_create.restype = C.c_void_p
        self.lib.fastllm_nomas_pipeline_destroy.argtypes = [C.c_void_p]
        self.lib.fastllm_nomas_generate.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_int, C.c_int,
                                                   C.c_uint64, C.c_int, C.POINTER(C.c_uint8), C.c_size_t,
                                                   self.callback_type, C.c_void_p]
        self.worlds = {}
        self.lib.fastllm_nomas_world_create.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_int, C.c_uint64, C.c_int]
        self.lib.fastllm_nomas_world_create.restype = C.c_void_p
        self.lib.fastllm_nomas_world_destroy.argtypes = [C.c_void_p, C.c_void_p]
        self.lib.fastllm_nomas_world_step.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.POINTER(C.c_uint8), C.c_size_t, self.callback_type, C.c_void_p]
        self.lib.fastllm_nomas_world_step_prompt.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_uint8), C.c_size_t, self.callback_type, C.c_void_p]
        self.handle = self.lib.fastllm_nomas_pipeline_create(str(model).encode(), str(text_encoder).encode(), device)
        if not self.handle:
            raise RuntimeError(self.lib.fastllm_nomas_error().decode())

    def generate(self, request, progress):
        size = request.width * request.height * request.num_frames * 3
        rgb = (C.c_uint8 * size)()
        callback_error = []

        @self.callback_type
        def callback(stage, current, total, _):
            try:
                return int(progress(stage, current, total))
            except Exception as exc:
                callback_error.append(exc)
                return 1

        if isinstance(request, WorldGenerationRequest):
            if request.world_id not in self.worlds:
                if progress(1, 0, 1):
                    raise RuntimeError("generation cancelled")
                handle = self.lib.fastllm_nomas_world_create(self.handle, request.caption().encode(), request.width, request.height, request.seed, request.steps)
                if not handle:
                    raise RuntimeError(self.lib.fastllm_nomas_error().decode())
                self.worlds[request.world_id] = handle
                if progress(1, 1, 1):
                    raise RuntimeError("generation cancelled")
            rc = self.lib.fastllm_nomas_world_step_prompt(self.handle, self.worlds[request.world_id], request.caption().encode(), ACTIONS.index(request.action), rgb, size, callback, None)
        else:
            rc = self.lib.fastllm_nomas_generate(self.handle, request.caption().encode(), request.width, request.height,
                                                request.num_frames, request.seed, request.steps, rgb, size, callback, None)
        if callback_error:
            raise callback_error[0]
        if rc:
            raise RuntimeError(self.lib.fastllm_nomas_error().decode())
        return bytes(rgb)

    def drop_world(self, world_id):
        handle = self.worlds.pop(world_id, None)
        if handle:
            self.lib.fastllm_nomas_world_destroy(self.handle, handle)

    def close(self):
        for world_id in list(self.worlds):
            self.drop_world(world_id)
        if self.handle:
            self.lib.fastllm_nomas_pipeline_destroy(self.handle)
            self.handle = None


def png_rgb(rgb, width, height):
    if len(rgb) != width * height * 3:
        raise ValueError("wrong RGB frame size")

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data) & 0xffffffff)

    scanlines = b"".join(b"\0" + rgb[y*width*3:(y+1)*width*3] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines, 3)) + chunk(b"IEND", b""))


class Jobs:
    def __init__(self, backend, output, max_queue=4, max_results=16):
        self.backend, self.output = backend, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.pending = queue.Queue(maxsize=max_queue)
        self.jobs, self.lock = {}, threading.RLock()
        self.worlds = {}
        self.stopping = threading.Event()
        self.max_results = max_results
        self.worker = threading.Thread(target=self.run, name="hy-nomas-gpu-worker", daemon=True)

    def start(self):
        self.worker.start()

    def close(self):
        self.stopping.set()
        self.worker.join(timeout=60)
        if not self.worker.is_alive():
            self.backend.close()

    def submit(self, request):
        with self.lock:
            if self.stopping.is_set():
                raise HTTPException(503, "server is shutting down")
            terminal = [j for j in self.jobs.values() if j["status"] in ("completed", "failed", "cancelled")]
            pinned = {w["latest_video"] for w in self.worlds.values()}
            while len(terminal) >= self.max_results:
                old = next((j for j in terminal if j["id"] not in pinned), None)
                if old is None:
                    break
                terminal.remove(old)
                self.jobs.pop(old["id"])
                shutil.rmtree(self.output / old["id"], ignore_errors=True)
            job_id = "video_" + uuid.uuid4().hex
            job = dict(id=job_id, object="video", model="hy-nomas", status="queued", created_at=int(time.time()),
                       progress=0, stage="queued", frames_ready=0, width=request.width, height=request.height,
                       num_frames=request.num_frames, fps=24, seed=request.seed, response_format="png_frames",
                       request=request, cancel=threading.Event())
            if isinstance(request, WorldGenerationRequest):
                world = self.worlds[request.world_id]
                job.update(world_id=request.world_id, action=request.action, start_frame=world["generated_frames"],
                           segment_index=world["segments"]+1, prompt_revision=world["prompt_revision"],
                           text_condition="camera_continuation" if world["segments"]>0 and
                           request.action!="follow_prompt" and request.caption()==world["applied_prompt"]
                           else "scene_description")
            self.jobs[job_id] = job
            try:
                self.pending.put_nowait(job_id)
            except queue.Full:
                del self.jobs[job_id]
                raise HTTPException(429, "generation queue is full")
            if isinstance(request, WorldGenerationRequest):
                world.update(status="generating", current_job=job_id)
            return self.public(job)

    def create_world(self, request):
        with self.lock:
            if len(self.worlds) >= 2:
                raise HTTPException(429, "at most two worlds; delete an old world first")
            if sum(w["width"]*w["height"] for w in self.worlds.values())+request.width*request.height > 512*288:
                raise HTTPException(429, "world GPU cache budget reached; delete an old world or use a smaller resolution")
            world_id = "world_" + uuid.uuid4().hex
            world = dict(id=world_id, object="world", status="ready", width=request.width, height=request.height,
                         seed=request.seed, history_segments=6, segments=0, generated_frames=0, current_job=None, latest_video=None,
                         prompt=request.caption(), prompt_revision=1, applied_prompt=None, applied_prompt_revision=0, request=request)
            self.worlds[world_id] = world
            try:
                self.action_world(world_id, "stay")
            except Exception:
                del self.worlds[world_id]
                raise
            return self.get_world(world_id)

    def get_world(self, world_id):
        with self.lock:
            if world_id not in self.worlds:
                raise HTTPException(404, "unknown world; server restart requires a new world")
            return {k: v for k, v in self.worlds[world_id].items() if k != "request"}

    def action_world(self, world_id, action):
        with self.lock:
            self.get_world(world_id)
            world = self.worlds[world_id]
            if world["status"] != "ready":
                raise HTTPException(409, "world is busy, failed, or complete; wait or reset it")
            request = WorldGenerationRequest(**world["request"].model_dump(), world_id=world_id, action=action)
            return self.submit(request)

    def update_world_prompt(self, world_id, request):
        # This stages text only. Already-submitted jobs keep their immutable
        # request; the GPU worker encodes the new prompt at a segment boundary.
        with self.lock:
            self.get_world(world_id)
            world = self.worlds[world_id]
            if world["status"] not in ("ready", "generating"):
                raise HTTPException(409, "world is failed, deleting, or complete; reset it")
            caption = request.caption()
            if caption != world["prompt"]:
                world["request"] = world["request"].model_copy(update={"prompt": caption, "messages": None})
                world.update(prompt=caption, prompt_revision=world["prompt_revision"]+1)
            return self.get_world(world_id)

    def delete_world(self, world_id):
        with self.lock:
            self.get_world(world_id)
            if self.worlds[world_id]["status"] in ("generating", "deleting"):
                raise HTTPException(409, "cancel the active action and wait before deleting this world")
            self.worlds[world_id]["status"] = "deleting"
        # Do not hold the HTTP lock while waiting for the native pipeline lock.
        self.backend.drop_world(world_id)
        with self.lock:
            del self.worlds[world_id]
        return {"id": world_id, "deleted": True}

    @staticmethod
    def public(job):
        result = {k: v for k, v in job.items() if k not in ("request", "cancel", "commit_started")}
        result["events_url"] = f"/v1/videos/{job['id']}/events"
        if result.get("frames_ready", 0):
            result["output"] = {"format": "png_frames", "frames": [f"/v1/videos/{job['id']}/frames/{i}" for i in range(job["frames_ready"])]}
        return result

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, "unknown or expired video")
            return self.public(self.jobs[job_id])

    def cancel(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, "unknown or expired video")
            job = self.jobs[job_id]
            if job.get("world_id") and job.get("commit_started") and job["status"] in ("in_progress", "cancelling"):
                raise HTTPException(409, "world action has committed; waiting for frame saving")
            if job["status"] not in ("completed", "failed", "cancelled"):
                if job["status"] == "queued" and job.get("world_id"):
                    self.worlds[job["world_id"]]["status"] = "ready"
                job["cancel"].set()
                job["status"] = "cancelling" if job["status"] in ("in_progress", "cancelling") else "cancelled"
            return self.public(job)

    def run(self):
        while not self.stopping.is_set():
            try:
                job_id = self.pending.get(timeout=.2)
            except queue.Empty:
                continue
            with self.lock:
                job = self.jobs.get(job_id)
                if job is None or job["cancel"].is_set():
                    self.pending.task_done()
                    continue
                job["status"] = "in_progress"
            directory = self.output / job_id

            def progress(stage, current, total):
                with self.lock:
                    job["stage"] = {1: "text_encoding", 2: "denoising", 3: "decoding", 4: "saving"}[stage]
                    start, span = {1: (0, 10), 2: (10, 70), 3: (80, 15), 4: (95, 0)}[stage]
                    job["progress"] = round(start + span * current / max(total, 1))
                    cancelled = self.stopping.is_set() or job["cancel"].is_set()
                    if stage == 4 and job.get("world_id") and not cancelled:
                        job["commit_started"] = True
                    return cancelled

            try:
                started = time.monotonic()
                rgb = self.backend.generate(job["request"], progress)
                with self.lock:
                    job["generation_seconds"] = round(time.monotonic()-started, 4)
                    job["generation_fps"] = round(job["num_frames"]/max(job["generation_seconds"], .0001), 2)
                directory.mkdir()
                size = job["width"] * job["height"] * 3
                for i in range(job["num_frames"]):
                    if self.stopping.is_set() or job["cancel"].is_set():
                        raise RuntimeError("generation cancelled")
                    (directory / f"{i}.png").write_bytes(png_rgb(rgb[i*size:(i+1)*size], job["width"], job["height"]))
                    with self.lock:
                        job["frames_ready"] = i+1
                with self.lock:
                    if job["cancel"].is_set():
                        raise RuntimeError("generation cancelled")
                    job.update(status="completed", progress=100, stage="completed", completed_at=int(time.time()), total_seconds=round(time.monotonic()-started, 4))
                    if job.get("world_id"):
                        world = self.worlds[job["world_id"]]
                        world.update(status="complete" if world["generated_frames"]+20 >= 2880 else "ready",
                                     segments=world["segments"]+1, generated_frames=world["generated_frames"]+20,
                                     latest_video=job_id, last_action=job["action"],
                                     last_text_condition=job["text_condition"],
                                     applied_prompt=job["request"].caption(), applied_prompt_revision=job["prompt_revision"])
            except Exception as exc:
                shutil.rmtree(directory, ignore_errors=True)
                with self.lock:
                    cancelled = self.stopping.is_set() or job["cancel"].is_set()
                    job.update(status="cancelled" if cancelled else "failed", stage="cancelled" if cancelled else "failed")
                    job["frames_ready"] = 0
                    if job.get("world_id"):
                        self.worlds[job["world_id"]]["status"] = "failed" if job.get("commit_started") else "ready"
                    if not cancelled:
                        job["error"] = {"type": "generation_error", "message": str(exc)}
            finally:
                self.pending.task_done()


def create_app(backend, output, precision="fp32"):
    jobs = Jobs(backend, output)

    @asynccontextmanager
    async def lifespan(app):
        jobs.start()
        yield
        jobs.close()

    app = FastAPI(title="fastllm HY-Nomas", version="0.1.0", lifespan=lifespan)
    app.state.jobs = jobs

    @app.exception_handler(HTTPException)
    async def error_handler(request, exc):
        return JSONResponse({"error": {"type": "request_error", "message": exc.detail, "code": exc.status_code}}, status_code=exc.status_code)

    @app.get("/", response_class=HTMLResponse)
    def demo():
        return Path(__file__).with_name("nomas_demo.html").read_text()

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": "native_cpp_cuda", "precision": precision,
                "precision_status": "baseline" if precision == "fp32" else "experimental",
                "torch_loaded": "torch" in sys.modules}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "hy-nomas", "object": "model", "owned_by": "tencent-hunyuan",
                                             "capabilities": ["text_to_video", "png_frames", "world_sessions", "camera_actions", "text_only_continuation", "live_prompt_edits", "sse_progress"], "preview": True}]}

    @app.post("/v1/worlds", status_code=202)
    def create_world(request: WorldRequest):
        return jobs.create_world(request)

    @app.get("/v1/worlds/{world_id}")
    def world_status(world_id: str):
        return jobs.get_world(world_id)

    @app.patch("/v1/worlds/{world_id}")
    def update_world_prompt(world_id: str, request: PromptRequest):
        """Stage a scene description without resetting history or starting a job.

        Accepts prompt or user messages. Already submitted segments retain
        their text; the next action uses the latest accepted revision.
        """
        return jobs.update_world_prompt(world_id, request)

    @app.post("/v1/worlds/{world_id}/actions", status_code=202)
    def world_action(world_id: str, request: ActionRequest):
        return jobs.action_world(world_id, request.action)

    @app.delete("/v1/worlds/{world_id}")
    def delete_world(world_id: str):
        return jobs.delete_world(world_id)

    @app.post("/v1/videos", status_code=202)
    @app.post("/v1/generate", status_code=202)
    def generate(request: GenerateRequest):
        return jobs.submit(request)

    @app.get("/v1/videos/{job_id}")
    @app.get("/v1/generate/{job_id}")
    def status(job_id: str):
        return jobs.get(job_id)

    @app.get("/v1/videos/{job_id}/events", response_class=StreamingResponse,
             responses={200: {"content": {"text/event-stream": {}}}})
    async def events(job_id: str):
        """SSE snapshots, including committed PNG URLs as frames become available.

        Reconnecting returns the current snapshot. Disconnecting does not cancel
        inference; use the explicit cancel endpoint. Decoder inference still
        operates on complete 20-frame world segments.
        """
        jobs.get(job_id)
        async def stream():
            previous = None
            while True:
                try:
                    state = jobs.get(job_id)
                except HTTPException:
                    yield 'event: expired\ndata: {}\n\n'
                    return
                data = json.dumps(state, separators=(",", ":"))
                if data != previous:
                    yield "data: " + data + "\n\n"
                    previous = data
                if state["status"] in ("completed", "failed", "cancelled"):
                    return
                await asyncio.sleep(.1)
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/v1/videos/{job_id}/cancel")
    def cancel(job_id: str):
        return jobs.cancel(job_id)

    @app.get("/v1/videos/{job_id}/frames/{index}")
    def frame(job_id: str, index: int):
        job = jobs.get(job_id)
        if not 0 <= index < job["num_frames"]:
            raise HTTPException(404, "frame index out of range")
        if index >= job["frames_ready"]:
            raise HTTPException(409, "frame is not ready")
        return FileResponse(jobs.output / job_id / f"{index}.png", media_type="image/png")

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8007)
    args = parser.parse_args()
    os.environ["FASTLLM_NOMAS_PRECISION"] = args.precision
    uvicorn.run(create_app(NativeBackend(args.library, args.model, args.text_encoder, args.device), args.output, args.precision),
                host=args.host, port=args.port, workers=1)
