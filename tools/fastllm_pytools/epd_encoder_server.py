"""EPD encoder 独立服务：只加载 qwen3_5 视觉塔，把图片编码成 embeddings 写入共享缓存目录。

用法：
    python -m ftllm.epd_encoder_server --model_path <模型目录> --port 10001 \
        --vision_device cuda:0 --cache_dir /dev/shm/fastllm_ec_cache

配合 consumer（ftllm server，env FASTLLM_EPD_CACHE_DIR 指向同一目录）与
epd_proxy 组成 E/PD 分离拓扑。缓存键与 consumer 预处理共用同一 sha256 算法，
两侧对同一张图算出的键一致。
"""

import argparse
import asyncio
import base64
import binascii
import ctypes
import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

try:
    from . import llm
    from .qwen35_multimodal_native import (
        EPD_EMBEDS_FILENAME,
        EPD_META_FILENAME,
        _append_payload_tensor,
        _compute_image_grid,
        _to_rgb_array,
        compute_qwen35_image_cache_digest,
        get_qwen35_multimodal_config,
        qwen35_epd_scoped_cache_dir,
        write_qwen35_epd_entries,
    )
except ImportError:
    import llm
    from qwen35_multimodal_native import (
        EPD_EMBEDS_FILENAME,
        EPD_META_FILENAME,
        _append_payload_tensor,
        _compute_image_grid,
        _to_rgb_array,
        compute_qwen35_image_cache_digest,
        get_qwen35_multimodal_config,
        qwen35_epd_scoped_cache_dir,
        write_qwen35_epd_entries,
    )


def _load_image_from_url(image_url: str) -> Image.Image:
    if not image_url:
        raise ValueError("Image URL cannot be empty.")
    parsed = urlparse(image_url)
    scheme = parsed.scheme.lower()
    try:
        if scheme in ("http", "https"):
            with urlopen(image_url, timeout=20) as response:
                return Image.open(response).convert("RGB")
        if scheme == "data":
            if not image_url.startswith("data:image/"):
                raise ValueError("Only image data URLs are supported.")
            header, encoded = image_url.split(",", 1)
            if ";base64" not in header:
                raise ValueError("Only base64-encoded image data URLs are supported.")
            try:
                image_bytes = base64.b64decode(encoded)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("Invalid base64 image data.") from exc
            import io
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if scheme == "file":
            file_path = unquote(parsed.path or "")
            if os.name == "nt" and len(file_path) >= 3 and file_path[0] == "/" and file_path[2] == ":":
                file_path = file_path[1:]
            with Image.open(file_path) as image:
                return image.convert("RGB")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to load image from {image_url!r}: {exc}") from exc
    raise ValueError(
        f"Unsupported image URL scheme in {image_url!r}. "
        "Supported schemes are http(s), data:image/... and file://."
    )


class EncodeRequest(BaseModel):
    images: List[str]


class EncoderState:
    model: Any = None
    multimodal_config: Dict[str, Any] = {}
    cache_dir: str = ""
    cache_max_bytes: int = 0
    lock: asyncio.Lock = None


app = FastAPI(title="fastllm EPD encoder")
state = EncoderState()


def _enforce_cache_cap(min_age_sec: float = 60.0) -> None:
    """缓存容量上限：超过上限时按创建时间从老到新驱逐（FIFO）。
    跳过 min_age_sec 内的新条目，避免误删 consumer 正在读取的缓存。"""
    if state.cache_max_bytes <= 0:
        return
    try:
        entries = []
        total = 0
        for name in os.listdir(state.cache_dir):
            path = os.path.join(state.cache_dir, name)
            if not os.path.isdir(path) or name.startswith("."):
                continue
            try:
                size = sum(
                    os.path.getsize(os.path.join(path, f))
                    for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))
                )
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            entries.append((mtime, size, path))
            total += size
        if total <= state.cache_max_bytes:
            return
        now = time.time()
        entries.sort()  # 最老的在前
        removed = 0
        for mtime, size, path in entries:
            if total <= state.cache_max_bytes:
                break
            if now - mtime < min_age_sec:
                continue
            shutil.rmtree(path, ignore_errors=True)
            total -= size
            removed += 1
        if removed:
            print(f"[EPD-encoder] cache over cap, evicted {removed} oldest entrie(s)", flush=True)
    except Exception as exc:
        print(f"[EPD-encoder] cache cap sweep failed: {exc}", flush=True)


def _encode_missing_images(raw_images: List[np.ndarray], grids: List[np.ndarray]) -> None:
    """对未命中缓存的图片跑 ViT 并写入共享缓存（raw_images/grids 仅含未命中项）。"""
    arrays: List[np.ndarray] = []
    descriptors: List[Dict[str, Any]] = []
    offset = 0
    grid_array = np.stack(grids, axis=0)
    offset = _append_payload_tensor(arrays, descriptors, "image_grid_thw", grid_array, np.int32, offset)
    for raw in raw_images:
        offset = _append_payload_tensor(
            arrays, descriptors, "image_frames",
            np.ascontiguousarray(raw, dtype=np.float32), np.float32, offset,
        )
    payload = b"".join(array.tobytes(order="C") for array in arrays)
    payload_config = {"mode": "qwen35", "tensors": descriptors}

    fd, features_path = tempfile.mkstemp(prefix=".epd-enc-", dir=state.cache_dir)
    os.close(fd)
    meta_buf = ctypes.create_string_buffer(65536)
    try:
        payload_buffer = ctypes.create_string_buffer(payload)
        ret = llm.fastllm_lib.encode_visual_items_llm_model(
            state.model.model,
            json.dumps(payload_config).encode(),
            payload_buffer,
            features_path.encode(),
            meta_buf,
            len(meta_buf),
        )
        if ret != 0:
            raise RuntimeError(f"encode_visual_items_llm_model failed with code {ret}")
        meta = json.loads(meta_buf.value.decode("utf-8"))
        features = np.fromfile(features_path, dtype=np.float32)
        features = features.reshape(int(meta["tokens"]), int(meta["hidden"]))
    finally:
        if os.path.exists(features_path):
            os.unlink(features_path)

    write_qwen35_epd_entries(
        state.cache_dir, raw_images, grid_array, state.multimodal_config, features
    )


@app.post("/encode_images")
async def encode_images(request: EncodeRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="encoder model not ready")
    if not request.images:
        raise HTTPException(status_code=400, detail="images must not be empty")

    try:
        raw_images = [_to_rgb_array(_load_image_from_url(url)) for url in request.images]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    grids = []
    keys = []
    for raw in raw_images:
        grid_thw, _ = _compute_image_grid(raw, state.multimodal_config)
        grids.append(grid_thw)
        keys.append(
            compute_qwen35_image_cache_digest(raw, grid_thw, state.multimodal_config).hex()
        )

    missing = []
    for index, key in enumerate(keys):
        embeds_path = os.path.join(state.cache_dir, key, EPD_EMBEDS_FILENAME)
        meta_path = os.path.join(state.cache_dir, key, EPD_META_FILENAME)
        if not (os.path.isfile(embeds_path) and os.path.isfile(meta_path)):
            missing.append(index)

    if missing:
        async with state.lock:
            # 拿锁后复查，避免并发请求重复编码
            still_missing = []
            for index in missing:
                key = keys[index]
                if not (
                    os.path.isfile(os.path.join(state.cache_dir, key, EPD_EMBEDS_FILENAME))
                    and os.path.isfile(os.path.join(state.cache_dir, key, EPD_META_FILENAME))
                ):
                    still_missing.append(index)
            if still_missing:
                try:
                    await asyncio.to_thread(
                        _encode_missing_images,
                        [raw_images[i] for i in still_missing],
                        [grids[i] for i in still_missing],
                    )
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"vision encode failed: {exc}") from exc
                _enforce_cache_cap()

    return {
        "results": [
            {
                "index": index,
                "key": keys[index],
                "grid_thw": [int(v) for v in grids[index]],
                "cached": index not in missing,
            }
            for index in range(len(keys))
        ]
    }


@app.get("/health")
async def health():
    if state.model is None:
        raise HTTPException(status_code=503, detail="encoder model not ready")
    return {"status": "healthy", "cache_dir": state.cache_dir}


def main():
    parser = argparse.ArgumentParser(description="fastllm EPD vision encoder server")
    parser.add_argument("model_path", type=str, help="HF 模型目录（与 consumer 同一模型）")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--vision_device", type=str, default="cuda:0",
                        help="ViT 运行设备（写入 FASTLLM_QWEN35_VISION_DEVICE）")
    parser.add_argument("--cache_dir", type=str, default="/dev/shm/fastllm_ec_cache",
                        help="EPD 共享缓存目录（须与 consumer 的 FASTLLM_EPD_CACHE_DIR 一致）")
    parser.add_argument("--cache_max_mb", type=int,
                        default=int(os.environ.get("FASTLLM_EPD_CACHE_MAX_MB", "4096")),
                        help="缓存容量上限（MB），超过后按创建时间从老到新驱逐；0 表示不限制")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    if not hasattr(llm.fastllm_lib, "encode_visual_items_llm_model"):
        raise RuntimeError(
            "当前 libfastllm_tools 不支持 encode_visual_items_llm_model，请先重新编译安装。"
        )

    os.environ["FASTLLM_VISION_ONLY"] = "1"
    os.environ["FASTLLM_QWEN35_VISION_DEVICE"] = args.vision_device
    os.makedirs(args.cache_dir, exist_ok=True)

    state.lock = asyncio.Lock()
    print(f"[EPD-encoder] loading vision tower from {args.model_path} "
          f"(vision_device={args.vision_device}, cache_dir={args.cache_dir}) ...", flush=True)
    state.model = llm.model(args.model_path, dtype=args.dtype)  # encoder 不 warmup
    state.multimodal_config = get_qwen35_multimodal_config(
        args.model_path, model_config=getattr(state.model, "config", None)
    )
    # 缓存目录按模型指纹隔离：换模型后不会读到旧模型的 embeddings
    state.cache_dir = qwen35_epd_scoped_cache_dir(
        args.cache_dir, args.model_path, getattr(state.model, "config", None)
    )
    state.cache_max_bytes = max(0, args.cache_max_mb) * 1024 * 1024
    os.makedirs(state.cache_dir, exist_ok=True)
    print(f"[EPD-encoder] vision tower ready. scoped cache: {state.cache_dir}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
