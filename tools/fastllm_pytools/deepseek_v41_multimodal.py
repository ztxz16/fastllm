"""
DeepSeek-V4.1 图像输入预处理（移植自官方 inference/image_processor.py）。

一张图先按动态分辨率规则缩放 / 补边成 patch 网格 ``n_vit_h x n_vit_w``（patch 14），
经 aligner 3x3 下采样后 LLM 看到的 token 网格为 ``n_llm_h x n_llm_w``，在 prompt 中展开为

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

这些位置在 input_ids 中都是 ``image_token_id``。本模块只负责：
  * 把 OpenAI 风格的 conversation（含 {"type": "image"} 内容块）渲染成带占位符的 prompt；
  * 把 PIL 图像转成 ViT patch（float32，数值与官方 bf16 预处理一致）；
  * 展开占位符，得到 input_ids、图像 span 位置与传给 C++ 的 payload。
ViT / aligner 前向在 C++ 侧完成（src/models/deepseekv41_vision.cpp）。
"""

import copy
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


def get_deepseek_v41_vision_config(model_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = model_config or {}
    vision = cfg.get("vision_config") if isinstance(cfg.get("vision_config"), dict) else {}

    # 真实 checkpoint 用 HF 命名嵌在 vision_config 里，迷你测试模型用扁平的 vision_xxx，两种都要认
    def pick(flat_key, hf_key, default):
        if hf_key is not None and hf_key in vision:
            return vision[hf_key]
        if flat_key in cfg:
            return cfg[flat_key]
        if flat_key in vision:
            return vision[flat_key]
        return default

    max_wh_ratio = pick("vision_max_wh_ratio", "max_wh_ratio", None)
    return {
        "vision_n_layers": int(pick("vision_n_layers", "num_hidden_layers", 0)),
        "patch_size": int(pick("vision_patch_size", "patch_size", 14)),
        "downsample_ratio": int(pick("vision_downsample_ratio", "downsample_ratio", 3)),
        "max_n_token": int(pick("vision_max_n_token", "max_image_tokens", 1024)),
        "min_pixels": int(pick("vision_min_pixels", "min_pixels", 544 * 544)),
        "max_wh_ratio": max_wh_ratio,
        "image_token_id": int(pick("image_token_id", None, 129264)),
    }


# ---------------- 动态分辨率（与官方逐行对应）----------------

def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int) -> Tuple[int, int]:
    return (math.ceil((best_height // patch_size) / downsample_ratio),
            math.ceil((best_width // patch_size) / downsample_ratio))


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, cfg: Dict[str, Any]):
    p = cfg["patch_size"]
    max_wh_ratio = cfg.get("max_wh_ratio")
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < cfg["min_pixels"]:
        ratio = (cfg["min_pixels"] / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, cfg["downsample_ratio"], cfg["max_n_token"])


def _round_to_bfloat16(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16（RNE）-> float32，复现官方 ``.to(torch.bfloat16)`` 的取整。"""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return rounded.astype(np.uint32).view(np.float32)


def image_to_patches(image: Image.Image, cfg: Dict[str, Any]):
    """PIL 图像 -> (patches float32 [n_vit_h * n_vit_w, 3 * p * p], n_vit_h, n_vit_w, n_llm_h, n_llm_w)"""
    p = cfg["patch_size"]
    image = image.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, cfg)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    max_wh_ratio = cfg.get("max_wh_ratio")
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    x = _round_to_bfloat16((x - 0.5) / 0.5)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).transpose(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3 * p * p)
    return np.ascontiguousarray(patches, dtype=np.float32), n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> List[int]:
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return types


# ---------------- 会话规范化 ----------------

def normalize_deepseek_v41_conversation(conversation: List[Dict[str, Any]], image_count: int) -> List[Dict[str, Any]]:
    """把 OpenAI 服务端传入的 {"type": "image"} 内容块改写成 encoding_dsv41 可识别的 image_url 块；
    若会话中没有任何图像块但给了图像，则把图像挂到最后一条 user 消息前面。"""
    conversation = copy.deepcopy(conversation)
    counter = [0]

    def convert_parts(parts):
        out = []
        for part in parts:
            if not isinstance(part, dict):
                out.append(part)
                continue
            if part.get("type") == "image":
                if "image_url" in part or "source" in part or "url" in part or "data" in part:
                    out.append(part)
                else:
                    out.append({"type": "image_url", "image_url": {"url": "fastllm://image/%d" % counter[0]}})
                counter[0] += 1
            elif part.get("type") == "image_url":
                out.append(part)
                counter[0] += 1
            elif part.get("type") == "video":
                raise ValueError("DeepSeek-V4.1 does not support video input.")
            else:
                out.append(part)
        return out

    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = convert_parts(content)
        blocks = message.get("content_blocks")
        if isinstance(blocks, list):
            message["content_blocks"] = convert_parts(blocks)

    if counter[0] == 0 and image_count > 0:
        placeholders = [{"type": "image_url", "image_url": {"url": "fastllm://image/%d" % i}} for i in range(image_count)]
        for idx in range(len(conversation) - 1, -1, -1):
            if conversation[idx].get("role") == "user":
                content = conversation[idx].get("content", "")
                if isinstance(content, list):
                    conversation[idx]["content"] = placeholders + content
                else:
                    conversation[idx]["content"] = placeholders + [{"type": "text", "text": content or ""}]
                break
        else:
            conversation.append({"role": "user", "content": placeholders})
        counter[0] = image_count
    if counter[0] != image_count:
        raise ValueError("Found %d image blocks in the conversation but got %d images." % (counter[0], image_count))
    return conversation


# ---------------- 输入构造 ----------------

def expand_image_placeholders(prompt_tokens: List[int], images: List[Image.Image], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把 prompt token 中的每个 image_token_id 占位符展开成图像 span。"""
    image_token_id = cfg["image_token_id"]
    num_placeholders = sum(1 for tok in prompt_tokens if tok == image_token_id)
    if num_placeholders != len(images):
        raise ValueError("Found %d image tokens but got %d images" % (num_placeholders, len(images)))
    if num_placeholders and cfg["vision_n_layers"] <= 0:
        raise ValueError("The model config has no vision tower (vision_n_layers == 0) but the prompt contains images")

    input_ids: List[int] = []
    token_types: List[int] = []
    image_patches: List[np.ndarray] = []
    image_grid: List[List[int]] = []
    image_iter = iter(images)
    for tok in prompt_tokens:
        if tok != image_token_id:
            input_ids.append(int(tok))
            token_types.append(TEXT)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_to_patches(next(image_iter), cfg)
        types = image_token_types(n_llm_h, n_llm_w)
        image_grid.append([len(input_ids), n_vit_h, n_vit_w])
        image_patches.append(patches)
        input_ids += [image_token_id] * len(types)
        token_types += types
    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "image_patches": image_patches,
        "image_grid": np.asarray(image_grid, dtype=np.int32).reshape(-1, 3),
        "image_token_id": image_token_id,
    }


def prepare_deepseek_v41_multimodal_inputs(
    conversation: List[Dict[str, Any]],
    images: List[Image.Image],
    model_config: Optional[Dict[str, Any]],
    encode_messages: Callable[..., Any],
    encode_fn: Callable[[str], List[int]],
    thinking_mode: str = "chat",
    reasoning_effort: Any = None,
) -> Dict[str, Any]:
    """conversation 已经过 normalize_deepseek_v41_conversation 与工具注入；
    encode_messages 为 encoding_dsv41.encode_messages，encode_fn 把 prompt 字符串编码为 token id。"""
    cfg = get_deepseek_v41_vision_config(model_config)
    kwargs = {"thinking_mode": thinking_mode, "return_multi_modal_data": True}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    prompt, media = encode_messages(conversation, **kwargs)
    records = media.get("images", []) if isinstance(media, dict) else []
    if len(records) != len(images):
        raise ValueError("Prompt has %d image placeholders but %d images were provided." % (len(records), len(images)))
    native = expand_image_placeholders(encode_fn(prompt), images, cfg)
    native["prompt"] = prompt
    native["vision_config"] = cfg
    return native


def build_deepseek_v41_multimodal_payload(native_inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], bytes]:
    """payload 布局（与 pytools.cpp 的 typed tensor 解析一致）：
       pixel_values: 每张图一个 float32 [n_vit_h * n_vit_w, 3 * p * p]
       image_grid:   int32 [num_images, 3] = (span 起始位置, n_vit_h, n_vit_w)"""
    arrays: List[np.ndarray] = []
    descriptors: List[Dict[str, Any]] = []
    offset = 0

    def append(name, array, dtype):
        nonlocal offset
        payload_array = np.ascontiguousarray(array.astype(dtype, copy=False))
        descriptors.append({
            "name": name,
            "dtype": np.dtype(dtype).name,
            "shape": [int(d) for d in payload_array.shape],
            "offset_bytes": int(offset),
            "nbytes": int(payload_array.nbytes),
        })
        arrays.append(payload_array)
        offset += int(payload_array.nbytes)

    for patches in native_inputs["image_patches"]:
        append("pixel_values", patches, np.float32)
    append("image_grid", native_inputs["image_grid"], np.int32)
    payload = b"".join(a.tobytes() for a in arrays)
    payload_config = {
        "mode": "deepseek_v41",
        "tensors": descriptors,
        "image_token_id": int(native_inputs["image_token_id"]),
        "num_images": int(len(native_inputs["image_patches"])),
    }
    return payload_config, payload


def summarize_grid(native_inputs: Dict[str, Any]) -> str:
    return json.dumps(native_inputs["image_grid"].tolist())
