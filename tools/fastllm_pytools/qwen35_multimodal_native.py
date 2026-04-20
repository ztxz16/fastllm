import json
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageSequence


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_qwen35_multimodal_config(
    model_dir: str,
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    image_processor_config = _load_json(os.path.join(model_dir, "preprocessor_config.json"))
    video_processor_path = os.path.join(model_dir, "video_preprocessor_config.json")
    video_processor_config = _load_json(video_processor_path) if os.path.exists(video_processor_path) else {}
    vision_config = (model_config or {}).get("vision_config", {})

    image_size = image_processor_config.get("size", {})
    video_size = video_processor_config.get("size", {})

    return {
        "patch_size": int(image_processor_config.get("patch_size", vision_config.get("patch_size", 16))),
        "temporal_patch_size": int(
            image_processor_config.get("temporal_patch_size", vision_config.get("temporal_patch_size", 2))
        ),
        "merge_size": int(image_processor_config.get("merge_size", vision_config.get("spatial_merge_size", 2))),
        "image_mean": [float(x) for x in image_processor_config.get("image_mean", [0.5, 0.5, 0.5])],
        "image_std": [float(x) for x in image_processor_config.get("image_std", [0.5, 0.5, 0.5])],
        "image_min_pixels": int(image_size.get("shortest_edge", 56 * 56)),
        "image_max_pixels": int(image_size.get("longest_edge", 28 * 28 * 1280)),
        "video_min_pixels": int(video_size.get("shortest_edge", 128 * 128)),
        "video_max_pixels": int(video_size.get("longest_edge", 16 * 16 * 2 * 2 * 2 * 6144)),
        "video_sampling_fps": 2.0,
        "video_min_frames": 4,
        "video_max_frames": 768,
    }


def get_qwen35_tokenizer_config(model_dir: str) -> Dict[str, Any]:
    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        return {}
    return _load_json(tokenizer_config_path)


_TOKENIZER_CONFIG_SPECIAL_TOKEN_KEYS = {
    "image_token": "image_token",
    "video_token": "video_token",
    "vision_start_token": "vision_bos_token",
    "vision_end_token": "vision_eos_token",
}


def _lookup_added_token_id(tokenizer_config: Dict[str, Any], token: str) -> Optional[int]:
    decoder = tokenizer_config.get("added_tokens_decoder", {})
    for token_id, info in decoder.items():
        if isinstance(info, dict) and info.get("content") == token:
            return int(token_id)
    return None


def _to_rgb_array(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.ndim != 3:
            raise ValueError(f"Unsupported ndarray image shape: {array.shape}.")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        if array.shape[2] != 3:
            raise ValueError(f"Unsupported ndarray image channel count: {array.shape[2]}.")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)
    raise ValueError(f"Unsupported image input type: {type(image)!r}.")


def smart_resize_image(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[int, int]:
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def _sample_frame_indices(
    total_num_frames: int,
    source_fps: Optional[float],
    target_fps: Optional[float],
    min_frames: int,
    max_frames: int,
) -> np.ndarray:
    if total_num_frames <= 0:
        raise ValueError("Video must contain at least one frame.")
    if target_fps is not None:
        fps = source_fps if source_fps is not None else 24.0
        num_frames = int(total_num_frames / fps * target_fps)
        num_frames = min(max(num_frames, min_frames), max_frames, total_num_frames)
    else:
        num_frames = min(max(total_num_frames, min_frames), max_frames)
    return np.linspace(0, total_num_frames - 1, num_frames).round().astype(np.int32)


def _load_video_frames_from_path(path: str) -> Tuple[List[np.ndarray], Optional[float], np.ndarray]:
    lower = path.lower()
    if lower.endswith(".gif"):
        with Image.open(path) as image:
            frames = [np.asarray(frame.copy().convert("RGB"), dtype=np.uint8) for frame in ImageSequence.Iterator(image)]
            duration = image.info.get("duration")
        fps = None
        if duration and duration > 0:
            fps = 1000.0 / float(duration)
        indices = np.arange(len(frames), dtype=np.int32)
        return frames, fps, indices

    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise ValueError(
            "Reading non-GIF video files requires `imageio`. Install it or use GIF input."
        ) from exc

    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        fps_value = meta.get("fps")
        frames = []
        for frame in reader:
            frame_array = np.asarray(frame)
            if frame_array.ndim == 2:
                frame_array = np.stack([frame_array] * 3, axis=-1)
            if frame_array.shape[-1] == 4:
                frame_array = frame_array[:, :, :3]
            frames.append(np.ascontiguousarray(frame_array.astype(np.uint8, copy=False)))
    finally:
        reader.close()

    if not frames:
        raise ValueError(f"Failed to decode video: {path}.")
    indices = np.arange(len(frames), dtype=np.int32)
    return frames, float(fps_value) if fps_value else None, indices


def _load_video_source(video: Any) -> Tuple[List[np.ndarray], Optional[float], np.ndarray]:
    if isinstance(video, dict):
        frames = video.get("frames")
        fps = video.get("fps")
        if frames is None:
            raise ValueError("Video dict input requires a `frames` field.")
        arrays = [_to_rgb_array(frame) for frame in frames]
        indices = np.arange(len(arrays), dtype=np.int32)
        return arrays, float(fps) if fps is not None else None, indices
    if isinstance(video, (list, tuple)):
        arrays = [_to_rgb_array(frame) for frame in video]
        indices = np.arange(len(arrays), dtype=np.int32)
        return arrays, None, indices
    if isinstance(video, (str, os.PathLike)):
        return _load_video_frames_from_path(os.fspath(video))
    raise ValueError(
        "Unsupported video input type. Use a GIF/video path, a list of frames, or "
        "a dict like {'frames': [...], 'fps': 25}."
    )


def _calculate_timestamps(
    indices: Sequence[int],
    video_fps: float,
    temporal_patch_size: int,
) -> List[float]:
    rounded = list(indices)
    if len(rounded) % temporal_patch_size != 0:
        rounded.extend(rounded[-1] for _ in range(temporal_patch_size - len(rounded) % temporal_patch_size))
    timestamps = [index / video_fps for index in rounded]
    return [
        (timestamps[i] + timestamps[i + temporal_patch_size - 1]) / 2.0
        for i in range(0, len(timestamps), temporal_patch_size)
    ]


def sanitize_qwen35_conversation(conversation: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized = []
    for message in conversation:
        item = {"role": message["role"]}
        content = message.get("content", "")
        if isinstance(content, list):
            new_content = []
            for part in content:
                part_type = part.get("type")
                if part_type == "text":
                    new_content.append({"type": "text", "text": part.get("text", "")})
                elif part_type == "image":
                    new_content.append({"type": "image"})
                elif part_type == "video":
                    new_content.append({"type": "video"})
                else:
                    raise ValueError(
                        f"Qwen3.5 native multimodal only supports text/image/video content, got {part_type!r}."
                    )
            item["content"] = new_content
        else:
            item["content"] = content
        sanitized.append(item)
    return sanitized


def normalize_qwen35_conversation(
    conversation: Sequence[Dict[str, Any]],
    image_count: int,
    video_count: int,
) -> List[Dict[str, Any]]:
    conversation = sanitize_qwen35_conversation(conversation)
    if not conversation:
        content = [{"type": "image"} for _ in range(image_count)] + [{"type": "video"} for _ in range(video_count)]
        return [{"role": "user", "content": content}]

    has_placeholder = False
    for message in conversation:
        content = message.get("content", "")
        if isinstance(content, list):
            has_placeholder = has_placeholder or any(part.get("type") in {"image", "video"} for part in content)
    if has_placeholder or (image_count == 0 and video_count == 0):
        return conversation

    inserted_parts = [{"type": "image"} for _ in range(image_count)] + [{"type": "video"} for _ in range(video_count)]
    updated = []
    attached = False
    for idx in range(len(conversation) - 1, -1, -1):
        message = dict(conversation[idx])
        if message.get("role") != "user":
            updated.append(message)
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            message["content"] = inserted_parts + content
        else:
            message["content"] = inserted_parts + [{"type": "text", "text": content}]
        updated.append(message)
        attached = True
        updated.extend(reversed(conversation[:idx]))
        break
    if not attached:
        updated = list(conversation)
        updated.append({"role": "user", "content": inserted_parts})
    else:
        updated = list(reversed(updated))
    return updated


def apply_chat_template_with_optional_thinking(tokenizer, conversation, add_generation_prompt, enable_thinking):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(conversation, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(conversation, **kwargs)


def _render_qwen35_chat_template_fallback(
    conversation: Sequence[Dict[str, Any]],
    tokenizer_config: Dict[str, Any],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    image_token = _get_special_token(None, "image_token", "<|image_pad|>", tokenizer_config=tokenizer_config)
    video_token = _get_special_token(None, "video_token", "<|video_pad|>", tokenizer_config=tokenizer_config)
    vision_start_token = _get_special_token(
        None, "vision_start_token", "<|vision_start|>", tokenizer_config=tokenizer_config
    )
    vision_end_token = _get_special_token(
        None, "vision_end_token", "<|vision_end|>", tokenizer_config=tokenizer_config
    )
    im_start_token = "<|im_start|>"
    im_end_token = "<|im_end|>"

    def render_content(content: Any, *, allow_vision: bool) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError(f"Unexpected content item type: {type(item)!r}.")
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(item.get("text", ""))
                elif item_type == "image":
                    if not allow_vision:
                        raise ValueError("System/assistant/tool message cannot contain images in Qwen3.5 fallback template.")
                    parts.append(vision_start_token + image_token + vision_end_token)
                elif item_type == "video":
                    if not allow_vision:
                        raise ValueError("System/assistant/tool message cannot contain videos in Qwen3.5 fallback template.")
                    parts.append(vision_start_token + video_token + vision_end_token)
                else:
                    raise ValueError(f"Unexpected item type in content: {item_type!r}.")
            return "".join(parts)
        raise ValueError(f"Unexpected content type: {type(content)!r}.")

    messages = sanitize_qwen35_conversation(conversation)
    if not messages:
        raise ValueError("No messages provided.")

    parts: List[str] = []
    if messages[0]["role"] == "system":
        system_text = render_content(messages[0].get("content", ""), allow_vision=False).strip()
        parts.append(im_start_token + "system\n" + system_text + im_end_token + "\n")

    for idx, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            if idx != 0:
                raise ValueError("System message must be at the beginning.")
            continue
        if role == "user":
            content = render_content(message.get("content", ""), allow_vision=True).strip()
            parts.append(im_start_token + "user\n" + content + im_end_token + "\n")
            continue
        if role == "assistant":
            content = render_content(message.get("content", ""), allow_vision=False).strip()
            reasoning_content = message.get("reasoning_content", "")
            if isinstance(reasoning_content, str):
                reasoning_content = reasoning_content.strip()
            else:
                reasoning_content = ""
            if reasoning_content:
                if content:
                    content = "<think>\n" + reasoning_content + "\n</think>\n\n" + content
                else:
                    content = "<think>\n" + reasoning_content + "\n</think>"
            parts.append(im_start_token + "assistant\n" + content + im_end_token + "\n")
            continue
        if role == "tool":
            content = render_content(message.get("content", ""), allow_vision=False).strip()
            parts.append(
                im_start_token
                + "user\n<tool_response>\n"
                + content
                + "\n</tool_response>"
                + im_end_token
                + "\n"
            )
            continue
        raise ValueError(f"Unexpected message role: {role!r}.")

    if add_generation_prompt:
        parts.append(im_start_token + "assistant\n")
        if enable_thinking:
            parts.append("<think>\n")
        else:
            parts.append("<think>\n\n</think>\n\n")

    return "".join(parts)


def _get_special_token_id(
    tokenizer,
    attr_name: str,
    token_attr_name: str,
    default_token: str,
    model_config: Optional[Dict[str, Any]] = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> int:
    if tokenizer is not None:
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            return int(token_id)
        token = getattr(tokenizer, token_attr_name, default_token)
        return int(tokenizer.convert_tokens_to_ids(token))
    if isinstance(model_config, dict) and attr_name in model_config:
        return int(model_config[attr_name])
    token = _get_special_token(None, token_attr_name, default_token, tokenizer_config=tokenizer_config)
    token_id = _lookup_added_token_id(tokenizer_config or {}, token)
    if token_id is None:
        raise ValueError(f"Failed to resolve special token id for {attr_name}.")
    return token_id


def _get_special_token(
    tokenizer,
    attr_name: str,
    default_token: str,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> str:
    if tokenizer is not None:
        token = getattr(tokenizer, attr_name, None)
        if isinstance(token, str) and token != "":
            return token
    extra_special_tokens = (tokenizer_config or {}).get("extra_special_tokens", {})
    config_key = _TOKENIZER_CONFIG_SPECIAL_TOKEN_KEYS.get(attr_name)
    if config_key is not None:
        token = extra_special_tokens.get(config_key)
        if isinstance(token, str) and token != "":
            return token
    return default_token


def _compute_image_grid(raw_image: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    patch_size = config["patch_size"]
    merge_size = config["merge_size"]
    factor = patch_size * merge_size
    height, width = raw_image.shape[:2]
    resized_height, resized_width = smart_resize_image(
        height=height,
        width=width,
        factor=factor,
        min_pixels=config["image_min_pixels"],
        max_pixels=config["image_max_pixels"],
    )
    grid_thw = np.asarray(
        [1, resized_height // patch_size, resized_width // patch_size],
        dtype=np.int32,
    )
    token_count = int(grid_thw[0] * (grid_thw[1] // merge_size) * (grid_thw[2] // merge_size))
    return grid_thw, token_count


def _compute_video_grid(
    frames: Sequence[np.ndarray],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, int]:
    patch_size = config["patch_size"]
    merge_size = config["merge_size"]
    factor = patch_size * merge_size
    height, width = frames[0].shape[:2]
    resized_height, resized_width = smart_resize_video(
        num_frames=len(frames),
        height=height,
        width=width,
        temporal_factor=config["temporal_patch_size"],
        factor=factor,
        min_pixels=config["video_min_pixels"],
        max_pixels=config["video_max_pixels"],
    )
    padded_frames = int(math.ceil(len(frames) / config["temporal_patch_size"]) * config["temporal_patch_size"])
    grid_thw = np.asarray(
        [
            padded_frames // config["temporal_patch_size"],
            resized_height // patch_size,
            resized_width // patch_size,
        ],
        dtype=np.int32,
    )
    frame_seqlen = int((grid_thw[1] // merge_size) * (grid_thw[2] // merge_size))
    token_count = int(grid_thw[0] * frame_seqlen)
    return grid_thw, token_count


def build_qwen35_prompt(
    tokenizer,
    conversation: Sequence[Dict[str, Any]],
    image_grid_thw: Optional[np.ndarray],
    video_grid_thw: Optional[np.ndarray],
    video_timestamps: Optional[List[List[float]]],
    merge_size: int,
    add_generation_prompt: bool,
    enable_thinking: bool,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> str:
    sanitized = sanitize_qwen35_conversation(conversation)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        prompt = apply_chat_template_with_optional_thinking(
            tokenizer,
            sanitized,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    else:
        prompt = _render_qwen35_chat_template_fallback(
            sanitized,
            tokenizer_config=tokenizer_config or {},
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    if image_grid_thw is not None:
        image_index = 0
        image_token = _get_special_token(
            tokenizer, "image_token", "<|image_pad|>", tokenizer_config=tokenizer_config
        )
        while image_token in prompt:
            num_image_tokens = int(np.prod(image_grid_thw[image_index]) // (merge_size**2))
            prompt = prompt.replace(image_token, "<|placeholder|>" * num_image_tokens, 1)
            image_index += 1
        prompt = prompt.replace("<|placeholder|>", image_token)

    if video_grid_thw is not None:
        video_index = 0
        video_token = _get_special_token(
            tokenizer, "video_token", "<|video_pad|>", tokenizer_config=tokenizer_config
        )
        vision_start_token = _get_special_token(
            tokenizer, "vision_start_token", "<|vision_start|>", tokenizer_config=tokenizer_config
        )
        vision_end_token = _get_special_token(
            tokenizer, "vision_end_token", "<|vision_end|>", tokenizer_config=tokenizer_config
        )
        video_wrapper = f"{vision_start_token}{video_token}{vision_end_token}"
        while video_token in prompt:
            grid_thw = video_grid_thw[video_index]
            frame_seqlen = int((grid_thw[1] // merge_size) * (grid_thw[2] // merge_size))
            current_timestamps = video_timestamps[video_index]
            video_placeholder = ""
            for frame_idx in range(int(grid_thw[0])):
                video_placeholder += f"<{current_timestamps[frame_idx]:.1f} seconds>"
                video_placeholder += (
                    vision_start_token
                    + "<|placeholder|>" * frame_seqlen
                    + vision_end_token
                )
            if video_wrapper in prompt:
                prompt = prompt.replace(video_wrapper, video_placeholder, 1)
            else:
                prompt = prompt.replace(video_token, video_placeholder, 1)
            video_index += 1
        prompt = prompt.replace("<|placeholder|>", video_token)

    return prompt


def _prepare_images(
    images: Sequence[Any],
    config: Dict[str, Any],
) -> Tuple[List[np.ndarray], Optional[np.ndarray], List[int]]:
    raw_images: List[np.ndarray] = []
    grids: List[np.ndarray] = []
    token_counts: List[int] = []
    for image in images:
        raw = _to_rgb_array(image)
        grid_thw, token_count = _compute_image_grid(raw, config)
        raw_images.append(raw)
        grids.append(grid_thw)
        token_counts.append(token_count)
    grid_array = np.stack(grids, axis=0) if grids else None
    return raw_images, grid_array, token_counts


def _prepare_videos(
    videos: Sequence[Any],
    config: Dict[str, Any],
) -> Tuple[List[np.ndarray], Optional[np.ndarray], List[int], List[List[float]]]:
    video_arrays: List[np.ndarray] = []
    grids: List[np.ndarray] = []
    token_counts: List[int] = []
    timestamps: List[List[float]] = []

    for video in videos:
        frames, source_fps, raw_indices = _load_video_source(video)
        sampled_indices = _sample_frame_indices(
            total_num_frames=len(frames),
            source_fps=source_fps,
            target_fps=config["video_sampling_fps"],
            min_frames=config["video_min_frames"],
            max_frames=config["video_max_frames"],
        )
        sampled_frames = [frames[int(idx)] for idx in sampled_indices.tolist()]
        if not sampled_frames:
            raise ValueError("Video sampling produced no frames.")
        sampled_video = np.stack(sampled_frames, axis=0)
        grid_thw, token_count = _compute_video_grid(sampled_frames, config)

        fps_value = float(source_fps) if source_fps is not None else 24.0
        timestamps.append(
            _calculate_timestamps(
                indices=[int(raw_indices[int(idx)]) for idx in sampled_indices.tolist()],
                video_fps=fps_value,
                temporal_patch_size=config["temporal_patch_size"],
            )
        )

        video_arrays.append(np.ascontiguousarray(sampled_video))
        grids.append(grid_thw)
        token_counts.append(token_count)

    grid_array = np.stack(grids, axis=0) if grids else None
    return video_arrays, grid_array, token_counts, timestamps


def prepare_qwen35_multimodal_inputs(
    tokenizer,
    model_dir: str,
    model_config: Dict[str, Any],
    conversation: Sequence[Dict[str, Any]],
    images: Optional[Sequence[Any]] = None,
    videos: Optional[Sequence[Any]] = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    encode_vision: bool = True,
    vision_device: Optional[str] = None,
    vision_dtype: Optional[Any] = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    encode_fn: Optional[Callable[[str], Sequence[int]]] = None,
) -> Dict[str, Any]:
    del encode_vision, vision_device, vision_dtype

    images = list(images or [])
    videos = list(videos or [])
    if not images and not videos:
        raise ValueError("Qwen3.5 multimodal preprocessing requires at least one image or video.")

    config = get_qwen35_multimodal_config(model_dir, model_config=model_config)
    tokenizer_config = tokenizer_config or get_qwen35_tokenizer_config(model_dir)
    conversation = normalize_qwen35_conversation(conversation, len(images), len(videos))

    raw_images, image_grid_thw, num_image_tokens = _prepare_images(images, config)
    raw_videos, video_grid_thw, num_video_tokens, video_timestamps = _prepare_videos(videos, config)

    prompt = build_qwen35_prompt(
        tokenizer=tokenizer,
        conversation=conversation,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        video_timestamps=video_timestamps,
        merge_size=config["merge_size"],
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        tokenizer_config=tokenizer_config,
    )
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    elif encode_fn is not None:
        input_ids = list(encode_fn(prompt))
    else:
        raise ValueError("Qwen3.5 multimodal preprocessing requires tokenizer.encode or encode_fn.")

    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "image_arrays": raw_images,
        "image_grid_thw": image_grid_thw,
        "video_arrays": raw_videos,
        "video_grid_thw": video_grid_thw,
        "num_image_tokens": num_image_tokens,
        "num_video_tokens": num_video_tokens,
        "video_timestamps": video_timestamps,
        "multimodal_config": config,
        "tokenizer_config": tokenizer_config,
    }


def _append_payload_tensor(
    arrays: List[np.ndarray],
    descriptors: List[Dict[str, Any]],
    name: str,
    array: Optional[np.ndarray],
    dtype: np.dtype,
    current_offset: int,
) -> int:
    if array is None:
        return current_offset
    payload_array = np.ascontiguousarray(array.astype(dtype, copy=False))
    descriptors.append(
        {
            "name": name,
            "dtype": np.dtype(dtype).name,
            "shape": [int(dim) for dim in payload_array.shape],
            "offset_bytes": int(current_offset),
            "nbytes": int(payload_array.nbytes),
        }
    )
    arrays.append(payload_array)
    return current_offset + int(payload_array.nbytes)


def build_qwen35_multimodal_payload(
    native_inputs: Dict[str, Any],
    tokenizer,
    model_config: Optional[Dict[str, Any]] = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bytes]:
    tokenizer_config = tokenizer_config or native_inputs.get("tokenizer_config") or {}
    image_token_id = _get_special_token_id(
        tokenizer, "image_token_id", "image_token", "<|image_pad|>", model_config=model_config, tokenizer_config=tokenizer_config
    )
    video_token_id = _get_special_token_id(
        tokenizer, "video_token_id", "video_token", "<|video_pad|>", model_config=model_config, tokenizer_config=tokenizer_config
    )
    vision_start_token_id = _get_special_token_id(
        tokenizer,
        "vision_start_token_id",
        "vision_start_token",
        "<|vision_start|>",
        model_config=model_config,
        tokenizer_config=tokenizer_config,
    )
    vision_end_token_id = _get_special_token_id(
        tokenizer,
        "vision_end_token_id",
        "vision_end_token",
        "<|vision_end|>",
        model_config=model_config,
        tokenizer_config=tokenizer_config,
    )

    arrays: List[np.ndarray] = []
    descriptors: List[Dict[str, Any]] = []
    offset = 0

    offset = _append_payload_tensor(arrays, descriptors, "image_grid_thw", native_inputs.get("image_grid_thw"), np.int32, offset)
    offset = _append_payload_tensor(arrays, descriptors, "video_grid_thw", native_inputs.get("video_grid_thw"), np.int32, offset)

    for image_array in native_inputs.get("image_arrays", []):
        offset = _append_payload_tensor(arrays, descriptors, "image_frames", image_array, np.float32, offset)

    for video_array in native_inputs.get("video_arrays", []):
        offset = _append_payload_tensor(arrays, descriptors, "video_frames", video_array, np.float32, offset)

    payload = b"".join(array.tobytes(order="C") for array in arrays)
    payload_config = {
        "mode": "qwen35",
        "image_token_id": image_token_id,
        "video_token_id": video_token_id,
        "vision_start_token_id": vision_start_token_id,
        "vision_end_token_id": vision_end_token_id,
        "tensors": descriptors,
    }
    return payload_config, payload


# Keep the raw-frame path in this module for future parity work, but use the
# HF-aligned vision encoder for Qwen3.5 inference today because it matches the
# reference model's visual features and position data.
try:
    from .qwen35_multimodal import (
        build_qwen35_multimodal_payload as _hf_build_qwen35_multimodal_payload,
        prepare_qwen35_multimodal_inputs as _hf_prepare_qwen35_multimodal_inputs,
    )
except ImportError:
    import importlib.util

    _legacy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen35_multimodal.py")
    _legacy_spec = importlib.util.spec_from_file_location("fastllm_qwen35_multimodal_legacy", _legacy_path)
    if _legacy_spec is None or _legacy_spec.loader is None:
        raise ImportError(f"Failed to load Qwen3.5 legacy multimodal module from {_legacy_path}.")
    _legacy_module = importlib.util.module_from_spec(_legacy_spec)
    _legacy_spec.loader.exec_module(_legacy_module)
    _hf_build_qwen35_multimodal_payload = _legacy_module.build_qwen35_multimodal_payload
    _hf_prepare_qwen35_multimodal_inputs = _legacy_module.prepare_qwen35_multimodal_inputs


def prepare_qwen35_multimodal_inputs(
    tokenizer,
    model_dir: str,
    model_config: Dict[str, Any],
    conversation: Sequence[Dict[str, Any]],
    images: Optional[Sequence[Any]] = None,
    videos: Optional[Sequence[Any]] = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    encode_vision: bool = True,
    vision_device = None,
    vision_dtype = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    encode_fn = None,
) -> Dict[str, Any]:
    del tokenizer_config, encode_fn
    return _hf_prepare_qwen35_multimodal_inputs(
        tokenizer = tokenizer,
        model_dir = model_dir,
        model_config = model_config,
        conversation = conversation,
        images = images,
        videos = videos,
        add_generation_prompt = add_generation_prompt,
        enable_thinking = enable_thinking,
        encode_vision = encode_vision,
        vision_device = vision_device,
        vision_dtype = vision_dtype,
    )


def build_qwen35_multimodal_payload(
    native_inputs: Dict[str, Any],
    tokenizer,
    model_config: Optional[Dict[str, Any]] = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bytes]:
    del model_config, tokenizer_config
    payload_config, payload = _hf_build_qwen35_multimodal_payload(native_inputs, tokenizer)
    if isinstance(payload, np.ndarray):
        return payload_config, payload.astype(np.float32, copy=False).tobytes(order="C")
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return payload_config, bytes(payload)
    return payload_config, np.asarray(payload, dtype=np.float32).tobytes(order="C")
