import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _get_special_token_id(tokenizer, attr_name: str, token_attr_name: str, default_token: str) -> int:
    token_id = getattr(tokenizer, attr_name, None)
    if token_id is not None:
        return int(token_id)
    token = getattr(tokenizer, token_attr_name, default_token)
    return int(tokenizer.convert_tokens_to_ids(token))


def _get_special_token(tokenizer, attr_name: str, default_token: str) -> str:
    token = getattr(tokenizer, attr_name, None)
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
) -> str:
    sanitized = sanitize_qwen35_conversation(conversation)
    prompt = apply_chat_template_with_optional_thinking(
        tokenizer,
        sanitized,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )

    if image_grid_thw is not None:
        image_index = 0
        image_token = _get_special_token(tokenizer, "image_token", "<|image_pad|>")
        while image_token in prompt:
            num_image_tokens = int(np.prod(image_grid_thw[image_index]) // (merge_size**2))
            prompt = prompt.replace(image_token, "<|placeholder|>" * num_image_tokens, 1)
            image_index += 1
        prompt = prompt.replace("<|placeholder|>", image_token)

    if video_grid_thw is not None:
        video_index = 0
        video_token = _get_special_token(tokenizer, "video_token", "<|video_pad|>")
        vision_start_token = _get_special_token(tokenizer, "vision_start_token", "<|vision_start|>")
        vision_end_token = _get_special_token(tokenizer, "vision_end_token", "<|vision_end|>")
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
) -> Dict[str, Any]:
    del encode_vision, vision_device, vision_dtype

    images = list(images or [])
    videos = list(videos or [])
    if not images and not videos:
        raise ValueError("Qwen3.5 multimodal preprocessing requires at least one image or video.")

    config = get_qwen35_multimodal_config(model_dir, model_config=model_config)
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
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)

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


def build_qwen35_multimodal_payload(native_inputs: Dict[str, Any], tokenizer) -> Tuple[Dict[str, Any], bytes]:
    image_token_id = _get_special_token_id(tokenizer, "image_token_id", "image_token", "<|image_pad|>")
    video_token_id = _get_special_token_id(tokenizer, "video_token_id", "video_token", "<|video_pad|>")
    vision_start_token_id = _get_special_token_id(
        tokenizer, "vision_start_token_id", "vision_start_token", "<|vision_start|>"
    )
    vision_end_token_id = _get_special_token_id(
        tokenizer, "vision_end_token_id", "vision_end_token", "<|vision_end|>"
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
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def _get_special_token_id(tokenizer, attr_name: str, token_attr_name: str, default_token: str) -> int:
    token_id = getattr(tokenizer, attr_name, None)
    if token_id is not None:
        return int(token_id)
    token = getattr(tokenizer, token_attr_name, default_token)
    return int(tokenizer.convert_tokens_to_ids(token))


def _get_special_token(tokenizer, attr_name: str, default_token: str) -> str:
    token = getattr(tokenizer, attr_name, None)
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
) -> str:
    sanitized = sanitize_qwen35_conversation(conversation)
    prompt = apply_chat_template_with_optional_thinking(
        tokenizer,
        sanitized,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )

    if image_grid_thw is not None:
        image_index = 0
        image_token = _get_special_token(tokenizer, "image_token", "<|image_pad|>")
        while image_token in prompt:
            num_image_tokens = int(np.prod(image_grid_thw[image_index]) // (merge_size**2))
            prompt = prompt.replace(image_token, "<|placeholder|>" * num_image_tokens, 1)
            image_index += 1
        prompt = prompt.replace("<|placeholder|>", image_token)

    if video_grid_thw is not None:
        video_index = 0
        video_token = _get_special_token(tokenizer, "video_token", "<|video_pad|>")
        vision_start_token = _get_special_token(tokenizer, "vision_start_token", "<|vision_start|>")
        vision_end_token = _get_special_token(tokenizer, "vision_end_token", "<|vision_end|>")
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
) -> Dict[str, Any]:
    del encode_vision, vision_device, vision_dtype

    images = list(images or [])
    videos = list(videos or [])
    if not images and not videos:
        raise ValueError("Qwen3.5 multimodal preprocessing requires at least one image or video.")

    config = get_qwen35_multimodal_config(model_dir, model_config=model_config)
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
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)

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


def build_qwen35_multimodal_payload(native_inputs: Dict[str, Any], tokenizer) -> Tuple[Dict[str, Any], bytes]:
    image_token_id = _get_special_token_id(tokenizer, "image_token_id", "image_token", "<|image_pad|>")
    video_token_id = _get_special_token_id(tokenizer, "video_token_id", "video_token", "<|video_pad|>")
    vision_start_token_id = _get_special_token_id(
        tokenizer, "vision_start_token_id", "vision_start_token", "<|vision_start|>"
    )
    vision_end_token_id = _get_special_token_id(
        tokenizer, "vision_end_token_id", "vision_end_token", "<|vision_end|>"
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
import copy
import itertools
import json
import math
import os
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageSequence
from safetensors import safe_open


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_qwen35_multimodal_config(model_dir: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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


def _resolve_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _resolve_vision_device(device: Optional[str] = None) -> str:
    if device:
        return device
    env_device = os.environ.get("FASTLLM_QWEN35_VISION_DEVICE")
    if env_device:
        return env_device
    return "cpu"


def _to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.ndim != 3:
            raise ValueError(f"Unsupported ndarray image shape: {image.shape}.")
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
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


def _normalize_frame_tensor(
    frame: Image.Image,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> torch.Tensor:
    frame_np = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
    frame_np = frame_np.transpose(2, 0, 1)
    for i in range(3):
        frame_np[i] = (frame_np[i] - image_mean[i]) / image_std[i]
    return torch.from_numpy(frame_np)


def _flatten_patches(
    frames: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if frames.ndim != 4:
        raise ValueError(f"Expected [T, C, H, W] frames, got {tuple(frames.shape)}.")
    if frames.shape[0] % temporal_patch_size != 0:
        pad = temporal_patch_size - (frames.shape[0] % temporal_patch_size)
        frames = torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)

    total_frames, channels, height, width = frames.shape
    grid_t = total_frames // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size

    patches = frames.view(
        1,
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flat = patches.reshape(
        grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    )
    return flat.to(torch.float32), torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long)


def preprocess_images(
    images: Sequence[Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    patch_size = config["patch_size"]
    merge_size = config["merge_size"]
    factor = patch_size * merge_size

    pixel_values = []
    grids = []
    token_counts = []

    for image in images:
        pil_image = _to_pil_image(image)
        width, height = pil_image.size
        resized_height, resized_width = smart_resize_image(
            height=height,
            width=width,
            factor=factor,
            min_pixels=config["image_min_pixels"],
            max_pixels=config["image_max_pixels"],
        )
        if (resized_width, resized_height) != pil_image.size:
            resampling = getattr(Image, "Resampling", Image)
            pil_image = pil_image.resize((resized_width, resized_height), resample=resampling.BICUBIC)
        frame = _normalize_frame_tensor(pil_image, config["image_mean"], config["image_std"])
        flat, grid_thw = _flatten_patches(
            frame.unsqueeze(0),
            patch_size=patch_size,
            temporal_patch_size=config["temporal_patch_size"],
            merge_size=merge_size,
        )
        pixel_values.append(flat)
        grids.append(grid_thw)
        token_counts.append(int(grid_thw[0].item() * (grid_thw[1].item() // merge_size) * (grid_thw[2].item() // merge_size)))

    return {
        "pixel_values": torch.cat(pixel_values, dim=0) if pixel_values else None,
        "image_grid_thw": torch.stack(grids, dim=0) if grids else None,
        "num_image_tokens": token_counts,
    }


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
    return np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)


def _load_video_frames_from_path(path: str) -> Tuple[List[Image.Image], Optional[float], np.ndarray]:
    lower = path.lower()
    if lower.endswith(".gif"):
        with Image.open(path) as image:
            frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(image)]
            duration = image.info.get("duration")
        fps = None
        if duration and duration > 0:
            fps = 1000.0 / float(duration)
        indices = np.arange(len(frames), dtype=np.int32)
        return frames, fps, indices

    try:
        from torchvision.io import read_video
    except Exception as exc:
        raise ValueError(
            "Reading video files requires either GIF input or torchvision video backend support."
        ) from exc

    video_frames, _, info = read_video(path, pts_unit="sec")
    if video_frames.numel() == 0:
        raise ValueError(f"Failed to decode video: {path}.")
    fps = info.get("video_fps")
    frames = [Image.fromarray(frame.numpy()).convert("RGB") for frame in video_frames]
    indices = np.arange(len(frames), dtype=np.int32)
    return frames, float(fps) if fps else None, indices


def _load_video_source(video: Any) -> Tuple[List[Image.Image], Optional[float], np.ndarray]:
    if isinstance(video, dict):
        frames = video.get("frames")
        fps = video.get("fps")
        if frames is None:
            raise ValueError("Video dict input requires a `frames` field.")
        pil_frames = [_to_pil_image(frame) for frame in frames]
        indices = np.arange(len(pil_frames), dtype=np.int32)
        return pil_frames, float(fps) if fps is not None else None, indices
    if isinstance(video, (list, tuple)):
        pil_frames = [_to_pil_image(frame) for frame in video]
        indices = np.arange(len(pil_frames), dtype=np.int32)
        return pil_frames, None, indices
    if isinstance(video, (str, os.PathLike)):
        return _load_video_frames_from_path(os.fspath(video))
    raise ValueError(
        "Unsupported video input type. Use a GIF/video path, a list of frames, or "
        "a dict like {'frames': [...], 'fps': 25}."
    )


def _calculate_timestamps(indices: Sequence[int], video_fps: float, temporal_patch_size: int) -> List[float]:
    indices = list(indices)
    if len(indices) % temporal_patch_size != 0:
        indices.extend(indices[-1] for _ in range(temporal_patch_size - len(indices) % temporal_patch_size))
    timestamps = [index / video_fps for index in indices]
    return [
        (timestamps[i] + timestamps[i + temporal_patch_size - 1]) / 2
        for i in range(0, len(timestamps), temporal_patch_size)
    ]


def preprocess_videos(
    videos: Sequence[Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    patch_size = config["patch_size"]
    merge_size = config["merge_size"]
    factor = patch_size * merge_size

    pixel_values = []
    grids = []
    token_counts = []
    timestamps = []

    for video in videos:
        frames, source_fps, raw_indices = _load_video_source(video)
        sampled_indices = _sample_frame_indices(
            total_num_frames=len(frames),
            source_fps=source_fps,
            target_fps=config["video_sampling_fps"],
            min_frames=config["video_min_frames"],
            max_frames=config["video_max_frames"],
        )
        sampled_frames = [frames[idx] for idx in sampled_indices.tolist()]
        source_fps = float(source_fps) if source_fps is not None else 24.0

        width, height = sampled_frames[0].size
        resized_height, resized_width = smart_resize_video(
            num_frames=len(sampled_frames),
            height=height,
            width=width,
            temporal_factor=config["temporal_patch_size"],
            factor=factor,
            min_pixels=config["video_min_pixels"],
            max_pixels=config["video_max_pixels"],
        )
        resampling = getattr(Image, "Resampling", Image)
        sampled_frames = [
            frame.resize((resized_width, resized_height), resample=resampling.BICUBIC)
            if frame.size != (resized_width, resized_height)
            else frame
            for frame in sampled_frames
        ]
        frame_tensors = torch.stack(
            [_normalize_frame_tensor(frame, config["image_mean"], config["image_std"]) for frame in sampled_frames],
            dim=0,
        )
        flat, grid_thw = _flatten_patches(
            frame_tensors,
            patch_size=patch_size,
            temporal_patch_size=config["temporal_patch_size"],
            merge_size=merge_size,
        )
        pixel_values.append(flat)
        grids.append(grid_thw)
        frame_seqlen = int((grid_thw[1].item() // merge_size) * (grid_thw[2].item() // merge_size))
        token_counts.append(int(grid_thw[0].item() * frame_seqlen))
        timestamps.append(
            _calculate_timestamps(
                indices=[int(raw_indices[idx]) for idx in sampled_indices.tolist()],
                video_fps=source_fps,
                temporal_patch_size=config["temporal_patch_size"],
            )
        )

    return {
        "pixel_values_videos": torch.cat(pixel_values, dim=0) if pixel_values else None,
        "video_grid_thw": torch.stack(grids, dim=0) if grids else None,
        "num_video_tokens": token_counts,
        "video_timestamps": timestamps,
    }


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


def build_qwen35_prompt(
    tokenizer,
    conversation: Sequence[Dict[str, Any]],
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    video_timestamps: Optional[List[List[float]]],
    merge_size: int,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    sanitized = sanitize_qwen35_conversation(conversation)
    prompt = apply_chat_template_with_optional_thinking(
        tokenizer,
        sanitized,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )

    if image_grid_thw is not None:
        image_index = 0
        image_token = _get_special_token(tokenizer, "image_token", "<|image_pad|>")
        while image_token in prompt:
            num_image_tokens = int(image_grid_thw[image_index].prod().item() // (merge_size**2))
            prompt = prompt.replace(image_token, "<|placeholder|>" * num_image_tokens, 1)
            image_index += 1
        prompt = prompt.replace("<|placeholder|>", image_token)

    if video_grid_thw is not None:
        video_index = 0
        video_token = _get_special_token(tokenizer, "video_token", "<|video_pad|>")
        vision_start_token = _get_special_token(tokenizer, "vision_start_token", "<|vision_start|>")
        vision_end_token = _get_special_token(tokenizer, "vision_end_token", "<|vision_end|>")
        video_wrapper = f"{vision_start_token}{video_token}{vision_end_token}"
        while video_token in prompt:
            grid_thw = video_grid_thw[video_index]
            frame_seqlen = int(grid_thw[1:].prod().item() // (merge_size**2))
            current_timestamps = video_timestamps[video_index]
            video_placeholder = ""
            for frame_idx in range(int(grid_thw[0].item())):
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


def create_mm_token_type_ids(
    input_ids: Sequence[int],
    image_token_id: int,
    video_token_id: int,
) -> List[int]:
    token_types = []
    for token_id in input_ids:
        if token_id == image_token_id:
            token_types.append(1)
        elif token_id == video_token_id:
            token_types.append(2)
        else:
            token_types.append(0)
    return token_types


def _get_special_token_id(tokenizer, attr_name: str, token_attr_name: str, default_token: str) -> int:
    token_id = getattr(tokenizer, attr_name, None)
    if token_id is not None:
        return int(token_id)
    token = getattr(tokenizer, token_attr_name, default_token)
    return int(tokenizer.convert_tokens_to_ids(token))


def _get_special_token(tokenizer, attr_name: str, default_token: str) -> str:
    token = getattr(tokenizer, attr_name, None)
    if isinstance(token, str) and token != "":
        return token
    return default_token


def get_vision_position_ids(
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    llm_grid_t = int(grid_thw[0].item())
    llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
    llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size
    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = torch.arange(start_position, start_position + llm_grid_w, dtype=torch.long).repeat(
        llm_grid_h * llm_grid_t
    )
    position_height = torch.arange(start_position, start_position + llm_grid_h, dtype=torch.long).repeat_interleave(
        llm_grid_w * llm_grid_t
    )
    position_temporal = torch.full((image_seq_length,), start_position, dtype=torch.long)
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def get_rope_index(
    input_ids: Sequence[int],
    mm_token_type_ids: Sequence[int],
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        repeated = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
        repeated[:, 0] = 1
        video_grid_thw = repeated

    position_chunks = []
    current_pos = 0
    image_iter = iter(image_grid_thw) if image_grid_thw is not None else iter(())
    video_iter = iter(video_grid_thw) if video_grid_thw is not None else iter(())
    for modality_type, group in itertools.groupby(mm_token_type_ids):
        group_len = len(list(group))
        if modality_type == 0:
            position_chunks.append(torch.arange(group_len, dtype=torch.long).view(1, -1).expand(3, -1) + current_pos)
            current_pos += group_len
        elif modality_type == 1:
            grid_thw = next(image_iter)
            vision_position_ids = get_vision_position_ids(
                start_position=current_pos,
                grid_thw=grid_thw,
                spatial_merge_size=spatial_merge_size,
            )
            position_chunks.append(vision_position_ids)
            current_pos += max(int(grid_thw[1].item()), int(grid_thw[2].item())) // spatial_merge_size
        elif modality_type == 2:
            grid_thw = next(video_iter)
            vision_position_ids = get_vision_position_ids(
                start_position=current_pos,
                grid_thw=grid_thw,
                spatial_merge_size=spatial_merge_size,
            )
            position_chunks.append(vision_position_ids)
            current_pos += max(int(grid_thw[1].item()), int(grid_thw[2].item())) // spatial_merge_size
        else:
            raise ValueError(f"Unsupported multimodal token type id: {modality_type}.")

    if not position_chunks:
        positions = torch.zeros((3, 0), dtype=torch.long)
    else:
        positions = torch.cat(position_chunks, dim=1).reshape(3, -1)
    rope_delta = torch.tensor([[int(positions.max().item() + 1 - len(input_ids))]], dtype=torch.long) if positions.numel() else torch.zeros((1, 1), dtype=torch.long)
    return positions, rope_delta


class _Qwen35VisionEncoderCache:
    _lock = threading.Lock()
    _cache: Dict[Tuple[str, str, torch.dtype], torch.nn.Module] = {}

    @classmethod
    def get(cls, model_dir: str, device: str, dtype: torch.dtype):
        key = (os.path.abspath(model_dir), device, dtype)
        with cls._lock:
            if key not in cls._cache:
                cls._cache[key] = cls._load(model_dir, device, dtype)
            return cls._cache[key]

    @classmethod
    def _load(cls, model_dir: str, device: str, dtype: torch.dtype):
        from transformers import AutoConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        config = AutoConfig.from_pretrained(model_dir)
        vision_model = Qwen3_5VisionModel(config.vision_config)

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise ValueError("Qwen3.5 multimodal vision loading requires `model.safetensors.index.json`.")
        index = _load_json(index_path)
        weight_map = index.get("weight_map", {})
        prefix = "model.visual."
        files = sorted({weight_map[name] for name in weight_map if name.startswith(prefix)})

        state_dict = {}
        for filename in files:
            file_path = os.path.join(model_dir, filename)
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name.startswith(prefix):
                        state_dict[name[len(prefix) :]] = handle.get_tensor(name)
        missing, unexpected = vision_model.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError(f"Missing Qwen3.5 vision weights: {missing}.")
        if unexpected:
            raise ValueError(f"Unexpected Qwen3.5 vision weights: {unexpected}.")

        vision_model.eval()
        vision_model.to(device=device, dtype=dtype)
        return vision_model


def encode_qwen35_vision_features(
    model_dir: str,
    native_inputs: Dict[str, Any],
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    target_device = _resolve_vision_device(device)
    target_dtype = dtype or _resolve_dtype(target_device)
    vision_model = _Qwen35VisionEncoderCache.get(model_dir, target_device, target_dtype)

    image_embeds = None
    if native_inputs["pixel_values"] is not None:
        with torch.inference_mode():
            image_output = vision_model(
                native_inputs["pixel_values"].to(device=target_device, dtype=target_dtype),
                grid_thw=native_inputs["image_grid_thw"].to(device=target_device),
                return_dict=True,
            )
        image_embeds = image_output.pooler_output.detach().to(device="cpu", dtype=torch.float32)

    video_embeds = None
    if native_inputs["pixel_values_videos"] is not None:
        with torch.inference_mode():
            video_output = vision_model(
                native_inputs["pixel_values_videos"].to(device=target_device, dtype=target_dtype),
                grid_thw=native_inputs["video_grid_thw"].to(device=target_device),
                return_dict=True,
            )
        video_embeds = video_output.pooler_output.detach().to(device="cpu", dtype=torch.float32)

    return {
        "image_embeds": image_embeds,
        "video_embeds": video_embeds,
    }


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
    vision_dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    images = list(images or [])
    videos = list(videos or [])
    if not images and not videos:
        raise ValueError("Qwen3.5 multimodal preprocessing requires at least one image or video.")

    config = get_qwen35_multimodal_config(model_dir, model_config=model_config)
    conversation = normalize_qwen35_conversation(conversation, len(images), len(videos))

    image_inputs = preprocess_images(images, config) if images else {
        "pixel_values": None,
        "image_grid_thw": None,
        "num_image_tokens": [],
    }
    video_inputs = preprocess_videos(videos, config) if videos else {
        "pixel_values_videos": None,
        "video_grid_thw": None,
        "num_video_tokens": [],
        "video_timestamps": [],
    }

    prompt = build_qwen35_prompt(
        tokenizer=tokenizer,
        conversation=conversation,
        image_grid_thw=image_inputs["image_grid_thw"],
        video_grid_thw=video_inputs["video_grid_thw"],
        video_timestamps=video_inputs["video_timestamps"],
        merge_size=config["merge_size"],
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    image_token_id = _get_special_token_id(tokenizer, "image_token_id", "image_token", "<|image_pad|>")
    video_token_id = _get_special_token_id(tokenizer, "video_token_id", "video_token", "<|video_pad|>")
    mm_token_type_ids = create_mm_token_type_ids(
        input_ids,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
    )
    mrope_position_ids, mrope_position_delta = get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_inputs["image_grid_thw"],
        video_grid_thw=video_inputs["video_grid_thw"],
        spatial_merge_size=config["merge_size"],
    )

    native_inputs = {
        "prompt": prompt,
        "input_ids": input_ids,
        "pixel_values": image_inputs["pixel_values"],
        "image_grid_thw": image_inputs["image_grid_thw"],
        "pixel_values_videos": video_inputs["pixel_values_videos"],
        "video_grid_thw": video_inputs["video_grid_thw"],
        "mm_token_type_ids": mm_token_type_ids,
        "mrope_position_ids": mrope_position_ids,
        "mrope_position_delta": mrope_position_delta,
        "num_image_tokens": image_inputs["num_image_tokens"],
        "num_video_tokens": video_inputs["num_video_tokens"],
        "video_timestamps": video_inputs["video_timestamps"],
        "multimodal_config": config,
    }

    if encode_vision:
        native_inputs.update(
            encode_qwen35_vision_features(
                model_dir=model_dir,
                native_inputs=native_inputs,
                device=vision_device,
                dtype=vision_dtype,
            )
        )
    else:
        native_inputs["image_embeds"] = None
        native_inputs["video_embeds"] = None

    return native_inputs


def build_qwen35_multimodal_payload(native_inputs: Dict[str, Any], tokenizer) -> Tuple[Dict[str, Any], np.ndarray]:
    image_token_id = _get_special_token_id(tokenizer, "image_token_id", "image_token", "<|image_pad|>")
    video_token_id = _get_special_token_id(tokenizer, "video_token_id", "video_token", "<|video_pad|>")
    vision_start_token_id = _get_special_token_id(
        tokenizer, "vision_start_token_id", "vision_start_token", "<|vision_start|>"
    )
    vision_end_token_id = _get_special_token_id(
        tokenizer, "vision_end_token_id", "vision_end_token", "<|vision_end|>"
    )

    image_embeds = native_inputs["image_embeds"]
    video_embeds = native_inputs["video_embeds"]
    mrope_position_ids = native_inputs["mrope_position_ids"].to(dtype=torch.float32)
    mrope_position_delta = native_inputs["mrope_position_delta"].to(dtype=torch.float32)
    mm_token_type_ids = torch.tensor(native_inputs["mm_token_type_ids"], dtype=torch.float32).reshape(1, -1)

    arrays = []
    offsets = {}
    lengths = {}

    def add_array(name: str, tensor: Optional[torch.Tensor]) -> List[int]:
        if tensor is None:
            offsets[name] = 0
            lengths[name] = 0
            return []
        flat = tensor.reshape(-1).cpu().numpy().astype(np.float32, copy=False)
        offsets[name] = sum(arr.size for arr in arrays)
        lengths[name] = int(flat.size)
        arrays.append(flat)
        return list(tensor.shape)

    image_embeds_shape = add_array("image_embeds", image_embeds)
    video_embeds_shape = add_array("video_embeds", video_embeds)
    mrope_position_ids_shape = add_array("mrope_position_ids", mrope_position_ids)
    mrope_position_delta_shape = add_array("mrope_position_delta", mrope_position_delta)
    mm_token_type_ids_shape = add_array("mm_token_type_ids", mm_token_type_ids)

    payload = np.concatenate(arrays, axis=0).astype(np.float32, copy=False) if arrays else np.empty((0,), dtype=np.float32)
    payload_config = {
        "mode": "qwen35",
        "image_embeds_shape": image_embeds_shape,
        "video_embeds_shape": video_embeds_shape,
        "mrope_position_ids_shape": mrope_position_ids_shape,
        "mrope_position_delta_shape": mrope_position_delta_shape,
        "mm_token_type_ids_shape": mm_token_type_ids_shape,
        "image_embeds_offset": int(offsets["image_embeds"]),
        "image_embeds_length": int(lengths["image_embeds"]),
        "video_embeds_offset": int(offsets["video_embeds"]),
        "video_embeds_length": int(lengths["video_embeds"]),
        "mrope_position_ids_offset": int(offsets["mrope_position_ids"]),
        "mrope_position_ids_length": int(lengths["mrope_position_ids"]),
        "mrope_position_delta_offset": int(offsets["mrope_position_delta"]),
        "mrope_position_delta_length": int(lengths["mrope_position_delta"]),
        "mm_token_type_ids_offset": int(offsets["mm_token_type_ids"]),
        "mm_token_type_ids_length": int(lengths["mm_token_type_ids"]),
        "image_token_id": image_token_id,
        "video_token_id": video_token_id,
        "vision_start_token_id": vision_start_token_id,
        "vision_end_token_id": vision_end_token_id,
    }
    return payload_config, payload
