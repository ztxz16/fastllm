import math
import os
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageSequence


MAX_IMAGE_SIZE = 3024
STEP3_BOS = "<｜begin▁of▁sentence｜>"
STEP3_IM_START = "<|im_start|>"
STEP3_IM_END = "<|im_end|>"

IMAGE_TOKEN = "<im_patch>"
IM_START_TOKEN = "<im_start>"
IM_END_TOKEN = "<im_end>"
PATCH_START_TOKEN = "<patch_start>"
PATCH_END_TOKEN = "<patch_end>"
PATCH_NEWLINE_TOKEN = "<patch_newline>"

IMAGE_TOKEN_LEN = 169
PATCH_TOKEN_LEN = 81
IMAGE_SIZE = 728
PATCH_SIZE = 504
IMAGE_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

VIDEO_DEFAULT_MAX_FRAMES = int(os.environ.get("FASTLLM_STEP3P7_VIDEO_MAX_FRAMES", "8"))
VIDEO_DEFAULT_SAMPLING_FPS = float(os.environ.get("FASTLLM_STEP3P7_VIDEO_FPS", "0"))


def _to_pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
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
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB")
    if isinstance(image, (str, os.PathLike)):
        return Image.open(os.fspath(image)).convert("RGB")
    raise ValueError(f"Unsupported image input type: {type(image)!r}.")


def _to_rgb_array(image: Any) -> np.ndarray:
    return np.asarray(_to_pil_rgb(image), dtype=np.uint8)


def _load_animated_image_frames(path: str) -> Tuple[List[np.ndarray], Optional[float], np.ndarray]:
    with Image.open(path) as image:
        if not getattr(image, "is_animated", False):
            raise ValueError(f"{path!r} is not an animated image.")
        frames = [
            np.asarray(frame.copy().convert("RGB"), dtype=np.uint8)
            for frame in ImageSequence.Iterator(image)
        ]
        duration = image.info.get("duration")
    if not frames:
        raise ValueError(f"Failed to decode video frames from {path!r}.")
    fps = None
    if duration and duration > 0:
        fps = 1000.0 / float(duration)
    return frames, fps, np.arange(len(frames), dtype=np.int32)


def _load_video_frames_from_path(path: str) -> Tuple[List[np.ndarray], Optional[float], np.ndarray]:
    lower = path.lower()
    if lower.endswith((".gif", ".webp")):
        return _load_animated_image_frames(path)

    try:
        return _load_animated_image_frames(path)
    except Exception:
        pass

    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise ValueError(
            "Reading non-GIF video files for Step3.7 requires `imageio` with an "
            "ffmpeg-capable backend, or pass a GIF / list of decoded frames."
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
    return frames, float(fps_value) if fps_value else None, np.arange(len(frames), dtype=np.int32)


def _load_video_source(video: Any) -> Tuple[List[np.ndarray], Optional[float], np.ndarray]:
    if isinstance(video, dict):
        frames = video.get("frames")
        fps = video.get("fps")
        if frames is not None:
            arrays = [_to_rgb_array(frame) for frame in frames]
            return arrays, float(fps) if fps is not None else None, np.arange(len(arrays), dtype=np.int32)
        path = video.get("path") or video.get("file") or video.get("url")
        if path is not None:
            return _load_video_frames_from_path(os.fspath(path))
        raise ValueError("Video dict input requires a `frames`, `path`, `file`, or `url` field.")
    if isinstance(video, (list, tuple)):
        arrays = [_to_rgb_array(frame) for frame in video]
        return arrays, None, np.arange(len(arrays), dtype=np.int32)
    if isinstance(video, (str, os.PathLike)):
        return _load_video_frames_from_path(os.fspath(video))
    raise ValueError(
        "Unsupported video input type. Use a GIF/video path, a list of frames, "
        "or a dict like {'frames': [...], 'fps': 25}."
    )


def _sample_video_frames(
    frames: Sequence[np.ndarray],
    source_fps: Optional[float],
    raw_indices: Sequence[int],
    max_frames: int,
    sampling_fps: float,
) -> Tuple[List[Image.Image], List[float]]:
    total_frames = len(frames)
    if total_frames <= 0:
        raise ValueError("Video must contain at least one frame.")
    if max_frames <= 0:
        raise ValueError("Step3.7 video max_frames must be positive.")

    if sampling_fps > 0 and source_fps and source_fps > 0:
        target_count = int(total_frames / source_fps * sampling_fps)
        target_count = min(max(1, target_count), max_frames, total_frames)
    else:
        target_count = min(max_frames, total_frames)

    sampled_positions = np.linspace(0, total_frames - 1, target_count).round().astype(np.int32)
    raw_indices_array = np.asarray(raw_indices, dtype=np.int32)
    if len(raw_indices_array) != total_frames:
        raw_indices_array = np.arange(total_frames, dtype=np.int32)

    fps_for_timestamp = source_fps if source_fps and source_fps > 0 else None
    sampled_frames: List[Image.Image] = []
    timestamps: List[float] = []
    for position in sampled_positions:
        frame_index = int(raw_indices_array[int(position)])
        sampled_frames.append(_to_pil_rgb(frames[int(position)]))
        timestamps.append(float(frame_index) / fps_for_timestamp if fps_for_timestamp else float(frame_index))
    return sampled_frames, timestamps


def _resize_normalize_chw(image: Image.Image, size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)


class ImagePatcher:
    def determine_window_size(self, long_side: int, short_side: int) -> int:
        if long_side <= 728:
            return short_side if long_side / short_side > 1.5 else 0
        return min(short_side, 504) if long_side / short_side > 4 else 504

    def slide_window(
        self,
        width: int,
        height: int,
        sizes: List[Tuple[int, int]],
        steps: List[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int]]:
        windows = []
        x_num = 0
        y_num = 0
        for size, step in zip(sizes, steps):
            size_w, size_h = size
            step_w, step_h = step
            x_num = 1 if width <= size_w else math.ceil((width - size_w) / step_w + 1)
            x_start = [step_w * i for i in range(x_num)]
            if len(x_start) > 1 and x_start[-1] + size_w > width:
                x_start[-1] = width - size_w

            y_num = 1 if height <= size_h else math.ceil((height - size_h) / step_h + 1)
            y_start = [step_h * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size_h > height:
                y_start[-1] = height - size_h

            start = np.asarray(list(product(y_start, x_start)), dtype=np.int32)
            start[:, [0, 1]] = start[:, [1, 0]]
            windows.append(np.concatenate([start, start + np.asarray(size, dtype=np.int32)], axis=1))
        merged = np.concatenate(windows, axis=0)
        return [
            (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))
            for box in merged
        ], (x_num, y_num)

    def square_pad(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        padded = Image.new(img.mode, (size, size), 0)
        padded.paste(img, (0, 0))
        return padded

    def get_image_size_for_padding(self, img_width: int, img_height: int) -> Tuple[int, int]:
        ratio = img_width / img_height
        if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 0.25):
            new_size = max(img_height, img_width)
            return new_size, new_size
        return img_width, img_height

    def get_image_size_for_preprocess(self, img_width: int, img_height: int) -> Tuple[int, int]:
        if max(img_height, img_width) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(img_height, img_width)
            img_width = int(img_width * scale)
            img_height = int(img_height * scale)
        return img_width, img_height

    def get_image_size_for_crop(self, img_width: int, img_height: int, window_size: int) -> Tuple[int, int]:
        w_ratio = img_width / window_size
        h_ratio = img_height / window_size
        if w_ratio < 1:
            width_new = img_width
        else:
            decimal_w = w_ratio - img_width // window_size
            width_new = window_size * (int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio))
        if h_ratio < 1:
            height_new = img_height
        else:
            decimal_h = h_ratio - img_height // window_size
            height_new = window_size * (int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio))
        return int(width_new), int(height_new)

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, List[Image.Image], Optional[List[bool]]]:
        img = img.convert("RGB")
        img_width, img_height = img.size
        new_width, new_height = self.get_image_size_for_padding(img_width, img_height)
        if (new_width, new_height) != (img_width, img_height):
            img = self.square_pad(img)
            img_width, img_height = img.size

        new_width, new_height = self.get_image_size_for_preprocess(img_width, img_height)
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        window_size = self.determine_window_size(max(new_height, new_width), min(new_height, new_width))
        if window_size == 0:
            return img, [], None

        crop_width, crop_height = self.get_image_size_for_crop(new_width, new_height, window_size)
        img_for_crop = img.resize((crop_width, crop_height), Image.Resampling.BILINEAR) if (
            crop_width,
            crop_height,
        ) != (new_width, new_height) else img

        patches: List[Image.Image] = []
        newlines: List[int] = []
        centers, (x_num, _) = self.slide_window(
            crop_width, crop_height, [(window_size, window_size)], [(window_size, window_size)]
        )
        for patch_id, (x, y, patch_w, patch_h) in enumerate(centers):
            patches.append(img_for_crop.crop((x, y, x + patch_w, y + patch_h)))
            if (patch_id + 1) % x_num == 0:
                newlines.append(patch_id)
        if newlines and newlines[-1] == len(patches) - 1:
            newlines.pop()
        return img, patches, [i in newlines for i in range(len(patches))] if patches else None


def _sanitize_conversation(conversation: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for message in conversation:
        item = dict(message)
        content = item.get("content", "")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError(f"Unexpected content item type: {type(part)!r}.")
                part_type = part.get("type")
                if part_type == "text":
                    new_content.append({"type": "text", "text": part.get("text", "")})
                elif part_type in {"image", "image_url", "input_image"}:
                    new_content.append({"type": "image"})
                elif part_type in {"video", "video_url", "input_video"}:
                    new_content.append({"type": "video"})
                else:
                    raise ValueError(
                        f"Step3.7 multimodal only supports text/image/video content, got {part_type!r}."
                    )
            item["content"] = new_content
        sanitized.append(item)
    return sanitized


def _count_media_placeholders(conversation: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    image_count = 0
    video_count = 0
    for message in conversation:
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = part.get("type")
            if part_type == "image":
                image_count += 1
            elif part_type == "video":
                video_count += 1
    return image_count, video_count


def normalize_step3p7_conversation(
    conversation: Sequence[Dict[str, Any]],
    image_count: int,
    video_count: int = 0,
) -> List[Dict[str, Any]]:
    conversation = _sanitize_conversation(conversation)
    if not conversation:
        content = [{"type": "image"} for _ in range(image_count)]
        content.extend({"type": "video"} for _ in range(video_count))
        return [{"role": "user", "content": content}]
    placeholder_images, placeholder_videos = _count_media_placeholders(conversation)
    missing_images = max(0, image_count - placeholder_images)
    missing_videos = max(0, video_count - placeholder_videos)
    if (missing_images == 0 and missing_videos == 0) or (image_count == 0 and video_count == 0):
        return conversation

    inserted_parts = [{"type": "image"} for _ in range(missing_images)]
    inserted_parts.extend({"type": "video"} for _ in range(missing_videos))
    updated = list(conversation)
    for idx in range(len(updated) - 1, -1, -1):
        if updated[idx].get("role") != "user":
            continue
        message = dict(updated[idx])
        content = message.get("content", "")
        if isinstance(content, list):
            message["content"] = inserted_parts + content
        else:
            message["content"] = inserted_parts + [{"type": "text", "text": content}]
        updated[idx] = message
        return updated
    updated.append({"role": "user", "content": inserted_parts})
    return updated


def _expand_video_placeholders(
    conversation: Sequence[Dict[str, Any]],
    images: Sequence[Any],
    videos: Sequence[Any],
    video_max_frames: int,
    video_sampling_fps: float,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    image_queue = list(images)
    video_queue = list(videos)
    expanded_images: List[Any] = []
    expanded_conversation: List[Dict[str, Any]] = []
    video_index = 0

    for message in conversation:
        item = dict(message)
        content = item.get("content", "")
        if isinstance(content, list):
            new_content = []
            for part in content:
                part_type = part.get("type")
                if part_type == "image":
                    if not image_queue:
                        raise ValueError("The number of Step3.7 image placeholders exceeds the number of images.")
                    expanded_images.append(image_queue.pop(0))
                    new_content.append({"type": "image"})
                elif part_type == "video":
                    if not video_queue:
                        raise ValueError("The number of Step3.7 video placeholders exceeds the number of videos.")
                    video_index += 1
                    frames, source_fps, raw_indices = _load_video_source(video_queue.pop(0))
                    sampled_frames, timestamps = _sample_video_frames(
                        frames,
                        source_fps,
                        raw_indices,
                        max_frames=video_max_frames,
                        sampling_fps=video_sampling_fps,
                    )
                    for frame_idx, (frame, timestamp) in enumerate(zip(sampled_frames, timestamps)):
                        new_content.append(
                            {
                                "type": "text",
                                "text": f"\n[video {video_index} frame {frame_idx + 1} at {timestamp:.1f}s]\n",
                            }
                        )
                        new_content.append({"type": "image"})
                        expanded_images.append(frame)
                else:
                    new_content.append(part)
            item["content"] = new_content
        expanded_conversation.append(item)

    if image_queue:
        raise ValueError("More Step3.7 images were provided than image placeholders.")
    if video_queue:
        raise ValueError("More Step3.7 videos were provided than video placeholders.")
    return expanded_conversation, expanded_images


def _render_content(content: Any, allow_image: bool) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: List[str] = []
        needs_text_separator = False
        for item in content:
            if not isinstance(item, dict):
                raise ValueError(f"Unexpected content item type: {type(item)!r}.")
            item_type = item.get("type")
            if item_type == "text":
                if needs_text_separator:
                    parts.append(" ")
                parts.append(item.get("text", ""))
                needs_text_separator = True
            elif item_type in {"image", "image_url"}:
                if not allow_image:
                    raise ValueError("Only user messages may contain images for Step3.7.")
                parts.append(IMAGE_TOKEN)
                needs_text_separator = False
            elif item_type == "video":
                raise ValueError("Step3.7 video placeholders must be expanded into sampled image frames first.")
            else:
                raise ValueError(f"Unexpected content item type: {item_type!r}.")
        return "".join(parts)
    raise ValueError(f"Unexpected content type: {type(content)!r}.")


def build_step3p7_prompt(
    conversation: Sequence[Dict[str, Any]],
    image_replacements: Optional[Sequence[str]] = None,
    add_generation_prompt: bool = True,
) -> str:
    prompt = STEP3_BOS
    for message in _sanitize_conversation(conversation):
        role = message.get("role")
        content = _render_content(message.get("content", ""), allow_image=(role == "user"))
        if role in {"system", "user", "assistant"}:
            prompt += STEP3_IM_START + role + "\n" + content + STEP3_IM_END + "\n"
        elif role == "tool":
            prompt += STEP3_IM_START + "tool_response\n<tool_response>" + content + "</tool_response>" + STEP3_IM_END + "\n"
        else:
            raise ValueError(f"Unexpected message role: {role!r}.")

    if add_generation_prompt:
        prompt += STEP3_IM_START + "assistant\n<think>\n"

    if image_replacements is not None:
        parts = prompt.split(IMAGE_TOKEN)
        if len(parts) - 1 != len(image_replacements):
            raise ValueError("The number of image placeholders does not match the number of images.")
        merged = [parts[0]]
        for idx, repl in enumerate(image_replacements):
            merged.append(repl)
            merged.append(parts[idx + 1])
        prompt = "".join(merged)
    return prompt


def _image_replacement(num_patches: int, patch_newline_mask: Optional[List[bool]]) -> str:
    patch_text = ""
    if num_patches > 0:
        mask = patch_newline_mask or [False] * num_patches
        if len(mask) != num_patches:
            raise ValueError("Patch newline mask length mismatch.")
        for idx in range(num_patches):
            patch_text += PATCH_START_TOKEN + (IMAGE_TOKEN * PATCH_TOKEN_LEN) + PATCH_END_TOKEN
            if mask[idx]:
                patch_text += PATCH_NEWLINE_TOKEN
    return patch_text + IM_START_TOKEN + (IMAGE_TOKEN * IMAGE_TOKEN_LEN) + IM_END_TOKEN


def prepare_step3p7_multimodal_inputs(
    tokenizer,
    model_dir: str,
    model_config: Dict[str, Any],
    conversation: Sequence[Dict[str, Any]],
    images: Optional[Sequence[Any]] = None,
    videos: Optional[Sequence[Any]] = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    encode_fn: Optional[Callable[[str], Sequence[int]]] = None,
    video_max_frames: int = VIDEO_DEFAULT_MAX_FRAMES,
    video_sampling_fps: float = VIDEO_DEFAULT_SAMPLING_FPS,
) -> Dict[str, Any]:
    del model_dir, model_config, enable_thinking
    images = list(images or [])
    videos = list(videos or [])
    if not images and not videos:
        raise ValueError("Step3.7 multimodal preprocessing requires at least one image or video.")

    conversation = normalize_step3p7_conversation(conversation, len(images), len(videos))
    conversation, images = _expand_video_placeholders(
        conversation,
        images,
        videos,
        video_max_frames=video_max_frames,
        video_sampling_fps=video_sampling_fps,
    )
    if not images:
        raise ValueError("Step3.7 video preprocessing did not produce any sampled frames.")

    patcher = ImagePatcher()
    pixel_values: List[np.ndarray] = []
    patch_pixel_values: List[np.ndarray] = []
    num_patches: List[int] = []
    replacements: List[str] = []

    for image in images:
        full_image, patches, newline_mask = patcher(_to_pil_rgb(image))
        pixel_values.append(_resize_normalize_chw(full_image, IMAGE_SIZE))
        for patch in patches:
            patch_pixel_values.append(_resize_normalize_chw(patch, PATCH_SIZE))
        num_patches.append(len(patches))
        replacements.append(_image_replacement(len(patches), newline_mask))

    prompt = build_step3p7_prompt(
        conversation,
        image_replacements=replacements,
        add_generation_prompt=add_generation_prompt,
    )
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    elif encode_fn is not None:
        input_ids = list(encode_fn(prompt))
    else:
        raise ValueError("Step3.7 multimodal preprocessing requires tokenizer.encode or encode_fn.")

    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "pixel_values": np.stack(pixel_values, axis=0).astype(np.float32, copy=False),
        "patch_pixel_values": (
            np.stack(patch_pixel_values, axis=0).astype(np.float32, copy=False)
            if patch_pixel_values
            else None
        ),
        "num_patches": np.asarray(num_patches, dtype=np.int32),
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


def build_step3p7_multimodal_payload(native_inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], bytes]:
    arrays: List[np.ndarray] = []
    descriptors: List[Dict[str, Any]] = []
    offset = 0
    offset = _append_payload_tensor(arrays, descriptors, "pixel_values", native_inputs["pixel_values"], np.float32, offset)
    offset = _append_payload_tensor(
        arrays, descriptors, "patch_pixel_values", native_inputs.get("patch_pixel_values"), np.float32, offset
    )
    offset = _append_payload_tensor(arrays, descriptors, "num_patches", native_inputs["num_patches"], np.int32, offset)
    del offset
    return {"mode": "qwen35", "tensors": descriptors}, b"".join(array.tobytes(order="C") for array in arrays)
