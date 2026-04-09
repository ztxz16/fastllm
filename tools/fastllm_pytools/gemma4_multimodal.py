import inspect
import json
import math
import os
import re

import numpy as np
from PIL import Image


SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gemma4_multimodal_config(model_dir, model_config=None):
    processor_config_path = os.path.join(model_dir, "processor_config.json")
    processor_config = _load_json(processor_config_path) if os.path.exists(processor_config_path) else {}
    image_processor_config = processor_config.get("image_processor", {})
    vision_config = (model_config or {}).get("vision_config", {})

    patch_size = image_processor_config.get("patch_size", vision_config.get("patch_size", 16))
    max_soft_tokens = image_processor_config.get(
        "max_soft_tokens",
        (model_config or {}).get("vision_soft_tokens_per_image", 280),
    )
    pooling_kernel_size = image_processor_config.get(
        "pooling_kernel_size",
        vision_config.get("pooling_kernel_size", 3),
    )
    rescale_factor = image_processor_config.get("rescale_factor", 1.0 / 255.0)

    if max_soft_tokens not in SUPPORTED_SOFT_TOKENS:
        raise ValueError(
            f"Gemma4 max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}."
        )

    return {
        "patch_size": int(patch_size),
        "max_soft_tokens": int(max_soft_tokens),
        "pooling_kernel_size": int(pooling_kernel_size),
        "rescale_factor": float(rescale_factor),
    }


def get_aspect_ratio_preserving_size(height, width, patch_size, max_patches, pooling_kernel_size):
    total_px = height * width
    target_px = max_patches * (patch_size ** 2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. "
            f"Dimensions must stay divisible by pooling_kernel_size * patch_size = {side_mult}."
        )

    max_side_length = (max_patches // (pooling_kernel_size ** 2)) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] exceeds patch budget {max_patches}."
        )

    return target_height, target_width


def convert_image_to_patches(image, patch_size):
    channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(
        channels,
        num_patches_height,
        patch_size,
        num_patches_width,
        patch_size,
    )
    patched_image = patched_image.transpose(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def pad_along_first_dim(image, positions, target_length):
    current_length = image.shape[0]
    padding_length = target_length - current_length
    if padding_length > 0:
        image_padding = [(0, padding_length)] + [(0, 0)] * (image.ndim - 1)
        positions_padding = [(0, padding_length), (0, 0)]
        image = np.pad(image, image_padding, mode="constant", constant_values=0.0)
        positions = np.pad(positions, positions_padding, mode="constant", constant_values=-1)
    return image, positions


def preprocess_single_image(image, config):
    patch_size = config["patch_size"]
    max_soft_tokens = config["max_soft_tokens"]
    pooling_kernel_size = config["pooling_kernel_size"]
    rescale_factor = config["rescale_factor"]

    max_patches = max_soft_tokens * (pooling_kernel_size ** 2)
    side_mult = pooling_kernel_size * patch_size

    image = image.convert("RGB")
    width, height = image.size
    target_height, target_width = get_aspect_ratio_preserving_size(
        height=height,
        width=width,
        patch_size=patch_size,
        max_patches=max_patches,
        pooling_kernel_size=pooling_kernel_size,
    )

    if target_height % side_mult != 0 or target_width % side_mult != 0:
        raise ValueError(
            f"Target size [{target_height}x{target_width}] must be divisible by {side_mult}."
        )

    if (height, width) != (target_height, target_width):
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize((target_width, target_height), resample=resampling.BICUBIC)

    image_np = np.asarray(image, dtype=np.float32) * rescale_factor
    image_np = image_np.transpose(2, 0, 1)

    patch_height = image_np.shape[-2] // patch_size
    patch_width = image_np.shape[-1] // patch_size
    patches = convert_image_to_patches(image_np, patch_size).astype(np.float32, copy=False)
    num_soft_tokens = patches.shape[0] // (pooling_kernel_size ** 2)

    patch_grid = np.meshgrid(
        np.arange(patch_width, dtype=np.int32),
        np.arange(patch_height, dtype=np.int32),
        indexing="xy",
    )
    stacked_grid = np.stack(patch_grid, axis=-1)
    real_positions = stacked_grid.reshape(patches.shape[0], 2)

    padded_patches, padded_positions = pad_along_first_dim(patches, real_positions, max_patches)
    return {
        "pixel_values": padded_patches,
        "image_position_ids": padded_positions.astype(np.int32, copy=False),
        "num_soft_tokens": int(num_soft_tokens),
        "resized_height": int(target_height),
        "resized_width": int(target_width),
    }


def preprocess_images(images, model_dir, model_config=None):
    config = get_gemma4_multimodal_config(model_dir, model_config=model_config)
    processed = [preprocess_single_image(image, config) for image in images]
    pixel_values = np.stack([item["pixel_values"] for item in processed], axis=0).astype(np.float32, copy=False)
    image_position_ids = np.stack(
        [item["image_position_ids"] for item in processed],
        axis=0,
    ).astype(np.int32, copy=False)

    return {
        "config": config,
        "pixel_values": pixel_values,
        "image_position_ids": image_position_ids,
        "num_soft_tokens_per_image": [item["num_soft_tokens"] for item in processed],
        "resized_sizes": [(item["resized_height"], item["resized_width"]) for item in processed],
    }


def sanitize_conversation_for_template(conversation):
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
                else:
                    raise ValueError(f"Gemma4 native multimodal only supports text/image content, got {part_type!r}.")
            item["content"] = new_content
        else:
            item["content"] = content
        sanitized.append(item)
    return sanitized


def normalize_gemma4_conversation(conversation, image_count):
    conversation = sanitize_conversation_for_template(conversation)
    image_placeholders = []
    for _ in range(image_count):
        image_placeholders.append({"type": "image"})

    has_image_placeholder = False
    for message in conversation:
        content = message.get("content", "")
        if isinstance(content, list):
            has_image_placeholder = has_image_placeholder or any(part.get("type") == "image" for part in content)

    if has_image_placeholder or image_count == 0:
        return conversation

    if not conversation:
        return [{"role": "user", "content": image_placeholders}]

    updated = []
    attached = False
    for idx in range(len(conversation) - 1, -1, -1):
        message = dict(conversation[idx])
        if message.get("role") != "user":
            updated.append(message)
            continue

        content = message.get("content", "")
        if isinstance(content, list):
            message["content"] = image_placeholders + content
        else:
            message["content"] = image_placeholders + [{"type": "text", "text": content}]
        updated.append(message)
        attached = True
        updated.extend(reversed(conversation[:idx]))
        break

    if not attached:
        updated = list(conversation)
        updated.append({"role": "user", "content": image_placeholders})
    else:
        updated = list(reversed(updated))

    return updated


def apply_chat_template_with_optional_thinking(tokenizer, conversation, add_generation_prompt, enable_thinking):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    signature = inspect.signature(tokenizer.apply_chat_template)
    if "enable_thinking" in signature.parameters:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(conversation, **kwargs)


def build_prompt_with_expanded_image_tokens(tokenizer, conversation, num_soft_tokens_per_image, add_generation_prompt, enable_thinking):
    sanitized = sanitize_conversation_for_template(conversation)
    prompt = apply_chat_template_with_optional_thinking(
        tokenizer,
        sanitized,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    replacements = [
        f"{tokenizer.boi_token}{tokenizer.image_token * token_count}{tokenizer.eoi_token}"
        for token_count in num_soft_tokens_per_image
    ]
    expected = prompt.count(tokenizer.image_token)
    if expected != len(replacements):
        raise ValueError(
            f"Gemma4 image placeholder count mismatch: template has {expected}, images provide {len(replacements)}."
        )
    replacements_iter = iter(replacements)
    pattern = re.escape(tokenizer.image_token)
    return re.sub(pattern, lambda _: next(replacements_iter), prompt)


def create_mm_token_type_ids(input_ids, image_token_id):
    return [1 if token_id == image_token_id else 0 for token_id in input_ids]


def prepare_gemma4_multimodal_inputs(tokenizer, model_dir, model_config, conversation, images, add_generation_prompt=True, enable_thinking=False):
    if not images:
        raise ValueError("Gemma4 multimodal preprocessing requires at least one image.")

    image_inputs = preprocess_images(images, model_dir=model_dir, model_config=model_config)
    prompt = build_prompt_with_expanded_image_tokens(
        tokenizer,
        conversation=conversation,
        num_soft_tokens_per_image=image_inputs["num_soft_tokens_per_image"],
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    mm_token_type_ids = create_mm_token_type_ids(input_ids, tokenizer.image_token_id)

    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "pixel_values": image_inputs["pixel_values"],
        "image_position_ids": image_inputs["image_position_ids"],
        "mm_token_type_ids": mm_token_type_ids,
        "num_soft_tokens_per_image": image_inputs["num_soft_tokens_per_image"],
        "resized_sizes": image_inputs["resized_sizes"],
        "image_config": image_inputs["config"],
    }


def build_gemma4_multimodal_payload(native_inputs, tokenizer):
    pixel_values = native_inputs["pixel_values"].astype(np.float32, copy=False)
    image_position_ids = native_inputs["image_position_ids"].astype(np.float32, copy=False)
    mm_token_type_ids = np.asarray(native_inputs["mm_token_type_ids"], dtype=np.float32).reshape(1, -1)

    pixel_values_flat = pixel_values.reshape(-1)
    image_position_ids_flat = image_position_ids.reshape(-1)
    mm_token_type_ids_flat = mm_token_type_ids.reshape(-1)

    pixel_values_offset = 0
    image_position_ids_offset = pixel_values_flat.size
    mm_token_type_ids_offset = image_position_ids_offset + image_position_ids_flat.size

    payload = np.concatenate(
        [
            pixel_values_flat,
            image_position_ids_flat,
            mm_token_type_ids_flat,
        ]
    ).astype(np.float32, copy=False)

    payload_config = {
        "mode": "gemma4",
        "pixel_values_shape": list(pixel_values.shape),
        "image_position_ids_shape": list(image_position_ids.shape),
        "mm_token_type_ids_shape": list(mm_token_type_ids.shape),
        "num_soft_tokens_per_image": list(native_inputs["num_soft_tokens_per_image"]),
        "pixel_values_offset": int(pixel_values_offset),
        "pixel_values_length": int(pixel_values_flat.size),
        "image_position_ids_offset": int(image_position_ids_offset),
        "image_position_ids_length": int(image_position_ids_flat.size),
        "mm_token_type_ids_offset": int(mm_token_type_ids_offset),
        "mm_token_type_ids_length": int(mm_token_type_ids_flat.size),
        "image_token_id": int(tokenizer.image_token_id),
        "boi_token_id": int(tokenizer.boi_token_id),
        "eoi_token_id": int(tokenizer.eoi_token_id),
        "patch_size": int(native_inputs["image_config"]["patch_size"]),
        "pooling_kernel_size": int(native_inputs["image_config"]["pooling_kernel_size"]),
        "max_soft_tokens": int(native_inputs["image_config"]["max_soft_tokens"]),
    }
    return payload_config, payload
