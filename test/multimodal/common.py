import json
import os
from dataclasses import dataclass
from typing import Any, List

from PIL import Image, ImageDraw


TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TEST_ROOT, "..", ".."))
FASTLLM_TOOLS_PATH = os.path.join(REPO_ROOT, "build", "tools")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    architectures: tuple[str, ...]
    default_prompt: str
    text_only_prompt: str
    description: str


MODEL_PROFILES = {
    "generic": ModelProfile(
        name="generic",
        architectures=(),
        default_prompt="Describe the uploaded image in one short sentence.",
        text_only_prompt="Reply with one short sentence confirming the text pipeline is working.",
        description="默认通用图文测试配置，适合大多数接收 images= 参数的模型。",
    ),
    "gemma4": ModelProfile(
        name="gemma4",
        architectures=("Gemma4ForConditionalGeneration",),
        default_prompt="Describe the uploaded image in one short sentence. Mention the main shapes and colors.",
        text_only_prompt="Reply with one short sentence confirming the text pipeline is working.",
        description="Gemma4 图文测试配置。",
    ),
    "qwen35": ModelProfile(
        name="qwen35",
        architectures=("Qwen3_5ForConditionalGeneration",),
        default_prompt="Describe the uploaded image and video together in one short sentence. Mention the main colors and moving shapes.",
        text_only_prompt="Reply with one short sentence confirming the text pipeline is working.",
        description="Qwen3.5 图像+视频测试配置。",
    ),
    "cogvlm": ModelProfile(
        name="cogvlm",
        architectures=("CogVLMForCausalLM",),
        default_prompt="Describe the uploaded image in one short sentence.",
        text_only_prompt="Reply with one short sentence confirming the text pipeline is working.",
        description="CogVLM 图文测试配置。",
    ),
}

ARCHITECTURE_TO_PROFILE = {}
for profile in MODEL_PROFILES.values():
    for architecture in profile.architectures:
        ARCHITECTURE_TO_PROFILE[architecture] = profile.name

SUPPORTED_PROFILE_CHOICES = ["auto"] + sorted(MODEL_PROFILES.keys())


def print_banner(title: str):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def get_profile_by_name(profile_name: str) -> ModelProfile:
    return MODEL_PROFILES[profile_name]


def infer_profile_name(architecture: str, requested_profile: str) -> str:
    if requested_profile != "auto":
        return requested_profile
    return ARCHITECTURE_TO_PROFILE.get(architecture, "generic")


def get_default_probe_image_path(profile_name: str) -> str:
    return os.path.join(TEST_ROOT, f"{profile_name}_probe.png")


def get_default_probe_video_path(profile_name: str) -> str:
    return os.path.join(TEST_ROOT, f"{profile_name}_probe.gif")


def create_probe_image(image_path: str, label: str = "MM"):
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if os.path.exists(image_path):
        return image_path, False

    image = Image.new("RGB", (384, 384), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.rectangle((36, 36, 172, 172), fill=(220, 40, 40), outline=(90, 0, 0), width=4)
    draw.ellipse((212, 44, 348, 180), fill=(40, 100, 230), outline=(0, 40, 120), width=4)
    draw.polygon([(104, 228), (36, 348), (172, 348)], fill=(24, 165, 84), outline=(0, 80, 20), width=4)
    draw.rectangle((216, 228, 348, 348), fill=(245, 220, 80), outline=(120, 90, 0), width=4)
    draw.text((246, 280), label, fill=(30, 30, 30))
    image.save(image_path)
    return image_path, True


def create_probe_video(video_path: str, label: str = "MM"):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    if os.path.exists(video_path):
        return video_path, False

    frames = []
    width, height = 320, 240
    background = (244, 244, 240)
    for frame_idx in range(6):
        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)
        offset = frame_idx * 20
        draw.rectangle((24 + offset, 36, 104 + offset, 116), fill=(220, 40, 40), outline=(90, 0, 0), width=4)
        draw.ellipse((192 - offset // 2, 42 + offset // 3, 280 - offset // 2, 130 + offset // 3),
                     fill=(40, 100, 230), outline=(0, 40, 120), width=4)
        draw.polygon(
            [(76, 180 - offset // 4), (28, 224 - offset // 4), (124, 224 - offset // 4)],
            fill=(24, 165, 84), outline=(0, 80, 20), width=4,
        )
        draw.rectangle((208, 156, 288, 220), fill=(245, 220, 80), outline=(120, 90, 0), width=4)
        draw.text((228, 180), f"{label}{frame_idx}", fill=(24, 24, 24))
        frames.append(image)

    frames[0].save(
        video_path,
        save_all=True,
        append_images=frames[1:],
        duration=180,
        loop=0,
    )
    return video_path, True


def _validate_messages(messages: Any) -> List[dict]:
    if not isinstance(messages, list):
        raise ValueError("messages 必须是 JSON 数组。")
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{idx}] 必须是对象。")
        if "role" not in message or "content" not in message:
            raise ValueError(f"messages[{idx}] 需要同时包含 role 和 content。")
    return messages


def load_messages(prompt: str, messages_json: str, messages_file: str, profile: ModelProfile) -> List[dict]:
    if messages_json and messages_file:
        raise ValueError("--messages-json 和 --messages-file 只能二选一。")

    if messages_json:
        return _validate_messages(json.loads(messages_json))

    if messages_file:
        with open(messages_file, "r", encoding="utf-8") as f:
            return _validate_messages(json.load(f))

    final_prompt = prompt or profile.default_prompt
    return [{"role": "user", "content": final_prompt}]
