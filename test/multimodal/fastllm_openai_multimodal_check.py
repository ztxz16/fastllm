#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common import (
    create_probe_image,
    create_probe_video,
    get_default_probe_image_path,
    get_default_probe_video_path,
    print_banner,
)


DEFAULT_PROMPT = "请用一句话同时概括图片和视频里的主要颜色、形状和变化。"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an OpenAI-compatible multimodal smoke test against fastllm serve.",
    )
    parser.add_argument("--base-url", required=True, help="服务地址，例如 http://127.0.0.1:8080")
    parser.add_argument("--model", required=True, help="OpenAI 接口里的模型名。")
    parser.add_argument("--api-key", default="no-key", help="鉴权 token。")
    parser.add_argument("--image", default="", help="本地图片路径；不传时自动生成 probe 图。")
    parser.add_argument("--video", default="", help="本地视频路径；建议传 GIF，不传时自动生成 probe GIF。")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="随图像/视频一起发送的文本问题。")
    parser.add_argument("--max-tokens", type=int, default=64, help="最大生成 token 数。")
    parser.add_argument("--skip-image", action="store_true", help="不发送图片。")
    parser.add_argument("--skip-video", action="store_true", help="不发送视频。")
    return parser.parse_args()


def ensure_probe_image(path: str):
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return path, False
    return create_probe_image(get_default_probe_image_path("qwen35"), label="QW")


def ensure_probe_video(path: str):
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")
        return path, False
    return create_probe_video(get_default_probe_video_path("qwen35"), label="QW")


def build_file_url(path: str) -> str:
    return Path(path).resolve().as_uri()


def build_payload(args, image_path: str, video_path: str):
    content = [{"type": "text", "text": args.prompt}]
    if image_path:
        content.append({
            "type": "image_url",
            "image_url": {"url": build_file_url(image_path)},
        })
    if video_path:
        content.append({
            "type": "video_url",
            "video_url": {"url": build_file_url(video_path)},
        })
    return {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }


def post_json(url: str, payload: dict, api_key: str):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    args = parse_args()
    use_image = not args.skip_image
    use_video = not args.skip_video
    if not use_image and not use_video:
        raise ValueError("图片和视频不能同时跳过。")

    image_path = ""
    image_created = False
    if use_image:
        image_path, image_created = ensure_probe_image(args.image)

    video_path = ""
    video_created = False
    if use_video:
        video_path, video_created = ensure_probe_video(args.video)

    payload = build_payload(args, image_path, video_path)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"

    print_banner("Request")
    print(f"url: {url}")
    print(f"model: {args.model}")
    print(f"image_path: {image_path or '<disabled>'}")
    print(f"image_created: {image_created}")
    print(f"video_path: {video_path or '<disabled>'}")
    print(f"video_created: {video_created}")
    print(f"prompt: {args.prompt}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        data = post_json(url, payload, args.api_key)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc

    print_banner("Response")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Response does not contain choices.")
    text = choices[0].get("message", {}).get("content", "")
    if not text.strip():
        raise RuntimeError("Response content is empty.")
    print_banner("Summary")
    print(text)


if __name__ == "__main__":
    main()
