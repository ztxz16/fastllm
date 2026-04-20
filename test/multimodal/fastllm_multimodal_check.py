import argparse
import json
import os
import sys
import time

from PIL import Image

from common import (
    FASTLLM_TOOLS_PATH,
    SUPPORTED_PROFILE_CHOICES,
    create_probe_image,
    create_probe_video,
    get_default_probe_image_path,
    get_default_probe_video_path,
    get_profile_by_name,
    infer_profile_name,
    load_messages,
    print_banner,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a generic fastllm multimodal smoke test.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型目录。",
    )
    parser.add_argument(
        "--profile",
        choices=SUPPORTED_PROFILE_CHOICES,
        default="auto",
        help="消息和默认 prompt 配置。auto 会按 architecture 自动推断。",
    )
    parser.add_argument(
        "--image",
        default="",
        help="测试图片路径。不传时会在 test/multimodal 下自动生成一张。",
    )
    parser.add_argument(
        "--video",
        default="",
        help="测试视频路径。建议传 GIF；不传时 Qwen3.5 profile 会自动生成一段 GIF probe。",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="默认 user prompt。若传了 messages 参数则忽略。",
    )
    parser.add_argument(
        "--messages-json",
        default="",
        help="直接传入 JSON 字符串形式的 messages。",
    )
    parser.add_argument(
        "--messages-file",
        default="",
        help="从 JSON 文件读取 messages。",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="ftllm.llm.model 的 dtype。",
    )
    parser.add_argument(
        "--device",
        default="",
        help="可选推理设备，例如 cpu、cuda 或 cuda:0。",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="可选 CPU 线程数，>0 时会调用 llm.set_cpu_threads。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="最大生成 token 数。",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=0,
        help="可选 paged cache token 上限，>0 时调用 llm.set_max_tokens。",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=0,
        help="可选 paged cache page size，>0 时调用 llm.set_page_size。",
    )
    parser.add_argument(
        "--gpu-mem-ratio",
        type=float,
        default=0.0,
        help="可选 GPU 显存占比，>0 时调用 llm.set_gpu_mem_ratio。",
    )
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=0,
        help="可选 chunked prefill size，>0 时调用 model.set_chunked_prefill_size。",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="加载模型后先执行一次 model.warmup()。",
    )
    parser.add_argument(
        "--text-only-first",
        action="store_true",
        help="先跑一遍文本链路检查。",
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="不传图片，只测视频链路。",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="不传视频，只测图片链路。",
    )
    parser.add_argument(
        "--show-messages",
        action="store_true",
        help="打印最终传给 fastllm 的 messages。",
    )
    return parser.parse_args()


def load_fastllm_llm():
    if FASTLLM_TOOLS_PATH not in sys.path:
        sys.path.insert(0, FASTLLM_TOOLS_PATH)
    from ftllm import llm

    return llm


def collect_stream_output(model, messages, image, video, max_new_tokens):
    multimodal_kwargs = {}
    if image is not None:
        multimodal_kwargs["images"] = [image]
    if video is not None:
        multimodal_kwargs["videos"] = [video]
    if not multimodal_kwargs:
        raise ValueError("At least one of image or video must be provided.")
    handle = model.launch_stream_response(
        messages,
        max_length=max_new_tokens,
        do_sample=False,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        repeat_penalty=1.0,
        one_by_one=True,
        enable_thinking=False,
        **multimodal_kwargs,
    )
    return "".join(model.stream_response_handle(handle))


def collect_text_only_output(model, prompt):
    handle = model.launch_stream_response(
        [{"role": "user", "content": prompt}],
        max_length=32,
        do_sample=False,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        repeat_penalty=1.0,
        one_by_one=True,
        enable_thinking=False,
    )
    return "".join(model.stream_response_handle(handle))


def main():
    args = parse_args()
    llm = load_fastllm_llm()
    if args.cpu_threads > 0:
        llm.set_cpu_threads(args.cpu_threads)
    if args.device:
        llm.set_device_map(args.device)
    if args.tokens > 0:
        llm.set_max_tokens(args.tokens)
    if args.page_size > 0:
        llm.set_page_size(args.page_size)
    if args.gpu_mem_ratio > 0:
        llm.set_gpu_mem_ratio(args.gpu_mem_ratio)

    print_banner("Environment")
    print(f"model_path: {args.model_path}")
    print(f"dtype: {args.dtype}")
    print(f"device: {args.device or '<default>'}")
    print(f"cpu_threads: {args.cpu_threads if args.cpu_threads > 0 else '<default>'}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"tokens: {args.tokens if args.tokens > 0 else '<default>'}")
    print(f"page_size: {args.page_size if args.page_size > 0 else '<default>'}")
    print(f"gpu_mem_ratio: {args.gpu_mem_ratio if args.gpu_mem_ratio > 0 else '<default>'}")
    print(f"chunked_prefill_size: {args.chunked_prefill_size if args.chunked_prefill_size > 0 else '<default>'}")
    print(f"warmup: {args.warmup}")
    print(f"requested_profile: {args.profile}")

    load_start = time.time()
    model = llm.model(args.model_path, dtype=args.dtype)
    if args.chunked_prefill_size > 0:
        model.set_chunked_prefill_size(args.chunked_prefill_size)
    if args.warmup:
        model.warmup()
    load_sec = time.time() - load_start

    architecture = "<unknown>"
    if hasattr(model, "config"):
        architecture = (model.config.get("architectures") or ["<unknown>"])[0]
    profile_name = infer_profile_name(architecture, args.profile)
    profile = get_profile_by_name(profile_name)

    messages = load_messages(args.prompt, args.messages_json, args.messages_file, profile)
    use_image = not args.skip_image
    use_video = (not args.skip_video) and (profile_name == "qwen35" or args.video != "")
    if not use_image and not use_video:
        raise ValueError("图片和视频不能同时跳过。")

    image_path = ""
    image_created = False
    if use_image:
        image_path = args.image or get_default_probe_image_path(profile_name)
        image_label = profile_name[:2].upper()
        image_path, image_created = create_probe_image(image_path, label=image_label)

    video_path = ""
    video_created = False
    if use_video:
        if args.video and not os.path.exists(args.video):
            raise FileNotFoundError(f"Video not found: {args.video}")
        video_path = args.video or get_default_probe_video_path(profile_name)
        video_label = profile_name[:2].upper()
        video_path, video_created = create_probe_video(video_path, label=video_label)

    print_banner("Model")
    print(f"load_sec: {load_sec:.3f}")
    print(f"architecture: {architecture}")
    print(f"resolved_profile: {profile.name}")
    print(f"profile_description: {profile.description}")
    print(f"image_enabled: {use_image}")
    print(f"image_path: {image_path or '<disabled>'}")
    print(f"image_created: {image_created}")
    print(f"video_enabled: {use_video}")
    print(f"video_path: {video_path or '<disabled>'}")
    print(f"video_created: {video_created}")

    if args.show_messages:
        print_banner("Messages")
        print(json.dumps(messages, ensure_ascii=False, indent=2))

    if args.text_only_first:
        print_banner("Text Only")
        print(collect_text_only_output(model, profile.text_only_prompt))

    image = None
    if use_image:
        image = Image.open(image_path).convert("RGB")
    try:
        print_banner("Multimodal Output")
        print(collect_stream_output(model, messages, image, video_path if use_video else None, args.max_new_tokens))
    finally:
        if image is not None:
            image.close()


if __name__ == "__main__":
    main()
