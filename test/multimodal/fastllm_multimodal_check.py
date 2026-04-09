import argparse
import json
import sys
import time

from PIL import Image

from common import (
    FASTLLM_TOOLS_PATH,
    SUPPORTED_PROFILE_CHOICES,
    create_probe_image,
    get_default_probe_image_path,
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
        "--text-only-first",
        action="store_true",
        help="先跑一遍文本链路检查。",
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


def collect_stream_output(model, messages, image, max_new_tokens):
    handle = model.launch_stream_response(
        messages,
        max_length=max_new_tokens,
        do_sample=False,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        repeat_penalty=1.0,
        one_by_one=True,
        images=[image],
        enable_thinking=False,
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

    print_banner("Environment")
    print(f"model_path: {args.model_path}")
    print(f"dtype: {args.dtype}")
    print(f"device: {args.device or '<default>'}")
    print(f"cpu_threads: {args.cpu_threads if args.cpu_threads > 0 else '<default>'}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"requested_profile: {args.profile}")

    load_start = time.time()
    model = llm.model(args.model_path, dtype=args.dtype)
    load_sec = time.time() - load_start

    architecture = "<unknown>"
    if hasattr(model, "config"):
        architecture = (model.config.get("architectures") or ["<unknown>"])[0]
    profile_name = infer_profile_name(architecture, args.profile)
    profile = get_profile_by_name(profile_name)

    image_path = args.image or get_default_probe_image_path(profile_name)
    image_label = profile_name[:2].upper()
    image_path, image_created = create_probe_image(image_path, label=image_label)
    messages = load_messages(args.prompt, args.messages_json, args.messages_file, profile)

    print_banner("Model")
    print(f"load_sec: {load_sec:.3f}")
    print(f"architecture: {architecture}")
    print(f"resolved_profile: {profile.name}")
    print(f"profile_description: {profile.description}")
    print(f"image_path: {image_path}")
    print(f"image_created: {image_created}")

    if args.show_messages:
        print_banner("Messages")
        print(json.dumps(messages, ensure_ascii=False, indent=2))

    if args.text_only_first:
        print_banner("Text Only")
        print(collect_text_only_output(model, profile.text_only_prompt))

    image = Image.open(image_path).convert("RGB")
    try:
        print_banner("Multimodal Output")
        print(collect_stream_output(model, messages, image, args.max_new_tokens))
    finally:
        image.close()


if __name__ == "__main__":
    main()
