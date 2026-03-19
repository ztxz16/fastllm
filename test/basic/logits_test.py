import importlib.util
import json
import logging
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTOOLS_DIR = REPO_ROOT / "tools" / "fastllm_pytools"

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_PROMPT = "北京有什么景点？"


def bootstrap_local_ftllm():
    if "ftllm" in sys.modules:
        return

    local_runtime_libs = [
        PYTOOLS_DIR / "libfastllm_tools.so",
        PYTOOLS_DIR / "libfastllm_tools-cu11.so",
        PYTOOLS_DIR / "libfastllm_tools-cpu.so",
        PYTOOLS_DIR / "fastllm_tools.dll",
        PYTOOLS_DIR / "libfastllm_tools.dylib",
    ]
    if not any(path.exists() for path in local_runtime_libs):
        return

    init_file = PYTOOLS_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "ftllm",
        init_file,
        submodule_search_locations=[str(PYTOOLS_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local ftllm package from {init_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["ftllm"] = module
    spec.loader.exec_module(module)


def args_parser():
    bootstrap_local_ftllm()
    from ftllm.util import make_normal_parser

    parser = make_normal_parser("fastllm_logits_test")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="单轮用户输入内容",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="",
        help="从文件读取 prompt 内容，优先于 --prompt",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="system prompt，使用 --messages 或 --direct_prompt 时忽略",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="",
        help="历史对话，支持 JSON 字符串或 JSON 文件路径，格式为 [[user, assistant], ...]",
    )
    parser.add_argument(
        "--messages",
        type=str,
        default="",
        help="完整 messages，支持 JSON 字符串或 JSON 文件路径，格式为 [{\"role\":..., \"content\":...}, ...]",
    )
    parser.add_argument(
        "--direct_prompt",
        action="store_true",
        help="将 prompt 视为最终输入，绕过 chat_template",
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="打印最终送入模型的 prompt",
    )
    parser.add_argument(
        "--output_tokens",
        "--max_new_tokens",
        dest="max_new_tokens",
        type=int,
        default=16,
        help="控制输出的 token 数，默认 16",
    )
    parser.add_argument(
        "--use_generation_config",
        action="store_true",
        help="从模型的 generation_config.json 读取采样参数",
    )
    parser.set_defaults(do_sample=False)
    parser.add_argument(
        "--do_sample",
        dest="do_sample",
        action="store_true",
        help="启用采样",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="覆盖 top_p",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="覆盖 top_k",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="覆盖 temperature",
    )
    parser.add_argument(
        "--repeat_penalty",
        "--repetition_penalty",
        dest="repeat_penalty",
        type=float,
        default=None,
        help="覆盖 repeat_penalty",
    )
    parser.set_defaults(skip_warmup=True)
    parser.add_argument(
        "--warmup",
        dest="skip_warmup",
        action="store_false",
        help="启用模型 warmup，默认跳过以避免 FASTLLM_PRINT_LOGITS=1 时混入 warmup 输出",
    )
    return parser.parse_args()


def load_json_arg(value: str, arg_name: str):
    if value is None or value == "":
        return None
    try:
        if os.path.exists(value):
            with open(value, "r", encoding="utf-8") as file:
                return json.load(file)
        return json.loads(value)
    except Exception as exc:
        raise ValueError(f"{arg_name} 解析失败: {exc}") from exc


def normalize_history(raw_history: str):
    history = load_json_arg(raw_history, "--history")
    if history is None:
        return []
    if not isinstance(history, list):
        raise ValueError("--history 必须是列表")

    normalized = []
    for idx, item in enumerate(history):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user, assistant = item
        elif isinstance(item, dict) and "user" in item and "assistant" in item:
            user = item["user"]
            assistant = item["assistant"]
        else:
            raise ValueError(
                f"--history 第 {idx} 项格式错误，期望 [user, assistant] 或 {{\"user\": ..., \"assistant\": ...}}"
            )
        normalized.append((str(user), str(assistant)))
    return normalized


def normalize_messages(raw_messages: str):
    messages = load_json_arg(raw_messages, "--messages")
    if messages is None:
        return None
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("--messages 必须是非空列表")

    normalized = []
    for idx, item in enumerate(messages):
        if not isinstance(item, dict):
            raise ValueError(f"--messages 第 {idx} 项必须是对象")
        role = str(item.get("role", "")).strip()
        if role == "":
            raise ValueError(f"--messages 第 {idx} 项缺少 role")
        if "content" not in item:
            raise ValueError(f"--messages 第 {idx} 项缺少 content")
        normalized.append({
            "role": role,
            "content": str(item["content"]),
        })
    return normalized


def load_prompt(args) -> str:
    if args.prompt_file != "":
        with open(args.prompt_file, "r", encoding="utf-8") as file:
            return file.read()
    return args.prompt


def build_generation_config(model, args):
    generation_config = {
        "repeat_penalty": 1.0,
        "top_p": 0.8,
        "top_k": 1,
        "temperature": 1.0,
    }
    if args.use_generation_config:
        model_generation_config = getattr(model, "default_generation_config", {})
        if "repetition_penalty" in model_generation_config:
            generation_config["repeat_penalty"] = model_generation_config["repetition_penalty"]
        for key in ["top_p", "top_k", "temperature"]:
            if key in model_generation_config:
                generation_config[key] = model_generation_config[key]

    if args.repeat_penalty is not None:
        generation_config["repeat_penalty"] = args.repeat_penalty
    if args.top_p is not None:
        generation_config["top_p"] = args.top_p
    if args.top_k is not None:
        generation_config["top_k"] = args.top_k
    if args.temperature is not None:
        generation_config["temperature"] = args.temperature

    return generation_config


def build_request(model, args):
    messages = normalize_messages(args.messages)
    if messages is not None:
        prompt = model.apply_chat_template(messages, add_generation_prompt=True)
        return messages, None, prompt

    prompt = load_prompt(args)
    if args.direct_prompt:
        model.direct_query = True
        return prompt, None, prompt

    history = normalize_history(args.history)
    model.system_prompt = args.system
    rendered_prompt = model.get_prompt(prompt, history)
    return prompt, history, rendered_prompt


if __name__ == "__main__":
    args = args_parser()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    if args.skip_warmup:
        os.environ["FASTLLM_SKIP_WARMUP"] = "1"
    else:
        os.environ["FASTLLM_SKIP_WARMUP"] = "0"

    from ftllm.util import make_normal_llm_model

    model = None
    try:
        model = make_normal_llm_model(args)
        query, history, rendered_prompt = build_request(model, args)
        generation_config = build_generation_config(model, args)

        logger.info("开始单 batch 推理测试")
        logger.info(
            "decode 配置: do_sample=%s, top_p=%s, top_k=%s, temperature=%s, repeat_penalty=%s, max_new_tokens=%s",
            args.do_sample,
            generation_config["top_p"],
            generation_config["top_k"],
            generation_config["temperature"],
            generation_config["repeat_penalty"],
            args.max_new_tokens,
        )
        if args.show_prompt:
            print("Prompt:")
            print(rendered_prompt)

        response = model.response(
            query,
            history=history,
            max_length=args.max_new_tokens,
            do_sample=args.do_sample,
            top_p=generation_config["top_p"],
            top_k=generation_config["top_k"],
            temperature=generation_config["temperature"],
            repeat_penalty=generation_config["repeat_penalty"],
        )
        print("Response:")
        print(response)
    finally:
        if model is not None:
            model.release_memory()
