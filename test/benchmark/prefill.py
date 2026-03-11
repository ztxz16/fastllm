import argparse
import importlib.util
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTOOLS_DIR = REPO_ROOT / "tools" / "fastllm_pytools"


def sanitize_sys_path():
    # Avoid importing sibling helper scripts such as test/api/openai.py
    # when the server expects the external openai package.
    script_dir = str(Path(__file__).resolve().parent)
    while script_dir in sys.path:
        sys.path.remove(script_dir)
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


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


sanitize_sys_path()


def build_parser():
    bootstrap_local_ftllm()
    from ftllm.util import add_server_args, make_normal_parser

    parser = make_normal_parser("FastLLM Prefill benchmark")
    add_server_args(parser)
    parser.set_defaults(host="127.0.0.1", port=18080, api_key="no-key", think="false", hide_input=True)

    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=2048,
        help="重复上下文片段的次数，用于构造长上下文",
    )
    parser.add_argument(
        "--prompt-unit",
        type=str,
        default="FastLLM prefill benchmark context block. ",
        help="构造长上下文时重复使用的片段",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="请阅读以上上下文，并只回复“测试完成”。",
        help="追加在长上下文后的最终问题",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="生成的最大 token 数",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=600,
        help="等待 API server 启动的超时时间（秒）",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=3600,
        help="单次测试请求超时时间（秒）",
    )
    parser.add_argument(
        "--warmup-max-tokens",
        type=int,
        default=8,
        help="预热请求生成的最大 token 数",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="跳过短请求预热",
    )
    return parser


def server_entry(args, error_queue):
    try:
        bootstrap_local_ftllm()
        from ftllm.server import fastllm_server

        fastllm_server(args)
    except BaseException:
        error_queue.put(traceback.format_exc())
        raise


def wait_for_server(base_url, api_key, process, error_queue, timeout):
    deadline = time.time() + timeout
    headers = {"Authorization": f"Bearer {api_key}"}
    last_error = ""

    while time.time() < deadline:
        if not process.is_alive():
            server_error = ""
            if not error_queue.empty():
                server_error = error_queue.get()
            raise RuntimeError("API server exited before becoming ready.\n" + server_error)

        try:
            response = requests.get(f"{base_url}/v1/models", headers=headers, timeout=2)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)

        time.sleep(1)

    raise TimeoutError(f"API server startup timed out after {timeout}s. last_error={last_error}")


def build_long_context(prompt_unit, prompt_repeat, question):
    parts = []
    for idx in range(prompt_repeat):
        parts.append(f"[context-{idx:06d}] {prompt_unit}")
    parts.append("\n\n")
    parts.append(question)
    return "".join(parts)


def run_chat_completion(base_url, model_name, api_key, prompt_text, max_tokens, request_timeout):
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [
            {"role": "user", "content": prompt_text},
        ],
    }

    start_time = time.perf_counter()
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=request_timeout,
        stream=True,
    )
    response.raise_for_status()

    first_token_time = None
    finish_reason = None
    usage = None
    output_text = []

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            break

        chunk = json.loads(data_str)
        choices = chunk.get("choices", [])
        if choices:
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                output_text.append(content)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

        if chunk.get("usage"):
            usage = chunk["usage"]

    end_time = time.perf_counter()
    if first_token_time is None:
        raise RuntimeError("Did not receive any completion token from the server.")
    if usage is None:
        raise RuntimeError("Did not receive usage information from the streaming response.")

    ttft = first_token_time - start_time
    total_time = end_time - start_time
    decode_time = max(total_time - ttft, 1e-9)
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage["total_tokens"]),
        "ttft": ttft,
        "total_time": total_time,
        "decode_time": decode_time,
        "prefill_speed": prompt_tokens / max(ttft, 1e-9),
        "decode_speed": completion_tokens / decode_time,
        "finish_reason": finish_reason,
        "output_preview": "".join(output_text)[:200],
    }


def warmup(base_url, model_name, api_key, request_timeout, max_tokens):
    prompt_text = "这是一次预热请求，请只回复“ok”。"
    return run_chat_completion(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        prompt_text=prompt_text,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )


def stop_server(process):
    if process is None:
        return
    if process.is_alive():
        process.terminate()
        process.join(timeout=30)
    if process.is_alive():
        process.kill()
        process.join(timeout=10)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.model and not args.path:
        parser.error("必须提供模型路径或名称，例如 `python test/benchmark/prefill.py Qwen/Qwen3-0.6B`")

    server_model_name = args.model_name or args.path or args.model
    args.model_name = server_model_name
    base_url = f"http://{args.host}:{args.port}".rstrip("/")

    start_method = "fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method()
    context = mp.get_context(start_method)
    error_queue = context.Queue()
    server_process = context.Process(target=server_entry, args=(args, error_queue), daemon=True)

    long_prompt = build_long_context(args.prompt_unit, args.prompt_repeat, args.question)

    try:
        print("FastLLM Prefill benchmark")
        print(f"repo: {REPO_ROOT}")
        print(f"base_url: {base_url}")
        print(f"model_name: {server_model_name}")
        print(f"prompt_repeat: {args.prompt_repeat}")
        print(f"prompt_chars: {len(long_prompt)}")
        print(f"max_tokens: {args.max_tokens}")
        print("=" * 60)

        server_process.start()
        wait_for_server(base_url, args.api_key, server_process, error_queue, args.startup_timeout)
        print("server_ready: true")

        if not args.skip_warmup:
            warmup_result = warmup(
                base_url=base_url,
                model_name=server_model_name,
                api_key=args.api_key,
                request_timeout=args.request_timeout,
                max_tokens=args.warmup_max_tokens,
            )
            print(
                "warmup: "
                f"ttft={warmup_result['ttft']:.3f}s, "
                f"prompt_tokens={warmup_result['prompt_tokens']}, "
                f"completion_tokens={warmup_result['completion_tokens']}"
            )

        result = run_chat_completion(
            base_url=base_url,
            model_name=server_model_name,
            api_key=args.api_key,
            prompt_text=long_prompt,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
        )

        print("=" * 60)
        print("benchmark_result")
        print(f"prompt_tokens: {result['prompt_tokens']}")
        print(f"completion_tokens: {result['completion_tokens']}")
        print(f"finish_reason: {result['finish_reason']}")
        print(f"ttft_seconds: {result['ttft']:.6f}")
        print(f"total_seconds: {result['total_time']:.6f}")
        print(f"decode_seconds: {result['decode_time']:.6f}")
        print(f"prefill_tokens_per_second: {result['prefill_speed']:.2f}")
        print(f"decode_tokens_per_second: {result['decode_speed']:.2f}")
        print(f"output_preview: {result['output_preview']}")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        stop_server(server_process)


if __name__ == "__main__":
    main()
