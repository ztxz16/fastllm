import copy
import importlib.util
import json
import multiprocessing as mp
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    parser = make_normal_parser("FastLLM Decode benchmark")
    add_server_args(parser)
    parser.set_defaults(
        host="127.0.0.1",
        port=18080,
        api_key="no-key",
        think="false",
        hide_input=True,
        threads=16,
        device="cuda",
        moe_device="numa",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="从 JSON 文件读取测试配置，支持批量执行多组 benchmark",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="同时发起的请求数，用于压测多 batch decode 吞吐",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=16,
        help="重复上下文片段的次数，用于构造请求 prompt",
    )
    parser.add_argument(
        "--prompt-unit",
        type=str,
        default="FastLLM decode benchmark context block. ",
        help="构造 prompt 时重复使用的片段",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="请连续输出数字序列，不要解释。",
        help="追加在上下文后的最终问题",
    )
    parser.add_argument(
        "--prefill-length",
        type=int,
        default=None,
        help="按字符数直接构造 prompt，优先于 prompt_repeat",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="每个请求生成的最大 token 数",
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
        "--request-stagger-ms",
        type=int,
        default=0,
        help="相邻请求的启动间隔（毫秒），默认尽量同时发起",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="跳过预热请求",
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


def build_prompt(prompt_unit, prompt_repeat, question):
    parts = []
    for idx in range(prompt_repeat):
        parts.append(f"[context-{idx:06d}] {prompt_unit}")
    parts.append("\n\n")
    parts.append(question)
    return "".join(parts)


def build_prompt_by_chars(prompt_unit, target_chars, question):
    if target_chars < 0:
        raise ValueError("prefill_length 必须大于等于 0")

    question = question or ""
    if target_chars <= len(question):
        return question[-target_chars:] if target_chars > 0 else question

    available_chars = target_chars - len(question)
    if available_chars <= 0:
        return question

    if not prompt_unit:
        raise ValueError("当使用 prefill_length 时，prompt_unit 不能为空")

    parts = []
    current_chars = 0
    idx = 0
    while current_chars < available_chars:
        chunk = f"[context-{idx:06d}] {prompt_unit}"
        parts.append(chunk)
        current_chars += len(chunk)
        idx += 1
    parts.append(question)
    return "".join(parts)[:target_chars]


def build_prompt_from_args(args):
    if getattr(args, "prefill_length", None) is not None:
        return build_prompt_by_chars(args.prompt_unit, int(args.prefill_length), args.question)
    return build_prompt(args.prompt_unit, args.prompt_repeat, args.question)


def run_stream_request(base_url, model_name, api_key, prompt_text, max_tokens, request_timeout, request_id):
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
    with requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=request_timeout,
        stream=True,
    ) as response:
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
                reasoning = delta.get("reasoning_content")

                if first_token_time is None and (content or reasoning):
                    first_token_time = time.perf_counter()
                if content:
                    output_text.append(content)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]

            if chunk.get("usage"):
                usage = chunk["usage"]

    end_time = time.perf_counter()
    if first_token_time is None:
        raise RuntimeError(f"request-{request_id}: Did not receive any completion token from the server.")
    if usage is None:
        raise RuntimeError(f"request-{request_id}: Did not receive usage information from the streaming response.")

    ttft = first_token_time - start_time
    total_time = end_time - start_time
    decode_time = max(end_time - first_token_time, 1e-9)
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])

    return {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage["total_tokens"]),
        "ttft": ttft,
        "total_time": total_time,
        "decode_time": decode_time,
        "decode_speed": completion_tokens / decode_time,
        "finish_reason": finish_reason,
        "output_preview": "".join(output_text)[:200],
        "start_time": start_time,
        "first_token_time": first_token_time,
        "end_time": end_time,
    }


def warmup(base_url, model_name, api_key, request_timeout, max_tokens):
    prompt_text = "这是一次预热请求，请只回复“ok”。"
    return run_stream_request(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        prompt_text=prompt_text,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        request_id=-1,
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


def load_json_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("配置文件根节点必须是 JSON object")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("配置文件必须包含非空的 cases 数组")
    return config


def apply_overrides(namespace, overrides):
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


def resolve_case_args(base_args, defaults, case_config, case_index):
    args = copy.deepcopy(base_args)
    apply_overrides(args, defaults)
    apply_overrides(args, case_config)

    if not getattr(args, "name", ""):
        args.name = case_config.get("name") or f"case-{case_index}"

    if not args.model and not args.path:
        raise ValueError(f"{args.name}: 必须提供 model 或 path")

    if hasattr(args, "model_path") and getattr(args, "model_path", None):
        args.path = args.model_path

    if hasattr(args, "output_length") and getattr(args, "output_length", None) is not None:
        args.max_tokens = args.output_length

    if getattr(args, "prefill_length", None) is not None:
        args.prefill_length = int(args.prefill_length)

    args.batch_size = int(args.batch_size)
    args.max_tokens = int(args.max_tokens)
    args.request_stagger_ms = int(args.request_stagger_ms)
    if getattr(args, "max_batch", -1) <= 0:
        args.max_batch = args.batch_size
    return args


def summarize_finish_reasons(request_results):
    counts = {}
    for item in request_results:
        key = item["finish_reason"] or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def run_decode_batch(
    base_url,
    model_name,
    api_key,
    prompt_text,
    max_tokens,
    request_timeout,
    batch_size,
    request_stagger_ms,
):
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if request_stagger_ms < 0:
        raise ValueError("request_stagger_ms 不能小于 0")

    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for request_id in range(batch_size):
            futures.append(
                executor.submit(
                    run_stream_request,
                    base_url,
                    model_name,
                    api_key,
                    prompt_text,
                    max_tokens,
                    request_timeout,
                    request_id,
                )
            )
            if request_stagger_ms > 0 and request_id + 1 < batch_size:
                time.sleep(request_stagger_ms / 1000.0)

        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["request_id"])

    start_times = [item["start_time"] for item in results]
    first_token_times = [item["first_token_time"] for item in results]
    end_times = [item["end_time"] for item in results]
    ttfts = [item["ttft"] for item in results]
    decode_speeds = [item["decode_speed"] for item in results]
    completion_tokens = [item["completion_tokens"] for item in results]
    decode_times = [item["decode_time"] for item in results]
    total_times = [item["total_time"] for item in results]

    batch_start = min(start_times)
    batch_first_token = min(first_token_times)
    batch_end = max(end_times)
    batch_wall_time = batch_end - batch_start
    batch_decode_time = max(batch_end - batch_first_token, 1e-9)

    request_results = []
    for item in results:
        request_results.append(
            {
                "request_id": item["request_id"],
                "prompt_tokens": item["prompt_tokens"],
                "completion_tokens": item["completion_tokens"],
                "total_tokens": item["total_tokens"],
                "ttft": item["ttft"],
                "total_time": item["total_time"],
                "decode_time": item["decode_time"],
                "decode_speed": item["decode_speed"],
                "finish_reason": item["finish_reason"],
                "output_preview": item["output_preview"],
            }
        )

    total_prompt_tokens = sum(item["prompt_tokens"] for item in results)
    total_completion_tokens = sum(completion_tokens)
    total_tokens = sum(item["total_tokens"] for item in results)

    return {
        "requests": batch_size,
        "request_stagger_ms": request_stagger_ms,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "batch_wall_time": batch_wall_time,
        "batch_decode_time": batch_decode_time,
        "batch_decode_speed": total_completion_tokens / batch_decode_time,
        "end_to_end_speed": total_completion_tokens / max(batch_wall_time, 1e-9),
        "ttft_min": min(ttfts),
        "ttft_avg": statistics.mean(ttfts),
        "ttft_max": max(ttfts),
        "per_request_completion_tokens_avg": statistics.mean(completion_tokens),
        "per_request_total_time_avg": statistics.mean(total_times),
        "per_request_decode_time_avg": statistics.mean(decode_times),
        "per_request_decode_speed_min": min(decode_speeds),
        "per_request_decode_speed_avg": statistics.mean(decode_speeds),
        "per_request_decode_speed_max": max(decode_speeds),
        "launch_span": max(start_times) - min(start_times),
        "finish_reason_counts": summarize_finish_reasons(request_results),
        "request_results": request_results,
    }


def run_single_benchmark(args):
    if args.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    server_model_name = args.model_name or args.path or args.model
    args.model_name = server_model_name
    if getattr(args, "max_batch", -1) <= 0:
        args.max_batch = args.batch_size

    base_url = f"http://{args.host}:{args.port}".rstrip("/")

    start_method = "fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method()
    context = mp.get_context(start_method)
    error_queue = context.Queue()
    server_process = context.Process(target=server_entry, args=(args, error_queue), daemon=True)

    prompt_text = build_prompt_from_args(args)

    try:
        print("FastLLM Decode benchmark")
        print(f"case_name: {getattr(args, 'name', 'single')}")
        print(f"repo: {REPO_ROOT}")
        print(f"base_url: {base_url}")
        print(f"model_name: {server_model_name}")
        print(f"model_path: {args.path or args.model}")
        print(f"batch_size: {args.batch_size}")
        print(f"server_max_batch: {args.max_batch}")
        print(f"request_stagger_ms: {args.request_stagger_ms}")
        if args.max_batch > 0 and args.max_batch < args.batch_size:
            print(
                "warning: "
                f"server_max_batch({args.max_batch}) < batch_size({args.batch_size}), "
                "结果会包含服务端排队时间。"
            )
        if getattr(args, "prefill_length", None) is not None:
            print(f"prefill_length_chars: {args.prefill_length}")
        else:
            print(f"prompt_repeat: {args.prompt_repeat}")
        print(f"prompt_chars: {len(prompt_text)}")
        print(f"max_tokens_per_request: {args.max_tokens}")
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
                f"completion_tokens={warmup_result['completion_tokens']}, "
                f"decode_speed={warmup_result['decode_speed']:.2f} tok/s"
            )

        result = run_decode_batch(
            base_url=base_url,
            model_name=server_model_name,
            api_key=args.api_key,
            prompt_text=prompt_text,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            batch_size=args.batch_size,
            request_stagger_ms=args.request_stagger_ms,
        )
        result.update(
            {
                "case_name": getattr(args, "name", "single"),
                "model_name": server_model_name,
                "model_path": args.path or args.model,
                "prompt_chars": len(prompt_text),
                "batch_size": args.batch_size,
                "server_max_batch": args.max_batch,
                "max_tokens": args.max_tokens,
                "queueing_expected": args.max_batch > 0 and args.max_batch < args.batch_size,
            }
        )
        if getattr(args, "prefill_length", None) is not None:
            result["prefill_length_chars"] = args.prefill_length
        else:
            result["prompt_repeat"] = args.prompt_repeat

        print("=" * 60)
        print("benchmark_result")
        print(f"total_prompt_tokens: {result['total_prompt_tokens']}")
        print(f"total_completion_tokens: {result['total_completion_tokens']}")
        print(f"batch_wall_seconds: {result['batch_wall_time']:.6f}")
        print(f"batch_decode_seconds: {result['batch_decode_time']:.6f}")
        print(f"launch_span_seconds: {result['launch_span']:.6f}")
        print(f"batch_decode_tokens_per_second: {result['batch_decode_speed']:.2f}")
        print(f"end_to_end_tokens_per_second: {result['end_to_end_speed']:.2f}")
        print(f"ttft_min_seconds: {result['ttft_min']:.6f}")
        print(f"ttft_avg_seconds: {result['ttft_avg']:.6f}")
        print(f"ttft_max_seconds: {result['ttft_max']:.6f}")
        print(
            "per_request_decode_tokens_per_second: "
            f"avg={result['per_request_decode_speed_avg']:.2f}, "
            f"min={result['per_request_decode_speed_min']:.2f}, "
            f"max={result['per_request_decode_speed_max']:.2f}"
        )
        print(f"finish_reason_counts: {json.dumps(result['finish_reason_counts'], ensure_ascii=False)}")
        if result["request_results"]:
            print(f"sample_output_preview: {result['request_results'][0]['output_preview']}")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        stop_server(server_process)


def run_batch_benchmark(base_args):
    config = load_json_config(base_args.config)
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults 必须是 JSON object")

    summary = []
    total_cases = len(config["cases"])
    for index, case_config in enumerate(config["cases"], start=1):
        if not isinstance(case_config, dict):
            raise ValueError(f"cases[{index - 1}] 必须是 JSON object")

        case_args = resolve_case_args(base_args, defaults, case_config, index)
        print(f"\n[{index}/{total_cases}] start_case: {case_args.name}")
        result = run_single_benchmark(case_args)
        summary.append(result)

    print("\n" + "=" * 60)
    print("benchmark_summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        run_batch_benchmark(args)
        return

    if not args.model and not args.path:
        parser.error("必须提供模型路径或名称，例如 `python test/benchmark/decode.py Qwen/Qwen3-0.6B`")

    if getattr(args, "max_batch", -1) <= 0:
        args.max_batch = args.batch_size

    args.name = "single"
    run_single_benchmark(args)


if __name__ == "__main__":
    main()
