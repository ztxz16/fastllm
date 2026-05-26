import argparse
import ctypes
import statistics
import time
from typing import Dict, List, Optional

from .util import make_normal_llm_model, make_normal_parser


def add_benchmark_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input_tokens", type=int, default=64,
                        help="Input token length for benchmark prompts")
    parser.add_argument("--output_tokens", type=int, default=256,
                        help="Max output token length for each benchmark request")
    parser.add_argument("--batch", type=int, default=1,
                        help="Number of concurrent benchmark requests")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup requests before benchmark")
    parser.add_argument("--prompt_unit", type=str,
                        default="FastLLM benchmark context block. ",
                        help="Text unit repeated to build the synthetic prompt")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Generation temperature; set <= 0 for greedy decoding")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Generation top_p")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Generation top_k")
    parser.add_argument("--repeat_penalty", "--repetition_penalty",
                        dest="repeat_penalty", type=float, default=None,
                        help="Generation repetition penalty")


def args_parser():
    parser = make_normal_parser("fastllm_benchmark")
    add_benchmark_args(parser)
    return parser.parse_args()


def _validate_args(args):
    if args.input_tokens <= 0:
        raise ValueError("--input_tokens must be greater than 0")
    if args.output_tokens <= 0:
        raise ValueError("--output_tokens must be greater than 0")
    if args.batch <= 0:
        raise ValueError("--batch must be greater than 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be greater than or equal to 0")
    if args.prompt_unit == "":
        raise ValueError("--prompt_unit must not be empty")


def _encode_prompt(model, prompt: str) -> List[int]:
    if getattr(model, "hf_tokenizer", None) is not None:
        from .llm import encode_hf_prompt

        return encode_hf_prompt(model.hf_tokenizer, prompt)
    return model.encode(prompt)


def _build_input_tokens(model, target_tokens: int, prompt_unit: str) -> List[int]:
    repeat = 1
    token_ids = _encode_prompt(model, prompt_unit)
    if len(token_ids) == 0:
        raise ValueError("prompt_unit produced empty tokens")
    while len(token_ids) < target_tokens:
        repeat = max(repeat + 1,
                     int(repeat * target_tokens / max(len(token_ids), 1)) + 1)
        token_ids = _encode_prompt(model, prompt_unit * repeat)
    return token_ids[:target_tokens]


def _generation_args(model, args) -> Dict[str, object]:
    default_config = getattr(model, "default_generation_config", {})
    top_p = args.top_p if args.top_p is not None else default_config.get("top_p", 0.8)
    top_k = args.top_k if args.top_k is not None else default_config.get("top_k", 1)
    temperature = (args.temperature if args.temperature is not None
                   else default_config.get("temperature", 1.0))
    repeat_penalty = (args.repeat_penalty if args.repeat_penalty is not None
                      else default_config.get("repetition_penalty", 1.0))

    do_sample = True
    if temperature is not None and temperature <= 0:
        do_sample = False
        temperature = 1.0
        top_p = 1.0
        top_k = 1

    return {
        "do_sample": do_sample,
        "top_p": float(top_p),
        "top_k": int(top_k),
        "temperature": float(temperature),
        "repeat_penalty": float(repeat_penalty),
    }


def _launch_raw_response(model, input_tokens: List[int], output_tokens: int,
                         generation_args: Dict[str, object]) -> int:
    from .llm import fastllm_lib

    stop_token_len, stop_token_list = model.stop_token_ctypes(None)
    input_buffer = (ctypes.c_int * len(input_tokens))(*input_tokens)
    return fastllm_lib.launch_response_llm_model(
        model.model,
        len(input_tokens),
        input_buffer,
        ctypes.c_int(output_tokens),
        ctypes.c_int(0),
        ctypes.c_bool(generation_args["do_sample"]),
        ctypes.c_float(generation_args["top_p"]),
        ctypes.c_int(generation_args["top_k"]),
        ctypes.c_float(generation_args["temperature"]),
        ctypes.c_float(generation_args["repeat_penalty"]),
        ctypes.c_bool(False),
        stop_token_len,
        stop_token_list,
    )


def _run_batch(model, input_tokens: List[int], output_tokens: int,
               batch: int, generation_args: Dict[str, object],
               label: str = "benchmark") -> Dict[str, object]:
    from .llm import fastllm_lib

    requests = []
    batch_start = time.perf_counter()
    for request_id in range(batch):
        start_time = time.perf_counter()
        handle = _launch_raw_response(model, input_tokens, output_tokens, generation_args)
        requests.append({
            "request_id": request_id,
            "handle": handle,
            "start_time": start_time,
            "first_token_time": None,
            "end_time": None,
            "output_tokens": 0,
            "finish_code": None,
        })

    pending = set(range(batch))
    while pending:
        progressed = False
        for request_index in list(pending):
            item = requests[request_index]
            if not fastllm_lib.can_fetch_response_llm_model(model.model, item["handle"]):
                continue
            token = fastllm_lib.fetch_response_llm_model(model.model, item["handle"])
            now = time.perf_counter()
            progressed = True
            if token <= -1:
                item["end_time"] = now
                item["finish_code"] = token
                pending.remove(request_index)
                continue
            if item["first_token_time"] is None:
                item["first_token_time"] = now
            item["output_tokens"] += 1
        if not progressed:
            time.sleep(0.0005)

    batch_end = max(item["end_time"] for item in requests)
    total_output_tokens = sum(item["output_tokens"] for item in requests)
    ttfts = [
        item["first_token_time"] - item["start_time"]
        for item in requests
        if item["first_token_time"] is not None
    ]
    tpops = [
        (item["end_time"] - item["first_token_time"]) / (item["output_tokens"] - 1)
        for item in requests
        if item["first_token_time"] is not None and item["output_tokens"] > 1
    ]
    request_speeds = [
        item["output_tokens"] / max(item["end_time"] - item["start_time"], 1e-9)
        for item in requests
    ]
    decode_tokens = sum(max(item["output_tokens"] - 1, 0) for item in requests)
    first_token_times = [
        item["first_token_time"] for item in requests
        if item["first_token_time"] is not None
    ]
    batch_decode_span = (
        max(item["end_time"] for item in requests) - min(first_token_times)
        if first_token_times else 0.0
    )
    return {
        "label": label,
        "input_tokens": len(input_tokens),
        "target_output_tokens": output_tokens,
        "batch": batch,
        "requests": requests,
        "total_output_tokens": total_output_tokens,
        "total_time": batch_end - batch_start,
        "ttft_avg": statistics.mean(ttfts) if ttfts else None,
        "ttft_min": min(ttfts) if ttfts else None,
        "ttft_max": max(ttfts) if ttfts else None,
        "tpop_avg": statistics.mean(tpops) if tpops else None,
        "tpop_min": min(tpops) if tpops else None,
        "tpop_max": max(tpops) if tpops else None,
        "per_request_tokens_per_second_avg": (
            statistics.mean(request_speeds) if request_speeds else 0.0
        ),
        "prefill_tokens_per_second": (
            len(input_tokens) / max(ttfts[0], 1e-9) if batch == 1 and ttfts else None
        ),
        "batch_tokens_per_second": total_output_tokens / max(batch_end - batch_start, 1e-9),
        "batch_decode_tokens_per_second": (
            decode_tokens / max(batch_decode_span, 1e-9) if batch_decode_span > 0 else 0.0
        ),
    }


def _format_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1000:.2f} ms"


def _format_ms_per_token(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1000:.2f} ms/token"


def _format_tokens_per_second(value: float) -> str:
    return f"{value:.2f} tokens/s"


def _format_generation_args(generation_args: Dict[str, object]) -> str:
    return (
        f"sample={str(generation_args['do_sample']).lower()}, "
        f"top_p={generation_args['top_p']}, "
        f"top_k={generation_args['top_k']}, "
        f"temperature={generation_args['temperature']}, "
        f"repeat_penalty={generation_args['repeat_penalty']}"
    )


def _print_kv(label: str, value):
    print(f"  {label:<30} {value}")


def _finish_code_message(finish_code: int) -> str:
    if finish_code == -2:
        return "prompt too long"
    return f"generation failed with finish code {finish_code}"


def _print_header(title: str):
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_start(args, generation_args: Dict[str, object], input_tokens: List[int]):
    _print_header("FastLLM Benchmark")
    print("Config")
    _print_kv("Model", args.path or args.model)
    _print_kv("Input tokens", len(input_tokens))
    _print_kv("Output tokens", args.output_tokens)
    _print_kv("Batch", args.batch)
    _print_kv("Max batch", args.max_batch)
    _print_kv("Generation", _format_generation_args(generation_args))
    print()


def _print_result(result: Dict[str, object]):
    _print_header("FastLLM Benchmark Result")
    print("Summary")
    _print_kv("Input tokens", result["input_tokens"])
    _print_kv("Target output tokens", result["target_output_tokens"])
    _print_kv("Batch", result["batch"])
    _print_kv("Actual output tokens", result["total_output_tokens"])
    _print_kv("Total time", f"{result['total_time']:.4f} s")

    print()
    print("Latency")
    _print_kv("TTFT avg", _format_ms(result["ttft_avg"]))
    _print_kv("TTFT min", _format_ms(result["ttft_min"]))
    _print_kv("TTFT max", _format_ms(result["ttft_max"]))
    _print_kv("TPOP avg", _format_ms_per_token(result["tpop_avg"]))
    _print_kv("TPOP min", _format_ms_per_token(result["tpop_min"]))
    _print_kv("TPOP max", _format_ms_per_token(result["tpop_max"]))

    print()
    print("Throughput")
    if result["batch"] == 1 and result["prefill_tokens_per_second"] is not None:
        _print_kv("Prefill", _format_tokens_per_second(result["prefill_tokens_per_second"]))
    _print_kv("Batch total", _format_tokens_per_second(result["batch_tokens_per_second"]))
    _print_kv("Batch decode after TTFT",
              _format_tokens_per_second(result["batch_decode_tokens_per_second"]))
    _print_kv("Per request avg",
              _format_tokens_per_second(result["per_request_tokens_per_second_avg"]))

    early_finished = [
        item for item in result["requests"]
        if item["output_tokens"] < result["target_output_tokens"]
    ]
    errors = [
        item for item in result["requests"]
        if item["finish_code"] is not None and item["finish_code"] != -1
    ]

    if errors:
        print()
        print("Errors")
        for item in errors[:10]:
            _print_kv(f"Request #{item['request_id']}",
                      _finish_code_message(item["finish_code"]))
        if len(errors) > 10:
            _print_kv("More errors", len(errors) - 10)
    elif early_finished:
        print()
        print("Warnings")
        _print_kv("Early finished requests",
                  f"{len(early_finished)} / {result['batch']}")
    print("=" * 72)


def fastllm_benchmark(args):
    _validate_args(args)
    if getattr(args, "max_batch", -1) <= 0:
        args.max_batch = args.batch

    model = make_normal_llm_model(args)
    try:
        generation_args = _generation_args(model, args)
        input_tokens = _build_input_tokens(model, args.input_tokens, args.prompt_unit)

        _print_start(args, generation_args, input_tokens)

        for warmup_index in range(args.warmup):
            print(f"Warmup {warmup_index + 1}/{args.warmup} ...")
            _run_batch(model, input_tokens, min(args.output_tokens, 8), 1,
                       generation_args, label="warmup")
        if args.warmup > 0:
            print()

        result = _run_batch(model, input_tokens, args.output_tokens, args.batch,
                            generation_args)
        _print_result(result)
        return result
    finally:
        model.release_memory()


if __name__ == "__main__":
    fastllm_benchmark(args_parser())
