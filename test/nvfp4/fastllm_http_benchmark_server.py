#!/usr/bin/env python3
"""为模型性能对比提供直接消费Token ID的FastLLM HTTP流式接口。"""

import argparse
import ctypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--atype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batch", type=int, required=True)
    parser.add_argument("--kv-cache-layout", default="auto",
                        choices=("auto", "continuous", "paged"))
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--attention-backend-strict", action="store_true")
    parser.add_argument("--attention-backend-trace", action="store_true")
    parser.add_argument(
        "--enable-prefix-cache", action="store_true",
        help="保存已完成请求的KV Cache，供后续相同前缀请求复用")
    return parser.parse_args()


def write_sse(handler, payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    handler.wfile.write(f"data: {data}\n\n".encode("utf-8"))
    handler.wfile.flush()


def make_handler(model):
    from ftllm import llm

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            sys.stderr.write("[fastllm-http] " + fmt % args + "\n")

        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/v1/models":
                self.send_json(404, {"error": "not found"})
                return
            self.send_json(200, {"data": [{"id": "nvfp4-performance"}]})

        def do_POST(self):
            if self.path != "/v1/completions":
                self.send_json(404, {"error": "not found"})
                return
            try:
                trace_cpu = os.environ.get(
                    "FASTLLM_HTTP_CPU_TRACE", "").lower() in ("1", "true", "on")
                request_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                launch_us = 0
                can_fetch_us = 0
                poll_sleep_us = 0
                statistics_us = 0
                fetch_us = 0
                sse_us = 0
                poll_checks = 0
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                input_tokens = request.get("prompt")
                output_tokens = int(request.get("max_tokens", 0))
                if (not isinstance(input_tokens, list) or
                        not all(isinstance(token, int) for token in input_tokens)):
                    raise ValueError("prompt必须是Token ID数组")
                if output_tokens <= 0:
                    raise ValueError("max_tokens必须大于0")

                token_buffer = (ctypes.c_int * len(input_tokens))(*input_tokens)
                empty_stops = (ctypes.c_int * 0)()
                # min_length与max_length相同，从logits中屏蔽EOS，保证双方都
                # 严格生成指定数量的Token。
                stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                handle = llm.fastllm_lib.launch_response_llm_model(
                    model.model, len(input_tokens), token_buffer,
                    ctypes.c_int(output_tokens), ctypes.c_int(output_tokens),
                    ctypes.c_bool(False), ctypes.c_float(1.0), ctypes.c_int(1),
                    ctypes.c_float(1.0), ctypes.c_float(1.0), ctypes.c_bool(False),
                    ctypes.c_int(0), empty_stops)
                if trace_cpu:
                    launch_us = (time.perf_counter_ns() - stage_begin_ns) // 1000

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                # Prompt回显独立于生成Token发送。即使后端在首Token前结束，
                # 客户端也能区分Prompt不一致与零Token提前结束；空token_ids
                # 不进入TTFT、ITL和输出吞吐计数。
                stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                write_sse(self, {
                    "choices": [{
                        "index": 0,
                        "token_ids": [],
                        "text": "",
                        "prompt_token_ids": input_tokens,
                    }],
                })
                if trace_cpu:
                    sse_us += (time.perf_counter_ns() - stage_begin_ns) // 1000
                generated = 0
                statistics = {}
                while True:
                    stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                    can_fetch = llm.fastllm_lib.can_fetch_response_llm_model(
                        model.model, handle)
                    if trace_cpu:
                        can_fetch_us += (time.perf_counter_ns() - stage_begin_ns) // 1000
                        poll_checks += 1
                    if not can_fetch:
                        stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                        time.sleep(0.0005)
                        if trace_cpu:
                            poll_sleep_us += (time.perf_counter_ns() - stage_begin_ns) // 1000
                        continue
                    # 最后一次fetch会删除ResponseContext，因此必须在fetch前
                    # 保存本请求的Prefix Cache命中统计。
                    stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                    statistics = model.get_response_statistics(handle) or statistics
                    if trace_cpu:
                        statistics_us += (time.perf_counter_ns() - stage_begin_ns) // 1000
                    stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                    token = llm.fastllm_lib.fetch_response_llm_model(
                        model.model, handle)
                    if trace_cpu:
                        fetch_us += (time.perf_counter_ns() - stage_begin_ns) // 1000
                    if token <= -1:
                        if generated == 0:
                            write_sse(self, {
                                "error": {
                                    "message": (
                                        "FastLLM在首Token前结束生成；请检查服务日志中的"
                                        "[fastllm][paged-debug]首个失败阶段"),
                                    "type": "fastllm_backend_error",
                                },
                            })
                        break
                    choice = {"index": 0, "token_ids": [token], "text": ""}
                    stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                    write_sse(self, {"choices": [choice]})
                    if trace_cpu:
                        sse_us += (time.perf_counter_ns() - stage_begin_ns) // 1000
                    generated += 1
                cached = max(0, int(statistics.get("cached_input_tokens", 0)))
                missed = max(
                    0, int(statistics.get(
                        "missed_input_tokens", len(input_tokens) - cached)))
                stage_begin_ns = time.perf_counter_ns() if trace_cpu else 0
                write_sse(self, {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(input_tokens),
                        "completion_tokens": generated,
                        "total_tokens": len(input_tokens) + generated,
                        "prompt_tokens_details": {
                            "cached_tokens": cached,
                            "missed_tokens": missed,
                        },
                    },
                })
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                if trace_cpu:
                    sse_us += (time.perf_counter_ns() - stage_begin_ns) // 1000
                    total_us = (time.perf_counter_ns() - request_begin_ns) // 1000
                    print(
                        "[fastllm][http][cpu] "
                        f"prompt={len(input_tokens)} generated={generated} "
                        f"launch_us={launch_us} poll_checks={poll_checks} "
                        f"can_fetch_us={can_fetch_us} poll_sleep_us={poll_sleep_us} "
                        f"statistics_us={statistics_us} fetch_us={fetch_us} "
                        f"sse_us={sse_us} total_us={total_us}",
                        file=sys.stderr,
                        flush=True,
                    )
                self.close_connection = True
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception as error:
                self.send_json(400, {"error": str(error)})

    return Handler


def main():
    args = parse_args()
    from ftllm import llm

    llm.set_device_map(args.device)
    llm.set_device_map(args.device, True)
    llm.set_kv_cache_layout(args.kv_cache_layout)
    llm.set_attention_backend(args.attention_backend)
    llm.set_attention_backend_strict(args.attention_backend_strict)
    llm.set_attention_backend_trace(args.attention_backend_trace)
    model = llm.model(args.model, dtype=args.dtype)
    model.set_atype(args.atype)
    model.set_max_batch(args.max_batch)
    if args.enable_prefix_cache:
        model.set_save_history(True)
    model.warmup()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model))
    server.daemon_threads = True
    print(
        f"FastLLM benchmark HTTP server ready: http://{args.host}:{args.port}",
        flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        model.release_memory()


if __name__ == "__main__":
    main()
