#!/usr/bin/env python3
"""双卡服务器 HTTP 验收：正式队列/取消路径，对照同参数单次入口输出，记录四种报告。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import http.client
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import threading
import time
import uuid

from run_process_tp import ROOT, capture, save_report
from run_qwen3_prefill import require
from qwen3_process_chat import prepare_request, run_native
from qwen3_process_session import BACKEND, EMPTY, Qwen3ProcessSession
from qwen3_process_server import ChatHTTPServer, ChatService


def call(address, method, path, payload=None, request_id=None):
    connection = http.client.HTTPConnection(*address, timeout=240)
    try:
        headers = {"Content-Type": "application/json"}
        if request_id:
            headers["X-Request-ID"] = request_id
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()



def read_event(response):
    """读取正式 SSE 帧；有界保留每帧，识别意外截断而非接受半个 JSON。"""
    data = []
    while True:
        line = response.readline(262144)
        if not line:
            require(not data, "truncated SSE event")
            return None
        require(line.endswith(b"\n"), "SSE line too long or truncated")
        if line == b"\n" and data:
            text = "\n".join(data)
            return text if text == "[DONE]" else json.loads(text)
        if line.startswith(b"data: "):
            data.append(line[6:].decode("utf-8").rstrip("\n"))


def stream_call(address, payload, request_id, output, on_content=None):
    connection = http.client.HTTPConnection(*address, timeout=240)
    events, response, status = [], None, None
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                           {"Content-Type": "application/json", "X-Request-ID": request_id})
        response = connection.getresponse()
        status = response.status
        if status != 200:
            return status, json.loads(response.read())
        require("text/event-stream" in response.getheader("Content-Type", ""), "missing SSE content type")
        while True:
            event = read_event(response)
            if event is None:
                break
            events.append(event)
            if on_content and isinstance(event, dict) and event.get("choices") and event["choices"][0]["delta"].get("content"):
                callback, on_content = on_content, None
                if callback() == "disconnect":
                    break
        return status, events
    finally:
        if response is not None:
            response.close()
        connection.close()
        (output / f"{request_id}.sse.json").write_text(json.dumps(dict(status=status, events=events), ensure_ascii=False, indent=2))


def verify_stream(output, request_id, result, reference, tokenizer, workers, include_usage):
    status, events = result
    require(status == 200 and events and events[-1] == "[DONE]" and events.count("[DONE]") == 1, "missing/duplicate DONE")
    chunks = events[:-1]
    require(all(isinstance(e, dict) and "error" not in e and e.get("object") == "chat.completion.chunk" for e in chunks),
            "invalid success SSE event")
    require({e["id"] for e in chunks} == {"chatcmpl-" + request_id} and len({e["created"] for e in chunks}) == 1 and
            {e["model"] for e in chunks} == {"qwen3"}, "SSE identity changed")
    choices = [e["choices"][0] for e in chunks if e["choices"]]
    require(choices[0]["delta"].get("role") == "assistant", "missing role chunk")
    require(all(e["index"] == 0 and e["finish_reason"] is None for e in choices[:-1]), "premature stream finish")
    final = audit(output, request_id)["response"]
    text = "".join(e["delta"].get("content", "") for e in choices)
    require(text == final["choices"][0]["message"]["content"], "SSE text differs from full generation")
    require(choices[-1]["delta"] == {} and choices[-1]["finish_reason"] == final["choices"][0]["finish_reason"], "SSE finish mismatch")
    usage = [e["usage"] for e in chunks if not e["choices"]]
    require(usage == ([final["usage"]] if include_usage else []), "SSE usage mismatch")
    return verify_completion(output, request_id, (200, final), reference, tokenizer, workers)


def verify_stream_error(result, code):
    status, events = result
    if status == code:
        require("error" in events, "missing HTTP error")
        return
    require(status == 200 and events and isinstance(events[-1], dict) and events[-1].get("error", {}).get("code") == code,
            f"wrong stream error: {status}, {events}")
    require("[DONE]" not in events and not any(e.get("choices") and e["choices"][0]["finish_reason"] for e in events),
            "failed stream reported success")


def wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition timed out")


def audit(output, request_id):
    for path in output.glob("http-*.json"):
        report = json.loads(path.read_text())
        if report["id"] == request_id:
            return report
    raise AssertionError(f"missing HTTP audit: {request_id}")


def native_report(output, request_id):
    record = audit(output, request_id)
    require(type(record["native_id"]) is int, "request never reached native generation")
    return json.loads((output / "session" / f"request-{record['native_id']:06d}.response.json").read_text())


def verify_completion(output, request_id, result, reference, tokenizer, workers):
    status, body = result
    require(status == 200, f"HTTP {status}: {body}")
    report = native_report(output, request_id)
    actual, expected = report["token_ids"], reference["token_ids"]
    require(report["backend"] == BACKEND and report["workers"] == workers, "worker/backend changed")
    require(report["request"] == reference["request"], "HTTP generation parameters differ from reference")
    require(report["cache_after_generate"] == [EMPTY for _ in workers], "request KV leaked")
    require(len(actual) == len(expected), "output length differs from reference")
    errors = [abs(a-b) for a, b in zip(actual, expected)]
    metrics = dict(max_abs_error=float(max(errors, default=0.0)), mean_abs_error=sum(errors)/max(1, len(errors)),
                   failed_elements=sum(value != 0 for value in errors))
    if metrics["failed_elements"]:
        error = AssertionError(f"token IDs differ: {metrics}")
        error.metrics = metrics
        raise error
    choice = body["choices"][0]
    require(choice["message"] == dict(role="assistant", content=tokenizer.decode(expected, skip_special_tokens=True,
                                                                               clean_up_tokenization_spaces=False)),
            "decoded text mismatch")
    require(choice["finish_reason"] == ("stop" if reference["finish_reason"] == "eos" else "length"), "finish reason mismatch")
    prompt = len(reference["request"]["token_ids"])
    require(body["usage"] == dict(prompt_tokens=prompt, completion_tokens=len(expected), total_tokens=prompt+len(expected)),
            "usage mismatch")
    return dict(**metrics, generated_tokens=len(actual), prompt_tokens=prompt)


def verify_cancelled(output, request_id, workers):
    report = native_report(output, request_id)
    require(report["status"] == "FAIL" and report.get("cancelled") is True and report["reusable"] is True,
            "request did not acknowledge native cancellation")
    require(report["cache_after_error"] == [EMPTY for _ in workers] and report["workers"] == workers,
            "cancelled request leaked KV or changed workers")
    return dict(cancelled_after_tokens=len(report.get("steps", [])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generate", "worker", "model", "build-cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--output", type=Path, default=ROOT / "test/distributed/results")
    args = parser.parse_args()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_http_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(["git", "rev-parse", "HEAD"]),
        git_status=capture(["git", "status", "--short"]), seed=0, warmup=0, iterations=1, repeats=1,
        gpu=capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), devices=args.devices,
        expected_case_count=17, execution="eager", prefix_cache=False, kv_cache="FP16 request-local",
        report_title="Qwen3 HTTP correctness", report_description="通过正式 HTTP/会话入口验证。误差字段为输出 Token ID 的精确一致性；不代表性能。")
    rows = []
    service = server = http_thread = None

    def case(name, action):
        row = dict(case=name, status="FAIL", backend=BACKEND, reason="")
        try:
            row.update(action() or {})
            row["status"] = "PASS"
        except Exception as error:
            row.update(getattr(error, "metrics", {}))
            row["reason"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        save_report(output, rows, metadata)
        # 生命周期失败时停止后续步骤，防止请求仍在执行却误记下一项 PASS。
        if row["status"] != "PASS":
            raise RuntimeError(f"{name}: {row['reason']}")

    try:
        metadata["build_cache"] = args.build_cache.read_text()
        metadata["compile_architecture"] = [line for line in metadata["build_cache"].splitlines()
                                            if line.startswith(("CUDA_ARCH:", "CMAKE_CUDA_ARCHITECTURES:"))]
        os.environ["USE_TORCH"] = "0"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        payloads = {name: dict(model="qwen3", messages=[dict(role="user", content=prompt)], max_tokens=32, temperature=0, seed=0)
                    for name, prompt in (("A", "请用一句话介绍你自己。"), ("B", "一加一等于几？"))}
        references = {}
        for name, payload in payloads.items():
            directory = output / ("reference-" + name)
            directory.mkdir()
            request = prepare_request(args.model, tokenizer, None, messages=payload["messages"], max_new_tokens=32)
            references[name] = run_native(args.generate.resolve(), args.worker.resolve(), args.model.resolve(), request,
                                          args.devices, directory)
        with Qwen3ProcessSession(args.generate, args.worker, args.model, args.devices, output / "session", timeout=180) as session:
            service = ChatService(session, tokenizer, args.model, "qwen3", output, queue_size=2, request_timeout=120)
            try:
                server = ChatHTTPServer(("127.0.0.1", 0), service)
                address = server.server_address
                http_thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=0.05))
                http_thread.start()
                workers = session.ready["workers"]
                metadata["workers"] = workers
                health = lambda: call(address, "GET", "/health")[1]

                def ready():
                    status, body = call(address, "GET", "/health")
                    require(status == 200 and body["workers"] == workers and body["model_load_count"] == 1, "invalid readiness")
                    require(call(address, "GET", "/v1/models")[1]["data"][0]["id"] == "qwen3", "invalid model endpoint")
                case("health", ready)
                for name in ("A", "B", "A_repeat"):
                    base = name[0]
                    case(name, lambda name=name, base=base: verify_completion(output, name,
                        call(address, "POST", "/v1/chat/completions", payloads[base], name), references[base], tokenizer, workers))

                def rejects():
                    before = len(list((output / "session").glob("request-*.response.json")))
                    for payload, status in ((dict(payloads["A"], stream="true"), 400), (dict(payloads["A"], tools=[]), 400),
                                            (dict(payloads["A"], max_tokens=0), 400), (dict(payloads["A"], model="wrong"), 404)):
                        actual, body = call(address, "POST", "/v1/chat/completions", payload)
                        require(actual == status and "error" in body, "unsupported request was accepted")
                    require(len(list((output / "session").glob("request-*.response.json"))) == before, "rejected payload reached GPU")
                case("reject_unsupported", rejects)

                long_payload = dict(payloads["A"], max_tokens=1024, messages=[dict(role="user", content="请从1开始连续列出整数，一直列到1000，不要省略。")])
                with ThreadPoolExecutor(max_workers=3) as pool:
                    def queue_cancel():
                        before = len(list((output / "session").glob("request-*.json")))
                        active = pool.submit(call, address, "POST", "/v1/chat/completions", long_payload, "active")
                        wait_for(lambda: health()["active_request_id"] == "active")
                        wait_for(lambda: len(list((output / "session").glob("request-*.json"))) > before)
                        queued = pool.submit(call, address, "POST", "/v1/chat/completions", payloads["B"], "queued")
                        wait_for(lambda: health()["queued_request_ids"] == ["queued"])
                        cancelled = pool.submit(call, address, "POST", "/v1/chat/completions", payloads["A"], "cancel-waiter")
                        wait_for(lambda: health()["queued_request_ids"] == ["queued", "cancel-waiter"])
                        require(call(address, "POST", "/v1/chat/completions", payloads["A"], "overflow")[0] == 429, "queue overflow accepted")
                        require(call(address, "DELETE", "/v1/requests/cancel-waiter")[0] == 202, "queued cancellation rejected")
                        require(cancelled.result()[0] == 499 and audit(output, "cancel-waiter")["native_id"] is None,
                                "queued request reached native execution")
                        require(call(address, "DELETE", "/v1/requests/active")[0] == 202, "active cancellation rejected")
                        require(active.result()[0] == 499, "active request did not cancel")
                        metrics = verify_cancelled(output, "active", workers)
                        metrics.update(verify_completion(output, "queued", queued.result(), references["B"], tokenizer, workers))
                        return metrics
                    case("queue_limit_and_cancel", queue_cancel)

                    def disconnect():
                        connection = socket.create_connection(address)
                        try:
                            body = json.dumps(long_payload).encode()
                            before = len(list((output / "session").glob("request-*.json")))
                            connection.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                                f"X-Request-ID: disconnect\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body)
                            wait_for(lambda: health()["active_request_id"] == "disconnect")
                            wait_for(lambda: len(list((output / "session").glob("request-*.json"))) > before)
                            time.sleep(0.1)
                        finally:
                            connection.close()
                        wait_for(lambda: health()["active_request_id"] is None)
                        require(audit(output, "disconnect")["status"] == 499, "disconnect was ignored")
                        return verify_cancelled(output, "disconnect", workers)
                    case("disconnect", disconnect)

                    def timeout():
                        # 只缩短正式服务的请求期限；计算路径、sampling 和取消机制不改变。
                        service.request_timeout = 0.5
                        try:
                            status, body = call(address, "POST", "/v1/chat/completions", long_payload, "timeout")
                            require(status == 504, f"deadline not enforced: {status}, {body}")
                            return verify_cancelled(output, "timeout", workers)
                        finally:
                            service.request_timeout = 120
                    case("deadline", timeout)
                for name, base, include_usage in (("stream_A", "A", True), ("stream_B", "B", False)):
                    payload = dict(payloads[base], stream=True, stream_options=dict(include_usage=include_usage))
                    case(name, lambda name=name, base=base, payload=payload, include_usage=include_usage: verify_stream(
                        output, name, stream_call(address, payload, name, output), references[base], tokenizer, workers, include_usage))

                def stream_cancel():
                    def cancel():
                        require(health()["active_request_id"] == "stream-cancel", "generation finished before streaming content")
                        require(call(address, "DELETE", "/v1/requests/stream-cancel")[0] == 202, "stream cancellation rejected")
                    result = stream_call(address, dict(long_payload, stream=True), "stream-cancel", output, cancel)
                    verify_stream_error(result, 499)
                    return verify_cancelled(output, "stream-cancel", workers)
                case("stream_cancel", stream_cancel)

                def stream_disconnect():
                    stream_call(address, dict(long_payload, stream=True), "stream-disconnect", output, lambda: "disconnect")
                    wait_for(lambda: health()["active_request_id"] is None)
                    require(audit(output, "stream-disconnect")["status"] == 499, "SSE disconnect was ignored")
                    return verify_cancelled(output, "stream-disconnect", workers)
                case("stream_disconnect", stream_disconnect)

                def stream_timeout():
                    service.request_timeout = 0.5
                    try:
                        result = stream_call(address, dict(long_payload, stream=True), "stream-timeout", output)
                        verify_stream_error(result, 504)
                        return verify_cancelled(output, "stream-timeout", workers)
                    finally:
                        service.request_timeout = 120
                case("stream_timeout", stream_timeout)
                case("stream_recovery", lambda: verify_stream(output, "stream-recovery",
                    stream_call(address, dict(payloads["A"], stream=True), "stream-recovery", output),
                    references["A"], tokenizer, workers, False))
                case("recovery", lambda: verify_completion(output, "recovery",
                    call(address, "POST", "/v1/chat/completions", payloads["A"], "recovery"), references["A"], tokenizer, workers))

                def same_workers():
                    body = health()
                    require(body["workers"] == workers and body["model_load_count"] == 1 and body["status"] == "ready" and
                            body["active_request_id"] is None and not body["queued_request_ids"], "model changed or work leaked")
                case("workers_reused", same_workers)
            finally:
                try:
                    service.close()
                finally:
                    if http_thread is not None:
                        server.shutdown()
                        http_thread.join(10)
                    if server is not None:
                        server.server_close()
            def shutdown():
                require(session.state == "closed" and session.shutdown["shutdown_complete"], "shutdown incomplete")
                require(not Path(f"/proc/{session._process.pid}").exists(), "coordinator still running")
                require(all(not Path(f"/proc/{w['pid']}").exists() for w in workers), "worker still running")
            case("shutdown", shutdown)
    except BaseException as error:
        if not rows or rows[-1]["status"] != "FAIL":
            rows.append(dict(case="runner", backend=BACKEND, status="FAIL", reason=f"{type(error).__name__}: {error}"))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        status = save_report(output, rows, metadata)
    print(f"{status}: {output}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
