import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

import requests


def make_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def extract_response_text(data: dict) -> str:
    texts = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                texts.append(part.get("text", ""))
    return "".join(texts)


def responses_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def test_responses(base_url: str, model: str, api_key: str,
                   session: requests.Session) -> bool:
    url = f"{base_url}/v1/responses"
    payload = {
        "model": model,
        "instructions": "你是一个简洁的助手。",
        "max_output_tokens": 128,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "请用一句话介绍你自己。"},
                ],
            },
        ],
        "stream": False,
    }

    print("=" * 60)
    print("[1] 测试非流式 Responses 接口")
    print(f"    POST {url}")
    try:
        resp = session.post(url, headers=responses_headers(api_key),
                            json=payload, timeout=120)
    except requests.exceptions.ConnectionError as e:
        print(f"    ✗ 连接失败: {e}")
        return False
    except requests.exceptions.Timeout:
        print("    ✗ 请求超时")
        return False

    if resp.status_code != 200:
        print(f"    ✗ HTTP {resp.status_code}")
        print(f"    响应: {resp.text[:500]}")
        return False

    data = resp.json()
    if data.get("object") != "response":
        print(f"    ✗ 预期 object='response', 实际 object='{data.get('object')}'")
        print(f"    响应: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
        return False
    if data.get("status") not in ("completed", "incomplete"):
        print(f"    ✗ 非预期 status: {data.get('status')}")
        return False

    text = extract_response_text(data)
    if not text.strip():
        print("    ✗ 响应文本为空")
        return False

    print(f"    ✓ id: {data.get('id', 'N/A')}")
    print(f"    ✓ status: {data.get('status', 'N/A')}")
    print(f"    ✓ 回复内容: {text[:200]}")
    usage = data.get("usage", {})
    if usage:
        print(f"    ✓ token 用量: input={usage.get('input_tokens', '?')}, "
              f"output={usage.get('output_tokens', '?')}, "
              f"total={usage.get('total_tokens', '?')}")
    return True


def test_responses_stream(base_url: str, model: str, api_key: str,
                          session: requests.Session) -> bool:
    url = f"{base_url}/v1/responses"
    payload = {
        "model": model,
        "max_output_tokens": 96,
        "stream": True,
        "input": "从1数到3，每个数字一行。",
    }

    print("=" * 60)
    print("[2] 测试流式 Responses 接口")
    print(f"    POST {url} (stream=true)")
    try:
        resp = session.post(url, headers=responses_headers(api_key),
                            json=payload, timeout=120, stream=True)
    except requests.exceptions.ConnectionError as e:
        print(f"    ✗ 连接失败: {e}")
        return False
    except requests.exceptions.Timeout:
        print("    ✗ 请求超时")
        return False

    if resp.status_code != 200:
        print(f"    ✗ HTTP {resp.status_code}")
        print(f"    响应: {resp.text[:500]}")
        return False

    current_event = ""
    events = []
    text = ""
    saw_content_part = False
    first_token_time = None
    start_time = time.time()

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
            events.append(current_event)
            continue
        if not line.startswith("data:"):
            continue
        try:
            data = json.loads(line[len("data:"):].strip())
        except json.JSONDecodeError:
            continue
        event_type = current_event or data.get("type", "")
        if event_type == "response.content_part.added":
            saw_content_part = True
        if event_type == "response.output_text.delta":
            if not saw_content_part:
                print("    ✗ output_text.delta 出现在 content_part.added 之前")
                return False
            if first_token_time is None:
                first_token_time = time.time()
            text += data.get("delta", "")

    elapsed = time.time() - start_time
    required_events = [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    missing = [event for event in required_events if event not in events]
    if missing:
        print(f"    ✗ 缺少事件: {missing}")
        print(f"    实际事件: {events}")
        return False
    if not text.strip():
        print("    ✗ 流式响应未收到任何文本")
        return False

    print(f"    ✓ 收到事件: {events}")
    print(f"    ✓ 回复内容: {text.strip()[:200]}")
    if first_token_time is not None:
        print(f"    ✓ 首 token 延迟: {first_token_time - start_time:.3f}s")
    print(f"    ✓ 总耗时: {elapsed:.3f}s")
    return True


def provider_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def no_proxy_value(base_url: str) -> str:
    parsed = urlparse(base_url)
    hosts = ["127.0.0.1", "localhost"]
    if parsed.hostname and parsed.hostname not in hosts:
        hosts.append(parsed.hostname)
    return ",".join(hosts)


def test_codex_cli(base_url: str, model: str, api_key: str) -> bool:
    print("=" * 60)
    print("[3] 测试 Codex CLI + Responses 接口（工具调用）")
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        print("    △ 未找到 codex 命令，跳过 Codex CLI 测试")
        return True
    has_gpp = shutil.which("g++") is not None
    expected_text = "FASTLLM_COMPILE_OK" if has_gpp else "FASTLLM_TOOL_LOOP_OK"
    if has_gpp:
        prompt = (
            "创建 main.cpp，内容为一个只打印 FASTLLM_COMPILE_OK 的 C++ 程序；"
            "用 g++ -std=c++17 main.cpp -o main 编译；运行 ./main 验证；"
            "最终只输出程序运行结果。"
        )
    else:
        prompt = (
            "创建 result.txt，内容为 FASTLLM_TOOL_LOOP_OK；运行 cat result.txt 验证；"
            "最终只输出 cat 的结果。"
        )

    with tempfile.TemporaryDirectory(prefix="fastllm-codex-home-") as codex_home, \
            tempfile.TemporaryDirectory(prefix="fastllm-codex-work-") as workdir:
        provider = provider_base_url(base_url)
        catalog_path = os.path.join(codex_home, "models.json")
        catalog_session = make_session()
        try:
            resp = catalog_session.get(f"{provider}/models", timeout=30)
            resp.raise_for_status()
            with open(catalog_path, "w", encoding="utf-8") as f:
                json.dump(resp.json(), f, ensure_ascii=False)
        except Exception as e:
            print(f"    △ 生成 Codex model_catalog_json 失败: {e}")
            catalog_path = None

        provider_config = (
            "{"
            'name="FastLLM",'
            f'base_url="{provider}",'
            'wire_api="responses",'
            f'experimental_bearer_token="{api_key}",'
            "stream_max_retries=0,"
            "request_max_retries=0"
            "}"
        )
        cmd = [
            codex_bin, "exec",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "-C", workdir,
            "-s", "workspace-write",
            "-m", model,
            "-c", 'approval_policy="never"',
            "-c", 'model_provider="fastllm_local"',
            "-c", f"model_providers.fastllm_local={provider_config}",
            "--json",
            prompt,
        ]
        if catalog_path is not None:
            cmd.extend(["-c", f'model_catalog_json="{catalog_path}"'])
        env = os.environ.copy()
        env["CODEX_HOME"] = codex_home
        env["NO_PROXY"] = no_proxy_value(base_url)
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                    "http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)

        print(f"    provider: {provider}")
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=240,
            )
        except subprocess.TimeoutExpired:
            print("    ✗ Codex CLI 请求超时")
            return False

        agent_texts = []
        errors = []
        completed_tools = []
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                item_type = item.get("type")
                if item_type == "agent_message":
                    agent_texts.append(item.get("text", ""))
                elif item_type == "error":
                    errors.append(item.get("message", ""))
                elif item_type in ("command_execution", "exec_command", "function_call"):
                    completed_tools.append(item_type)
            elif event.get("type") == "error":
                errors.append(event.get("message", ""))

        agent_text = "\n".join(agent_texts).strip()
        if proc.returncode != 0:
            print(f"    ✗ Codex CLI 退出码: {proc.returncode}")
            if errors:
                print(f"    错误: {errors[-1][:500]}")
            if proc.stderr:
                print(f"    stderr: {proc.stderr[-500:]}")
            return False
        if errors:
            print(f"    ✗ Codex CLI 返回错误事件: {errors[-1][:500]}")
            return False
        if not agent_text:
            print("    ✗ Codex CLI 未返回 agent_message")
            if errors:
                print(f"    错误: {errors[-1][:500]}")
            return False
        if expected_text not in agent_text:
            print(f"    ✗ Codex 最终回复未包含 {expected_text}")
            print(f"    回复: {agent_text[:500]}")
            return False

        print(f"    ✓ Codex 回复: {agent_text[:200]}")
        if completed_tools:
            print(f"    ✓ 已完成工具调用: {completed_tools[:5]}")
        else:
            print("    △ 未在 JSON 事件中识别到工具调用记录")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Responses API 兼容性测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python responses.py --base-url http://localhost:8080 --model my-model
  python responses.py --base-url http://localhost:8080 --model Qwen3-8B-FP8 --skip-codex
""",
    )
    parser.add_argument("--base-url", type=str, required=True,
                        help="API 服务地址，例如 http://localhost:8080")
    parser.add_argument("--model", type=str, required=True,
                        help="模型名称")
    parser.add_argument("--api-key", type=str, default="no-key",
                        help="API Key（默认 no-key）")
    parser.add_argument("--skip-codex", action="store_true",
                        help="跳过 Codex CLI 兼容性测试")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print("Responses API 兼容性测试")
    print(f"服务地址: {base_url}")
    print(f"模型名称: {args.model}")
    print()

    session = make_session()
    tests = [
        ("非流式 Responses", lambda: test_responses(
            base_url, args.model, args.api_key, session)),
        ("流式 Responses", lambda: test_responses_stream(
            base_url, args.model, args.api_key, session)),
    ]
    if not args.skip_codex:
        tests.append(("Codex CLI", lambda: test_codex_cli(
            base_url, args.model, args.api_key)))

    results = []
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:
            print(f"    ✗ 异常: {e}")
            ok = False
        results.append((name, ok))
        print()

    print("=" * 60)
    print("测试结果汇总:")
    passed = 0
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if ok:
            passed += 1
    print(f"\n通过 {passed}/{len(results)} 项测试")
    print("=" * 60)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
