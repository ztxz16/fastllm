import argparse
import json
import sys
import time
import requests


def test_messages(base_url: str, model: str, api_key: str) -> bool:
    """测试非流式 messages 接口"""
    url = f"{base_url}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": 128,
        "messages": [
            {"role": "user", "content": "请用一句话介绍你自己。"}
        ],
    }

    print("=" * 60)
    print("[1] 测试非流式 Messages 接口")
    print(f"    POST {url}")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
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
    if data.get("type") != "message":
        print(f"    ✗ 预期 type='message', 实际 type='{data.get('type')}'")
        print(f"    响应: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
        return False

    content_blocks = data.get("content", [])
    if not content_blocks:
        print("    ✗ 响应中没有 content 块")
        return False

    text = "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )
    if not text.strip():
        print("    ✗ 响应文本为空")
        return False

    print(f"    ✓ 模型: {data.get('model', 'N/A')}")
    print(f"    ✓ stop_reason: {data.get('stop_reason', 'N/A')}")
    print(f"    ✓ 回复内容: {text[:200]}")
    usage = data.get("usage", {})
    if usage:
        print(f"    ✓ token 用量: input={usage.get('input_tokens', '?')}, output={usage.get('output_tokens', '?')}")
    return True


def test_messages_stream(base_url: str, model: str, api_key: str) -> bool:
    """测试流式 messages 接口"""
    url = f"{base_url}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": 128,
        "stream": True,
        "messages": [
            {"role": "user", "content": "从1数到5，每个数字一行。"}
        ],
    }

    print("=" * 60)
    print("[2] 测试流式 Messages 接口")
    print(f"    POST {url} (stream=true)")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60, stream=True)
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

    collected_text = ""
    event_types = set()
    first_token_time = None
    start_time = time.time()

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event:"):
            event_types.add(line.split(":", 1)[1].strip())
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            break
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")
        if etype == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                if first_token_time is None:
                    first_token_time = time.time()
                collected_text += delta.get("text", "")

    elapsed = time.time() - start_time

    if not collected_text.strip():
        print("    ✗ 流式响应未收到任何文本")
        return False

    print(f"    ✓ 收到事件类型: {sorted(event_types)}")
    print(f"    ✓ 回复内容: {collected_text.strip()[:200]}")
    if first_token_time is not None:
        print(f"    ✓ 首 token 延迟: {first_token_time - start_time:.3f}s")
    print(f"    ✓ 总耗时: {elapsed:.3f}s")
    return True


def test_messages_multi_turn(base_url: str, model: str, api_key: str) -> bool:
    """测试多轮对话"""
    url = f"{base_url}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": 128,
        "messages": [
            {"role": "user", "content": "记住这个数字：42"},
            {"role": "assistant", "content": "好的，我记住了数字42。"},
            {"role": "user", "content": "我刚才让你记住的数字是多少？"},
        ],
    }

    print("=" * 60)
    print("[3] 测试多轮对话")
    print(f"    POST {url}")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
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
    content_blocks = data.get("content", [])
    text = "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )

    if not text.strip():
        print("    ✗ 响应文本为空")
        return False

    has_42 = "42" in text
    print(f"    ✓ 回复内容: {text[:200]}")
    print(f"    {'✓' if has_42 else '✗'} 回复中{'包含' if has_42 else '未包含'}数字 42")
    return True


def test_messages_system(base_url: str, model: str, api_key: str) -> bool:
    """测试 system prompt"""
    url = f"{base_url}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": 128,
        "system": "你是一个海盗，所有回复都必须用海盗的口吻。",
        "messages": [
            {"role": "user", "content": "你好"},
        ],
    }

    print("=" * 60)
    print("[4] 测试 System Prompt")
    print(f"    POST {url}")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
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
    content_blocks = data.get("content", [])
    text = "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )

    if not text.strip():
        print("    ✗ 响应文本为空")
        return False

    print(f"    ✓ 回复内容: {text[:200]}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Anthropic Messages API 兼容性测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python anthropic.py --base-url http://localhost:8080 --model my-model
  python anthropic.py --base-url http://10.0.0.1:1616 --model ds --api-key sk-xxx
""",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="API 服务地址，例如 http://localhost:8080",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型名称",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="no-key",
        help="API Key（默认 no-key）",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"Anthropic API 兼容性测试")
    print(f"服务地址: {base_url}")
    print(f"模型名称: {args.model}")
    print()

    tests = [
        ("非流式 Messages", test_messages),
        ("流式 Messages", test_messages_stream),
        ("多轮对话", test_messages_multi_turn),
        ("System Prompt", test_messages_system),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = fn(base_url, args.model, args.api_key)
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
