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


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，例如 北京",
            },
        },
        "required": ["city"],
    },
}


def test_tool_use(base_url: str, model: str, api_key: str) -> bool:
    """测试非流式 tool use"""
    url = f"{base_url}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？"},
        ],
        "tools": [WEATHER_TOOL],
    }

    print("=" * 60)
    print("[5] 测试非流式 Tool Use")
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
    stop_reason = data.get("stop_reason")
    content_blocks = data.get("content", [])

    tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
    text_blocks = [b for b in content_blocks if b.get("type") == "text"]

    if not tool_use_blocks:
        print(f"    ✗ 未返回 tool_use 块（stop_reason={stop_reason}）")
        if text_blocks:
            print(f"    文本内容: {text_blocks[0].get('text', '')[:200]}")
        return False

    print(f"    ✓ stop_reason: {stop_reason}")
    for i, block in enumerate(tool_use_blocks):
        name = block.get("name", "?")
        tool_id = block.get("id", "?")
        input_data = block.get("input", {})
        print(f"    ✓ tool_use[{i}]: id={tool_id}, name={name}, input={json.dumps(input_data, ensure_ascii=False)[:200]}")
        if "city" in input_data:
            print(f"    ✓ 解析参数 city={input_data['city']}")
    if text_blocks:
        print(f"    ✓ 附带文本: {text_blocks[0].get('text', '')[:100]}")

    return True


def test_tool_use_stream(base_url: str, model: str, api_key: str) -> bool:
    """测试流式 tool use"""
    url = f"{base_url}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": 256,
        "stream": True,
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？"},
        ],
        "tools": [WEATHER_TOOL],
    }

    print("=" * 60)
    print("[6] 测试流式 Tool Use")
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

    blocks = {}
    current_index = -1
    collected_text = ""
    stop_reason = None
    start_time = time.time()

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event:"):
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

        if etype == "content_block_start":
            idx = event.get("index", 0)
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                blocks[idx] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input_json": "",
                }
            elif block.get("type") == "text":
                blocks[idx] = {"type": "text", "text": block.get("text", "")}
            current_index = idx

        elif etype == "content_block_delta":
            idx = event.get("index", current_index)
            delta = event.get("delta", {})
            dtype = delta.get("type", "")
            if dtype == "input_json_delta" and idx in blocks and "input_json" in blocks[idx]:
                blocks[idx]["input_json"] += delta.get("partial_json", "")
            elif dtype == "text_delta":
                collected_text += delta.get("text", "")

        elif etype == "message_delta":
            d = event.get("delta", {})
            if d.get("stop_reason"):
                stop_reason = d["stop_reason"]

    elapsed = time.time() - start_time

    tool_blocks = {k: v for k, v in blocks.items() if "input_json" in v}
    if not tool_blocks:
        print(f"    ✗ 流式响应未收到 tool_use 块（stop_reason={stop_reason}）")
        if collected_text:
            print(f"    收到文本: {collected_text[:200]}")
        return False

    print(f"    ✓ stop_reason: {stop_reason}")
    print(f"    ✓ 总耗时: {elapsed:.3f}s")
    for idx in sorted(tool_blocks):
        entry = tool_blocks[idx]
        print(f"    ✓ tool_use[{idx}]: id={entry['id']}, name={entry['name']}, input_json={entry['input_json'][:200]}")
        try:
            parsed = json.loads(entry["input_json"])
            if "city" in parsed:
                print(f"    ✓ 解析参数 city={parsed['city']}")
        except json.JSONDecodeError:
            print(f"    △ input_json 不是合法 JSON: {entry['input_json'][:100]}")
    if collected_text:
        print(f"    ✓ 附带文本: {collected_text[:100]}")

    return True


def test_tool_use_multi_turn(base_url: str, model: str, api_key: str) -> bool:
    """测试 tool use 多轮：发送 tool_result 后模型应给出最终回复"""
    url = f"{base_url}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    # 第一轮：模型应调用 get_weather
    payload_1 = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？"},
        ],
        "tools": [WEATHER_TOOL],
    }

    print("=" * 60)
    print("[7] 测试 Tool Use 多轮对话")
    print(f"    POST {url} (第一轮: 触发 tool use)")
    try:
        resp1 = requests.post(url, headers=headers, json=payload_1, timeout=60)
    except requests.exceptions.ConnectionError as e:
        print(f"    ✗ 连接失败: {e}")
        return False
    except requests.exceptions.Timeout:
        print("    ✗ 请求超时")
        return False

    if resp1.status_code != 200:
        print(f"    ✗ HTTP {resp1.status_code}")
        return False

    data1 = resp1.json()
    content1 = data1.get("content", [])
    tool_use_blocks = [b for b in content1 if b.get("type") == "tool_use"]

    if not tool_use_blocks:
        print("    ✗ 第一轮未返回 tool_use 块")
        text_blocks = [b for b in content1 if b.get("type") == "text"]
        if text_blocks:
            print(f"    内容: {text_blocks[0].get('text', '')[:200]}")
        return False

    tu = tool_use_blocks[0]
    tool_use_id = tu.get("id", "toolu_test")
    tool_name = tu.get("name", "?")
    print(f"    ✓ 第一轮返回 tool_use: name={tool_name}, id={tool_use_id}")

    # 第二轮：将 tool_result 返回给模型
    payload_2 = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？"},
            {"role": "assistant", "content": content1},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(
                            {"city": "北京", "weather": "晴", "temperature": "25°C"},
                            ensure_ascii=False,
                        ),
                    }
                ],
            },
        ],
        "tools": [WEATHER_TOOL],
    }

    print(f"    POST {url} (第二轮: 提交 tool_result)")
    try:
        resp2 = requests.post(url, headers=headers, json=payload_2, timeout=60)
    except requests.exceptions.ConnectionError as e:
        print(f"    ✗ 连接失败: {e}")
        return False
    except requests.exceptions.Timeout:
        print("    ✗ 请求超时")
        return False

    if resp2.status_code != 200:
        print(f"    ✗ HTTP {resp2.status_code}")
        print(f"    响应: {resp2.text[:500]}")
        return False

    data2 = resp2.json()
    content2 = data2.get("content", [])
    text_blocks = [b for b in content2 if b.get("type") == "text"]

    if not text_blocks:
        print("    ✗ 第二轮响应中没有 text 块")
        return False

    text2 = text_blocks[0].get("text", "")
    if not text2.strip():
        print("    ✗ 第二轮响应文本为空")
        return False

    print(f"    ✓ 第二轮回复: {text2[:200]}")
    has_weather_info = any(kw in text2 for kw in ["晴", "25", "天气", "北京"])
    print(f"    {'✓' if has_weather_info else '△'} 回复中{'包含' if has_weather_info else '未包含'}天气相关信息")
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
        ("非流式 Tool Use", test_tool_use),
        ("流式 Tool Use", test_tool_use_stream),
        ("Tool Use 多轮对话", test_tool_use_multi_turn),
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
