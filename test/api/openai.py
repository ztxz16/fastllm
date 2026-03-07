import argparse
import json
import sys
import time
import requests


def test_chat_completions(base_url: str, model: str, api_key: str) -> bool:
    """测试非流式 chat/completions 接口"""
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "max_tokens": 128,
        "messages": [
            {"role": "user", "content": "请用一句话介绍你自己。"}
        ],
        "stream": False,
    }

    print("=" * 60)
    print("[1] 测试非流式 Chat Completions 接口")
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
    if data.get("object") != "chat.completion":
        print(f"    ✗ 预期 object='chat.completion', 实际 object='{data.get('object')}'")
        print(f"    响应: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
        return False

    choices = data.get("choices", [])
    if not choices:
        print("    ✗ 响应中没有 choices")
        return False

    message = choices[0].get("message", {})
    text = message.get("content", "")
    if not text.strip():
        print("    ✗ 响应文本为空")
        return False

    print(f"    ✓ id: {data.get('id', 'N/A')}")
    print(f"    ✓ 模型: {data.get('model', 'N/A')}")
    print(f"    ✓ finish_reason: {choices[0].get('finish_reason', 'N/A')}")
    print(f"    ✓ 回复内容: {text[:200]}")
    usage = data.get("usage", {})
    if usage:
        print(f"    ✓ token 用量: prompt={usage.get('prompt_tokens', '?')}, "
              f"completion={usage.get('completion_tokens', '?')}, "
              f"total={usage.get('total_tokens', '?')}")
    return True


def test_chat_completions_stream(base_url: str, model: str, api_key: str) -> bool:
    """测试流式 chat/completions 接口"""
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
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
    print("[2] 测试流式 Chat Completions 接口")
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
    collected_reasoning = ""
    chunk_count = 0
    first_token_time = None
    start_time = time.time()

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if chunk.get("object") != "chat.completion.chunk":
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content")
        reasoning = delta.get("reasoning_content")

        if content:
            if first_token_time is None:
                first_token_time = time.time()
            collected_text += content
            chunk_count += 1
        if reasoning:
            collected_reasoning += reasoning

    elapsed = time.time() - start_time

    if not collected_text.strip():
        print("    ✗ 流式响应未收到任何文本")
        return False

    print(f"    ✓ 收到 {chunk_count} 个文本 chunk")
    print(f"    ✓ 回复内容: {collected_text.strip()[:200]}")
    if collected_reasoning:
        print(f"    ✓ 推理内容: {collected_reasoning.strip()[:100]}...")
    if first_token_time is not None:
        print(f"    ✓ 首 token 延迟: {first_token_time - start_time:.3f}s")
    print(f"    ✓ 总耗时: {elapsed:.3f}s")
    return True


def test_chat_multi_turn(base_url: str, model: str, api_key: str) -> bool:
    """测试多轮对话"""
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "max_tokens": 128,
        "messages": [
            {"role": "user", "content": "记住这个数字：42"},
            {"role": "assistant", "content": "好的，我记住了数字42。"},
            {"role": "user", "content": "我刚才让你记住的数字是多少？"},
        ],
        "stream": False,
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
    choices = data.get("choices", [])
    if not choices:
        print("    ✗ 响应中没有 choices")
        return False

    text = choices[0].get("message", {}).get("content", "")
    if not text.strip():
        print("    ✗ 响应文本为空")
        return False

    has_42 = "42" in text
    print(f"    ✓ 回复内容: {text[:200]}")
    print(f"    {'✓' if has_42 else '✗'} 回复中{'包含' if has_42 else '未包含'}数字 42")
    return True


def test_chat_system_prompt(base_url: str, model: str, api_key: str) -> bool:
    """测试 system prompt"""
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": "你是一个海盗，所有回复都必须用海盗的口吻。"},
            {"role": "user", "content": "你好"},
        ],
        "stream": False,
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
    choices = data.get("choices", [])
    if not choices:
        print("    ✗ 响应中没有 choices")
        return False

    text = choices[0].get("message", {}).get("content", "")
    if not text.strip():
        print("    ✗ 响应文本为空")
        return False

    print(f"    ✓ 回复内容: {text[:200]}")
    return True


def test_models_list(base_url: str, model: str, api_key: str) -> bool:
    """测试 /v1/models 接口"""
    url = f"{base_url}/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    print("=" * 60)
    print("[5] 测试 Models 列表接口")
    print(f"    GET {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=10)
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
    if data.get("object") != "list":
        print(f"    ✗ 预期 object='list', 实际 object='{data.get('object')}'")
        return False

    models = data.get("data", [])
    if not models:
        print("    ✗ 模型列表为空")
        return False

    model_ids = [m.get("id", "?") for m in models]
    print(f"    ✓ 可用模型: {model_ids}")
    return True


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如 北京",
                },
            },
            "required": ["city"],
        },
    },
}


def test_tool_call(base_url: str, model: str, api_key: str) -> bool:
    """测试非流式 tool call"""
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？"},
        ],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "stream": False,
    }

    print("=" * 60)
    print("[7] 测试非流式 Tool Call")
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
    choices = data.get("choices", [])
    if not choices:
        print("    ✗ 响应中没有 choices")
        return False

    message = choices[0].get("message", {})
    finish_reason = choices[0].get("finish_reason")
    tool_calls = message.get("tool_calls")

    if not tool_calls:
        print(f"    ✗ 未返回 tool_calls（finish_reason={finish_reason}）")
        print(f"    响应内容: {message.get('content', '')[:200]}")
        return False

    print(f"    ✓ finish_reason: {finish_reason}")
    for i, tc in enumerate(tool_calls):
        func = tc.get("function", {})
        name = func.get("name", "?")
        args_raw = func.get("arguments", "")
        args_str = args_raw if isinstance(args_raw, str) else json.dumps(args_raw, ensure_ascii=False)
        print(f"    ✓ tool_call[{i}]: id={tc.get('id', '?')}, name={name}, args={args_str[:200]}")
        try:
            parsed = json.loads(args_str) if isinstance(args_raw, str) else args_raw
            if "city" in parsed:
                print(f"    ✓ 解析参数 city={parsed['city']}")
        except json.JSONDecodeError:
            print(f"    △ arguments 不是合法 JSON: {args_str[:100]}")

    return True


def test_tool_call_stream(base_url: str, model: str, api_key: str) -> bool:
    """测试流式 tool call"""
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？"},
        ],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "stream": True,
    }

    print("=" * 60)
    print("[8] 测试流式 Tool Call")
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

    tool_calls_by_index = {}
    collected_content = ""
    finish_reason = None
    start_time = time.time()

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        fr = choices[0].get("finish_reason")
        if fr:
            finish_reason = fr

        content = delta.get("content")
        if content:
            collected_content += content

        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta.get("index", 0)
            if idx not in tool_calls_by_index:
                tool_calls_by_index[idx] = {
                    "id": tc_delta.get("id", ""),
                    "name": "",
                    "arguments": "",
                }
            entry = tool_calls_by_index[idx]
            if tc_delta.get("id"):
                entry["id"] = tc_delta["id"]
            func = tc_delta.get("function", {})
            if func.get("name"):
                entry["name"] += func["name"]
            if func.get("arguments"):
                entry["arguments"] += func["arguments"]

    elapsed = time.time() - start_time

    if not tool_calls_by_index:
        print(f"    ✗ 流式响应未收到 tool_calls（finish_reason={finish_reason}）")
        if collected_content:
            print(f"    收到文本内容: {collected_content[:200]}")
        return False

    print(f"    ✓ finish_reason: {finish_reason}")
    print(f"    ✓ 总耗时: {elapsed:.3f}s")
    for idx in sorted(tool_calls_by_index):
        entry = tool_calls_by_index[idx]
        print(f"    ✓ tool_call[{idx}]: id={entry['id']}, name={entry['name']}, args={entry['arguments'][:200]}")
        try:
            parsed = json.loads(entry["arguments"])
            if "city" in parsed:
                print(f"    ✓ 解析参数 city={parsed['city']}")
        except json.JSONDecodeError:
            print(f"    △ arguments 不是合法 JSON: {entry['arguments'][:100]}")

    return True


def test_tool_call_multi_turn(base_url: str, model: str, api_key: str) -> bool:
    """测试 tool call 多轮：发送 tool 结果后模型应给出最终回复"""
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # 第一轮：模型应调用 get_weather
    payload_1 = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？"},
        ],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "stream": False,
    }

    print("=" * 60)
    print("[9] 测试 Tool Call 多轮对话")
    print(f"    POST {url} (第一轮: 触发 tool call)")
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
    choices1 = data1.get("choices", [])
    if not choices1:
        print("    ✗ 第一轮响应中没有 choices")
        return False

    msg1 = choices1[0].get("message", {})
    tool_calls = msg1.get("tool_calls")
    if not tool_calls:
        print(f"    ✗ 第一轮未返回 tool_calls")
        print(f"    内容: {msg1.get('content', '')[:200]}")
        return False

    tc = tool_calls[0]
    tc_id = tc.get("id", "call_test")
    func_name = tc.get("function", {}).get("name", "?")
    print(f"    ✓ 第一轮返回 tool_call: name={func_name}, id={tc_id}")

    # 第二轮：将 tool 结果返回给模型
    assistant_msg = {"role": "assistant", "content": msg1.get("content") or None, "tool_calls": tool_calls}
    tool_result_msg = {
        "role": "tool",
        "tool_call_id": tc_id,
        "name": func_name,
        "content": json.dumps({"city": "北京", "weather": "晴", "temperature": "25°C"}, ensure_ascii=False),
    }
    payload_2 = {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？"},
            assistant_msg,
            tool_result_msg,
        ],
        "tools": [WEATHER_TOOL],
        "stream": False,
    }

    print(f"    POST {url} (第二轮: 提交 tool 结果)")
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
    choices2 = data2.get("choices", [])
    if not choices2:
        print("    ✗ 第二轮响应中没有 choices")
        return False

    text2 = choices2[0].get("message", {}).get("content", "")
    if not text2.strip():
        print("    ✗ 第二轮响应文本为空")
        return False

    print(f"    ✓ 第二轮回复: {text2[:200]}")
    has_weather_info = any(kw in text2 for kw in ["晴", "25", "天气", "北京"])
    print(f"    {'✓' if has_weather_info else '△'} 回复中{'包含' if has_weather_info else '未包含'}天气相关信息")
    return True


def test_chat_temperature(base_url: str, model: str, api_key: str) -> bool:
    """测试 temperature 参数（temperature=0 应产生较稳定的输出）"""
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "max_tokens": 32,
        "temperature": 0,
        "top_k": 1,
        "messages": [
            {"role": "user", "content": "1+1等于几？只回答数字。"}
        ],
        "stream": False,
    }

    print("=" * 60)
    print("[6] 测试 Temperature 参数")
    print(f"    POST {url} (temperature=0, top_k=1)")

    results = []
    for i in range(2):
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
            return False

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            print("    ✗ 响应中没有 choices")
            return False

        text = choices[0].get("message", {}).get("content", "").strip()
        results.append(text)
        print(f"    第 {i + 1} 次回复: {text[:100]}")

    same = results[0] == results[1]
    print(f"    {'✓' if same else '△'} 两次回复{'一致' if same else '不一致'}（temperature=0 期望一致）")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Chat Completions API 兼容性测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python openai.py --base-url http://localhost:8080 --model my-model
  python openai.py --base-url http://10.0.0.1:1616 --model ds --api-key sk-xxx
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
    print(f"OpenAI API 兼容性测试")
    print(f"服务地址: {base_url}")
    print(f"模型名称: {args.model}")
    print()

    tests = [
        ("非流式 Chat Completions", test_chat_completions),
        ("流式 Chat Completions", test_chat_completions_stream),
        ("多轮对话", test_chat_multi_turn),
        ("System Prompt", test_chat_system_prompt),
        ("Models 列表", test_models_list),
        ("Temperature 参数", test_chat_temperature),
        ("非流式 Tool Call", test_tool_call),
        ("流式 Tool Call", test_tool_call_stream),
        ("Tool Call 多轮对话", test_tool_call_multi_turn),
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
