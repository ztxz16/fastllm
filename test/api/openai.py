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
