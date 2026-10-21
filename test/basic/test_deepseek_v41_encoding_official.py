#!/usr/bin/env python3
"""用官方 encoding/tests 的用例核对 ftllm 的 V4.1 prompt 编码与分词。

需要 DeepSeek-V4.1-Flash 的 ``encoding`` 目录（含 tests/test_input_*.json 与
test_output_*.txt）；不存在时整个文件跳过。tokenizer 部分还需要一个带
tokenizer.json 的模型目录。

运行：
  PYTHONPATH=<fastllm>/build/tools \\
  DSV41_ENCODING_DIR=/mnt/shared2/models/DeepSeek-V4.1-Flash-1017/encoding \\
  DSV41_TOKENIZER_DIR=/root/v41-tokenizer \\
      python test/basic/test_deepseek_v41_encoding_official.py
"""

import difflib
import json
import os
import sys

try:
    # fastllm 的动态库加载之后再 import torch/transformers 会撞上
    # ncclCommWindowDeregister 符号错误，所以先把 torch 拉进来。
    import torch  # noqa: F401
except Exception:  # noqa: BLE001
    pass

import ftllm.encoding_dsv41 as enc

DEFAULT_ENCODING_DIR = "/mnt/shared2/models/DeepSeek-V4.1-Flash-1017/encoding"
DEFAULT_TOKENIZER_DIR = "/root/v41-tokenizer"

ENCODING_DIR = os.environ.get("DSV41_ENCODING_DIR", DEFAULT_ENCODING_DIR)
TOKENIZER_DIR = os.environ.get("DSV41_TOKENIZER_DIR", DEFAULT_TOKENIZER_DIR)
CASES_DIR = os.path.join(ENCODING_DIR, "tests")


def _skip(reason):
    print(f"     skip: {reason}")


def _case_ids():
    if not os.path.isdir(CASES_DIR):
        return []
    ids = []
    for name in sorted(os.listdir(CASES_DIR)):
        if name.startswith("test_input_") and name.endswith(".json"):
            ids.append(name[len("test_input_"):-len(".json")])
    return ids


def test_vendored_encoding_matches_official():
    """encoding_dsv41.py 除了开头的说明 docstring 之外必须与官方 encoding.py 逐字相同。"""
    official_path = os.path.join(ENCODING_DIR, "encoding.py")
    if not os.path.isfile(official_path):
        _skip(f"没有 {official_path}")
        return
    with open(official_path, encoding="utf-8") as handle:
        official = handle.read()
    with open(enc.__file__, encoding="utf-8") as handle:
        vendored = handle.read()
    # 去掉 vendored 版本开头额外插入的说明 docstring
    marker = '"""\n\n"""\n'
    assert marker in vendored[:2048], "vendored 文件开头格式变了"
    stripped = vendored[vendored.index(marker) + len(marker):]
    official_body = official[official.index('"""\n') + len('"""\n'):]
    if stripped != official_body:
        diff = "\n".join(list(difflib.unified_diff(
            official_body.splitlines(), stripped.splitlines(),
            "official/encoding.py", "ftllm/encoding_dsv41.py", lineterm=""))[:40])
        raise AssertionError("vendored 与官方 encoding.py 不一致:\n" + diff)
    print("     encoding_dsv41.py 与官方 encoding.py 逐字一致（仅多出说明 docstring）")


def test_official_cases_encode_bytewise_identical():
    """每个官方用例的 prompt 编码必须与 golden 输出逐字节一致。"""
    case_ids = _case_ids()
    if not case_ids:
        _skip(f"没有 {CASES_DIR}")
        return
    for case_id in case_ids:
        input_file = os.path.join(CASES_DIR, f"test_input_{case_id}.json")
        output_file = os.path.join(CASES_DIR, f"test_output_{case_id}.txt")
        assert os.path.isfile(output_file), f"缺少 golden 输出 {output_file}"
        case = enc.load_cases(input_file)[0]
        prompt, _ = enc.encode_case(case, thinking_mode="chat")
        with open(output_file, encoding="utf-8") as handle:
            expected = handle.read()
        assert prompt == expected, (
            f"case {case_id} 编码与官方不一致\n"
            + "\n".join(difflib.unified_diff(
                expected.splitlines(), prompt.splitlines(),
                "official", "ftllm", lineterm="")))
        print(f"     case {case_id}: {len(prompt)} 字符逐字一致")


def _load_tokenizer():
    if not os.path.isfile(os.path.join(TOKENIZER_DIR, "tokenizer.json")):
        return None
    try:
        from ftllm.llm import try_load_hf_tokenizer
    except Exception as error:  # noqa: BLE001
        print(f"     ftllm.llm 不可用（{type(error).__name__}）")
        return None
    return try_load_hf_tokenizer(TOKENIZER_DIR)


def test_official_cases_tokenize_identically():
    """ftllm 的 encode_hf_prompt 必须与直接用官方 tokenizer 分词结果一致，且可无损还原。"""
    case_ids = _case_ids()
    if not case_ids:
        _skip(f"没有 {CASES_DIR}")
        return
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        _skip(f"没有可用的 tokenizer（{TOKENIZER_DIR}）")
        return
    from ftllm.llm import encode_hf_prompt

    total = 0
    for case_id in case_ids:
        case = enc.load_cases(
            os.path.join(CASES_DIR, f"test_input_{case_id}.json"))[0]
        prompt, _ = enc.encode_case(case, thinking_mode="chat")
        ids = encode_hf_prompt(tokenizer, prompt)
        reference = tokenizer.encode(prompt)
        assert ids == reference, f"case {case_id}: encode_hf_prompt 与官方分词不同"
        # BOS 不能被重复加一次
        assert prompt.startswith(enc.bos_token)
        assert ids.count(ids[0]) == 1 or not prompt.count(enc.bos_token) == 1, (
            f"case {case_id}: BOS 被重复添加")
        decoded = tokenizer.decode(ids)
        assert decoded == prompt, f"case {case_id}: 分词后无法逐字还原 prompt"
        total += len(ids)
        print(f"     case {case_id}: {len(ids)} tokens，解码后逐字还原")
    print(f"     共 {total} tokens 全部一致")


def test_tokenizer_loads_without_mistral_warning():
    """加载 V4.1 tokenizer 不应再出现 transformers 的 fix_mistral_regex 告警。"""
    if not os.path.isfile(os.path.join(TOKENIZER_DIR, "tokenizer.json")):
        _skip(f"没有 {TOKENIZER_DIR}")
        return
    try:
        import logging
        from transformers import AutoTokenizer
        from ftllm.llm import _hf_tokenizer_compat_kwargs
    except Exception as error:  # noqa: BLE001
        _skip(f"transformers/ftllm 不可用（{type(error).__name__}）")
        return

    records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collector()
    root = logging.getLogger()
    original_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    try:
        AutoTokenizer.from_pretrained(
            TOKENIZER_DIR, trust_remote_code=True,
            **_hf_tokenizer_compat_kwargs(TOKENIZER_DIR))
    finally:
        root.removeHandler(handler)
        root.setLevel(original_level)

    offending = [msg for msg in records if "fix_mistral_regex" in msg]
    assert not offending, f"仍然出现 fix_mistral_regex 告警: {offending}"
    # 非 Mistral 模型必须显式传 False；Mistral 系列则不能被我们篡改
    assert _hf_tokenizer_compat_kwargs(TOKENIZER_DIR) in (
        {}, {"fix_mistral_regex": False})
    print("     未出现 fix_mistral_regex 告警")


def test_uses_hf_tokenizer_for_v41():
    """_uses_hf_deepseek_v4_tokenizer 必须覆盖 V4.1。"""
    try:
        from ftllm.llm import model as llm_model
    except Exception as error:  # noqa: BLE001
        _skip(f"ftllm.llm 不可用（{type(error).__name__}）")
        return

    class _Fake:
        force_chat_template = False
        hf_tokenizer = object()
        config = {"model_type": "deepseek_v41"}

        def _get_architecture(self):
            return "DeepseekV41ForCausalLM"

        _is_deepseek_v41 = llm_model._is_deepseek_v41
        _is_deepseek_v4 = llm_model._is_deepseek_v4
        _uses_hf_deepseek_v4_tokenizer = llm_model._uses_hf_deepseek_v4_tokenizer

    fake = _Fake()
    assert fake._is_deepseek_v41()
    assert fake._is_deepseek_v4()
    assert fake._uses_hf_deepseek_v4_tokenizer()
    # text-only 变体同样成立
    fake_text = _Fake()
    fake_text.config = {"model_type": "deepseek_v41_text"}
    fake_text._get_architecture = lambda: ""
    assert fake_text._uses_hf_deepseek_v4_tokenizer()
    # force_chat_template 时回到通用路径
    forced = _Fake()
    forced.force_chat_template = True
    assert not forced._uses_hf_deepseek_v4_tokenizer()
    print("     _uses_hf_deepseek_v4_tokenizer 覆盖 deepseek_v41 / deepseek_v41_text")


def test_tool_results_do_not_start_new_turn():
    """1017 起：只含工具结果的 user 消息不算新一轮，当前轮跨工具调用保留思考。

    旧版 find_last_user_index 把每条工具结果都当成新的用户轮次，未声明 tools 时
    drop_thinking 会删掉 agent 当前轮里每一步的 reasoning_content。
    """
    import json as _json
    call = lambda i, cmd: [{"id": f"c{i}", "type": "function",
                            "function": {"name": "bash", "arguments": _json.dumps({"command": cmd})}}]
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "OLD-TURN"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "", "reasoning_content": "STEP-1", "tool_calls": call(1, "ls")},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
        {"role": "assistant", "content": "", "reasoning_content": "STEP-2", "tool_calls": call(2, "pwd")},
        {"role": "tool", "tool_call_id": "c2", "content": "y"},
    ]
    prompt = enc.encode_messages(messages, thinking_mode="thinking", drop_thinking=True)
    assert "OLD-TURN" not in prompt, "已结束轮次的思考应当丢弃"
    assert "STEP-1" in prompt and "STEP-2" in prompt, "当前轮跨工具调用的思考被丢了"
    assert prompt.endswith("<｜Assistant｜><think>")
    print("     当前轮两步思考保留，上一轮思考丢弃")


def test_namespaced_tool_roundtrip():
    """1017 起：工具可带 namespace，渲染为 ns::name，解析回 name + namespace。"""
    tools = [{"type": "function", "namespace": {"name": "search", "description": "Search tools."},
              "function": {"name": "lookup", "description": "Look up a value",
                           "parameters": {"type": "object",
                                          "properties": {"query": {"type": "string"}}}}}]
    messages = [{"role": "system", "content": "s", "tools": tools}, {"role": "user", "content": "q"}]
    prompt = enc.encode_messages(messages, thinking_mode="chat")
    assert "search::lookup" in prompt
    # 助手消息带 namespace 的工具调用：渲染后再解析回来
    history = [{"role": "user", "content": "q"},
               {"role": "assistant", "content": "", "reasoning_content": "r",
                "tool_calls": [{"type": "function", "namespace": "search",
                                "function": {"name": "lookup", "arguments": '{"query": "v"}'}}]}]
    rendered = enc.render_message(1, history, thinking_mode="thinking")
    assert 'name="search::lookup"' in rendered, rendered
    parsed = enc.parse_message_from_completion_text(rendered, thinking_mode="thinking")
    call = parsed["tool_calls"][0]
    assert call.get("namespace") == "search", call
    assert call["function"]["name"] == "lookup", call
    print("     search::lookup 渲染与解析往返一致")


def main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, test in tests:
        print(f"---- {name}")
        try:
            test()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
