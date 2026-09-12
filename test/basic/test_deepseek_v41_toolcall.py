#!/usr/bin/env python3
"""DeepSeek-V4.1 工具调用解析 / 请求参数透传的纯 Python 单测（不需要 GPU）。

运行：
  PYTHONPATH=<fastllm>/build/tools python test/basic/test_deepseek_v41_toolcall.py
或
  PYTHONPATH=<fastllm>/build/tools python -m pytest test/basic/test_deepseek_v41_toolcall.py

所有待解析的 completion 文本都用官方 ``ftllm.encoding_dsv41`` 的模板拼出来，
避免测试里出现第二份 DSML 格式定义。
"""

import json
import sys

from ftllm.encoding_dsv41 import (
    REASONING_EFFORT_MAPPINGS,
    dsml_token,
    encode_arguments_to_dsml,
    encode_messages,
    eos_token,
    parse_message_from_completion_text,
    thinking_end_token,
    tool_call_tag_name,
    tool_call_template,
    tool_calls_block_name,
    tool_calls_template,
)
from ftllm.openai_server.protocal.openai_protocol import ChatCompletionRequest
from ftllm.openai_server.tool_parsers import ToolParserManager
from ftllm.openai_server.toolcall_constraints import (
    compile_tool_call_constraint,
)


# ------------------------------------------------------------------
# 官方模板拼装的 completion 文本
# ------------------------------------------------------------------

def build_tool_calls_block(calls):
    """用官方模板拼出 <｜DSML｜ calls> 块。calls 为 [(name, arguments dict)]。"""
    rendered = [
        tool_call_template.format(
            dsml_token=dsml_token,
            tool_call_tag_name=tool_call_tag_name,
            name=name,
            arguments=encode_arguments_to_dsml(
                {"name": name, "arguments": arguments}),
        )
        for name, arguments in calls
    ]
    return tool_calls_template.format(
        dsml_token=dsml_token,
        tool_calls="\n".join(rendered),
        tc_block_name=tool_calls_block_name,
    )


def build_completion(calls, summary="", reasoning=None, with_eos=True):
    text = ""
    if reasoning is not None:
        text += reasoning + thinking_end_token
    text += summary
    if calls:
        text += "\n\n" + build_tool_calls_block(calls)
    if with_eos:
        text += eos_token
    return text


class _StubTokenizer:
    """工具解析器只要求 tokenizer 为真值。"""

    def get_vocab(self):
        return {}


def make_parser():
    parser_cls = ToolParserManager.get_tool_parser("deepseek_v41")
    return parser_cls(_StubTokenizer())


def make_request(tools=None):
    return ChatCompletionRequest(
        model="deepseek-v41",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
    )


def parsed_calls(info):
    return [
        (call.function.name, json.loads(call.function.arguments))
        for call in info.tool_calls
    ]


# ------------------------------------------------------------------
# 非流式
# ------------------------------------------------------------------

def test_auto_parser_selection():
    """model_type=deepseek_v41 时必须自动选到 V4.1 parser，V4 的行为不变。"""
    v41 = ToolParserManager.get_tool_parser_auto("deepseek_v41", "")
    v41_text = ToolParserManager.get_tool_parser_auto("deepseek_v41_text", "")
    v4 = ToolParserManager.get_tool_parser_auto("deepseek_v4", "")
    assert v41.__name__ == "DeepSeekV41ToolParser"
    assert v41_text.__name__ == "DeepSeekV41ToolParser"
    assert v4.__name__ == "DeepSeekV4ToolParser"
    # force_chat_template 时按模板里的标签检测
    detected = ToolParserManager.get_tool_parser_auto(
        "unknown", "<｜DSML｜ calls>", force_chat_template=True)
    assert detected.__name__ == "DeepSeekV41ToolParser"
    detected_v4 = ToolParserManager.get_tool_parser_auto(
        "unknown", "<｜DSML｜tool_calls>", force_chat_template=True)
    assert detected_v4.__name__ == "DeepSeekV4ToolParser"


def test_single_tool_call():
    text = build_completion(
        [("get_weather", {"location": "北京", "days": 3})],
        summary="我来查一下。")
    info = make_parser().extract_tool_calls(text, make_request())
    assert info.tools_called
    assert parsed_calls(info) == [
        ("get_weather", {"location": "北京", "days": 3})]
    assert info.content == "我来查一下。"
    # 与官方解析器逐字段一致
    official = parse_message_from_completion_text(text, "chat")
    assert [(c["function"]["name"], c["function"]["arguments"])
            for c in official["tool_calls"]] == [
        (call.function.name, call.function.arguments)
        for call in info.tool_calls]


def test_multiple_tool_calls():
    calls = [
        ("get_weather", {"location": "上海"}),
        ("search", {"query": "fastllm", "top_k": 5}),
        ("noop", {}),
    ]
    text = build_completion(calls)
    info = make_parser().extract_tool_calls(text, make_request())
    assert info.tools_called
    assert parsed_calls(info) == calls


def test_chinese_and_json_arguments():
    arguments = {
        "标题": "上海天气：晴，26℃",
        "payload": {"列表": [1, 2, {"k": "值"}], "flag": True, "ratio": 0.5},
        "count": -12,
        "empty": None,
    }
    text = build_completion([("记录", arguments)])
    info = make_parser().extract_tool_calls(text, make_request())
    assert info.tools_called
    name, decoded = parsed_calls(info)[0]
    assert name == "记录"
    assert decoded == arguments
    # 中文不被转义成 \uXXXX
    assert "上海天气" in info.tool_calls[0].function.arguments


def test_plain_text_without_tool_calls():
    text = "这是一个普通回答，没有工具调用。" + eos_token
    info = make_parser().extract_tool_calls(text, make_request())
    assert not info.tools_called
    assert info.tool_calls == []
    assert info.content == text


def test_reasoning_plus_tool_call_split_by_server():
    """服务端已经按 </think> 拆出 reasoning 时，parser 只看到后半段。"""
    full = build_completion(
        [("lookup", {"query": "value", "limit": 2})],
        summary="summary", reasoning="  reason  ")
    reasoning, content = full.split(thinking_end_token, 1)
    assert reasoning == "  reason  "
    info = make_parser().extract_tool_calls(content, make_request())
    assert info.tools_called
    assert parsed_calls(info) == [("lookup", {"query": "value", "limit": 2})]
    assert info.content == "summary"


def test_reasoning_inline_with_tool_call():
    """emit_reasoning_content=False 时 <think>…</think> 留在文本里。"""
    text = "<think>" + build_completion(
        [("lookup", {"query": "值"})], summary="好的", reasoning="推理过程")
    info = make_parser().extract_tool_calls(text, make_request())
    assert info.tools_called
    assert parsed_calls(info) == [("lookup", {"query": "值"})]
    assert info.content is not None
    assert info.content.startswith("<think>推理过程</think>")


def test_truncated_output_falls_back_to_tolerant_scan():
    """缺 EOS / 结尾有多余内容时，官方严格解析失败，回退扫描仍要给出工具调用。"""
    text = build_completion(
        [("get_weather", {"location": "广州"})], with_eos=False)
    text += "\n\n(truncated)"
    info = make_parser().extract_tool_calls(text, make_request())
    assert info.tools_called
    assert parsed_calls(info) == [("get_weather", {"location": "广州"})]


def test_v4_tags_are_not_accepted():
    """V4 的无空格标签不应被 V4.1 parser 当成工具调用。"""
    v4_text = (
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="location" string="true">北京'
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>" + eos_token
    )
    info = make_parser().extract_tool_calls(v4_text, make_request())
    assert not info.tools_called


# ------------------------------------------------------------------
# 流式
# ------------------------------------------------------------------

def run_streaming(text, chunk_size):
    """按固定长度切分（会把标签切开），返回 (content, [(name, args)])。"""
    parser = make_parser()
    request = make_request()
    previous = ""
    content_parts = []
    tool_calls = []
    for start in range(0, len(text), chunk_size):
        delta = text[start:start + chunk_size]
        current = previous + delta
        message = parser.extract_tool_calls_streaming(
            previous, current, delta, [0], [0], [0], request)
        if message is not None:
            if message.content:
                content_parts.append(message.content)
            for call in message.tool_calls or []:
                tool_calls.append(
                    (call.function.name, json.loads(call.function.arguments)))
        previous = current
    return "".join(content_parts), tool_calls


def test_streaming_every_chunk_size():
    calls = [
        ("get_weather", {"location": "北京", "days": 3}),
        ("search", {"query": "深度求索", "filters": {"lang": "zh"}}),
    ]
    text = build_completion(calls, summary="稍等，我查一下。")
    # 逐字符到整段，覆盖所有把 DSML 标签切开的边界
    for chunk_size in list(range(1, 12)) + [17, 33, 64, len(text)]:
        content, streamed = run_streaming(text, chunk_size)
        assert streamed == calls, f"chunk_size={chunk_size}: {streamed}"
        assert content.startswith("稍等，我查一下。"), (
            f"chunk_size={chunk_size}: {content!r}")
        # 工具调用块本身绝不能作为 content 流出去
        assert dsml_token not in content, f"chunk_size={chunk_size}"


def test_streaming_plain_text():
    text = "普通回答，没有工具调用。" + eos_token
    for chunk_size in (1, 3, 7, len(text)):
        content, streamed = run_streaming(text, chunk_size)
        assert streamed == []
        assert content == text, f"chunk_size={chunk_size}: {content!r}"


def test_streaming_partial_start_tag_is_not_leaked():
    """标签前缀在分片边界上时不能提前当作正文吐出。"""
    parser = make_parser()
    request = make_request()
    prefix = "答案："
    partial = prefix + "<｜DSML｜ ca"
    message = parser.extract_tool_calls_streaming(
        "", partial, partial, [0], [0], [0], request)
    assert message is not None
    assert message.content == prefix


def test_streaming_matches_non_streaming():
    calls = [("记录", {"标题": "中文", "payload": {"a": [1, 2]}})]
    text = build_completion(calls, summary="好的")
    _, streamed = run_streaming(text, 2)
    info = make_parser().extract_tool_calls(text, make_request())
    assert streamed == parsed_calls(info)


# ------------------------------------------------------------------
# 约束解码
# ------------------------------------------------------------------

def _descriptor(constraint_type, model_type):
    return {
        "constraint_type": constraint_type,
        "model_type": model_type,
        "tool_names": ["get_weather"],
        "allowed_tool_names": ["get_weather"],
        "tool_choice": "auto",
        "requires_tool_call": False,
        "named_tool_choice": None,
        "parallel_tool_calls": None,
        "schemas": {},
        "parameter_names": {"get_weather": ["location", "days"]},
        "strict_tool_names": [],
    }


def test_v41_constraint_spec():
    spec = compile_tool_call_constraint(
        _descriptor("deepseek_v41_dsml", "deepseek_v41")).to_dict()
    name_constraint = spec["name_constraint"]
    assert name_constraint["format"] == "deepseek_v41_dsml"
    assert name_constraint["invoke_name_prefixes"] == [
        '<｜DSML｜ invoke name="', '<\\DSML\\ invoke name="']
    parameter_constraint = spec["parameter_name_constraint"]
    assert parameter_constraint["parameter_name_prefixes"] == [
        '<｜DSML｜ parameter name="', '<\\DSML\\ parameter name="']
    structural = spec["structural_tag"]
    assert structural["tool_call_start"] == "<｜DSML｜ calls>"
    assert structural["tool_call_end"] == "</｜DSML｜ calls>"
    # 前缀必须真的出现在官方渲染出来的工具调用块里
    block = build_tool_calls_block([("get_weather", {"location": "北京"})])
    assert name_constraint["invoke_name_prefixes"][0] in block
    assert parameter_constraint["parameter_name_prefixes"][0] in block


def test_v4_constraint_spec_unchanged():
    spec = compile_tool_call_constraint(
        _descriptor("deepseek_v4_dsml", "deepseek_v4")).to_dict()
    assert spec["name_constraint"]["invoke_name_prefixes"] == [
        '<｜DSML｜invoke name="', '<\\DSML\\invoke name="']
    assert spec["structural_tag"]["tool_call_start"] == "<｜DSML｜tool_calls>"


def test_constraint_type_from_parser_name():
    from ftllm.openai_server.toolcall_parser import FunctionCallParser
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    }]
    for parser_name, expected in (
            ("deepseek_v41", "deepseek_v41_dsml"),
            ("deepseek_v41_text", "deepseek_v41_dsml"),
            ("deepseek_v4", "deepseek_v4_dsml")):
        descriptor = FunctionCallParser.build_constraint_descriptor_from_request(
            make_request(tools), tool_parser_name=parser_name)
        assert descriptor.to_dict()["constraint_type"] == expected, parser_name


# ------------------------------------------------------------------
# 请求参数透传：reasoning effort 与中途 system 消息
# ------------------------------------------------------------------

class _StubV41Model:
    force_chat_template = False

    def _is_deepseek_v41(self):
        return True


def _resolve_effort(value, chat_template_kwargs=None):
    from ftllm.openai_server.fastllm_completion import FastLLmCompletion

    class _Stub:
        # 复用真实的模型判定逻辑，只把底层 model 换成桩
        model = _StubV41Model()
        _is_deepseek_v41_model = FastLLmCompletion._is_deepseek_v41_model

    request = ChatCompletionRequest(
        model="deepseek-v41",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=value,
        chat_template_kwargs=chat_template_kwargs,
    )
    return FastLLmCompletion._resolve_deepseek_v41_reasoning_effort(
        _Stub(), request)


def test_reasoning_effort_resolution():
    assert _resolve_effort(None) is None
    assert _resolve_effort("none") is None
    for alias in REASONING_EFFORT_MAPPINGS:
        assert _resolve_effort(alias) == alias
    assert _resolve_effort(1) == 1
    assert _resolve_effort(42) == 42
    assert _resolve_effort(100) == 100
    # OpenAI 客户端常常把数值当字符串发过来
    assert _resolve_effort("42") == 42
    # chat_template_kwargs 作为兜底
    assert _resolve_effort(None, {"reasoning_effort": 30}) == 30
    assert _resolve_effort(None, {"thinking_effort": "max"}) == "max"
    # 越界数值 / 其它模型的档位名都必须被拒绝（非法字符串在请求校验阶段就被挡掉）
    for bad in (0, 101, -3, "medium", "xhigh", "abc"):
        try:
            _resolve_effort(bad)
        except Exception:
            continue
        raise AssertionError(f"reasoning_effort={bad!r} 应当被拒绝")


def test_reasoning_effort_reaches_encode_messages():
    messages = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "你好"},
    ]
    for effort, budget in [(1, 1), (42, 42), (100, 100)] + [
            (alias, value) for alias, value in REASONING_EFFORT_MAPPINGS.items()]:
        prompt = encode_messages(
            messages, thinking_mode="thinking", reasoning_effort=effort)
        assert f"Reasoning Effort: {budget} " in prompt, effort
    # 默认值 high -> 75
    default_prompt = encode_messages(messages, thinking_mode="thinking")
    assert "Reasoning Effort: 75 " in default_prompt
    # chat 模式不渲染 effort 前缀
    chat_prompt = encode_messages(
        messages, thinking_mode="chat", reasoning_effort=42)
    assert "Reasoning Effort" not in chat_prompt


def test_midconversation_system_message():
    messages = [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "system", "content": "现在改用中文回答。"},
    ]
    prompt = encode_messages(messages, thinking_mode="chat")
    # 中途 system 消息用 <｜System｜> 引出，并触发 assistant 生成头
    assert "<｜System｜>现在改用中文回答。" in prompt
    assert prompt.endswith("<｜Assistant｜></think>")
    # 中途 system 也算最后一条“用户”消息
    assert prompt.count("<｜Assistant｜>") == 2


def test_llm_deepseek_encode_messages_binds_effort():
    """llm.py 的 _deepseek_encode_messages 必须把 effort 绑到 encode_messages 上。"""
    try:
        # 需要已编译的 libfastllm_tools；没有就跳过（其余用例不依赖它）。
        from ftllm.llm import model as llm_model
    except Exception as error:  # noqa: BLE001
        print(f"     skip: ftllm.llm 不可用（{type(error).__name__}）")
        return

    class _Fake:
        _is_deepseek_v41 = _StubV41Model._is_deepseek_v41
        _deepseek_encode_messages = llm_model._deepseek_encode_messages

    fake = _Fake()
    messages = [{"role": "user", "content": "你好"}]
    bound = fake._deepseek_encode_messages(42)
    assert "Reasoning Effort: 42 " in bound(messages, thinking_mode="thinking")
    unbound = fake._deepseek_encode_messages()
    assert "Reasoning Effort: 75 " in unbound(
        messages, thinking_mode="thinking")


# ------------------------------------------------------------------

def main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
            import traceback
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
