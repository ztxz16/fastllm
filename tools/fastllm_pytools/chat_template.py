import copy
import json


def normalize_conversation(model_struct, conversation):
    if (model_struct in ["minimax", "minimax_m2"]):
        for i in range(len(conversation)):
            if ("content" in conversation[i] and isinstance(conversation[i]["content"], str)):
                conversation[i]["content"] = [{"type": "text", "text": conversation[i]["content"]}]
    elif model_struct in ("dots3_note", "glm5_next", "naive_n05_flash"):
        # These chat templates iterate over argument objects, while the
        # OpenAI wire protocol stores function.arguments as a JSON string.
        # Normalize assistant tool-call history before rendering so a
        # tool-result round trip can be fed back into the model.
        conversation = copy.deepcopy(conversation)
        for message in conversation:
            if (not isinstance(message, dict) or
                    message.get("role") != "assistant"):
                continue
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function", tool_call)
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = (json.loads(arguments)
                                     if arguments.strip() else {})
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"{model_struct} tool-call history contains "
                            "invalid JSON arguments") from error
                elif arguments is None:
                    arguments = {}
                if not isinstance(arguments, dict):
                    raise ValueError(
                        f"{model_struct} tool-call arguments must decode to "
                        "a JSON object")
                function["arguments"] = arguments
    return conversation


def apply_hf_chat_template(tokenizer, conversation, add_generation_prompt = True,
                           tokenize = False, enable_thinking = None, tools = None,
                           thinking_alias = False, thinking_effort = None,
                           tool_choice = None, chat_template_kwargs = None):
    kwargs = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": tokenize,
    }
    if chat_template_kwargs:
        kwargs.update(chat_template_kwargs)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
        if thinking_alias:
            kwargs["thinking"] = enable_thinking
    if tools is not None:
        kwargs["tools"] = tools
    if thinking_effort is not None:
        kwargs["thinking_effort"] = thinking_effort
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    try:
        ret = tokenizer.apply_chat_template(conversation, **kwargs)
    except TypeError:
        kwargs.pop("thinking", None)
        try:
            ret = tokenizer.apply_chat_template(conversation, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            kwargs.pop("thinking_effort", None)
            kwargs.pop("tool_choice", None)
            kwargs.pop("tools", None)
            ret = tokenizer.apply_chat_template(conversation, **kwargs)
    if tokenize and hasattr(ret, "keys") and "input_ids" in ret.keys():
        ret = ret["input_ids"]
    if tokenize and hasattr(ret, "tolist"):
        ret = ret.tolist()
    return ret
