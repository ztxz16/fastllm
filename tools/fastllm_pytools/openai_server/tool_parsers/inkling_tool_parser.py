"""OpenAI adapter for Inkling's typed JSON invocation messages."""
import json

from ..inkling_messages import END, INVOKE, consume
from ..protocal.openai_protocol import (
    DeltaFunctionCall, DeltaMessage, DeltaToolCall, ExtractedToolCallInformation,
    FunctionCall, ToolCall,
)
from .abstract_tool_parser import ToolParser, ToolParserManager, random_tool_call_id


@ToolParserManager.register_module("inkling2_loop")
class InklingToolParser(ToolParser):
    tool_call_start_token = INVOKE

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._reset()
        self.index = 0

    def _reset(self):
        self.state = {"phase": "content", "buffer": ""}
        self.payload = None

    @property
    def incomplete_tool_call(self):
        return self.payload is not None

    def _consume(self, text, final=False):
        content, calls = [], []
        for channel, part in consume(text, self.state):
            if channel == "content":
                content.append(part)
            elif channel == "tool_json" and self.payload is not None:
                self.payload.append(part)
            elif channel == "tool":
                if part == INVOKE:
                    self.payload = []
                elif part == END and self.payload is not None:
                    body = "".join(self.payload)
                    self.payload = None
                    try:
                        obj = json.loads(body)
                    except ValueError:
                        continue
                    if (isinstance(obj, dict) and isinstance(obj.get("name"), str)
                            and obj["name"] and isinstance(obj.get("args"), dict)):
                        arguments = json.dumps(obj["args"], ensure_ascii=False, separators=(",", ":"))
                        calls.append(FunctionCall(name=obj["name"], arguments=arguments))
        if final and self.state["phase"] == "content":
            content.append(self.state["buffer"])
        return "".join(content) or None, calls

    def extract_tool_calls(self, model_output, request):
        self._reset()
        content, functions = self._consume(model_output, final=True)
        calls = [ToolCall(function=fn) for fn in functions]
        return ExtractedToolCallInformation(tools_called=bool(calls), tool_calls=calls, content=content)

    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, previous_token_ids,
                                     current_token_ids, delta_token_ids, request):
        content, functions = self._consume(delta_text)
        calls = []
        for fn in functions:
            calls.append(DeltaToolCall(index=self.index, type="function", id=random_tool_call_id(),
                                      function=DeltaFunctionCall(name=fn.name, arguments=fn.arguments)))
            self.index += 1
        return DeltaMessage(content=content, tool_calls=calls) if content or calls else None
