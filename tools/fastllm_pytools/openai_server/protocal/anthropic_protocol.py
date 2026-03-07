from typing import Any, Dict, List, Literal, Optional, Union

import shortuuid
import time
from pydantic import BaseModel, Field


class AnthropicTextContentBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicToolUseContentBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class AnthropicToolResultContentBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]]
    is_error: Optional[bool] = None


class AnthropicInputMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]


class AnthropicToolParam(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


class AnthropicMessageRequest(BaseModel):
    model: str
    messages: List[AnthropicInputMessage]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: int
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[AnthropicToolParam]] = None


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicMessageResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{shortuuid.random()}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Union[AnthropicTextContentBlock, AnthropicToolUseContentBlock]]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


class AnthropicTextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class AnthropicInputJsonDelta(BaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: AnthropicMessageResponse


class ContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: Union[AnthropicTextContentBlock, AnthropicToolUseContentBlock]


class ContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Union[AnthropicTextDelta, AnthropicInputJsonDelta]


class ContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class AnthropicMessageDelta(BaseModel):
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: AnthropicMessageDelta
    usage: AnthropicUsage


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"
