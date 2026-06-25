# Adapted from
# https://github.com/lm-sys/FastChat/blob/v0.2.36/fastchat/protocol/openai_api_protocol.py
from typing import Literal, Optional, List, Dict, Any, Union

import time
import uuid

import shortuuid
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{shortuuid.random()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "fastchat"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    strict: Optional[bool] = None


class ChatCompletionToolsParam(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(BaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(BaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"

class ChatCompletionRequest(BaseModel):
    model: str
    messages: Optional[Union[str, List[Dict[str, Any]]]] = []
    prompt: Optional[str] = ""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = 0
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    tools: Optional[list[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[
        Literal["none"],
        Literal["auto"],
        Literal["required"],
        ChatCompletionNamedToolChoiceParam,
    ]] = "none"
    parallel_tool_calls: Optional[bool] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None


class ResponsesRequest(BaseModel):
    model: str
    input: Optional[Union[str, List[Any]]] = None
    instructions: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = "auto"
    parallel_tool_calls: Optional[bool] = False
    store: Optional[bool] = False
    previous_response_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    user: Optional[str] = None
    text: Optional[Dict[str, Any]] = None
    reasoning: Optional[Dict[str, Any]] = None
    truncation: Optional[str] = "disabled"
    include: Optional[List[Any]] = Field(default_factory=list)
    prompt_cache_key: Optional[str] = None
    client_metadata: Optional[Dict[str, Any]] = None


class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: "fastllm-tool-" + str(uuid.uuid4().hex))
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

class DeltaToolCall(BaseModel):
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    index: int
    function: Optional[DeltaFunctionCall] = None

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)

class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool

    # extracted tool calls
    tool_calls: list[ToolCall]

    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: Optional[str] = None



class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


class ResponsesUsageInfo(BaseModel):
    input_tokens: int = 0
    input_tokens_details: Dict[str, Any] = Field(default_factory=lambda: {
        "cached_tokens": 0,
    })
    output_tokens: int = 0
    output_tokens_details: Dict[str, Any] = Field(default_factory=lambda: {
        "reasoning_tokens": 0,
    })
    total_tokens: int = 0


class ResponsesResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"resp_{shortuuid.random()}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: Literal["in_progress", "completed", "incomplete", "failed"] = "completed"
    completed_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    incomplete_details: Optional[Dict[str, Any]] = None
    instructions: Optional[Any] = None
    max_output_tokens: Optional[int] = None
    model: str
    output: List[Dict[str, Any]] = Field(default_factory=list)
    output_text: Optional[str] = None
    parallel_tool_calls: bool = False
    previous_response_id: Optional[str] = None
    reasoning: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "effort": None,
        "summary": None,
    })
    store: bool = False
    temperature: Optional[float] = None
    text: Dict[str, Any] = Field(default_factory=lambda: {
        "format": {
            "type": "text",
        },
    })
    tool_choice: Any = "auto"
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    top_p: Optional[float] = None
    truncation: str = "disabled"
    usage: Optional[ResponsesUsageInfo] = None
    user: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TokenCheckRequestItem(BaseModel):
    model: str
    prompt: str
    max_tokens: int


class TokenCheckRequest(BaseModel):
    prompts: List[TokenCheckRequestItem]


class TokenCheckResponseItem(BaseModel):
    fits: bool
    tokenCount: int
    contextLength: int


class TokenCheckResponse(BaseModel):
    prompts: List[TokenCheckResponseItem]


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    engine: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None
    encoding_format: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[Any]]
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    max_tokens: Optional[int] = 16
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    use_beam_search: Optional[bool] = False
    best_of: Optional[int] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]

class EmbedRequest(BaseModel):
    inputs: str
    normalize: Optional[bool] = False
    prompt_name: Optional[str] = "null"
    truncate: Optional[bool] = False
    truncation_direction: Optional[str] = 'right'

class RerankRequest(BaseModel):
    query: str
    texts: List[str]
    raw_scores: Optional[bool] = True
    return_text: Optional[bool] = False
    truncate: Optional[bool] = False
    truncation_direction: Optional[str] = "right"
