import asyncio
import base64
import binascii
import copy
import inspect
import io
import json
import logging
import os
import time
import traceback
import uuid
from http import HTTPStatus
from typing import (Any, AsyncGenerator, AsyncIterator, Awaitable, Dict, Iterable,
                    List, Optional, Tuple, TypedDict, Union, final)
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import shortuuid
from fastapi import Request
from openai.types.chat import (ChatCompletionContentPartParam,
                               ChatCompletionRole)
from PIL import Image
from starlette.background import BackgroundTask

from .protocal.openai_protocol import *
from .protocal.anthropic_protocol import *

try:
    from ..gemma4_multimodal import (
        normalize_gemma4_conversation,
        prepare_gemma4_multimodal_inputs,
    )
except ImportError:
    from gemma4_multimodal import (
        normalize_gemma4_conversation,
        prepare_gemma4_multimodal_inputs,
    )

ConversationContent = Union[str, List[Dict[str, Any]]]


class ConversationMessage:
    def __init__(self, role:str, content:ConversationContent, tool_calls=None, tool_call_id=None, name=None):
      self.role = role
      self.content = content
      self.tool_calls = tool_calls
      self.tool_call_id = tool_call_id
      self.name = name

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

class ChatCompletionStreamResponseWithUsage(BaseModel):
    id: str = Field(default_factory = lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory = lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default = None)

class FastLLmCompletion:
  def __init__(self,
               model_name,
               model,
               think, 
               hide_input):
    self.model_name = model_name
    self.model = model
    self.init_fast_llm_model()
    self.think = think
    self.hide_input = hide_input
    # Store mapping between conversation IDs and handles
    self.conversation_handles = {}
    self.tool_parser = None
    
  def init_fast_llm_model(self):
    pass
  
  def create_error_response(
          self,
          message: str,
          err_type: str = "BadRequestError",
          status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
      return ErrorResponse(message=message,
                            type=err_type,
                            code=status_code.value)

  def _normalize_sampling_args(self, top_p: float, top_k: int, temperature: float):
      do_sample = True
      if temperature is not None and temperature <= 0:
          do_sample = False
          temperature = 1.0
          top_k = 1
          top_p = 1.0
      return do_sample, top_p, top_k, temperature

  async def _check_model(self, request: ChatCompletionRequest):
    if request.model != self.model_name:
      return self.create_error_response(
          message=f"The model `{request.model}` does not exist.",
          err_type="NotFoundError",
          status_code=HTTPStatus.NOT_FOUND)
    else:
      return None

  def _build_message_content(
      self, parts: List[Dict[str, Any]]
  ) -> ConversationContent:
      if any(part.get("type") == "image" for part in parts):
          return parts
      text_parts = [
          part.get("text", "")
          for part in parts
          if part.get("type") == "text"
      ]
      return "\n".join([text for text in text_parts if text != ""])

  def _load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
      try:
          with Image.open(io.BytesIO(image_bytes)) as image:
              return image.convert("RGB")
      except Exception as exc:
          raise ValueError("Invalid image bytes.") from exc

  def _load_image_from_url(self, image_url: str) -> Image.Image:
      if not image_url:
          raise ValueError("Image URL cannot be empty.")

      parsed = urlparse(image_url)
      scheme = parsed.scheme.lower()
      try:
          if scheme in ("http", "https"):
              with urlopen(image_url, timeout = 20) as response:
                  return self._load_image_from_bytes(response.read())
          if scheme == "data":
              if not image_url.startswith("data:image/"):
                  raise ValueError("Only image data URLs are supported.")
              header, encoded = image_url.split(",", 1)
              if ";base64" not in header:
                  raise ValueError("Only base64-encoded image data URLs are supported.")
              try:
                  image_bytes = base64.b64decode(encoded)
              except (binascii.Error, ValueError) as exc:
                  raise ValueError("Invalid base64 image data.") from exc
              return self._load_image_from_bytes(image_bytes)
          if scheme == "file":
              file_path = unquote(parsed.path or "")
              if (os.name == "nt" and len(file_path) >= 3 and file_path[0] == "/"
                      and file_path[2] == ":"):
                  file_path = file_path[1:]
              with Image.open(file_path) as image:
                  return image.convert("RGB")
      except ValueError:
          raise
      except Exception as exc:
          raise ValueError(f"Failed to load image from {image_url!r}: {exc}") from exc

      raise ValueError(
          f"Unsupported image URL scheme in {image_url!r}. "
          "Supported schemes are http(s), data:image/... and file://."
      )

  def _extract_openai_image_url(self, part: Dict[str, Any]) -> str:
      image_url = part.get("image_url")
      if isinstance(image_url, dict):
          image_url = image_url.get("url")
      if isinstance(image_url, str) and image_url != "":
          return image_url
      raise ValueError(
          "OpenAI image content requires `image_url` to be a non-empty string "
          "or an object with a non-empty `url` field."
      )

  def _parse_openai_content_parts(
      self, content: Iterable[ChatCompletionContentPartParam]
  ) -> Tuple[ConversationContent, List[Image.Image]]:
      content_parts: List[Dict[str, Any]] = []
      images: List[Image.Image] = []
      for it in content:
          if isinstance(it, str):
              content_parts.append({"type": "text", "text": it})
              continue
          if not isinstance(it, dict):
              raise NotImplementedError("Complex input not supported yet")

          part_type = it.get("type")
          if part_type == "text":
              content_parts.append({"type": "text", "text": it.get("text", "")})
              continue

          if part_type == "image_url" or (
              part_type is None and "image_url" in it
          ):
              image_url = self._extract_openai_image_url(it)
              images.append(self._load_image_from_url(image_url))
              content_parts.append({"type": "image"})
              continue

          raise NotImplementedError(
              f"OpenAI content part type `{part_type}` is not supported yet."
          )

      return self._build_message_content(content_parts), images

  def _convert_anthropic_image_source_to_url(self, source: Dict[str, Any]) -> str:
      if not isinstance(source, dict):
          raise ValueError("Anthropic image source must be an object.")
      source_type = source.get("type")
      if source_type == "url":
          image_url = source.get("url", "")
          if not image_url:
              raise ValueError("Anthropic image source.url cannot be empty.")
          return image_url
      media_type = source.get("media_type", "image/jpeg")
      data = source.get("data", "")
      if not data:
          raise ValueError("Anthropic base64 image source.data cannot be empty.")
      return f"data:{media_type};base64,{data}"

  def _parse_anthropic_tool_result_content(
      self, content: Any
  ) -> Tuple[str, List[Image.Image]]:
      if content is None:
          return "", []
      if isinstance(content, str):
          return content, []
      if isinstance(content, list):
          text_blocks: List[str] = []
          images: List[Image.Image] = []
          for block in content:
              if not isinstance(block, dict):
                  text_blocks.append(str(block))
                  continue
              block_type = block.get("type")
              if block_type == "text":
                  text_blocks.append(block.get("text", ""))
              elif block_type == "image":
                  image_url = self._convert_anthropic_image_source_to_url(
                      block.get("source") or {}
                  )
                  images.append(self._load_image_from_url(image_url))
              else:
                  text_blocks.append(json.dumps(block, ensure_ascii = False))
          return "\n".join([text for text in text_blocks if text != ""]), images
      if isinstance(content, dict):
          return json.dumps(content, ensure_ascii = False), []
      return str(content), []

  def _compute_multimodal_input_token_len(
      self,
      messages: List[Dict[str, Any]],
      enable_thinking: bool,
      images: Optional[List[Image.Image]],
  ) -> int:
      if not images:
          return self.model.get_input_token_len(
              messages, enable_thinking = enable_thinking)

      try:
          signature = inspect.signature(self.model.get_input_token_len)
          if "images" in signature.parameters:
              return self.model.get_input_token_len(
                  messages,
                  enable_thinking = enable_thinking,
                  images = images)
      except (TypeError, ValueError):
          pass

      architecture = ""
      try:
          architecture = self.model.config["architectures"][0]
      except Exception:
          architecture = ""

      if architecture == "Gemma4ForConditionalGeneration":
          tokenizer = getattr(self.model, "hf_tokenizer", None)
          if tokenizer is None:
              raise ValueError(
                  "Gemma4 multimodal token counting needs a Hugging Face tokenizer.")
          gemma_conversation = normalize_gemma4_conversation(
              copy.deepcopy(messages), len(images))
          native_inputs = prepare_gemma4_multimodal_inputs(
              tokenizer = tokenizer,
              model_dir = (getattr(self.model, "model_path", "") or tokenizer.name_or_path),
              model_config = getattr(self.model, "config", {}),
              conversation = gemma_conversation,
              images = images,
              add_generation_prompt = True,
              enable_thinking = enable_thinking,
          )
          return len(native_inputs["input_ids"])

      return self.model.get_input_token_len(
          messages, enable_thinking = enable_thinking)

  def _parse_chat_message_content(
      self,
      role: ChatCompletionRole,
      content: Optional[Union[str,
                              Iterable[ChatCompletionContentPartParam]]],
      tool_calls=None,
      tool_call_id=None,
      name=None,
  ) -> Tuple[List[ConversationMessage], List[Image.Image]]:
      if content is None and tool_calls is None and tool_call_id is None:
          return [], []
      if content is None or isinstance(content, str):
          return [ConversationMessage(role=role, content=content or "",
                                      tool_calls=tool_calls,
                                      tool_call_id=tool_call_id,
                                      name=name)], []
      if isinstance(content, list):
          parsed_content, images = self._parse_openai_content_parts(content)
          return [ConversationMessage(role=role, content=parsed_content,
                                      tool_calls=tool_calls,
                                      tool_call_id=tool_call_id,
                                      name=name)], images
      raise NotImplementedError("Complex input not supported yet")

  def _stringify_anthropic_block_content(self, content: Any) -> str:
      if content is None:
          return ""
      if isinstance(content, str):
          return content
      if isinstance(content, list):
          parts = []
          for block in content:
              if isinstance(block, dict) and block.get("type") == "text":
                  parts.append(block.get("text", ""))
              else:
                  parts.append(json.dumps(block, ensure_ascii = False))
          return "\n".join([part for part in parts if part != ""])
      if isinstance(content, dict):
          return json.dumps(content, ensure_ascii = False)
      return str(content)

  def _serialize_tool_arguments(self, arguments: Any) -> str:
      if isinstance(arguments, str):
          return arguments
      return json.dumps(arguments, ensure_ascii = False)

  def _deserialize_tool_arguments(self, arguments: Any) -> Dict[str, Any]:
      if isinstance(arguments, dict):
          return arguments
      if isinstance(arguments, str):
          try:
              parsed = json.loads(arguments)
              if isinstance(parsed, dict):
                  return parsed
              return {"value": parsed}
          except (json.JSONDecodeError, TypeError):
              return {"raw": arguments}
      return {"value": arguments}

  def _convert_anthropic_tools(
      self, tools: Optional[List[AnthropicToolParam]]
  ) -> Tuple[Optional[List[ChatCompletionToolsParam]], Optional[List[Dict[str, Any]]]]:
      if not tools:
          return None, None

      parser_tools: List[ChatCompletionToolsParam] = []
      model_tools: List[Dict[str, Any]] = []
      for tool in tools:
          parser_tools.append(ChatCompletionToolsParam(
              function = FunctionDefinition(
                  name = tool.name,
                  description = tool.description,
                  parameters = tool.input_schema,
              )))
          model_tools.append({
              "type": "function",
              "function": {
                  "name": tool.name,
                  "description": tool.description,
                  "parameters": tool.input_schema,
              }
          })
      return parser_tools, model_tools

  def _build_anthropic_parser_request(
      self,
      request: AnthropicMessageRequest,
      messages: List[Dict[str, Any]],
      parser_tools: Optional[List[ChatCompletionToolsParam]],
  ) -> Optional[ChatCompletionRequest]:
      if not parser_tools:
          return None

      parser_messages = []
      for message in messages:
          parser_messages.append({
              "role": str(message.get("role", "user")),
              "content": self._stringify_anthropic_block_content(
                  message.get("content", "")),
          })

      return ChatCompletionRequest(
          model = request.model,
          messages = parser_messages,
          max_tokens = request.max_tokens,
          temperature = request.temperature,
          top_p = request.top_p,
          top_k = request.top_k,
          stop = request.stop_sequences,
          stream = request.stream,
          tools = parser_tools,
          tool_choice = "auto",
      )

  def _ensure_tool_parser(self):
      if self.tool_parser is None:
          tokenizer = self.model.hf_tokenizer
          chat_template = getattr(tokenizer, "chat_template", None)
          if tokenizer is None or chat_template is None:
              raise ValueError(
                  "Tool calling requires a Hugging Face tokenizer with chat_template. "
                  "Please use an HF model directory or provide the original tokenizer files."
              )
          from .tool_parsers import ToolParserManager
          self.tool_parser = ToolParserManager.get_tool_parser_auto(
              self.model.get_type(), chat_template,
              force_chat_template = self.model.force_chat_template,
              force_type = self.model.tool_call_parser)(tokenizer)

  def _parse_anthropic_message_content(
      self,
      role: str,
      content: Optional[Union[str, Iterable[Dict[str, Any]]]],
  ) -> Tuple[List[ConversationMessage], List[Image.Image]]:
      if content is None or isinstance(content, str):
          return self._parse_chat_message_content(role, content)

      if not isinstance(content, list):
          raise NotImplementedError("Complex input not supported yet")

      if role == "assistant":
          text_blocks: List[str] = []
          tool_calls: List[Dict[str, Any]] = []
          for block in content:
              if not isinstance(block, dict):
                  raise NotImplementedError("Complex input not supported yet")
              block_type = block.get("type")
              if block_type == "text":
                  text_blocks.append(block.get("text", ""))
              elif block_type == "tool_use":
                  tool_calls.append({
                      "id": block.get("id") or f"toolu_{shortuuid.random()}",
                      "type": "function",
                      "function": {
                          "name": block.get("name", ""),
                          "arguments": self._serialize_tool_arguments(
                              block.get("input", {})),
                      },
                  })
              else:
                  raise NotImplementedError("Complex input not supported yet")
          text_content = "\n".join([text for text in text_blocks if text != ""])
          return [ConversationMessage(
              role = role,
              content = text_content,
              tool_calls = tool_calls or None,
          )], []

      if role == "user":
          messages: List[ConversationMessage] = []
          all_images: List[Image.Image] = []
          pending_parts: List[Dict[str, Any]] = []
          pending_images: List[Image.Image] = []

          def flush_pending_user_parts():
              nonlocal pending_parts
              nonlocal pending_images
              if pending_parts:
                  messages.append(ConversationMessage(
                      role = "user",
                      content = self._build_message_content(pending_parts),
                  ))
                  if pending_images:
                      all_images.extend(pending_images)
                  pending_parts = []
                  pending_images = []

          for block in content:
              if not isinstance(block, dict):
                  raise NotImplementedError("Complex input not supported yet")
              block_type = block.get("type")
              if block_type == "text":
                  pending_parts.append({
                      "type": "text",
                      "text": block.get("text", ""),
                  })
              elif block_type == "image":
                  image_url = self._convert_anthropic_image_source_to_url(
                      block.get("source") or {}
                  )
                  pending_parts.append({"type": "image"})
                  pending_images.append(self._load_image_from_url(image_url))
              elif block_type == "tool_result":
                  flush_pending_user_parts()
                  tool_text, tool_images = self._parse_anthropic_tool_result_content(
                      block.get("content")
                  )
                  messages.append(ConversationMessage(
                      role = "tool",
                      content = tool_text,
                      tool_call_id = block.get("tool_use_id"),
                  ))
                  if tool_images:
                      messages.append(ConversationMessage(
                          role = "user",
                          content = [{"type": "image"} for _ in tool_images],
                      ))
                      all_images.extend(tool_images)
              else:
                  raise NotImplementedError("Complex input not supported yet")
          flush_pending_user_parts()
          return messages, all_images

      return self._parse_chat_message_content(role, content)

  def _build_anthropic_response_blocks(
      self,
      text_content: Optional[str],
      tool_calls: List[ToolCall],
  ) -> List[Union[AnthropicTextContentBlock, AnthropicToolUseContentBlock]]:
      blocks: List[Union[AnthropicTextContentBlock, AnthropicToolUseContentBlock]] = []
      if text_content:
          blocks.append(AnthropicTextContentBlock(text = text_content))
      for tool_call in tool_calls:
          blocks.append(AnthropicToolUseContentBlock(
              id = tool_call.id,
              name = tool_call.function.name,
              input = self._deserialize_tool_arguments(
                  tool_call.function.arguments),
          ))
      return blocks

  def _normalize_internal_tool_calls(self, tool_calls: Any) -> List[Any]:
      parsed_tool_calls = []
      for tc in tool_calls:
          tc_copy = dict(tc) if isinstance(tc, dict) else tc
          if isinstance(tc_copy, dict) and "function" in tc_copy:
              func = tc_copy["function"]
              if isinstance(func, dict) and isinstance(func.get("arguments"), str):
                  func = dict(func)
                  try:
                      func["arguments"] = json.loads(func["arguments"])
                  except (json.JSONDecodeError, TypeError):
                      pass
                  tc_copy = dict(tc_copy)
                  tc_copy["function"] = func
          parsed_tool_calls.append(tc_copy)
      return parsed_tool_calls

  def _get_anthropic_stop_reason(self, completion_tokens: int,
                                 max_tokens: Optional[int]) -> str:
      if max_tokens is not None and completion_tokens >= max_tokens:
          return "max_tokens"
      return "end_turn"

  def _create_anthropic_sse_event(self, event_name: str, payload: BaseModel) -> str:
      data = payload.model_dump_json(exclude_none = True)
      return f"event: {event_name}\ndata: {data}\n\n"

  async def create_anthropic_message(
      self, request: AnthropicMessageRequest, raw_request: Request
  ) -> Union[ErrorResponse, AsyncGenerator[str, None],
             AnthropicMessageResponse,
             Tuple[AsyncGenerator[str, None], AsyncGenerator]]:
      error_check_ret = await self._check_model(request)
      if error_check_ret is not None:
          return error_check_ret

      try:
          conversation: List[ConversationMessage] = []
          images: List[Image.Image] = []
          if request.system is not None:
              system_messages, system_images = self._parse_anthropic_message_content(
                  "system", request.system)
              conversation.extend(system_messages)
              images.extend(system_images)

          for m in request.messages:
              messages, message_images = self._parse_anthropic_message_content(
                  m.role, m.content)
              conversation.extend(messages)
              images.extend(message_images)

          if len(conversation) == 0:
              raise Exception("Empty msg")

          messages = []
          for msg in conversation:
              msg_dict = {
                  "role": msg.role,
                  "content": msg.content,
              }
              if msg.tool_calls is not None:
                  msg_dict["tool_calls"] = self._normalize_internal_tool_calls(
                      msg.tool_calls)
              if msg.tool_call_id is not None:
                  msg_dict["tool_call_id"] = msg.tool_call_id
              if msg.name is not None:
                  msg_dict["name"] = msg.name
              messages.append(msg_dict)
      except Exception as e:
          logging.error("Error in applying anthropic request: %s", e)
          traceback.print_exc()
          return self.create_error_response(str(e))

      request_id = f"msg_{shortuuid.random()}"
      parser_tools, model_tools = self._convert_anthropic_tools(request.tools)
      parser_request = self._build_anthropic_parser_request(
          request, messages, parser_tools)
      default_gen = self.model.default_generation_config
      top_p = request.top_p if request.top_p is not None else default_gen['top_p']
      top_k = request.top_k if request.top_k is not None else default_gen['top_k']
      temperature = request.temperature if request.temperature is not None else default_gen['temperature']
      do_sample, top_p, top_k, temperature = self._normalize_sampling_args(top_p, top_k, temperature)
      frequency_penalty = default_gen['repetition_penalty']
      max_length = request.max_tokens if request.max_tokens else 32768

      if request.stop_sequences:
          logging.warning("Anthropic stop_sequences are not supported yet and will be ignored.")

      if (not(self.hide_input)):
         logging.info(f"fastllm anthropic input message: {messages}")

      model_images = images if images else None
      input_token_len = self._compute_multimodal_input_token_len(
          messages,
          enable_thinking = self.think,
          images = model_images)
      handle = self.model.launch_stream_response(messages,
                        max_length = max_length, min_length = 0, do_sample = do_sample,
                        top_p = top_p, top_k = top_k, temperature = temperature,
                        repeat_penalty = frequency_penalty, tools = model_tools,
                        one_by_one = True, enable_thinking = self.think,
                        images = model_images)
      self.conversation_handles[request_id] = handle
      result_generator = self.model.stream_response_handle_async(handle)

      if request.stream:
          return (self.anthropic_message_stream_generator(
              request, raw_request, result_generator, request_id, input_token_len,
              parser_request),
              BackgroundTask(self.check_disconnect, raw_request, request_id, handle))
      else:
          try:
              return await self.anthropic_message_full_generator(
                  request, raw_request, handle, result_generator, request_id,
                  input_token_len, parser_request)
          except ValueError as e:
              return self.create_error_response(str(e))

  async def create_chat_completion(
      self, request: ChatCompletionRequest, raw_request: Request
  ) -> Union[ErrorResponse, AsyncGenerator[str, None],
              ChatCompletionResponse, 
              Tuple[AsyncGenerator[str, None], AsyncGenerator]]:
      """Completion API similar to OpenAI's API.

      See https://platform.openai.com/docs/api-reference/chat/create
      for the API specification. This API mimics the OpenAI
      ChatCompletion API.

      NOTE: Currently we do not support the following feature:
          - function_call (Users should implement this by themselves)
      """
      error_check_ret = await self._check_model(request)
      if error_check_ret is not None:
          return error_check_ret
      
      query:str = ""
      if request.prompt:
         request.messages.append({"role": "user", "content": request.prompt})
      try:
          # print("request", str(request))
          conversation: List[ConversationMessage] = []
          images: List[Image.Image] = []
          for m in request.messages:
              messages, message_images = self._parse_chat_message_content(
                  m["role"], m.get("content"),
                  tool_calls=m.get("tool_calls"),
                  tool_call_id=m.get("tool_call_id"),
                  name=m.get("name"))

              conversation.extend(messages)
              images.extend(message_images)

          if len(conversation) == 0:
            raise Exception("Empty msg")
          messages = []
          for msg in conversation:
            msg_dict = {"role": msg.role, "content": msg.content}
            if msg.tool_calls is not None:
                parsed_tool_calls = []
                for tc in msg.tool_calls:
                    tc_copy = dict(tc) if isinstance(tc, dict) else tc
                    if isinstance(tc_copy, dict) and "function" in tc_copy:
                        func = tc_copy["function"]
                        if isinstance(func, dict) and isinstance(func.get("arguments"), str):
                            func = dict(func)
                            try:
                                func["arguments"] = json.loads(func["arguments"])
                            except (json.JSONDecodeError, TypeError):
                                pass
                            tc_copy = dict(tc_copy)
                            tc_copy["function"] = func
                    parsed_tool_calls.append(tc_copy)
                msg_dict["tool_calls"] = parsed_tool_calls
            if msg.tool_call_id is not None:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.name is not None:
                msg_dict["name"] = msg.name
            messages.append(msg_dict)

      except Exception as e:
          logging.error("Error in applying chat template from request: %s", e)
          traceback.print_exc()
          return self.create_error_response(str(e))

      request_id = f"fastllm-{self.model_name}-{random_uuid()}"
      
      default_gen = self.model.default_generation_config
      top_p = request.top_p if request.top_p is not None else default_gen['top_p']
      top_k = request.top_k if request.top_k is not None else default_gen['top_k']
      temperature = request.temperature if request.temperature is not None else default_gen['temperature']
      do_sample, top_p, top_k, temperature = self._normalize_sampling_args(top_p, top_k, temperature)
      frequency_penalty = default_gen['repetition_penalty']
      if request.frequency_penalty and request.frequency_penalty != 0.0:
        frequency_penalty = request.frequency_penalty

      max_length = request.max_tokens if request.max_tokens else 32768
      min_length = request.min_tokens if request.min_tokens else 0

      enable_thinking = self.think
      if request.chat_template_kwargs and "enable_thinking" in request.chat_template_kwargs:
          enable_thinking = bool(request.chat_template_kwargs["enable_thinking"])

      #logging.info(request)
      if (not(self.hide_input)):
         logging.info(f"fastllm input message: {messages}")
      #logging.info(f"input tokens: {input_token_len}")

      model_images = images if images else None
      input_token_len = self._compute_multimodal_input_token_len(
          messages,
          enable_thinking = enable_thinking,
          images = model_images)

      tools = [tool.model_dump(exclude_none=True) for tool in request.tools] if request.tools is not None else None

      handle = self.model.launch_stream_response(messages,
                        max_length = max_length, min_length = min_length, do_sample = do_sample,
                        top_p = top_p, top_k = top_k, temperature = temperature,
                        repeat_penalty = frequency_penalty, tools = tools, one_by_one = True,
                        enable_thinking = enable_thinking, images = model_images)
      # Store the mapping between conversation ID and handle
      self.conversation_handles[request_id] = handle
      # logging.info(f"Created conversation: {request_id}, handle: {handle}")
      result_generator = self.model.stream_response_handle_async(handle)
      # Streaming response
      if request.stream:
          return (self.chat_completion_stream_generator(
              request, raw_request, result_generator, request_id, input_token_len, think = self.think),
              BackgroundTask(self.check_disconnect, raw_request, request_id, handle))
      else:
          try:
              return await self.chat_completion_full_generator(
                  request, raw_request, handle, result_generator, request_id, input_token_len)
          except ValueError as e:
              return self.create_error_response(str(e))

  async def check_disconnect(self, raw_request: Request, request_id, handle: int):
    # 进入BackgroundTask之后，说明流式请求已经断开了，那么这里直接abort
    self.model.abort_handle(handle)
    logging.info(f"Abort request: {request_id}")
    return
  
    while True:
      if await raw_request.is_disconnected():
        self.model.abort_handle(handle)
        logging.info(f"Abort request: {request_id}")
        return
      await asyncio.sleep(1)  # 检查间隔
      
  async def chat_completion_full_generator(
              self, request: ChatCompletionRequest, raw_request: Request,
              handle: int,
              result_generator: AsyncIterator,
              request_id: str,
              input_token_len: int) -> Union[ErrorResponse, ChatCompletionResponse]:
      model_name = self.model_name
      created_time = int(time.time())
      result = ""
      completion_tokens = 0
      async for res in result_generator:
        result += res
        completion_tokens += 1
        if await raw_request.is_disconnected():
           print("is_disconnected!!!")
           self.model.abort_handle(handle)
           logging.info(f"Abort request: {request_id}")
           return self.create_error_response("Client disconnected")

      if request.tools:
          self._ensure_tool_parser()

      if request.tools:
          tool_call_info = self.tool_parser.extract_tool_calls(result, request)
      else:
          tool_call_info = ExtractedToolCallInformation(
              tools_called=False, tool_calls=[], content=result)

      if tool_call_info.tools_called:
          choice_data = ChatCompletionResponseChoice(
              index=0,
              message=ChatMessage(
                  role="assistant",
                  content=tool_call_info.content or None,
                  tool_calls=tool_call_info.tool_calls,
              ),
              logprobs=None,
              finish_reason='tool_calls',
          )
      else:
          choice_data = ChatCompletionResponseChoice(
              index=0,
              message=ChatMessage(role="assistant", content=result),
              logprobs=None,
              finish_reason='stop',
          )

      response = ChatCompletionResponse(
          id = request_id,
          created = created_time,
          model = model_name,
          choices = [choice_data],
          usage = UsageInfo(prompt_tokens = input_token_len,
                            total_tokens = input_token_len + completion_tokens,
                            completion_tokens = completion_tokens)
      )

      # After completion, remove the conversation from tracking dictionary
      if request_id in self.conversation_handles:
          del self.conversation_handles[request_id]
          # logging.info(f"Removed completed conversation from tracking: {request_id}")

      return response
      
            
  async def chat_completion_stream_generator(
          self, request: ChatCompletionRequest, raw_request: Request,
          result_generator: AsyncIterator,
          request_id: str,
          input_token_len: int, think: bool) -> AsyncGenerator[str, None]:
      model_name = self.model_name
      created_time = int(time.time())
      chunk_object_type = "chat.completion.chunk"

      # TODO: 支持request.n 和 request.echo配置
      first_iteration = True
      try:
        if first_iteration:
            # 1. role部分
            choice_data = ChatCompletionResponseStreamChoice(
                            index = 0,
                            delta = DeltaMessage(role = "assistant"),
                            logprobs = None,
                            finish_reason = None)
            chunk = ChatCompletionStreamResponseWithUsage(
                id = request_id,
                object = chunk_object_type,
                created = created_time,
                choices = [choice_data],
                model = model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
            first_iteration = False

        # 新增：发送<think>标签
        has_sent_label = False
        if not has_sent_label and think:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content="<think>\n"),
                logprobs=None,
                finish_reason=None
            )
            chunk = ChatCompletionStreamResponseWithUsage(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[choice_data],
                model=model_name
            )
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"  # 发送标签块
            has_sent_label = True      # 标记已发送标签

        # 2. content部分

        if request.tools:
            self._ensure_tool_parser()
        
        completion_tokens = 0

        previous_token_ids = []
        current_token_ids = []
        previous_text = ""
        current_text = ""

        async for res in result_generator:
            if (res == "[unused16]"):
                res = "<think>"
            elif (res == "[unused17]"):
                res = "</think>" 
            completion_tokens += 1
            delta_text = res

            # print("delta_text", delta_text)

            # Send token-by-token response for each request.n
            if self.tool_parser and request.tools: 
                now_ids = self.tool_parser.get_token_ids(delta_text)
                # print("delta_text", delta_text, "now_ids", now_ids)

                current_text += delta_text
                current_token_ids += now_ids

                delta_message = self.tool_parser.extract_tool_calls_streaming(
                                previous_text = previous_text,
                                current_text = current_text,
                                delta_text = delta_text,
                                previous_token_ids = previous_token_ids,
                                current_token_ids = current_token_ids,
                                delta_token_ids = [0],
                                request = request)

                previous_text += delta_text
                previous_token_ids += now_ids
                # print("delta", delta_message)
            else:
                delta_message = DeltaMessage(content = delta_text)

            if (delta_message):
                choice_data = ChatCompletionResponseStreamChoice(
                    index = 0,
                    #delta = DeltaMessage(content = delta_text),
                    delta = delta_message,
                    logprobs = None,
                    finish_reason = None)
                chunk = ChatCompletionStreamResponseWithUsage(
                    id = request_id,
                    object = chunk_object_type,
                    created = created_time,
                    choices = [choice_data],
                    model = model_name)
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"
            #await asyncio.sleep(0)

        # 3. 结束标志
        choice_data = ChatCompletionResponseStreamChoice(
            index = 0,
            delta = DeltaMessage(),
            logprobs = None,
            finish_reason = 'stop')
        chunk = ChatCompletionStreamResponseWithUsage(
            id = request_id,
            object = chunk_object_type,
            created = created_time,
            choices = [choice_data],
            model = model_name,
            usage = UsageInfo(prompt_tokens = input_token_len,
                              total_tokens = input_token_len + completion_tokens,
                              completion_tokens = completion_tokens))
        data = chunk.model_dump_json(exclude_unset = True,
                                    exclude_none = True)
        yield f"data: {data}\n\n"
      except ValueError as e:
        data = self.create_streaming_error_response(str(e))
        yield f"data: {data}\n\n"
        await asyncio.sleep(0)
      
      # After completion, remove the conversation from tracking dictionary
      if request_id in self.conversation_handles:
          del self.conversation_handles[request_id]
          # logging.info(f"Removed completed stream conversation from tracking: {request_id}")
      
      yield "data: [DONE]\n\n"
      await asyncio.sleep(0)

  async def anthropic_message_full_generator(
              self, request: AnthropicMessageRequest, raw_request: Request,
              handle: int,
              result_generator: AsyncIterator,
              request_id: str,
              input_token_len: int,
              parser_request: Optional[ChatCompletionRequest]) -> Union[ErrorResponse, AnthropicMessageResponse]:
      model_name = self.model_name
      result = ""
      completion_tokens = 0
      async for res in result_generator:
        if (res == "[unused16]"):
            res = "<think>"
        elif (res == "[unused17]"):
            res = "</think>"
        result += res
        completion_tokens += 1
        if await raw_request.is_disconnected():
           print("is_disconnected!!!")
           self.model.abort_handle(handle)
           logging.info(f"Abort request: {request_id}")
           return self.create_error_response("Client disconnected")

      if request.tools and parser_request is not None:
          self._ensure_tool_parser()
          tool_call_info = self.tool_parser.extract_tool_calls(result, parser_request)
      else:
          tool_call_info = ExtractedToolCallInformation(
              tools_called = False, tool_calls = [], content = result)

      response = AnthropicMessageResponse(
          id = request_id,
          content = self._build_anthropic_response_blocks(
              tool_call_info.content, tool_call_info.tool_calls),
          model = model_name,
          stop_reason = ("tool_use" if tool_call_info.tools_called
                         else self._get_anthropic_stop_reason(
                             completion_tokens, request.max_tokens)),
          usage = AnthropicUsage(input_tokens = input_token_len,
                                 output_tokens = completion_tokens)
      )

      if request_id in self.conversation_handles:
          del self.conversation_handles[request_id]

      return response

  async def anthropic_message_stream_generator(
          self, request: AnthropicMessageRequest, raw_request: Request,
          result_generator: AsyncIterator,
          request_id: str,
          input_token_len: int,
          parser_request: Optional[ChatCompletionRequest]) -> AsyncGenerator[str, None]:
      model_name = self.model_name
      completion_tokens = 0
      next_content_index = 0
      text_block_index = None
      tool_blocks: Dict[int, Dict[str, Any]] = {}
      emitted_tool_use = False

      previous_token_ids = []
      current_token_ids = []
      previous_text = ""
      current_text = ""

      try:
          message = AnthropicMessageResponse(
              id = request_id,
              content = [],
              model = model_name,
              stop_reason = None,
              stop_sequence = None,
              usage = AnthropicUsage(input_tokens = input_token_len,
                                     output_tokens = 0),
          )
          yield self._create_anthropic_sse_event(
              "message_start", MessageStartEvent(message = message))

          if request.tools and parser_request is not None:
              self._ensure_tool_parser()

          async for res in result_generator:
              if (res == "[unused16]"):
                  res = "<think>"
              elif (res == "[unused17]"):
                  res = "</think>"
              completion_tokens += 1
              delta_text = res

              if request.tools and parser_request is not None and self.tool_parser:
                  now_ids = self.tool_parser.get_token_ids(delta_text)
                  current_text += delta_text
                  current_token_ids += now_ids

                  delta_message = self.tool_parser.extract_tool_calls_streaming(
                                  previous_text = previous_text,
                                  current_text = current_text,
                                  delta_text = delta_text,
                                  previous_token_ids = previous_token_ids,
                                  current_token_ids = current_token_ids,
                                  delta_token_ids = [0],
                                  request = parser_request)

                  previous_text += delta_text
                  previous_token_ids += now_ids
              else:
                  delta_message = DeltaMessage(content = delta_text)

              if not delta_message:
                  continue

              if delta_message.content:
                  if tool_blocks:
                      for tool_state in sorted(
                              tool_blocks.values(),
                              key = lambda item: item["content_index"]):
                          yield self._create_anthropic_sse_event(
                              "content_block_stop",
                              ContentBlockStopEvent(index = tool_state["content_index"]))
                      tool_blocks = {}

                  if text_block_index is None:
                      text_block_index = next_content_index
                      next_content_index += 1
                      yield self._create_anthropic_sse_event(
                          "content_block_start",
                          ContentBlockStartEvent(
                              index = text_block_index,
                              content_block = AnthropicTextContentBlock(text = ""),
                          ))
                  yield self._create_anthropic_sse_event(
                      "content_block_delta",
                      ContentBlockDeltaEvent(
                          index = text_block_index,
                          delta = AnthropicTextDelta(text = delta_message.content),
                      ))

              for tool_delta in delta_message.tool_calls:
                  emitted_tool_use = True
                  if text_block_index is not None:
                      yield self._create_anthropic_sse_event(
                          "content_block_stop",
                          ContentBlockStopEvent(index = text_block_index))
                      text_block_index = None

                  tool_state = tool_blocks.get(tool_delta.index)
                  if tool_state is None:
                      content_index = next_content_index
                      next_content_index += 1
                      tool_state = {
                          "content_index": content_index,
                          "id": tool_delta.id or f"toolu_{shortuuid.random()}",
                          "name": "",
                      }
                      if (tool_delta.function is not None
                              and tool_delta.function.name is not None):
                          tool_state["name"] = tool_delta.function.name
                      tool_blocks[tool_delta.index] = tool_state
                      yield self._create_anthropic_sse_event(
                          "content_block_start",
                          ContentBlockStartEvent(
                              index = content_index,
                              content_block = AnthropicToolUseContentBlock(
                                  id = tool_state["id"],
                                  name = tool_state["name"],
                                  input = {},
                              ),
                          ))

                  if (tool_delta.function is not None
                          and tool_delta.function.name is not None
                          and tool_state["name"] == ""):
                      tool_state["name"] = tool_delta.function.name

                  if (tool_delta.function is not None
                          and tool_delta.function.arguments is not None):
                      yield self._create_anthropic_sse_event(
                          "content_block_delta",
                          ContentBlockDeltaEvent(
                              index = tool_state["content_index"],
                              delta = AnthropicInputJsonDelta(
                                  partial_json = tool_delta.function.arguments),
                          ))

          if text_block_index is not None:
              yield self._create_anthropic_sse_event(
                  "content_block_stop",
                  ContentBlockStopEvent(index = text_block_index))
          for tool_state in sorted(
                  tool_blocks.values(), key = lambda item: item["content_index"]):
              yield self._create_anthropic_sse_event(
                  "content_block_stop",
                  ContentBlockStopEvent(index = tool_state["content_index"]))
          yield self._create_anthropic_sse_event(
              "message_delta",
              MessageDeltaEvent(
                  delta = AnthropicMessageDelta(
                      stop_reason = ("tool_use" if emitted_tool_use
                                     else self._get_anthropic_stop_reason(
                                         completion_tokens, request.max_tokens)),
                      stop_sequence = None,
                  ),
                  usage = AnthropicUsage(input_tokens = input_token_len,
                                         output_tokens = completion_tokens),
              ))
          yield self._create_anthropic_sse_event(
              "message_stop",
              MessageStopEvent())
      except ValueError as e:
          error_data = json.dumps({
              "type": "error",
              "error": {
                  "type": "invalid_request_error",
                  "message": str(e),
              }
          })
          yield f"event: error\ndata: {error_data}\n\n"
          await asyncio.sleep(0)

      if request_id in self.conversation_handles:
          del self.conversation_handles[request_id]

      await asyncio.sleep(0)
      
  def create_streaming_error_response(
          self,
          message: str,
          err_type: str = "BadRequestError",
          status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
      json_str = json.dumps({
          "error":
          self.create_error_response(message=message,
                                      err_type=err_type,
                                      status_code=status_code).model_dump()
      })
      return json_str

  def abort_conversation(self, conversation_id: str) -> bool:
    if conversation_id in self.conversation_handles:
      handle = self.conversation_handles[conversation_id]
      try:
        self.model.abort_handle(handle)
        logging.info(f"Aborted conversation: {conversation_id}, handle: {handle}")
        # Remove the conversation from the mapping
        del self.conversation_handles[conversation_id]
        return True
      except Exception as e:
        logging.error(f"Error aborting conversation {conversation_id}: {e}")
        return False
    else:
      logging.warning(f"Conversation ID not found: {conversation_id}")
      return False
      
  def get_active_conversations(self) -> List[Dict[str, Any]]:
    result = []
    for conversation_id, handle in self.conversation_handles.items():
      result.append({
        "conversation_id": conversation_id,
        "handle": handle
      })
    return result
