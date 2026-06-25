import asyncio
import base64
import binascii
import copy
import inspect
import io
import json
import logging
import os
import tempfile
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

try:
    from ..qwen35_multimodal_native import (
        normalize_qwen35_conversation,
        prepare_qwen35_multimodal_inputs,
    )
except ImportError:
    from qwen35_multimodal_native import (
        normalize_qwen35_conversation,
        prepare_qwen35_multimodal_inputs,
    )

try:
    from ..step3p7_multimodal_native import (
        normalize_step3p7_conversation,
        prepare_step3p7_multimodal_inputs,
    )
except ImportError:
    from step3p7_multimodal_native import (
        normalize_step3p7_conversation,
        prepare_step3p7_multimodal_inputs,
    )

ConversationContent = Union[str, List[Dict[str, Any]]]


class ConversationMessage:
    def __init__(self, role:str, content:ConversationContent, tool_calls=None, tool_call_id=None, name=None):
      self.role = role
      self.content = content
      self.tool_calls = tool_calls
      self.tool_call_id = tool_call_id
      self.name = name


class LoadedMedia:
    def __init__(self):
        self.images: List[Image.Image] = []
        self.videos: List[Any] = []
        self.temp_paths: List[str] = []

    def extend(self, other: "LoadedMedia"):
        self.images.extend(other.images)
        self.videos.extend(other.videos)
        self.temp_paths.extend(other.temp_paths)

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

class _EmptyToolTokenizer:
    def get_vocab(self):
        return {}

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
               hide_input,
               enable_thinking = None):
    self.model_name = model_name
    self.model = model
    self.init_fast_llm_model()
    self.think = think
    if enable_thinking is None:
        enable_thinking = getattr(model, "enable_thinking", True)
    self.enable_thinking = enable_thinking
    self.hide_input = hide_input
    # Store mapping between conversation IDs and handles
    self.conversation_handles = {}
    
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

  def _normalize_stop_strings(
      self, stop: Optional[Union[str, List[str]]]
  ) -> List[str]:
      if stop is None:
          return []
      if isinstance(stop, str):
          stops = [stop]
      else:
          stops = stop
      return [item for item in stops if isinstance(item, str) and item != ""]

  def _stop_token_ids_from_strings(self, stop_strings: List[str]) -> Optional[List[int]]:
      stop_token_ids: List[int] = []
      for stop in stop_strings:
          try:
              token_ids = self.model.encode(stop)
          except Exception:
              token_ids = []
          if len(token_ids) == 1 and token_ids[0] not in stop_token_ids:
              stop_token_ids.append(token_ids[0])
      return stop_token_ids or None

  def _truncate_at_stop(
      self, text: str, stop_strings: List[str]
  ) -> Tuple[str, bool]:
      if not stop_strings:
          return text, False
      stop_pos = -1
      for stop in stop_strings:
          pos = text.find(stop)
          if pos >= 0 and (stop_pos < 0 or pos < stop_pos):
              stop_pos = pos
      if stop_pos < 0:
          return text, False
      return text[:stop_pos], True

  def _filter_stop_delta(
      self,
      delta_text: str,
      stop_strings: List[str],
      state: Dict[str, Any],
  ) -> Tuple[str, bool]:
      if not stop_strings:
          return delta_text, False

      buffer = state.get("buffer", "") + delta_text
      stop_pos = -1
      for stop in stop_strings:
          pos = buffer.find(stop)
          if pos >= 0 and (stop_pos < 0 or pos < stop_pos):
              stop_pos = pos
      if stop_pos >= 0:
          state["buffer"] = ""
          return buffer[:stop_pos], True

      keep_len = max(len(stop) for stop in stop_strings) - 1
      if keep_len <= 0:
          state["buffer"] = ""
          return buffer, False

      emit_len = max(0, len(buffer) - keep_len)
      state["buffer"] = buffer[emit_len:]
      return buffer[:emit_len], False

  def _flush_stop_buffer(self, stop_strings: List[str], state: Dict[str, Any]) -> str:
      if not stop_strings:
          return ""
      buffer = state.get("buffer", "")
      state["buffer"] = ""
      return buffer

  def _chat_finish_reason(
      self,
      completion_tokens: int,
      max_length: Optional[int],
      stopped_by_stop_string: bool = False,
  ) -> str:
      if (not stopped_by_stop_string and max_length is not None
              and completion_tokens >= max_length):
          return "length"
      return "stop"

  def _is_deepseek_v4_model(self) -> bool:
      try:
          is_deepseek_v4 = self.model._is_deepseek_v4()
      except Exception:
          is_deepseek_v4 = False
      return is_deepseek_v4 and not getattr(self.model, "force_chat_template", False)

  def _is_deepseek_v4_reasoning_response(self, enable_thinking: bool) -> bool:
      return enable_thinking and self._is_deepseek_v4_model()

  def _normalize_model_delta(self, text: str) -> str:
      if text == "[unused16]":
          return "<think>"
      if text == "[unused17]":
          return "</think>"
      return text

  def _strip_optional_think_start(self, text: str) -> str:
      if text.startswith("<think>"):
          text = text[len("<think>"):]
          if text.startswith("\n"):
              text = text[1:]
      return text

  def _partial_tag_overlap(self, text: str, tag: str) -> int:
      max_len = min(len(text), len(tag) - 1)
      for length in range(max_len, 0, -1):
          if text.endswith(tag[:length]):
              return length
      return 0

  def _split_deepseek_v4_reasoning(
      self,
      result: str,
      emit_reasoning_content: bool,
  ) -> Tuple[str, Optional[str]]:
      if not emit_reasoning_content:
          return result, None
      result = self._strip_optional_think_start(result)
      think_end = "</think>"
      think_end_idx = result.find(think_end)
      if think_end_idx < 0:
          return result, None
      reasoning_content = result[:think_end_idx]
      content = result[think_end_idx + len(think_end):]
      return content, reasoning_content

  def _consume_deepseek_v4_reasoning_delta(
      self,
      delta_text: str,
      state: Dict[str, Any],
  ) -> Tuple[List[DeltaMessage], str]:
      if not state.get("active"):
          return [], delta_text

      think_end = "</think>"
      state["buffer"] = state.get("buffer", "") + delta_text
      if not state.get("started"):
          state["buffer"] = self._strip_optional_think_start(state["buffer"])
          state["started"] = True

      think_end_idx = state["buffer"].find(think_end)
      if think_end_idx >= 0:
          reasoning_delta = state["buffer"][:think_end_idx]
          content_delta = state["buffer"][think_end_idx + len(think_end):]
          state["buffer"] = ""
          state["active"] = False
          messages = []
          if reasoning_delta:
              messages.append(DeltaMessage(reasoning_content=reasoning_delta))
          return messages, content_delta

      overlap = self._partial_tag_overlap(state["buffer"], think_end)
      if overlap:
          reasoning_delta = state["buffer"][:-overlap]
          state["buffer"] = state["buffer"][-overlap:]
      else:
          reasoning_delta = state["buffer"]
          state["buffer"] = ""

      if reasoning_delta:
          return [DeltaMessage(reasoning_content=reasoning_delta)], ""
      return [], ""

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
      if any(part.get("type") in {"image", "video"} for part in parts):
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

  def _extract_openai_video_url(self, part: Dict[str, Any]) -> str:
      video_url = part.get("video_url")
      if isinstance(video_url, dict):
          video_url = video_url.get("url")
      if isinstance(video_url, str) and video_url != "":
          return video_url
      raise ValueError(
          "OpenAI video content requires `video_url` to be a non-empty string "
          "or an object with a non-empty `url` field."
      )

  def _guess_media_suffix(
      self,
      path: str,
      media_type: Optional[str],
      default_suffix: str,
  ) -> str:
      suffix = os.path.splitext(path or "")[1]
      if suffix:
          return suffix
      media_to_suffix = {
          "image/gif": ".gif",
          "video/mp4": ".mp4",
          "video/webm": ".webm",
          "video/quicktime": ".mov",
          "video/x-matroska": ".mkv",
          "video/avi": ".avi",
      }
      return media_to_suffix.get((media_type or "").lower(), default_suffix)

  def _write_temp_media_file(self, media_bytes: bytes, suffix: str) -> str:
      with tempfile.NamedTemporaryFile(delete = False, suffix = suffix) as tmp:
          tmp.write(media_bytes)
          return tmp.name

  def _load_video_from_url(self, video_url: str) -> Tuple[Any, Optional[str]]:
      if not video_url:
          raise ValueError("Video URL cannot be empty.")

      parsed = urlparse(video_url)
      scheme = parsed.scheme.lower()
      try:
          if scheme in ("http", "https"):
              with urlopen(video_url, timeout = 20) as response:
                  media_bytes = response.read()
                  media_type = None
                  headers = getattr(response, "headers", None)
                  if headers is not None and hasattr(headers, "get_content_type"):
                      media_type = headers.get_content_type()
              suffix = self._guess_media_suffix(parsed.path, media_type, ".mp4")
              temp_path = self._write_temp_media_file(media_bytes, suffix)
              return temp_path, temp_path
          if scheme == "data":
              if not video_url.startswith("data:"):
                  raise ValueError("Only data URLs are supported.")
              header, encoded = video_url.split(",", 1)
              if ";base64" not in header:
                  raise ValueError("Only base64-encoded video data URLs are supported.")
              media_type = header[5:].split(";", 1)[0].lower()
              if not (media_type.startswith("video/") or media_type == "image/gif"):
                  raise ValueError("Only video/* or image/gif data URLs are supported for video content.")
              try:
                  media_bytes = base64.b64decode(encoded)
              except (binascii.Error, ValueError) as exc:
                  raise ValueError("Invalid base64 video data.") from exc
              suffix = self._guess_media_suffix("", media_type, ".mp4")
              temp_path = self._write_temp_media_file(media_bytes, suffix)
              return temp_path, temp_path
          if scheme == "file":
              file_path = unquote(parsed.path or "")
              if (os.name == "nt" and len(file_path) >= 3 and file_path[0] == "/"
                      and file_path[2] == ":"):
                  file_path = file_path[1:]
              return file_path, None
      except ValueError:
          raise
      except Exception as exc:
          raise ValueError(f"Failed to load video from {video_url!r}: {exc}") from exc

      raise ValueError(
          f"Unsupported video URL scheme in {video_url!r}. "
          "Supported schemes are http(s), data:video/... / data:image/gif, and file://."
      )

  def _cleanup_temp_paths(self, paths: Iterable[str]):
      for path in sorted(set(paths)):
          if not path:
              continue
          try:
              os.remove(path)
          except OSError:
              pass

  def _parse_openai_content_parts(
      self, content: Iterable[ChatCompletionContentPartParam]
  ) -> Tuple[ConversationContent, LoadedMedia]:
      content_parts: List[Dict[str, Any]] = []
      media = LoadedMedia()
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
              media.images.append(self._load_image_from_url(image_url))
              content_parts.append({"type": "image"})
              continue

          if part_type == "video_url" or (
              part_type is None and "video_url" in it
          ):
              video_url = self._extract_openai_video_url(it)
              video, temp_path = self._load_video_from_url(video_url)
              media.videos.append(video)
              if temp_path is not None:
                  media.temp_paths.append(temp_path)
              content_parts.append({"type": "video"})
              continue

          raise NotImplementedError(
              f"OpenAI content part type `{part_type}` is not supported yet."
          )

      return self._build_message_content(content_parts), media

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
  ) -> Tuple[str, LoadedMedia]:
      media = LoadedMedia()
      if content is None:
          return "", media
      if isinstance(content, str):
          return content, media
      if isinstance(content, list):
          text_blocks: List[str] = []
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
                  media.images.append(self._load_image_from_url(image_url))
              else:
                  text_blocks.append(json.dumps(block, ensure_ascii = False))
          return "\n".join([text for text in text_blocks if text != ""]), media
      if isinstance(content, dict):
          return json.dumps(content, ensure_ascii = False), media
      return str(content), media

  def _compute_multimodal_input_token_len(
      self,
      messages: List[Dict[str, Any]],
      enable_thinking: bool,
      images: Optional[List[Image.Image]],
      videos: Optional[List[Any]],
      tools: Optional[List[Dict[str, Any]]] = None,
  ) -> int:
      token_len_messages = messages
      if tools and self._is_deepseek_v4_model():
          token_len_messages = self.model._inject_deepseek_v4_tools(messages, tools)
      if not images and not videos:
          return self.model.get_input_token_len(
              token_len_messages, enable_thinking = enable_thinking)

      try:
          signature = inspect.signature(self.model.get_input_token_len)
          if "images" in signature.parameters or "videos" in signature.parameters:
              kwargs = {"enable_thinking": enable_thinking}
              if "images" in signature.parameters:
                  kwargs["images"] = images
              if "videos" in signature.parameters:
                  kwargs["videos"] = videos
              return self.model.get_input_token_len(token_len_messages, **kwargs)
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
              copy.deepcopy(token_len_messages), len(images))
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
      if architecture == "Qwen3_5ForConditionalGeneration":
          tokenizer = getattr(self.model, "hf_tokenizer", None)
          qwen_conversation = normalize_qwen35_conversation(
              copy.deepcopy(token_len_messages), len(images or []), len(videos or []))
          model_dir = getattr(self.model, "model_path", "") or (tokenizer.name_or_path if tokenizer is not None else "")
          native_inputs = prepare_qwen35_multimodal_inputs(
              tokenizer = tokenizer,
              model_dir = model_dir,
              model_config = getattr(self.model, "config", {}),
              conversation = qwen_conversation,
              images = images,
              videos = videos,
              add_generation_prompt = True,
              enable_thinking = enable_thinking,
              encode_vision = False,
              encode_fn = self.model.encode,
          )
          return len(native_inputs["input_ids"])
      if architecture == "Step3p7ForConditionalGeneration":
          tokenizer = getattr(self.model, "hf_tokenizer", None)
          step_conversation = normalize_step3p7_conversation(
              copy.deepcopy(token_len_messages), len(images or []), len(videos or []))
          model_dir = getattr(self.model, "model_path", "") or (tokenizer.name_or_path if tokenizer is not None else "")
          native_inputs = prepare_step3p7_multimodal_inputs(
              tokenizer = tokenizer,
              model_dir = model_dir,
              model_config = getattr(self.model, "config", {}),
              conversation = step_conversation,
              images = images,
              videos = videos,
              add_generation_prompt = True,
              enable_thinking = enable_thinking,
              encode_fn = self.model.encode,
          )
          return len(native_inputs["input_ids"])

      return self.model.get_input_token_len(
          token_len_messages, enable_thinking = enable_thinking)

  def _parse_chat_message_content(
      self,
      role: ChatCompletionRole,
      content: Optional[Union[str,
                              Iterable[ChatCompletionContentPartParam]]],
      tool_calls=None,
      tool_call_id=None,
      name=None,
  ) -> Tuple[List[ConversationMessage], LoadedMedia]:
      empty_media = LoadedMedia()
      if content is None and tool_calls is None and tool_call_id is None:
          return [], empty_media
      if content is None or isinstance(content, str):
          return [ConversationMessage(role=role, content=content or "",
                                      tool_calls=tool_calls,
                                      tool_call_id=tool_call_id,
                                      name=name)], empty_media
      if isinstance(content, list):
          parsed_content, media = self._parse_openai_content_parts(content)
          return [ConversationMessage(role=role, content=parsed_content,
                                      tool_calls=tool_calls,
                                      tool_call_id=tool_call_id,
                                      name=name)], media
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

  def _create_tool_parser(self):
      tokenizer = getattr(self.model, "hf_tokenizer", None)
      model_type = self.model.get_type()
      chat_template = getattr(tokenizer, "chat_template", None)
      force_type = getattr(self.model, "tool_call_parser", "auto")
      allow_without_chat_template = (
          model_type == "deepseek_v4" or
          force_type in ("deepseek_v4",)
      )
      if tokenizer is None and allow_without_chat_template:
          tokenizer = _EmptyToolTokenizer()
      if tokenizer is None or (chat_template is None and not allow_without_chat_template):
          raise ValueError(
              "Tool calling requires a Hugging Face tokenizer with chat_template. "
              "Please use an HF model directory or provide the original tokenizer files."
          )
      from .tool_parsers import ToolParserManager
      return ToolParserManager.get_tool_parser_auto(
          model_type, chat_template,
          force_chat_template = self.model.force_chat_template,
          force_type = force_type)(tokenizer)

  def _create_function_call_parser(self, request: ChatCompletionRequest):
      from .toolcall_parser import FunctionCallParser
      return FunctionCallParser.from_request(
          request,
          parser = self._create_tool_parser(),
      )

  def _build_tool_call_constraint_descriptor(self,
                                             request: ChatCompletionRequest):
      if not request.tools:
          return None
      from .toolcall_parser import FunctionCallParser
      force_type = getattr(self.model, "tool_call_parser", "auto")
      model_type = self.model.get_type()
      tool_parser_name = force_type if force_type != "auto" else model_type
      return FunctionCallParser.build_constraint_descriptor_from_request(
          request,
          tool_parser_name = tool_parser_name,
      )

  def _build_tool_call_generation_constraint(
      self,
      request: ChatCompletionRequest,
  ) -> Optional[Dict[str, Any]]:
      descriptor = self._build_tool_call_constraint_descriptor(request)
      if descriptor is None:
          return None
      from .toolcall_constraints import compile_tool_call_constraint
      spec = compile_tool_call_constraint(descriptor)
      return spec.to_dict() if spec is not None else None

  def _model_supports_tool_call_constraint(self) -> bool:
      launch_fn = getattr(self.model, "launch_stream_response", None)
      if launch_fn is None:
          return False
      try:
          parameters = inspect.signature(launch_fn).parameters
      except (TypeError, ValueError):
          return False
      if "tool_call_constraint" in parameters:
          return True
      return any(
          parameter.kind == inspect.Parameter.VAR_KEYWORD
          for parameter in parameters.values()
      )

  def _attach_tool_call_constraint_if_supported(
      self,
      launch_kwargs: Dict[str, Any],
      request: ChatCompletionRequest,
  ) -> Optional[Dict[str, Any]]:
      constraint = self._build_tool_call_generation_constraint(request)
      if constraint is None:
          return None
      if self._model_supports_tool_call_constraint():
          launch_kwargs["tool_call_constraint"] = constraint
      else:
          logging.debug(
              "Tool call generation constraint prepared but backend does not "
              "support tool_call_constraint; continuing without backend "
              "constraint.")
      return constraint

  def _format_tool_call_diagnostics(self, diagnostics: Iterable[Any]) -> str:
      parts = []
      for diagnostic in diagnostics:
          code = getattr(diagnostic, "code", "invalid_tool_call")
          tool_name = getattr(diagnostic, "tool_name", None)
          index = getattr(diagnostic, "index", None)
          if tool_name is not None:
              message = f"{code} at index {index}: {tool_name!r}"
          else:
              message = f"{code} at index {index}"
          detail = getattr(diagnostic, "message", None)
          if detail:
              message += f" ({detail})"
          closest_tool_name = getattr(diagnostic, "closest_tool_name", None)
          similarity_ratio = getattr(diagnostic, "similarity_ratio", None)
          if closest_tool_name is not None and similarity_ratio is not None:
              message += (
                  f" closest={closest_tool_name!r} "
                  f"ratio={similarity_ratio:.3f}"
              )
          parts.append(message)
      return "; ".join(parts) or "invalid_tool_call"

  def _parse_non_stream_tool_calls(
      self,
      result: str,
      request: ChatCompletionRequest,
  ) -> Union[ErrorResponse, ExtractedToolCallInformation]:
      if not request.tools:
          return ExtractedToolCallInformation(
              tools_called = False,
              tool_calls = [],
              content = result,
          )

      parser = self._create_function_call_parser(request)
      parsed = parser.parse_non_stream(result)
      if parsed.has_invalid_tool_block:
          diagnostics = self._format_tool_call_diagnostics(parsed.diagnostics)
          logging.warning("Invalid non-stream tool call rejected: %s",
                          diagnostics)
          return self.create_error_response(
              f"Invalid tool call: {diagnostics}",
              err_type = "invalid_tool_call",
          )

      return ExtractedToolCallInformation(
          tools_called = parsed.tools_called,
          tool_calls = parsed.valid_tool_calls,
          content = parsed.content,
      )

  def _parse_anthropic_message_content(
      self,
      role: str,
      content: Optional[Union[str, Iterable[Dict[str, Any]]]],
  ) -> Tuple[List[ConversationMessage], LoadedMedia]:
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
          )], LoadedMedia()

      if role == "user":
          messages: List[ConversationMessage] = []
          all_media = LoadedMedia()
          pending_parts: List[Dict[str, Any]] = []
          pending_media = LoadedMedia()

          def flush_pending_user_parts():
              nonlocal pending_parts
              nonlocal pending_media
              if pending_parts:
                  messages.append(ConversationMessage(
                      role = "user",
                      content = self._build_message_content(pending_parts),
                  ))
                  all_media.extend(pending_media)
                  pending_parts = []
                  pending_media = LoadedMedia()

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
                  pending_media.images.append(self._load_image_from_url(image_url))
              elif block_type == "tool_result":
                  flush_pending_user_parts()
                  tool_text, tool_media = self._parse_anthropic_tool_result_content(
                      block.get("content")
                  )
                  messages.append(ConversationMessage(
                      role = "tool",
                      content = tool_text,
                      tool_call_id = block.get("tool_use_id"),
                  ))
                  if tool_media.images:
                      messages.append(ConversationMessage(
                          role = "user",
                          content = [{"type": "image"} for _ in tool_media.images],
                      ))
                      all_media.extend(tool_media)
              else:
                  raise NotImplementedError("Complex input not supported yet")
          flush_pending_user_parts()
          return messages, all_media

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

  def _stringify_responses_tool_output(self, output: Any) -> str:
      if output is None:
          return ""
      if isinstance(output, str):
          return output
      if isinstance(output, list):
          parts = []
          for item in output:
              if isinstance(item, dict) and item.get("type") in (
                      "input_text", "output_text", "text"):
                  parts.append(item.get("text", ""))
              else:
                  parts.append(json.dumps(item, ensure_ascii = False))
          return "\n".join([part for part in parts if part != ""])
      if isinstance(output, dict):
          return json.dumps(output, ensure_ascii = False)
      return str(output)

  def _convert_responses_content_to_chat_content(self, content: Any) -> ConversationContent:
      if content is None:
          return ""
      if isinstance(content, str):
          return content
      if isinstance(content, dict):
          if content.get("type") in ("input_text", "output_text", "text"):
              return content.get("text", "")
          return self._stringify_responses_tool_output(content)
      if not isinstance(content, list):
          return str(content)

      parts: List[Dict[str, Any]] = []
      has_media = False
      for part in content:
          if isinstance(part, str):
              parts.append({"type": "text", "text": part})
              continue
          if not isinstance(part, dict):
              parts.append({"type": "text", "text": str(part)})
              continue

          part_type = part.get("type")
          if part_type in ("input_text", "output_text", "text"):
              parts.append({"type": "text", "text": part.get("text", "")})
          elif part_type in ("input_image", "image_url"):
              image_url = part.get("image_url") or part.get("url")
              if isinstance(image_url, dict):
                  image_url = image_url.get("url")
              if image_url:
                  parts.append({
                      "type": "image_url",
                      "image_url": {"url": image_url},
                  })
                  has_media = True
          elif part_type in ("input_file", "file"):
              parts.append({
                  "type": "text",
                  "text": json.dumps(part, ensure_ascii = False),
              })
          else:
              parts.append({
                  "type": "text",
                  "text": json.dumps(part, ensure_ascii = False),
              })

      if has_media:
          return parts
      return "\n".join([
          part.get("text", "")
          for part in parts
          if part.get("type") == "text" and part.get("text", "") != ""
      ])

  def _append_responses_instruction_messages(
      self,
      messages: List[Dict[str, Any]],
      instructions: Any,
  ):
      if instructions is None:
          return
      if isinstance(instructions, list):
          for item in instructions:
              if isinstance(item, dict):
                  role = item.get("role", "system")
                  if role == "developer":
                      role = "system"
                  messages.append({
                      "role": role,
                      "content": self._convert_responses_content_to_chat_content(
                          item.get("content", "")),
                  })
              else:
                  messages.append({"role": "system", "content": str(item)})
          return
      if isinstance(instructions, dict):
          role = instructions.get("role", "system")
          if role == "developer":
              role = "system"
          messages.append({
              "role": role,
              "content": self._convert_responses_content_to_chat_content(
                  instructions.get("content", instructions)),
          })
          return
      messages.append({"role": "system", "content": str(instructions)})

  def _responses_system_content_to_text(self, content: Any) -> str:
      converted = self._convert_responses_content_to_chat_content(content)
      if converted is None:
          return ""
      if isinstance(converted, list):
          parts = []
          for part in converted:
              if isinstance(part, dict) and part.get("type") == "text":
                  parts.append(part.get("text", ""))
              elif part is not None:
                  parts.append(json.dumps(part, ensure_ascii = False)
                               if isinstance(part, dict) else str(part))
          return "\n".join([part for part in parts if part != ""])
      return str(converted)

  def _normalize_responses_chat_messages(
      self, messages: List[Dict[str, Any]]
  ) -> List[Dict[str, Any]]:
      system_parts: List[str] = []
      normalized: List[Dict[str, Any]] = []
      for message in messages:
          if not isinstance(message, dict):
              normalized.append({"role": "user", "content": str(message)})
              continue

          role = message.get("role", "user")
          if role == "developer":
              role = "system"
          if role == "system":
              text = self._responses_system_content_to_text(
                  message.get("content", ""))
              if text:
                  system_parts.append(text)
              continue

          message = dict(message)
          message["role"] = role
          normalized.append(message)

      if system_parts:
          normalized.insert(0, {
              "role": "system",
              "content": "\n\n".join(system_parts),
          })
      return normalized

  def _responses_input_to_chat_messages(
      self, request: ResponsesRequest
  ) -> List[Dict[str, Any]]:
      messages: List[Dict[str, Any]] = []
      self._append_responses_instruction_messages(messages, request.instructions)

      response_input = request.input
      if response_input is None:
          return self._normalize_responses_chat_messages(messages)
      if isinstance(response_input, str):
          messages.append({"role": "user", "content": response_input})
          return self._normalize_responses_chat_messages(messages)
      if not isinstance(response_input, list):
          messages.append({"role": "user", "content": str(response_input)})
          return self._normalize_responses_chat_messages(messages)

      for item in response_input:
          if isinstance(item, str):
              messages.append({"role": "user", "content": item})
              continue
          if not isinstance(item, dict):
              messages.append({"role": "user", "content": str(item)})
              continue

          item_type = item.get("type")
          if item_type == "function_call":
              call_id = item.get("call_id") or item.get("id") or f"call_{shortuuid.random()}"
              messages.append({
                  "role": "assistant",
                  "content": None,
                  "tool_calls": [{
                      "id": call_id,
                      "type": "function",
                      "function": {
                          "name": item.get("name", ""),
                          "arguments": item.get("arguments", "{}"),
                      },
                  }],
              })
              continue

          if item_type in (
                  "function_call_output", "custom_tool_call_output",
                  "local_shell_call_output", "shell_call_output",
                  "apply_patch_call_output"):
              messages.append({
                  "role": "tool",
                  "tool_call_id": item.get("call_id") or item.get("id"),
                  "content": self._stringify_responses_tool_output(
                      item.get("output", "")),
              })
              continue

          if item_type in ("reasoning", "item_reference"):
              continue

          if item_type in ("input_text", "input_image", "input_file", "text"):
              messages.append({
                  "role": "user",
                  "content": self._convert_responses_content_to_chat_content([item]),
              })
              continue

          if "role" in item or item_type in ("message", "input_message", "output_message"):
              role = item.get("role", "user")
              if role == "developer":
                  role = "system"
              messages.append({
                  "role": role,
                  "content": self._convert_responses_content_to_chat_content(
                      item.get("content", "")),
              })
              continue

          messages.append({
              "role": "user",
              "content": json.dumps(item, ensure_ascii = False),
          })
      return self._normalize_responses_chat_messages(messages)

  def _convert_responses_tools(
      self, tools: Optional[List[Dict[str, Any]]]
  ) -> Optional[List[ChatCompletionToolsParam]]:
      if not tools:
          return None
      converted: List[ChatCompletionToolsParam] = []
      for tool in tools:
          if not isinstance(tool, dict):
              continue
          if tool.get("type") != "function":
              continue

          function = tool.get("function")
          if not isinstance(function, dict):
              function = {
                  "name": tool.get("name"),
                  "description": tool.get("description"),
                  "parameters": tool.get("parameters"),
              }
          if not function.get("name"):
              continue
          converted.append(ChatCompletionToolsParam(
              function = FunctionDefinition(
                  name = function.get("name"),
                  description = function.get("description"),
                  parameters = function.get("parameters"),
              )))
      return converted or None

  def _convert_responses_tool_choice(self, tool_choice: Any) -> Any:
      if tool_choice is None:
          return "auto"
      if isinstance(tool_choice, str):
          if tool_choice in ("auto", "none", "required"):
              return tool_choice
          return "auto"
      if isinstance(tool_choice, dict):
          function = tool_choice.get("function")
          if isinstance(function, dict) and function.get("name"):
              return ChatCompletionNamedToolChoiceParam(
                  function = ChatCompletionNamedFunction(name = function.get("name")))
          if tool_choice.get("type") == "function" and tool_choice.get("name"):
              return ChatCompletionNamedToolChoiceParam(
                  function = ChatCompletionNamedFunction(name = tool_choice.get("name")))
      return "auto"

  def _build_chat_request_from_responses(
      self, request: ResponsesRequest
  ) -> ChatCompletionRequest:
      messages = self._responses_input_to_chat_messages(request)
      max_tokens = request.max_output_tokens
      if max_tokens is None:
          max_tokens = request.max_tokens
      return ChatCompletionRequest(
          model = request.model,
          messages = messages,
          temperature = request.temperature,
          top_p = request.top_p,
          top_k = request.top_k,
          max_tokens = max_tokens,
          stop = request.stop,
          stream = request.stream,
          tools = self._convert_responses_tools(request.tools),
          tool_choice = self._convert_responses_tool_choice(request.tool_choice),
      )

  def _responses_usage_from_chat_usage(self, usage: Optional[UsageInfo]) -> ResponsesUsageInfo:
      if usage is None:
          return ResponsesUsageInfo()
      prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
      completion_tokens = getattr(usage, "completion_tokens", 0) or 0
      total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
      return ResponsesUsageInfo(
          input_tokens = prompt_tokens,
          output_tokens = completion_tokens,
          total_tokens = total_tokens,
      )

  def _responses_usage_from_dict(self, usage: Optional[Dict[str, Any]]) -> ResponsesUsageInfo:
      if not usage:
          return ResponsesUsageInfo()
      prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
      completion_tokens = usage.get(
          "completion_tokens", usage.get("output_tokens", 0)) or 0
      total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
      return ResponsesUsageInfo(
          input_tokens = prompt_tokens,
          output_tokens = completion_tokens,
          total_tokens = total_tokens,
      )

  def _responses_tool_schemas_by_name(
      self, tools: Optional[List[Dict[str, Any]]]
  ) -> Dict[str, Dict[str, Any]]:
      schemas: Dict[str, Dict[str, Any]] = {}
      for tool in tools or []:
          if not isinstance(tool, dict):
              continue
          function = tool.get("function")
          if isinstance(function, dict):
              name = function.get("name") or tool.get("name")
              parameters = function.get("parameters") or tool.get("parameters")
          else:
              name = tool.get("name")
              parameters = tool.get("parameters")
          if isinstance(name, str) and isinstance(parameters, dict):
              schemas[name] = parameters
      return schemas

  def _coerce_json_value_by_schema(self, value: Any, schema: Any) -> Any:
      if not isinstance(schema, dict):
          return value

      schema_type = schema.get("type")
      if isinstance(schema_type, list):
          schema_type = next((item for item in schema_type if item != "null"), None)

      if schema_type == "boolean" and isinstance(value, str):
          lower = value.strip().lower()
          if lower == "true":
              return True
          if lower == "false":
              return False
      if schema_type == "integer" and isinstance(value, str):
          try:
              return int(value.strip())
          except ValueError:
              return value
      if schema_type == "number" and isinstance(value, str):
          try:
              return float(value.strip())
          except ValueError:
              return value
      if schema_type == "array":
          if isinstance(value, str):
              try:
                  parsed_value = json.loads(value)
              except json.JSONDecodeError:
                  parsed_value = value
              value = parsed_value
          if isinstance(value, list):
              item_schema = schema.get("items")
              return [
                  self._coerce_json_value_by_schema(item, item_schema)
                  for item in value
              ]
          return value
      if schema_type == "object" and isinstance(value, dict):
          properties = schema.get("properties")
          if not isinstance(properties, dict):
              return value
          coerced = dict(value)
          for key, prop_schema in properties.items():
              if key in coerced:
                  coerced[key] = self._coerce_json_value_by_schema(
                      coerced[key], prop_schema)
          return coerced
      return value

  def _coerce_responses_tool_argument_fallback(
      self, value: Any, key: Optional[str] = None
  ) -> Any:
      if isinstance(value, dict):
          return {
              item_key: self._coerce_responses_tool_argument_fallback(
                  item_value, item_key)
              for item_key, item_value in value.items()
          }
      if isinstance(value, list):
          return [
              self._coerce_responses_tool_argument_fallback(item)
              for item in value
          ]
      if isinstance(value, float) and value.is_integer():
          return int(value)

      int_keys = {
          "autoResolutionMs",
          "content_offset_chars",
          "duration",
          "expected_size_bytes",
          "limit",
          "max_output_tokens",
          "pageno",
          "recency",
          "session_id",
          "yield_time_ms",
      }
      bool_keys = {
          "login",
          "stream",
          "tty",
      }
      if isinstance(value, str) and key in int_keys:
          try:
              parsed = float(value.strip())
              if parsed.is_integer():
                  return int(parsed)
          except ValueError:
              return value
      if isinstance(value, str) and key in bool_keys:
          lower = value.strip().lower()
          if lower == "true":
              return True
          if lower == "false":
              return False
      return value

  def _coerce_responses_tool_arguments(
      self,
      name: Optional[str],
      arguments: Any,
      tool_schemas: Dict[str, Dict[str, Any]],
  ) -> str:
      if isinstance(arguments, str):
          try:
              parsed_arguments = json.loads(arguments)
          except json.JSONDecodeError:
              return arguments
      else:
          parsed_arguments = arguments or {}

      if isinstance(name, str) and name in tool_schemas:
          parsed_arguments = self._coerce_json_value_by_schema(
              parsed_arguments, tool_schemas[name])
      parsed_arguments = self._coerce_responses_tool_argument_fallback(
          parsed_arguments)
      return json.dumps(parsed_arguments or {}, ensure_ascii = False)

  def _responses_function_call_item(
      self,
      tool_call: Any,
      tool_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
  ) -> Dict[str, Any]:
      tool_call_id = getattr(tool_call, "id", None)
      function = getattr(tool_call, "function", None)
      if isinstance(tool_call, dict):
          tool_call_id = tool_call.get("id", tool_call_id)
          function = tool_call.get("function", function)

      name = getattr(function, "name", None)
      arguments = getattr(function, "arguments", None)
      if isinstance(function, dict):
          name = function.get("name", name)
          arguments = function.get("arguments", arguments)
      arguments = self._coerce_responses_tool_arguments(
          name, arguments, tool_schemas or {})

      call_id = tool_call_id or f"call_{shortuuid.random()}"
      return {
          "type": "function_call",
          "id": f"fc_{shortuuid.random()}",
          "call_id": call_id,
          "name": name or "",
          "arguments": arguments,
          "status": "completed",
      }

  def _responses_message_item(
      self, item_id: str, text: str, status: str = "completed"
  ) -> Dict[str, Any]:
      return {
          "type": "message",
          "id": item_id,
          "status": status,
          "role": "assistant",
          "content": [{
              "type": "output_text",
              "text": text,
              "annotations": [],
          }],
      }

  def _responses_response_object(
      self,
      request: ResponsesRequest,
      response_id: str,
      created_at: int,
      status: str,
      output: List[Dict[str, Any]],
      usage: Optional[ResponsesUsageInfo],
      output_text: Optional[str] = None,
  ) -> ResponsesResponse:
      completed_at = int(time.time()) if status in ("completed", "incomplete", "failed") else None
      incomplete_details = None
      if status == "incomplete":
          incomplete_details = {"reason": "max_output_tokens"}
      return ResponsesResponse(
          id = response_id,
          created_at = created_at,
          status = status,
          completed_at = completed_at,
          incomplete_details = incomplete_details,
          instructions = request.instructions,
          max_output_tokens = request.max_output_tokens or request.max_tokens,
          model = self.model_name,
          output = output,
          output_text = output_text,
          parallel_tool_calls = bool(request.parallel_tool_calls),
          previous_response_id = request.previous_response_id,
          reasoning = request.reasoning or {"effort": None, "summary": None},
          store = bool(request.store),
          temperature = request.temperature,
          text = request.text or {"format": {"type": "text"}},
          tool_choice = request.tool_choice,
          tools = request.tools or [],
          top_p = request.top_p,
          truncation = request.truncation or "disabled",
          usage = usage,
          user = request.user,
          metadata = request.metadata or {},
      )

  def _chat_response_to_responses_response(
      self, request: ResponsesRequest, chat_response: ChatCompletionResponse
  ) -> ResponsesResponse:
      response_id = chat_response.id
      created_at = chat_response.created
      output: List[Dict[str, Any]] = []
      output_text = ""
      finish_reason = None
      tool_schemas = self._responses_tool_schemas_by_name(request.tools)

      for choice in chat_response.choices:
          finish_reason = choice.finish_reason
          message = choice.message
          content = message.content or ""
          if content:
              item_id = f"msg_{shortuuid.random()}"
              output.append(self._responses_message_item(item_id, content))
              output_text += content
          if message.tool_calls:
              for tool_call in message.tool_calls:
                  output.append(self._responses_function_call_item(
                      tool_call, tool_schemas))

      status = "incomplete" if finish_reason == "length" else "completed"
      return self._responses_response_object(
          request = request,
          response_id = response_id,
          created_at = created_at,
          status = status,
          output = output,
          usage = self._responses_usage_from_chat_usage(chat_response.usage),
          output_text = output_text or None,
      )

  def _responses_sse_event(
      self,
      event_type: str,
      payload: Dict[str, Any],
      sequence_number: int,
  ) -> str:
      payload = dict(payload)
      payload["type"] = event_type
      payload["sequence_number"] = sequence_number
      data = json.dumps(payload, ensure_ascii = False, separators = (",", ":"))
      return f"event: {event_type}\ndata: {data}\n\n"

  async def create_response(
      self, request: ResponsesRequest, raw_request: Request
  ) -> Union[ErrorResponse, ResponsesResponse,
             Tuple[AsyncGenerator[str, None], AsyncGenerator]]:
      try:
          chat_request = self._build_chat_request_from_responses(request)
          if len(chat_request.messages) == 0:
              return self.create_error_response("Empty input")
      except Exception as e:
          logging.error("Error in applying responses request: %s", e)
          traceback.print_exc()
          return self.create_error_response(str(e))

      generator = await self.create_chat_completion(chat_request, raw_request)
      if isinstance(generator, ErrorResponse):
          return generator

      if request.stream:
          return (self.responses_stream_generator(request, generator[0]),
                  generator[1])

      assert isinstance(generator, ChatCompletionResponse)
      return self._chat_response_to_responses_response(request, generator)

  async def responses_stream_generator(
      self,
      request: ResponsesRequest,
      chat_stream: AsyncIterator[str],
  ) -> AsyncGenerator[str, None]:
      response_id = f"resp_{shortuuid.random()}"
      created_at = int(time.time())
      sequence_number = 0
      response_started = False
      message_started = False
      content_started = False
      message_item_id = f"msg_{shortuuid.random()}"
      output_text = ""
      output: List[Dict[str, Any]] = []
      tool_calls: Dict[int, Dict[str, Any]] = {}
      tool_schemas = self._responses_tool_schemas_by_name(request.tools)
      usage = ResponsesUsageInfo()
      finish_reason = None

      def next_event(event_type: str, payload: Dict[str, Any]) -> str:
          nonlocal sequence_number
          sequence_number += 1
          return self._responses_sse_event(event_type, payload, sequence_number)

      def ensure_response_started() -> List[str]:
          nonlocal response_started
          if response_started:
              return []
          response_started = True
          response = self._responses_response_object(
              request = request,
              response_id = response_id,
              created_at = created_at,
              status = "in_progress",
              output = [],
              usage = None,
          )
          return [next_event("response.created", {
              "response": response.model_dump(),
          })]

      def ensure_text_item_started() -> List[str]:
          nonlocal message_started
          nonlocal content_started
          events = ensure_response_started()
          if not message_started:
              message_started = True
              item = self._responses_message_item(
                  message_item_id, "", status = "in_progress")
              item["content"] = []
              events.append(next_event("response.output_item.added", {
                  "output_index": 0,
                  "item": item,
              }))
          if not content_started:
              content_started = True
              events.append(next_event("response.content_part.added", {
                  "item_id": message_item_id,
                  "output_index": 0,
                  "content_index": 0,
                  "part": {
                      "type": "output_text",
                      "text": "",
                      "annotations": [],
                  },
              }))
          return events

      async for chunk in chat_stream:
          for line in chunk.splitlines():
              line = line.strip()
              if not line.startswith("data:"):
                  continue
              data_str = line[len("data:"):].strip()
              if data_str == "[DONE]":
                  continue
              try:
                  data = json.loads(data_str)
              except json.JSONDecodeError:
                  continue

              if "error" in data:
                  for event_data in ensure_response_started():
                      yield event_data
                  yield next_event("error", {
                      "error": data["error"],
                  })
                  return

              response_id = data.get("id", response_id)
              created_at = data.get("created", created_at)
              for event_data in ensure_response_started():
                  yield event_data

              choices = data.get("choices", [])
              if not choices:
                  if "usage" in data:
                      usage = self._responses_usage_from_dict(data.get("usage"))
                  continue

              choice = choices[0]
              if choice.get("finish_reason") is not None:
                  finish_reason = choice.get("finish_reason")
              delta = choice.get("delta") or {}
              if "usage" in data:
                  usage = self._responses_usage_from_dict(data.get("usage"))

              text_delta = delta.get("content")
              if text_delta:
                  for event_data in ensure_text_item_started():
                      yield event_data
                  output_text += text_delta
                  yield next_event("response.output_text.delta", {
                      "item_id": message_item_id,
                      "output_index": 0,
                      "content_index": 0,
                      "delta": text_delta,
                  })

              for tool_delta in delta.get("tool_calls", []) or []:
                  index = tool_delta.get("index", 0)
                  current = tool_calls.setdefault(index, {
                      "id": tool_delta.get("id") or f"call_{shortuuid.random()}",
                      "name": "",
                      "arguments": "",
                  })
                  if tool_delta.get("id"):
                      current["id"] = tool_delta["id"]
                  function = tool_delta.get("function") or {}
                  if function.get("name"):
                      current["name"] += function["name"]
                  if function.get("arguments"):
                      current["arguments"] += function["arguments"]

      for event_data in ensure_response_started():
          yield event_data

      output_index = 0
      if message_started:
          message_item = self._responses_message_item(message_item_id, output_text)
          output.append(message_item)
          yield next_event("response.output_text.done", {
              "item_id": message_item_id,
              "output_index": output_index,
              "content_index": 0,
              "text": output_text,
          })
          yield next_event("response.content_part.done", {
              "item_id": message_item_id,
              "output_index": output_index,
              "content_index": 0,
              "part": {
                  "type": "output_text",
                  "text": output_text,
                  "annotations": [],
              },
          })
          yield next_event("response.output_item.done", {
              "output_index": output_index,
              "item": message_item,
          })
          output_index += 1

      for index in sorted(tool_calls):
          tool_call = tool_calls[index]
          arguments = self._coerce_responses_tool_arguments(
              tool_call["name"], tool_call["arguments"], tool_schemas)
          item = {
              "type": "function_call",
              "id": f"fc_{shortuuid.random()}",
              "call_id": tool_call["id"],
              "name": tool_call["name"],
              "arguments": arguments,
              "status": "completed",
          }
          output.append(item)
          added_item = dict(item)
          added_item["arguments"] = ""
          added_item["status"] = "in_progress"
          yield next_event("response.output_item.added", {
              "output_index": output_index,
              "item": added_item,
          })
          if arguments:
              yield next_event("response.function_call_arguments.delta", {
                  "item_id": item["id"],
                  "output_index": output_index,
                  "delta": arguments,
              })
          yield next_event("response.function_call_arguments.done", {
              "item_id": item["id"],
              "output_index": output_index,
              "arguments": arguments,
          })
          yield next_event("response.output_item.done", {
              "output_index": output_index,
              "item": item,
          })
          output_index += 1

      if not output and output_text == "":
          message_item = self._responses_message_item(message_item_id, "")
          output.append(message_item)

      status = "incomplete" if finish_reason == "length" else "completed"
      response = self._responses_response_object(
          request = request,
          response_id = response_id,
          created_at = created_at,
          status = status,
          output = output,
          usage = usage,
          output_text = output_text or None,
      )
      event_type = "response.incomplete" if status == "incomplete" else "response.completed"
      yield next_event(event_type, {
          "response": response.model_dump(),
      })

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
          media = LoadedMedia()
          if request.system is not None:
              system_messages, system_media = self._parse_anthropic_message_content(
                  "system", request.system)
              conversation.extend(system_messages)
              media.extend(system_media)

          for m in request.messages:
              messages, message_media = self._parse_anthropic_message_content(
                  m.role, m.content)
              conversation.extend(messages)
              media.extend(message_media)

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
          self._cleanup_temp_paths(media.temp_paths if "media" in locals() else [])
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

      model_images = media.images if media.images else None
      model_videos = media.videos if media.videos else None
      try:
          input_token_len = self._compute_multimodal_input_token_len(
              messages,
              enable_thinking = self.enable_thinking,
              images = model_images,
              videos = model_videos,
              tools = model_tools)
          launch_kwargs = {
              "max_length": max_length,
              "min_length": 0,
              "do_sample": do_sample,
              "top_p": top_p,
              "top_k": top_k,
              "temperature": temperature,
              "repeat_penalty": frequency_penalty,
              "tools": model_tools,
              "one_by_one": True,
              "enable_thinking": self.enable_thinking,
              "images": model_images,
              "videos": model_videos,
          }
          if parser_request is not None:
              self._attach_tool_call_constraint_if_supported(
                  launch_kwargs, parser_request)
          handle = self.model.launch_stream_response(
              messages, **launch_kwargs)
      finally:
          self._cleanup_temp_paths(media.temp_paths)
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
          media = LoadedMedia()
          for m in request.messages:
              messages, message_media = self._parse_chat_message_content(
                  m["role"], m.get("content"),
                  tool_calls=m.get("tool_calls"),
                  tool_call_id=m.get("tool_call_id"),
                  name=m.get("name"))

              conversation.extend(messages)
              media.extend(message_media)

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
          self._cleanup_temp_paths(media.temp_paths if "media" in locals() else [])
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
      stop_strings = self._normalize_stop_strings(request.stop)
      stop_token_ids = self._stop_token_ids_from_strings(stop_strings)

      enable_thinking = self.enable_thinking
      if request.chat_template_kwargs and "enable_thinking" in request.chat_template_kwargs:
          enable_thinking = bool(request.chat_template_kwargs["enable_thinking"])

      #logging.info(request)
      if (not(self.hide_input)):
         logging.info(f"fastllm input message: {messages}")
      #logging.info(f"input tokens: {input_token_len}")

      model_images = media.images if media.images else None
      model_videos = media.videos if media.videos else None

      tools = [tool.model_dump(exclude_none=True) for tool in request.tools] if request.tools is not None else None

      try:
          input_token_len = self._compute_multimodal_input_token_len(
              messages,
              enable_thinking = enable_thinking,
              images = model_images,
              videos = model_videos,
              tools = tools)

          launch_kwargs = {
              "max_length": max_length,
              "min_length": min_length,
              "do_sample": do_sample,
              "top_p": top_p,
              "top_k": top_k,
              "temperature": temperature,
              "repeat_penalty": frequency_penalty,
              "tools": tools,
              "one_by_one": True,
              "enable_thinking": enable_thinking,
              "images": model_images,
              "videos": model_videos,
              "stop_token_ids": stop_token_ids,
          }
          self._attach_tool_call_constraint_if_supported(
              launch_kwargs, request)
          handle = self.model.launch_stream_response(
              messages, **launch_kwargs)
      finally:
          self._cleanup_temp_paths(media.temp_paths)
      # Store the mapping between conversation ID and handle
      self.conversation_handles[request_id] = handle
      # logging.info(f"Created conversation: {request_id}, handle: {handle}")
      result_generator = self.model.stream_response_handle_async(handle)
      # --think 是用户显式指定"在输出前补 <think>\n 起始标签"的开关，
      # 严格按用户意愿执行，与 enable_thinking（是否进入思考模式）解耦。
      need_think_prefix = self.think
      emit_reasoning_content = self._is_deepseek_v4_reasoning_response(enable_thinking)
      # Streaming response
      if request.stream:
          return (self.chat_completion_stream_generator(
              request, raw_request, result_generator, request_id, input_token_len,
              think = need_think_prefix, emit_reasoning_content = emit_reasoning_content),
              BackgroundTask(self.check_disconnect, raw_request, request_id, handle))
      else:
          try:
              return await self.chat_completion_full_generator(
                  request, raw_request, handle, result_generator, request_id, input_token_len,
                  think = need_think_prefix, emit_reasoning_content = emit_reasoning_content)
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
              input_token_len: int,
              think: bool = False,
              emit_reasoning_content: bool = False) -> Union[ErrorResponse, ChatCompletionResponse]:
      model_name = self.model_name
      created_time = int(time.time())
      result = "" if emit_reasoning_content else ("<think>\n" if think else "")
      completion_tokens = 0
      async for res in result_generator:
        res = self._normalize_model_delta(res)
        result += res
        completion_tokens += 1
        if await raw_request.is_disconnected():
           print("is_disconnected!!!")
           self.model.abort_handle(handle)
           logging.info(f"Abort request: {request_id}")
           return self.create_error_response("Client disconnected")

      result, reasoning_content = self._split_deepseek_v4_reasoning(
          result, emit_reasoning_content)
      result, stopped_by_stop_string = self._truncate_at_stop(
          result, self._normalize_stop_strings(request.stop))

      tool_call_info = self._parse_non_stream_tool_calls(result, request)
      if isinstance(tool_call_info, ErrorResponse):
          if request_id in self.conversation_handles:
              del self.conversation_handles[request_id]
          return tool_call_info

      if tool_call_info.tools_called:
          choice_data = ChatCompletionResponseChoice(
              index=0,
              message=ChatMessage(
                  role="assistant",
                  content=tool_call_info.content or None,
                  reasoning_content=reasoning_content or None,
                  tool_calls=tool_call_info.tool_calls,
              ),
              logprobs=None,
              finish_reason='tool_calls',
          )
      else:
          choice_data = ChatCompletionResponseChoice(
              index=0,
              message=ChatMessage(
                  role="assistant",
                  content=tool_call_info.content,
                  reasoning_content=reasoning_content or None,
              ),
              logprobs=None,
              finish_reason=self._chat_finish_reason(
                  completion_tokens, request.max_tokens or 32768,
                  stopped_by_stop_string),
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
          input_token_len: int, think: bool,
          emit_reasoning_content: bool = False) -> AsyncGenerator[str, None]:
      model_name = self.model_name
      created_time = int(time.time())
      chunk_object_type = "chat.completion.chunk"
      json_dump = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
      fast_text_stream = not request.tools and not emit_reasoning_content
      stop_strings = self._normalize_stop_strings(request.stop)
      stop_filter_state = {"buffer": ""}
      stopped_by_stop_string = False

      # TODO: 支持request.n 和 request.echo配置
      first_iteration = True
      try:
        if first_iteration:
            # 1. role部分
            if fast_text_stream:
                data = json_dump({
                    "id": request_id,
                    "object": chunk_object_type,
                    "created": created_time,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "logprobs": None,
                        "finish_reason": None,
                    }],
                })
            else:
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
        if not has_sent_label and think and not emit_reasoning_content:
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

        tool_call_parser = (
            self._create_function_call_parser(request) if request.tools else None
        )
        
        completion_tokens = 0

        previous_token_ids = []
        current_token_ids = []
        previous_text = ""
        current_text = ""
        reasoning_state = {
            "active": emit_reasoning_content,
            "buffer": "",
            "started": False,
        }

        async for res in result_generator:
            res = self._normalize_model_delta(res)
            completion_tokens += 1
            delta_text = res

            reasoning_delta_messages, delta_text = self._consume_deepseek_v4_reasoning_delta(
                delta_text, reasoning_state)
            for reasoning_delta_message in reasoning_delta_messages:
                choice_data = ChatCompletionResponseStreamChoice(
                    index = 0,
                    delta = reasoning_delta_message,
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
            if delta_text == "":
                continue
            if not request.tools:
                delta_text, stop_hit = self._filter_stop_delta(
                    delta_text, stop_strings, stop_filter_state)
                if stop_hit:
                    stopped_by_stop_string = True
                if delta_text == "":
                    if stopped_by_stop_string:
                        break
                    continue

            # print("delta_text", delta_text)

            # Send token-by-token response for each request.n
            if tool_call_parser and request.tools:
                now_ids = tool_call_parser.get_token_ids(delta_text)
                # print("delta_text", delta_text, "now_ids", now_ids)

                current_text += delta_text
                current_token_ids += now_ids

                parse_result = tool_call_parser.parse_stream_chunk(
                                previous_text = previous_text,
                                current_text = current_text,
                                delta_text = delta_text,
                                previous_token_ids = previous_token_ids,
                                current_token_ids = current_token_ids,
                                delta_token_ids = now_ids)

                previous_text += delta_text
                previous_token_ids += now_ids
                if parse_result.has_invalid_tool_block:
                    diagnostics = self._format_tool_call_diagnostics(
                        parse_result.diagnostics)
                    logging.warning("Invalid stream tool call suppressed: %s",
                                    diagnostics)
                if parse_result.content or parse_result.valid_tool_calls:
                    delta_message = DeltaMessage(
                        content = parse_result.content,
                        tool_calls = parse_result.valid_tool_calls,
                    )
                else:
                    delta_message = None
                # print("delta", delta_message)
            else:
                delta_message = DeltaMessage(content = delta_text)

            if (delta_message):
                if fast_text_stream:
                    data = json_dump({
                        "id": request_id,
                        "object": chunk_object_type,
                        "created": created_time,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta_text},
                            "logprobs": None,
                            "finish_reason": None,
                        }],
                    })
                else:
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
            if stopped_by_stop_string:
                break
            #await asyncio.sleep(0)

        if (not request.tools) and not stopped_by_stop_string:
            delta_text = self._flush_stop_buffer(stop_strings, stop_filter_state)
            if delta_text:
                if fast_text_stream:
                    data = json_dump({
                        "id": request_id,
                        "object": chunk_object_type,
                        "created": created_time,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta_text},
                            "logprobs": None,
                            "finish_reason": None,
                        }],
                    })
                else:
                    choice_data = ChatCompletionResponseStreamChoice(
                        index = 0,
                        delta = DeltaMessage(content = delta_text),
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

        if (emit_reasoning_content and reasoning_state.get("active")
                and reasoning_state.get("buffer")):
            choice_data = ChatCompletionResponseStreamChoice(
                index = 0,
                delta = DeltaMessage(reasoning_content = reasoning_state["buffer"]),
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
            reasoning_state["buffer"] = ""

        # 3. 结束标志
        finish_reason = self._chat_finish_reason(
            completion_tokens, request.max_tokens or 32768,
            stopped_by_stop_string)
        final_stream_error_data = None
        if request.tools and tool_call_parser:
            final_diagnostics = tool_call_parser.finalize_stream()
            if final_diagnostics:
                diagnostics = self._format_tool_call_diagnostics(
                    final_diagnostics)
                logging.warning("Invalid stream tool call final state: %s",
                                diagnostics)
                final_stream_error_data = self.create_streaming_error_response(
                    f"Invalid tool call: {diagnostics}",
                    err_type = "invalid_tool_call",
                )
            elif tool_call_parser.has_valid_streamed_tool_calls:
                finish_reason = 'tool_calls'
        if final_stream_error_data is not None:
            data = final_stream_error_data
        elif fast_text_stream:
            data = json_dump({
                "id": request_id,
                "object": chunk_object_type,
                "created": created_time,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": input_token_len,
                    "total_tokens": input_token_len + completion_tokens,
                    "completion_tokens": completion_tokens,
                },
            })
        else:
            choice_data = ChatCompletionResponseStreamChoice(
                index = 0,
                delta = DeltaMessage(),
                logprobs = None,
                finish_reason = finish_reason)
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
        if (final_stream_error_data is None and request.tools
                and tool_call_parser):
            flush_result = tool_call_parser.flush_stream_tool_calls()
            if flush_result.valid_tool_calls:
                delta_message = DeltaMessage(
                    tool_calls = flush_result.valid_tool_calls,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index = 0,
                    delta = delta_message,
                    logprobs = None,
                    finish_reason = None)
                chunk = ChatCompletionStreamResponseWithUsage(
                    id = request_id,
                    object = chunk_object_type,
                    created = created_time,
                    choices = [choice_data],
                    model = model_name)
                flush_data = chunk.model_dump_json(exclude_unset = True,
                                                   exclude_none = True)
                yield f"data: {flush_data}\n\n"
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
          tool_parser = self._create_tool_parser()
          tool_call_info = tool_parser.extract_tool_calls(result, parser_request)
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

          tool_parser = (
              self._create_tool_parser()
              if request.tools and parser_request is not None
              else None
          )

          async for res in result_generator:
              if (res == "[unused16]"):
                  res = "<think>"
              elif (res == "[unused17]"):
                  res = "</think>"
              completion_tokens += 1
              delta_text = res

              if request.tools and parser_request is not None and tool_parser:
                  now_ids = tool_parser.get_token_ids(delta_text)
                  current_text += delta_text
                  current_token_ids += now_ids

                  delta_message = tool_parser.extract_tool_calls_streaming(
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
