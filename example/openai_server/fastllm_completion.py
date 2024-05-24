import asyncio
import logging
import json
import traceback
from fastapi import Request
from http import HTTPStatus
from typing import (AsyncGenerator, AsyncIterator, Awaitable, Iterable, List,
                    Optional, Tuple, TypedDict, Union, final)
import uuid
from openai.types.chat import (ChatCompletionContentPartParam,
                               ChatCompletionRole)

from protocal.openai_protocol import *
from fastllm_pytools import llm

class ConversationMessage:
    def __init__(self, role:str, content:str):
      self.role = role
      self.content = content

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

class FastLLmCompletion:
  def __init__(self,
               model_name,
               model_path,
               low_mem_mode = False,
               cpu_thds = 16):
    self.model_name = model_name
    self.model_path = model_path
    self.low_mem_mode = low_mem_mode
    self.cpu_thds = cpu_thds
    self.init_fast_llm_model()
    
  def init_fast_llm_model(self):
    llm.set_cpu_threads(self.cpu_thds)
    llm.set_cpu_low_mem(self.low_mem_mode)
    self.model = llm.model(self.model_path)
  
  def create_error_response(
          self,
          message: str,
          err_type: str = "BadRequestError",
          status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
      return ErrorResponse(message=message,
                            type=err_type,
                            code=status_code.value)

  async def _check_model(self, request: ChatCompletionRequest):
    if request.model != self.model_name:
      return self.create_error_response(
          message=f"The model `{request.model}` does not exist.",
          err_type="NotFoundError",
          status_code=HTTPStatus.NOT_FOUND)
    else:
      return None

  def _parse_chat_message_content(
      self,
      role: ChatCompletionRole,
      content: Optional[Union[str,
                              Iterable[ChatCompletionContentPartParam]]],
  ) -> Tuple[List[ConversationMessage], List[Awaitable[object]]]:
      if content is None:
          return [], []
      if isinstance(content, str):
          return [ConversationMessage(role=role, content=content)], []

      # 暂时不支持图像输入
      raise NotImplementedError("Complex input not supported yet")

  async def create_chat_completion(
      self, request: ChatCompletionRequest, raw_request: Request
  ) -> Union[ErrorResponse, AsyncGenerator[str, None],
              ChatCompletionResponse]:
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
      history:List[Tuple[str, str]] = []
      if request.prompt:
         request.messages.append({"role": "user", "content": request.prompt})
      try:
          conversation: List[ConversationMessage] = []
          for m in request.messages:
              messages, _ = self._parse_chat_message_content(
                  m["role"], m["content"])

              conversation.extend(messages)

          # fastllm 样例中history只能是一问一答, system promt 暂时不支持
          if len(conversation) == 0:
            raise Exception("Empty msg")

          for i in range(len(conversation)):
            msg = conversation[i]
            if msg.role == "system":
              # fastllm 暂时不支持system prompt
              continue
            elif msg.role == "user":
              if i + 1 < len(conversation):
                next_msg = conversation[i + 1]
                if next_msg.role == "assistant":
                  history.append((msg.content, next_msg.content))
                else:
                  # 只能是user、assistant、user、assistant的格式
                  raise Exception("fastllm requires that the prompt words must appear alternately in the roles of user and assistant.")
            elif msg.role == "assistant":
              if i - 1 < 0:
                raise Exception("fastllm Not Support assistant prompt in first message")
              else:
                pre_msg = conversation[i - 1]
                if pre_msg.role != "user":
                  raise Exception("In FastLLM, The message role before the assistant msg must be user")
            else:
              raise NotImplementedError(f"prompt role {msg.role } not supported yet")
            
          last_msg = conversation[-1]
          if last_msg.role != "user":
            raise Exception("last msg role must be user")
          query = last_msg.content

      except Exception as e:
          logging.error("Error in applying chat template from request: %s", e)
          traceback.print_exc()
          return self.create_error_response(str(e))

      request_id = f"fastllm-{self.model_name}-{random_uuid()}"
      
      frequency_penalty = 1.0
      if request.frequency_penalty and request.frequency_penalty != 0.0:
        frequency_penalty = request.frequency_penalty

      max_length = request.max_tokens if request.max_tokens else 8192
      logging.info(request)
      logging.info(f"fastllm input: {query}")
      logging.info(f"fastllm history: {history}")
      # stream_response 中的结果不包含token的统计信息
      result_generator = self.model.stream_response(query, history, 
                        max_length = max_length, do_sample = True,
                        top_p = request.top_p, top_k = request.top_k, temperature = request.temperature,
                        repeat_penalty = frequency_penalty, one_by_one = True)
      # Streaming response
      if request.stream:
          return self.chat_completion_stream_generator(
              request, result_generator, request_id)
      else:
          try:
              return await self.chat_completion_full_generator(
                  request, raw_request, result_generator, request_id)
          except ValueError as e:
              return self.create_error_response(str(e))
            
  async def chat_completion_full_generator(
              self, request: ChatCompletionRequest, raw_request: Request,
              result_generator: AsyncIterator,
              request_id: str) -> Union[ErrorResponse, ChatCompletionResponse]:
      model_name = self.model_name
      created_time = int(time.time())
      result = ""
      for res in result_generator:
        result += res
        await asyncio.sleep(0)

      choice_data = ChatCompletionResponseChoice(
              index=0,
              message=ChatMessage(role="assistant", content=result),
              logprobs=None,
              finish_reason='stop',
          )

      # TODO: 补充usage信息, 包括prompt token数, 生成的tokens数量
      response = ChatCompletionResponse(
          id=request_id,
          created=created_time,
          model=model_name,
          choices=[choice_data],
          usage=UsageInfo(),
      )

      return response
      
            
  async def chat_completion_stream_generator(
          self, request: ChatCompletionRequest,
          result_generator: AsyncIterator,
          request_id: str) -> AsyncGenerator[str, None]:
      model_name = self.model_name
      created_time = int(time.time())
      chunk_object_type = "chat.completion.chunk"

      # TODO: 支持request.n 和 request.echo配置
      first_iteration = True
      try:
        if first_iteration:
            # 1. role部分
            choice_data = ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant"),
                            logprobs=None,
                            finish_reason=None)
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[choice_data],
                model=model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
            await asyncio.sleep(0)
            first_iteration = False
        
        # 2. content部分
        for res in result_generator:
            delta_text = res
            # Send token-by-token response for each request.n
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=delta_text),
                logprobs=None,
                finish_reason=None)
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[choice_data],
                model=model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
            await asyncio.sleep(0)

        # 3. 结束标志
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(),
            logprobs=None,
            finish_reason='stop')
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            object=chunk_object_type,
            created=created_time,
            choices=[choice_data],
            model=model_name)
        data = chunk.model_dump_json(exclude_unset=True,
                                    exclude_none=True)
        yield f"data: {data}\n\n"
        await asyncio.sleep(0)
      except ValueError as e:
        data = self.create_streaming_error_response(str(e))
        yield f"data: {data}\n\n"
        await asyncio.sleep(0)
      yield "data: [DONE]\n\n"
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
    