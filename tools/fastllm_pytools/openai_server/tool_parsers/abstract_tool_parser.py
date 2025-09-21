# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copy from vLLM project

import os
from collections.abc import Sequence
from functools import cached_property
from typing import Callable, Optional, Union, TypeVar, Literal
from typing_extensions import Never, ParamSpec, TypeIs, assert_never
from ..protocal.openai_protocol import *

import logging
logger = logging.getLogger(__name__)

def random_tool_call_id() -> str:
    return "fastllm-tool-" + str(uuid.uuid4().hex)

T = TypeVar("T")
# `collections` helpers
def is_list_of(
    value: object,
    typ: Union[type[T], tuple[type[T], ...]],
    *,
    check: Literal["first", "all"] = "first",
) -> TypeIs[list[T]]:
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)

    assert_never(check)

def import_from_path(module_name: str, file_path: Union[str, os.PathLike]):
    """
    Import a Python file according to its file path.

    Based on the official recipe:
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ModuleNotFoundError(f"No module named '{module_name}'")

    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

class ToolParser:
    """
    Abstract ToolParser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    """

    def __init__(self, tokenizer):
        self.prev_tool_call_arr: list[dict] = []
        # the index of the tool call that is currently being parsed
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.model_tokenizer = tokenizer

    def get_token_ids(self, text: str) -> list[int]:
        return [1]

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Static method that used to adjust the request parameters.
        """
        return request

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Static method that should be implemented for extracting tool calls from
        a complete model-generated string.
        Used for non-streaming responses where we have the entire model response
        available before sending to the client.
        Static because it's stateless.
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls has not been implemented!")

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        """
        Instance method that should be implemented for extracting tool calls
        from an incomplete response; for use when handling tool calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls_streaming has not been "
            "implemented!")


class ToolParserManager:
    tool_parsers: dict[str, type] = {}

    @classmethod
    def get_tool_parser(cls, name) -> type:
        """
        Get tool parser by name which is registered by `register_module`.

        Raise a KeyError exception if the name is not registered.
        """
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]

        raise KeyError(f"tool helper: '{name}' not found in tool_parsers")

    @classmethod
    def _register_module(cls,
                         module: type,
                         module_name: Optional[Union[str, list[str]]] = None,
                         force: bool = True) -> None:
        if not issubclass(module, ToolParser):
            raise TypeError(
                f'module must be subclass of ToolParser, but got {type(module)}'
            )
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in cls.tool_parsers:
                existed_module = cls.tool_parsers[name]
                raise KeyError(f'{name} is already registered '
                               f'at {existed_module.__module__}')
            cls.tool_parsers[name] = module

    @classmethod
    def register_module(
            cls,
            name: Optional[Union[str, list[str]]] = None,
            force: bool = True,
            module: Union[type, None] = None) -> Union[type, Callable]:
        """
        Register module with the given name or name list. it can be used as a
        decoder(with module as None) or normal function(with module as not 
        None).
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)
                or is_list_of(name, str)):
            raise TypeError(
                'name must be None, an instance of str, or a sequence of str, '
                f'but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            cls._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    @classmethod
    def import_tool_parser(cls, plugin_path: str) -> None:
        """
        Import a user-defined tool parser by the path of the tool parser define
        file.
        """
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]

        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            logger.exception("Failed to load module '%s' from %s.",
                             module_name, plugin_path)
            return
    
    @classmethod
    def get_tool_parser_auto(cls, model_type, chat_template, force_chat_template = False, force_type = "auto") -> type:
        if (force_type not in ['auto', '']):
            # 如果指定了tool_parser类型，那么直接使用
            return cls.get_tool_parser(force_type)
        if (force_chat_template):
            # 如果指定了强制指定了chat_template，那么尝试检测tool调用类型
            target = ""
            if ("<｜tool▁calls▁begin｜>" in chat_template):
                target = "deepseek_v31"
            if (target == ""):
                print("Warning: can't detect tool parse, use default tool parser")
                target = "hermes"
            else:
                print("Auto tool parse detect type: " + target)
            return cls.get_tool_parser(target)

        if (model_type == 'qwen3' or model_type == 'qwen2' or model_type == 'qwen3_moe'
            or model_type == "qwen3_next"):
            # 判断是否是coder系列模型（使用xml工具调用）
            if ('<function>' in chat_template):
                target = 'qwen3_coder'
            else:
                target = 'hermes'
        elif model_type == 'glm4_moe':
            target = 'glm45'
        elif model_type == 'kimi_k2':
            target = 'kimi_k2'
        elif model_type == 'deepseek_v3' or model_type == 'deepseek_v2':
            target = 'deepseek_v31'
        else:
            print("Warning: can't detect tool parse, use default tool parser")
            return cls.get_tool_parser('hermes')
        
        print("Auto tool parse detect type: " + target)
        return cls.get_tool_parser(target)
