# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the fastllm project
# Adapted from vLLM project

import importlib
import os
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..protocal.openai_protocol import ChatCompletionRequest, DeltaMessage


class ReasoningParser:
    """
    Abstract reasoning parser class that should not be used directly.
    Provided methods should be used in derived classes.

    It is used to extract reasoning content from the model output.
    Reasoning content is typically wrapped in special tokens like <think>...</think>
    and represents the model's "thinking" process before giving an answer.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    @abstractmethod
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.

        It is used in structured engines like `xgrammar` to check if the
        reasoning content ends in the model output.

        Parameters:
        input_ids: list[int]
            The input_ids of the model output.

        Returns:
        bool
            True if the reasoning content ends in the input_ids.
        """

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        """
        Check if the reasoning content ends in the input_ids on a
        decode step.

        It is used in structured engines like `xgrammar` to check if the
        reasoning content ends in the model output during a decode step.
        `input_ids` the entire model output and `delta_ids` are the last few
        computed tokens of the model output (like during a decode step).

        Parameters:
        input_ids: list[int]
            The entire model output.
        delta_ids: list[int]
            The last few computed tokens of the model output at the current decode step.

        Returns:
        bool
            True if the reasoning content ends in the `delta_ids` on a
            decode step.
        """
        return self.is_reasoning_end(input_ids)

    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        Parameters:
        input_ids: list[int]
            The input_ids of the model output.
        Returns:
        list[int]
            The extracted content from the input_ids.
        """

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest",
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Parameters:
        model_output: str
            The model-generated string to extract reasoning content from.

        request: ChatCompletionRequest
            The request object that was used to generate the model_output.

        Returns:
        tuple[Optional[str], Optional[str]]
            A tuple containing the reasoning content and the content.
        """

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Optional["DeltaMessage"]:
        """
        Instance method that should be implemented for extracting reasoning
        from an incomplete response; for use when handling reasoning calls and
        streaming. Has to be an instance method because it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """


class ReasoningParserManager:
    """
    Central registry for ReasoningParser implementations.

    Supports two registration modes:
      - Eager registration via `register_module`
      - Lazy registration via `register_lazy_module`

    Each reasoning parser must inherit from `ReasoningParser`.
    """

    reasoning_parsers: dict[str, type[ReasoningParser]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}  # name -> (module_path, class_name)

    @classmethod
    def get_reasoning_parser(cls, name: str) -> type[ReasoningParser]:
        """
        Retrieve a registered or lazily registered ReasoningParser class.

        If the parser is lazily registered, it will be imported and cached
        on first access.

        Raises:
            KeyError: if no parser is found under the given name.
        """
        if name in cls.reasoning_parsers:
            return cls.reasoning_parsers[name]

        if name in cls.lazy_parsers:
            return cls._load_lazy_parser(name)

        registered = ", ".join(cls.list_registered())
        raise KeyError(
            f"Reasoning parser '{name}' not found. Available parsers: {registered}"
        )

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return names of all eagerly and lazily registered reasoning parsers."""
        return sorted(set(cls.reasoning_parsers.keys()) | set(cls.lazy_parsers.keys()))

    @classmethod
    def _load_lazy_parser(cls, name: str) -> type[ReasoningParser]:
        """Import and register a lazily loaded reasoning parser."""
        module_path, class_name = cls.lazy_parsers[name]
        try:
            mod = importlib.import_module(module_path)
            parser_cls = getattr(mod, class_name)
            if not issubclass(parser_cls, ReasoningParser):
                raise TypeError(
                    f"{class_name} in {module_path} is not a ReasoningParser subclass."
                )

            cls.reasoning_parsers[name] = parser_cls  # cache
            return parser_cls
        except Exception as e:
            logger.exception(
                "Failed to import lazy reasoning parser '%s' from %s: %s",
                name,
                module_path,
                e,
            )
            raise

    @classmethod
    def _register_module(
        cls,
        module: type[ReasoningParser],
        module_name: Optional[Union[str, list[str]]] = None,
        force: bool = True,
    ) -> None:
        """Register a ReasoningParser class immediately."""
        if not issubclass(module, ReasoningParser):
            raise TypeError(
                f"module must be subclass of ReasoningParser, but got {type(module)}"
            )

        if module_name is None:
            module_names = [module.__name__]
        elif isinstance(module_name, str):
            module_names = [module_name]
        elif isinstance(module_name, list):
            module_names = module_name
        else:
            raise TypeError("module_name must be str, list[str], or None.")

        for name in module_names:
            if not force and name in cls.reasoning_parsers:
                existed = cls.reasoning_parsers[name]
                raise KeyError(f"{name} is already registered at {existed.__module__}")
            cls.reasoning_parsers[name] = module

    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        """
        Register a lazy module mapping for delayed import.

        Example:
            ReasoningParserManager.register_lazy_module(
                name="qwen3",
                module_path="fastllm_pytools.openai_server.reasoning.qwen3_reasoning_parser",
                class_name="Qwen3ReasoningParser",
            )
        """
        cls.lazy_parsers[name] = (module_path, class_name)

    @classmethod
    def register_module(
        cls,
        name: Optional[Union[str, list[str]]] = None,
        force: bool = True,
        module: Optional[type[ReasoningParser]] = None,
    ) -> Union[type[ReasoningParser], Any]:
        """
        Register module with the given name or name list. It can be used as a
        decorator (with module as None) or normal function (with module as not
        None).
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # Immediate registration (explicit call)
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # Decorator usage
        def _decorator(obj: type[ReasoningParser]) -> type[ReasoningParser]:
            module_path = obj.__module__
            class_name = obj.__name__

            if isinstance(name, str):
                names = [name]
            elif isinstance(name, list):
                names = name
            else:
                names = [class_name]

            for n in names:
                cls.lazy_parsers[n] = (module_path, class_name)

            return obj

        return _decorator

    @classmethod
    def import_reasoning_parser(cls, plugin_path: str) -> None:
        """
        Import a user-defined reasoning parser by the path
        of the reasoning parser define file.
        """
        import sys
        import importlib.util
        
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]

        try:
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None:
                raise ModuleNotFoundError(f"No module named '{module_name}'")
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception:
            logger.exception(
                "Failed to load module '%s' from %s.", module_name, plugin_path
            )
            return
    
    @classmethod
    def get_reasoning_parser_auto(
        cls, 
        model_type: str, 
        chat_template: str = "",
        force_reasoning_parser: str = "auto"
    ) -> Optional[type[ReasoningParser]]:
        """
        Automatically detect and return the appropriate reasoning parser
        based on model type or chat template.
        
        Parameters:
        model_type: str
            The type of model (e.g., 'qwen3', 'deepseek_v3', 'glm4_moe')
        chat_template: str
            The chat template string to help detect model type
        force_reasoning_parser: str
            Force a specific reasoning parser, "auto" for auto-detection,
            empty string or "none" to disable reasoning parsing
            
        Returns:
        Optional[type[ReasoningParser]]
            The reasoning parser class, or None if no parser is applicable
        """
        if force_reasoning_parser in ['none', '']:
            return None
            
        if force_reasoning_parser not in ['auto']:
            # If a specific parser is requested, use it
            try:
                return cls.get_reasoning_parser(force_reasoning_parser)
            except KeyError:
                logger.warning(
                    f"Requested reasoning parser '{force_reasoning_parser}' not found, "
                    f"falling back to auto-detection"
                )
        
        # Auto-detect based on model type
        target = None
        
        # Check for <think> tags in chat template (common for reasoning models)
        if '<think>' in chat_template or '</think>' in chat_template:
            # Try to detect specific model type
            if model_type in ['qwen3', 'qwen3_moe', 'qwen3_next']:
                target = 'qwen3'
            elif model_type in ['deepseek_v3', 'deepseek_v2', 'deepseek_r1']:
                target = 'deepseek_r1'
            elif model_type == 'glm4_moe':
                target = 'glm45'
            else:
                # Default to qwen3 style parser for generic <think> tags
                target = 'qwen3'
        else:
            # Explicit model type mapping
            if model_type in ['qwen3', 'qwen3_moe', 'qwen3_next']:
                target = 'qwen3'
            elif model_type in ['deepseek_v3', 'deepseek_v2', 'deepseek_r1']:
                target = 'deepseek_r1'
            elif model_type == 'glm4_moe':
                target = 'glm45'
        
        if target is None:
            return None
            
        try:
            logger.info(f"Auto-detected reasoning parser: {target}")
            return cls.get_reasoning_parser(target)
        except KeyError:
            logger.warning(f"Reasoning parser '{target}' not found")
            return None
