# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the fastllm project
# Adapted from vLLM project

from .abs_reasoning_parsers import ReasoningParser, ReasoningParserManager

__all__ = [
    "ReasoningParser",
    "ReasoningParserManager",
]

# Register reasoning parsers
_REASONING_PARSERS_TO_REGISTER = {
    "deepseek_r1": (  # name
        "deepseek_r1_reasoning_parser",  # filename
        "DeepSeekR1ReasoningParser",  # class_name
    ),
    "qwen3": (
        "qwen3_reasoning_parser",
        "Qwen3ReasoningParser",
    ),
    "glm45": (
        "glm4_moe_reasoning_parser",
        "Glm4MoeModelReasoningParser",
    ),
}


def register_lazy_reasoning_parsers():
    for name, (file_name, class_name) in _REASONING_PARSERS_TO_REGISTER.items():
        module_path = f"ftllm.openai_server.reasoning.{file_name}"
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_reasoning_parsers()
