# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copy from vLLM project

import regex as re
from .abstract_tool_parser import ToolParserManager
from .glm4_moe_tool_parser import Glm4MoeModelToolParser

import logging
logger = logging.getLogger(__name__)


@ToolParserManager.register_module("glm47")
class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    """
    Tool parser for GLM-4.7 MOE model.
    
    This parser extends Glm4MoeModelToolParser with updated regex patterns
    to handle the slightly different tool call format used by GLM-4.7.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # Override regex patterns for GLM-4.7 format
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
