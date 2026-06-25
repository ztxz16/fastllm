
class FastLLmModel:
    def __init__(self,
                 model_name,
                 ):
        self.model_name = model_name
        codex_model = {
                "slug": model_name,
                "id": model_name,
                "display_name": model_name,
                "displayName": model_name,
                "description": "Local FastLLM model.",
                "app_ids": [],
                "hidden": False,
                "default_reasoning_level": "low",
                "supported_reasoning_levels": [],
                "supported_reasoning_efforts": [],
                "supportedReasoningEfforts": [],
                "default_reasoning_effort": None,
                "defaultReasoningEffort": None,
                "shell_type": "shell_command",
                "visibility": "list",
                "minimal_client_version": "0.0.0",
                "supported_in_api": True,
                "available_in_plans": [],
                "priority": 0,
                "input_modalities": ["text"],
                "inputModalities": ["text"],
                "supports_personality": False,
                "supportsPersonality": False,
                "additional_speed_tiers": [],
                "additionalSpeedTiers": [],
                "service_tiers": [
                    {
                        "id": "default",
                        "name": "Default",
                        "display_name": "Default",
                        "description": "Default local service tier."
                    }
                ],
                "serviceTiers": [
                    {
                        "id": "default",
                        "name": "Default",
                        "displayName": "Default",
                        "description": "Default local service tier."
                    }
                ],
                "default_service_tier": "default",
                "defaultServiceTier": "default",
                "upgrade": None,
                "base_instructions": (
                    "You are Codex, a coding agent running in a local workspace.\n"
                    "Use the available tools to complete the user's coding task "
                    "end to end before sending a final answer.\n"
                    "\n"
                    "Operational rules:\n"
                    "- For tasks that require creating or editing files, compiling, "
                    "running commands, inspecting the system, testing, or verifying "
                    "results, you MUST call the available tools. Do not only describe "
                    "what you will do.\n"
                    "- If you say you will run, compile, test, inspect, or check "
                    "something, your next action must be a tool call that does it.\n"
                    "- Do not send standalone progress messages while work remains. "
                    "Messages like 'I will start writing the program', 'let me fix "
                    "that', or 'now I will run the test' are not valid by themselves; "
                    "the same response must include the tool call that performs the "
                    "next action.\n"
                    "- For multi-step coding tasks, every assistant response before "
                    "the final answer must either update the plan and call a tool, "
                    "or call a tool directly. Never end a turn immediately after "
                    "announcing the next step.\n"
                    "- If a compile, test, or command fails, fix the concrete error "
                    "with another tool call and rerun the verification. Do not stop "
                    "after merely describing the fix.\n"
                    "- Only call tools that are listed as available in the current "
                    "turn. If apply_patch is unavailable or unsupported, edit files "
                    "with shell commands instead.\n"
                    "- Do not output placeholders such as '...' as progress or final "
                    "answers.\n"
                    "- A final answer is allowed only after all requested actions are "
                    "actually completed and verified, or after a real blocker is "
                    "observed.\n"
                    "- A final answer must directly answer the user's request with "
                    "the verified result. If the user requested a table and a command "
                    "has already produced measurements or raw data, include the table "
                    "in the final answer immediately. Never make the final answer a "
                    "promise to summarize, format, or continue later.\n"
                    "- Keep tool calls compact. Prefer small working programs and "
                    "short verification commands over large demonstrations.\n"
                    "- 中文任务同样适用：如果用户要求写程序、编译、运行、做表，"
                    "必须实际完成这些步骤后再最终回复；不能只回复“开始编写”、"
                    "“现在运行”、“我来修复”或“现在整理结果”这类进度文字。"
                ),
                "model_messages": None,
                "supports_reasoning_summaries": False,
                "default_reasoning_summary": "none",
                "reasoning_summary_format": "none",
                "support_verbosity": False,
                "default_verbosity": "medium",
                "apply_patch_tool_type": "freeform",
                "web_search_tool_type": "text",
                "truncation_policy": {
                    "mode": "tokens",
                    "limit": 10000
                },
                "supports_parallel_tool_calls": True,
                "supports_image_detail_original": False,
                "context_window": 32768,
                "max_context_window": 32768,
                "auto_compact_token_limit": 24000,
                "comp_hash": "",
                "experimental_supported_tools": [],
                "supports_search_tool": False,
                "use_responses_lite": False,
                "auto_review_model_override": None,
                "tool_mode": "code_mode",
                "multi_agent_version": "v1",
                "is_default": False,
                "isDefault": False,
                "upgrade_info": None,
                "upgradeInfo": None,
                "availability_nux": None,
                "availabilityNux": None
        }
        openai_model = {
            **codex_model,
            "id": model_name,
            "object": "model",
            "owned_by": "fastllm",
            "permission": []
        }
        self.response = {
            "data": [openai_model],
            "models": [codex_model],
            "verifications": [],
            "object": "list"
        }
