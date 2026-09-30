"""Optional local titles for Codex's time-limited auxiliary turns."""

import re


TITLE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string", "minLength": 1, "maxLength": 36}},
    "required": ["title"],
    "additionalProperties": False,
}
TITLE_PREFIX = "Generate a concise, single-line task title of at most 36 characters"
PROMPT_MARKER = "\n\nUser prompt:\n"


def local_codex_title(request, messages):
    """Match the auxiliary task, never an ordinary user/tool/JSON request."""
    format_spec = (request.text or {}).get("format")
    if not isinstance(format_spec, dict) or not (
        format_spec.get("type") == "json_schema"
        and format_spec.get("name") == "codex_output_schema"
        and format_spec.get("strict") is True
        and format_spec.get("schema") == TITLE_SCHEMA
    ):
        return None
    if not messages or messages[-1].get("role") != "user":
        return None
    content = messages[-1].get("content")
    if not isinstance(content, str) or not content.startswith(TITLE_PREFIX):
        return None
    instruction, marker, prompt = content.partition(PROMPT_MARKER)
    if not marker or "Do not answer the request." not in instruction:
        return None
    prompt = " ".join(prompt.split()).strip('`\"\' ')
    if not prompt:
        return None
    if prompt in {"你好", "您好", "hello", "Hello", "hi", "Hi"}:
        return "开始对话" if re.search(r"[\u4e00-\u9fff]", prompt) else "Start conversation"
    # This is a label copied from the request, not an inference result. Keep
    # complete short requests; truncate longer labels at a word when possible.
    title = prompt[:36]
    if len(prompt) > 36 and prompt[36] != " " and " " in title:
        candidate = title.rsplit(" ", 1)[0]
        if len(candidate) >= 18:
            title = candidate
    return title.rstrip(" .,:;!?。，、：；！？") or prompt[:36]
