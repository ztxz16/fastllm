import re
from typing import Tuple

_KIMI_THINK_OPEN_RE = re.compile(
    r"<\|open\|>\s*think\s*<\|sep\|>", re.IGNORECASE)
_KIMI_THINK_CLOSE_RE = re.compile(
    r"<\|close\|>\s*think\s*<\|sep\|>", re.IGNORECASE)
_KIMI_RESPONSE_TAG_RE = re.compile(
    r"<\|(open|close)\|>\s*(response|message)\s*<\|sep\|>",
    re.IGNORECASE)


def split_reasoning(text: str) -> Tuple[str, str]:
    """Split model output into reasoning and user-visible content."""
    text = str(text or "")
    kimi_open = _KIMI_THINK_OPEN_RE.search(text)
    if kimi_open is not None:
        kimi_close = _KIMI_THINK_CLOSE_RE.search(text, kimi_open.end())
        if kimi_close is None:
            return text[kimi_open.end():].strip(), ""
        reasoning = text[kimi_open.end():kimi_close.start()].strip()
        content = _KIMI_RESPONSE_TAG_RE.sub(
            "", text[kimi_close.end():]).strip()
        return reasoning, content

    match = re.search(
        r"<think>(.*?)(?:</think>|$)", text,
        flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        return "", text.strip()
    reasoning = match.group(1).strip()
    content = (text[:match.start()] + text[match.end():]).strip()
    return reasoning, content
