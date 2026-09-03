import re
from typing import List


def decode_text(data: bytes) -> str:
    """Decode user-provided text using common East Asian encodings."""
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "big5"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            pass
    return data.decode("utf-8", errors="replace")


def lexical_tokens(text: str) -> List[str]:
    """Return lightweight Latin tokens and Chinese bigrams for retrieval."""
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_][a-z0-9_.+\-]{1,}", lowered)
    for run in re.findall(r"[\u3400-\u9fff]+", lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(
                run[index:index + 2] for index in range(len(run) - 1))
    return tokens
