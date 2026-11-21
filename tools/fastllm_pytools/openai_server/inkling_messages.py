"""Decode the typed assistant messages emitted by the Inkling chat template."""

MARKERS = {
    "<|message_model|>": "content",
    "<|content_thinking|>": "reasoning",
    "<|content_text|>": "content",
    "<|end_message|>": "content",
    "<|content_model_end_sampling|>": "content",
}


def consume(text, state):
    """Yield (channel, text), retaining an incomplete control token between chunks."""
    text = state["buffer"] + text
    state["buffer"] = ""
    while text:
        start = text.find("<")
        if start < 0:
            yield state["phase"], text
            return
        if start:
            yield state["phase"], text[:start]
        text = text[start:]
        marker = next((tag for tag in MARKERS if text.startswith(tag)), None)
        if marker:
            state["phase"] = MARKERS[marker]
            text = text[len(marker):]
        elif any(tag.startswith(text) for tag in MARKERS):
            state["buffer"] = text
            return
        else:
            yield state["phase"], "<"
            text = text[1:]
