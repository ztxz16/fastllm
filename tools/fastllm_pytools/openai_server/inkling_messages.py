"""Decode the typed assistant messages emitted by the Inkling chat template."""

INVOKE = "<|content_invoke_tool_json|>"
END = "<|end_message|>"

MARKERS = {
    INVOKE: "tool",
    "<|message_model|>": "header",
    "<|content_thinking|>": "reasoning",
    "<|content_text|>": "content",
    END: "content",
    "<|content_model_end_sampling|>": "content",
}


def consume(text, state):
    """Yield (channel, text), retaining an incomplete control token between chunks."""
    text = state["buffer"] + text
    state["buffer"] = ""
    while text:
        if state["phase"] == "tool":
            # Reserved token spellings can occur literally inside a JSON
            # string, including when an escape or delimiter spans chunks.
            quoted = state.get("quoted", False)
            escaped = state.get("escaped", False)
            for i, char in enumerate(text):
                if escaped:
                    escaped = False
                elif quoted and char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = not quoted
                elif not quoted and char == "<":
                    if text.startswith(END, i):
                        if i:
                            yield "tool_json", text[:i]
                        yield "tool", END
                        state.update(phase="content", quoted=False, escaped=False)
                        text = text[i + len(END):]
                        break
                    if END.startswith(text[i:]):
                        state.update(buffer=text[i:], quoted=False, escaped=False)
                        if i:
                            yield "tool_json", text[:i]
                        return
            else:
                state.update(quoted=quoted, escaped=escaped)
                yield "tool_json", text
                return
            continue
        start = text.find("<")
        if start < 0:
            if state["phase"] != "header":
                yield state["phase"], text
            return
        if start and state["phase"] != "header":
            yield state["phase"], text[:start]
        text = text[start:]
        marker = next((tag for tag in MARKERS if text.startswith(tag)), None)
        if marker:
            if marker == INVOKE:
                yield "tool", marker
            state["phase"] = MARKERS[marker]
            text = text[len(marker):]
        elif any(tag.startswith(text) for tag in MARKERS):
            state["buffer"] = text
            return
        else:
            if state["phase"] != "header":
                yield state["phase"], "<"
            text = text[1:]
