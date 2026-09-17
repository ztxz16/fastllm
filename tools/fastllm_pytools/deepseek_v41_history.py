"""Bounded, model-local preservation of generated DSML for tool continuations."""
import json
from collections import OrderedDict
from threading import RLock


class ToolHistory:
    def __init__(self, max_entries=128, max_bytes=8 * 1024 * 1024):
        self.entries = OrderedDict()
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.bytes = 0
        self.lock = RLock()

    @staticmethod
    def signature(message):
        calls = message.get('tool_calls') or []
        if not calls:
            return None
        ids, values = [], []
        for call in calls:
            call_id = call.get('id')
            function = call.get('function') or {}
            if not call_id:
                return None
            arguments = function.get('arguments', '{}')
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True, allow_nan=False)
            except (ValueError, TypeError):
                return None
            ids.append(call_id)
            values.append((function.get('name'), arguments))
        if len(set(ids)) != len(ids):
            return None
        content = message.get('content') or ''
        if not isinstance(content, str):
            return None
        # Clients may omit an assistant message containing only whitespace.
        content = content if content.strip() else ''
        return tuple(ids), (content, tuple(values))

    def remember(self, raw, message, thinking=False):
        signature = self.signature(message)
        if signature is None:
            return
        key, value = signature
        reasoning = message.get('reasoning_content') or ''
        size = (len(raw.encode('utf-8')) + len(repr(signature).encode('utf-8'))
                + len(reasoning.encode('utf-8')))
        if size > self.max_bytes or self.max_entries <= 0:
            return
        with self.lock:
            old = self.entries.pop(key, None)
            if old:
                self.bytes -= old[2]
            self.entries[key] = (value, raw, size, thinking, reasoning)
            self.bytes += size
            while len(self.entries) > self.max_entries or self.bytes > self.max_bytes:
                self.bytes -= self.entries.popitem(last=False)[1][2]

    def restore(self, message, thinking=False):
        signature = self.signature(message)
        if signature is None:
            return None
        key, value = signature
        with self.lock:
            saved = self.entries.get(key)
            if saved is None or saved[0] != value or saved[3] != thinking:
                return None
            # Responses clients may omit reasoning entirely. Restore only the
            # server's own output, and reject explicitly changed reasoning.
            reasoning = message.get('reasoning_content')
            if reasoning and (not thinking or reasoning != saved[4]):
                return None
            self.entries.move_to_end(key)
            return saved[1]
