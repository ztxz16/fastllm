"""Preserve generated DSML for the model lifetime, with bounded RAM usage.

Old tool replies can remain in an active conversation long after their KV cache
was snapshotted. Evicting their original text changes the rendered prefix. Keep
a small RAM LRU and spill older entries to a private, model-local SQLite store;
only unloading the model releases that history, along with its KV caches.
"""
import json
import sqlite3
import tempfile
from collections import OrderedDict
from threading import RLock


class ToolHistory:
    def __init__(self, max_entries=128, max_bytes=8 * 1024 * 1024):
        self.entries = OrderedDict()
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.bytes = 0
        self.lock = RLock()
        self._spill_dir = None
        self._spill = None

    @staticmethod
    def _serialized(value):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

    def _spill_entry(self, key, saved):
        if self._spill is None:
            self._spill_dir = tempfile.TemporaryDirectory(prefix='fastllm-tool-history-')
            self._spill = sqlite3.connect(self._spill_dir.name + '/history.sqlite',
                                          check_same_thread=False)
            # This store only lives as long as the in-process model/KV cache.
            # It need not survive a crash or fsync every generated tool reply.
            self._spill.execute('PRAGMA journal_mode=MEMORY')
            self._spill.execute('PRAGMA synchronous=OFF')
            self._spill.execute('CREATE TABLE history (key TEXT PRIMARY KEY, saved TEXT NOT NULL)')
        self._spill.execute('INSERT OR REPLACE INTO history VALUES (?, ?)',
                            (self._serialized(key), self._serialized(saved)))
        self._spill.commit()

    def _trim(self):
        while self.entries and (len(self.entries) > self.max_entries or self.bytes > self.max_bytes):
            key, saved = next(iter(self.entries.items()))
            # Persist successfully before dropping the only exact copy.
            self._spill_entry(key, saved)
            self.entries.pop(key)
            self.bytes -= saved[2]

    def close(self):
        with self.lock:
            if self._spill is not None:
                self._spill.close()
                self._spill = None
            if self._spill_dir is not None:
                self._spill_dir.cleanup()
                self._spill_dir = None
            self.entries.clear()
            self.bytes = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
        with self.lock:
            old = self.entries.pop(key, None)
            if old:
                self.bytes -= old[2]
            self.entries[key] = (value, raw, size, thinking, reasoning)
            self.bytes += size
            # A reused call ID must not expose an older spilled version after
            # the new in-memory value is evicted.
            if self._spill is not None:
                self._spill.execute('DELETE FROM history WHERE key = ?', (self._serialized(key),))
                self._spill.commit()
            self._trim()

    def restore(self, message, thinking=False):
        signature = self.signature(message)
        if signature is None:
            return None
        key, value = signature
        with self.lock:
            saved = self.entries.get(key)
            if saved is None and self._spill is not None:
                row = self._spill.execute('SELECT saved FROM history WHERE key = ?',
                                          (self._serialized(key),)).fetchone()
                if row is not None:
                    saved = json.loads(row[0])
            if saved is None or self._serialized(saved[0]) != self._serialized(value) or saved[3] != thinking:
                return None
            # Responses clients may omit reasoning entirely. Restore only the
            # server's own output, and reject explicitly changed reasoning.
            reasoning = message.get('reasoning_content')
            if reasoning and (not thinking or reasoning != saved[4]):
                return None
            if key in self.entries:
                self.entries.move_to_end(key)
            return saved[1]
