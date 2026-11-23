"""CPU-only chat preparation for high-concurrency HTTP serving."""
import asyncio
from array import array
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import os
import pickle
import subprocess
import sys
import zlib

from .chat_template import apply_hf_chat_template, normalize_conversation


def _boundary_tokens(tokenizer):
    from transformers import PreTrainedTokenizerBase

    if (not getattr(tokenizer, "is_fast", False) or
            getattr(tokenizer.apply_chat_template, "__func__", None) is not PreTrainedTokenizerBase.apply_chat_template or
            getattr(tokenizer, "split_special_tokens", False) or
            getattr(tokenizer.backend_tokenizer, "encode_special_tokens", False)):
        return {}
    # Only use the standard HF text -> fast encoding implementation. Custom
    # tokenizers still run their original tokenize=True method in the worker.
    for name in ("__call__", "_call_one", "_encode_plus", "_batch_encode_plus"):
        method = getattr(type(tokenizer), name, None)
        if method and not method.__module__.startswith("transformers.tokenization_utils"):
            return {}
    cfg = json.loads(tokenizer.backend_tokenizer.to_str())
    model = cfg["model"]
    pre = cfg.get("pre_tokenizer") or {}
    steps = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else [pre]
    post = cfg.get("post_processor") or {}
    identity_post = (post.get("type") in (None, "ByteLevel") or
                     (post.get("type") == "TemplateProcessing" and
                      post.get("single") == [{"Sequence": {"id": "A", "type_id": 0}}]))
    if (model["type"] != "BPE" or model.get("dropout") or
            cfg.get("padding") or cfg.get("truncation") or
            (cfg.get("normalizer") or {}).get("type") not in (None, "NFC", "NFKC") or
            not identity_post or
            not steps or steps[-1].get("type") != "ByteLevel" or
            steps[-1].get("add_prefix_space", True) or
            any(s.get("type") != "Split" for s in steps[:-1])):
        return {}
    added = cfg["added_tokens"]
    result = {}
    for t in added:
        marker = t["content"]
        if (not t["special"] or not marker or marker[-1].isspace() or
                any(t[k] for k in ("normalized", "single_word", "lstrip", "rstrip"))):
            continue
        # A longer raw added token starting before or inside this marker
        # could win the match after the right-hand text changes.
        others = (a["content"] for a in added if a["id"] != t["id"] and not a["normalized"])
        if any(marker in other or any(other.startswith(marker[i:]) for i in range(1, len(marker)))
               for other in others):
            continue
        if any(marker.startswith(marker[i:]) for i in range(1, len(marker))):
            continue
        result[t["id"]] = marker
    return result


class ChatEncoder:
    def __init__(self, tokenizer, model_struct, budget):
        self.tokenizer = tokenizer
        self.model_struct = model_struct
        self.markers = _boundary_tokens(tokenizer) if budget else {}
        self.entries = OrderedDict()
        self.used = 0
        self.budget = budget

    def encode(self, messages, options, session):
        messages = normalize_conversation(self.model_struct, messages)
        kwargs = options.get("chat_template_kwargs") or {}
        # Other kwargs can change tokenization itself (padding, truncation,
        # tokenizer_kwargs, etc.). Preserve the checkpoint's full HF call.
        if not (session and self.markers and kwargs.keys() <= {"enable_thinking", "thinking", "reasoning_effort"}):
            return array("i", apply_hf_chat_template(self.tokenizer, messages, tokenize=True,
                                                     thinking_alias=True, **options))
        text = apply_hf_chat_template(self.tokenizer, messages, tokenize=False, thinking_alias=True, **options)
        raw = text.encode("utf-8")
        previous = self.entries.pop(session, None)
        offset = kept = 0
        ids = array("i")
        boundaries = array("I")
        if previous:
            old_text, old_ids, old_bounds, cost = previous
            self.used -= cost
            old_text = memoryview(old_text)
            lo, hi = 0, len(old_bounds) // 2
            while lo < hi:
                mid = (lo + hi) // 2
                end = old_bounds[mid * 2]
                if len(raw) >= end and raw.startswith(old_text[:end]):
                    lo = mid + 1
                else:
                    hi = mid
            if lo:
                boundaries = old_bounds[:2 * lo]
                offset, kept = boundaries[-2:]
                ids = old_ids[:kept]
        # Raw AddedTokens split the text before normalization and BPE. Only
        # resume after a marker whose actual encoding span matched its literal.
        suffix = raw[offset:].decode("utf-8") if offset else text
        encoded = self.tokenizer(suffix, add_special_tokens=False)
        encoding = encoded.encodings[0]
        tail = encoding.ids
        char_pos, byte_pos = 0, offset
        for i, token in enumerate(tail):
            marker = self.markers.get(token)
            if marker is None:
                continue
            span = encoding.token_to_chars(i)
            if span is None or suffix[span[0]:span[1]] != marker:
                continue
            end = span[1]
            byte_pos += len(suffix[char_pos:end].encode("utf-8"))
            char_pos = end
            boundaries.extend((byte_pos, kept + i + 1))
        ids.extend(tail)
        if boundaries:
            cost = sum(map(sys.getsizeof, (session, raw, ids, boundaries))) + 256
            if cost <= self.budget:
                while self.used + cost > self.budget:
                    _, entry = self.entries.popitem(last=False)
                    self.used -= entry[-1]
                self.entries[session] = raw, ids, boundaries, cost
                self.used += cost
        return ids


def _worker(fd):
    from jinja2.exceptions import TemplateError

    with Connection(fd) as conn:
        encoder = ChatEncoder(*conn.recv())
        conn.send_bytes(b"ready")
        while True:
            try:
                args = conn.recv()
            except EOFError:
                return
            try:
                ids = encoder.encode(*args)
            except (ValueError, TemplateError) as error:
                conn.send_bytes(b"E" + str(error).encode("utf-8", errors="replace"))
            else:
                conn.send_bytes(b"I" + ids.tobytes())


class _Lane:
    def __init__(self):
        self.executor = ThreadPoolExecutor(1, thread_name_prefix="chat-tokenizer")
        self.lock = asyncio.Lock()
        self.proc = self.conn = None
        self.disabled = False

    def stop(self):
        if self.conn:
            self.conn.close()
        if self.proc:
            self.proc.wait()
        self.conn = self.proc = None

    def encode(self, snapshot, args):
        if self.disabled:
            return None
        starting = self.conn is None
        try:
            if starting:
                self.conn, child = Pipe()
                # A fresh interpreter avoids inheriting CUDA or re-running an
                # embedding application's __main__ (including unguarded launchers).
                with child:
                    self.proc = subprocess.Popen(
                        [sys.executable, "-m", __name__, str(child.fileno())],
                        pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
                        # HF adds checkpoint tokenizer modules to sys.path at runtime.
                        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path),
                             "CUDA_VISIBLE_DEVICES": "", "USE_TORCH": "0",
                             "TOKENIZERS_PARALLELISM": "false"})
                self.conn.send_bytes(snapshot)
                if self.conn.recv_bytes() != b"ready":
                    raise OSError("Chat input worker failed to initialize")
                starting = False
            self.conn.send(args)
            data = self.conn.recv_bytes()
        except (OSError, EOFError, pickle.PickleError):
            self.stop()
            self.disabled = starting
            logging.exception("Chat input worker unavailable, using in-process input preparation")
            return None
        if data[:1] == b"E":
            raise ValueError(data[1:].decode("utf-8"))
        if data[:1] != b"I":
            raise RuntimeError("Invalid chat input worker response")
        ids = array("i")
        ids.frombytes(memoryview(data)[1:])
        return ids


class ChatInputPool:
    @classmethod
    def for_model(cls, model):
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
        if cpus < 2:
            return None
        count = min(8, cpus // 2)
        # The existing tokenizer-cache switch also disables session reuse here.
        budget = (128 << 20) if model.text_input_token_cache_limit > 0 else 0
        try:
            snapshot = pickle.dumps((model.hf_tokenizer, model.get_struct(), budget),
                                    protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            logging.info("Tokenizer cannot be copied to CPU workers", exc_info=True)
            return None
        return cls(snapshot, count)

    def __init__(self, snapshot, count):
        self.snapshot = snapshot
        self.lanes = [_Lane() for _ in range(count)]
        self.next_lane = 0

    async def prepare(self, messages, options, session):
        if session:
            key = session.encode("utf-8")
            if len(key) > 512 or b"\0" in key:
                raise ValueError("session key must contain at most 512 UTF-8 bytes without NUL")
            index = zlib.crc32(key)
        else:
            index = self.next_lane
            self.next_lane += 1
        lane = self.lanes[index % len(self.lanes)]
        async with lane.lock:
            return await asyncio.get_running_loop().run_in_executor(
                lane.executor, lane.encode, self.snapshot, (messages, options, session))

    def close(self):
        for lane in self.lanes:
            lane.executor.shutdown(wait=True, cancel_futures=True)
            lane.stop()


if __name__ == "__main__":
    _worker(int(sys.argv[1]))
