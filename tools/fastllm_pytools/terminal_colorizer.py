"""Color FastLLM event lines for a terminal while keeping the log plain text.

The service is commonly started as ``server 2>&1 | tee server.log``.  In that
layout the service stdout is a pipe, so an ``isatty(stdout)`` check cannot
decide whether a human terminal is present.  This small process owns the
plain-text log file and writes a separately colorized copy to ``/dev/tty``.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import OrderedDict
import sys
from typing import Iterable, TextIO


ANSI_RESET = "\033[0m"
def _ansi256(code: int) -> str:
    return f"\033[38;5;{code}m"


COLORS = {
    "cache": _ansi256(178),      # cache fallback; L1/L2/L3/MISS are selected below
    "vision": _ansi256(45),
    "prefill": _ansi256(117),
    "decode": "\033[32m" + _ansi256(120),  # light green (46 -> 84 -> 120)
    "output": _ansi256(40),
    "tool": _ansi256(201),
    "pstate": _ansi256(129),
    "warning": _ansi256(220),
    "error": "\033[31m" + _ansi256(196),
    "startup": _ansi256(250),
    "progress": _ansi256(82),
}

# Restoring the prefix cache is neither decode nor a warning: pale yellow.
CACHE_RESTORE_COLOR = _ansi256(229)
# Capacity is cache bookkeeping: light gold, still in the cache family.
CACHE_CAPACITY_COLOR = _ansi256(222)
# "Native decode average" summarises the decode phase: stay green, but a
# clearly darker shade than the live decode line (46) at a glance.
DECODE_AVERAGE_COLOR = _ansi256(34)

# Every cache line is yellow; the tier is readable by shade: L1 bright, L2 mid
# gold, L3/SSD darkest. Cache misses keep an orange-yellow so they do not look
# like a warning (warning stays 220).
CACHE_COLORS = {
    "L1": _ansi256(226),
    "L2": _ansi256(178),
    "L3": _ansi256(136),
    "MISS": _ansi256(214),
}

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_EVENT_RE = re.compile(r"\[fastllm-event\].*?\bcategory=(?P<category>[^\s]+)")
_EVENT_NAME_RE = re.compile(r"\bevent=(?P<event>[^\s]+)")
_SOURCE_TIER_RE = re.compile(r"\bsource_tier=(?P<tier>L1|L2|L3|MISS)\b")

# Presentation only. Producers emit the same complete stream in user/debug.
# Each displayed value is copied from one original event; no aggregation or
# replacement measurements belong in this module.
_USER_EVENT_FIELDS = {
    ("startup", "request_received"): {"path"},
    ("cache", "cache_candidates"): {
        "l1_resident_tokens", "l2_host_tokens", "combined_local_tokens", "l2_query_status",
    },
    ("cache", "cache_restore_begin"): {"tier", "phase", "candidate_tokens"},
    ("cache", "cache_result"): {
        "source_tier", "promoted_to", "source_tokens", "effective_reused_tokens",
        "lookup_ms", "restore_ms", "read_mib_s", "restore_tokens_s", "reason",
        "read_ms", "import_ms", "restore_total_ms", "restore_timing_scope", "restore_ms_scope",
    },
    ("prefill", "prefill"): {"tokens"},
    ("output", "first_token"): {"ttft_ms"},
    ("output", "request_finished"): {
        "status", "output_tokens", "cached_input_tokens",
        "effective_reused_tokens", "missed_input_tokens", "elapsed_ms",
    },
    ("tool", "tool_call"): {"count"},
    ("vision", "vision_cache_lookup_result"): {"hit", "lookup_ms", "bytes"},
    # The existing mixed GPU/RAM tier label is not evidence of GPU residency.
    ("vision", "vision_cache_restore"): {"bytes", "vision_ssd_load_mib_s"},
    ("vision", "vision_encode_end"): {
        "elapsed_ms", "features", "cache_hits", "cache_misses",
    },
}
_USER_EVENT_FIELDS.update({
    ("progress", "phase_status"): {
        "phase", "detail", "phase_elapsed_ms", "idle_ms", "completed_tokens", "total_tokens",
    },
    # One raw heartbeat contains native measurements and service capacity.
    # Internal scheduler progress and Python callback diagnostics stay raw/debug.
    ("progress", "request_status"): {
        "measurement", "phase", "native_interval_tokens_s", "snapshot_kind", "native_status_state",
        "native_rate_state", "prefill_completed_tokens",
        "prefill_total_tokens", "native_output_tokens",
        "l1_capacity_state", "l1_used_tokens", "l1_available_tokens", "l1_total_tokens",
        "l2_capacity_state", "l2_used_gib", "l2_available_gib", "l2_limit_gib",
        "ssd_capacity_state", "ssd_used_gib", "ssd_available_gib", "ssd_limit_gib",
    },
    ("prefill", "prefill_summary"): {"missed_input_tokens", "prefill_tokens"},
    ("prefill", "prefill_speed"): {"prefill_tokens", "elapsed_ms", "prefill_tokens_s"},
    ("output", "native_first_token"): {"ttft_ms"},
    ("output", "native_generation_summary"): {"native_rate_available", "native_average_tokens_s"},
    ("vision", "vision_encode_begin"): {"images", "video"},
})
_USER_ID_FIELDS = {"source", "category", "request", "request_id", "public_request",
                   "public_request_id", "native_handle", "native_generation", "event",
                   "timestamp_utc", "snapshot_timestamp_utc"}
_FIELD_RE = re.compile(
    r"(?<!\S)([A-Za-z_][A-Za-z0-9_]*)=(.*?)(?=\s+[A-Za-z_][A-Za-z0-9_]*=|$)")
_READY_RE = re.compile(r"^INFO:\s+Uvicorn running on ")
# The logger prefix already carries the wall-clock time; the projected line
# keeps it verbatim instead of dropping the only absolute timestamp.
_PREFIX_TIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)\s")
_TRACEBACK_CONTINUATION_RE = re.compile(
    r"^(?:[ \t]+\S|(?:[A-Za-z_][\w.]*)(?:Error|Exception|Interrupt):)")


def observability_mode(value: str | None = None) -> str:
    """Resolve the presentation mode without changing the server itself."""

    value = os.environ.get("FASTLLM_OBSERVABILITY_MODE", "off") if value is None else value
    value = str(value).strip().lower()
    return value if value in {"off", "user", "debug"} else "off"


def strip_ansi(text: str) -> str:
    """Return *text* without terminal control sequences."""

    return _ANSI_RE.sub("", text)


def _legacy_category(line: str) -> str | None:
    """Map older human-readable FastLLM prefixes to the same palette."""

    if "[Prefix" in line or "[PrefixCache]" in line:
        return "cache"
    if "[Vision]" in line:
        return "vision"
    if "[Prompt]" in line or "Prefill" in line:
        return "prefill"
    if "[Decode]" in line or "[Output]" in line:
        return "decode"
    if "Tool parser" in line or "tool_parser" in line or "tool call" in line:
        return "tool"
    if "P-State" in line or "pstate" in line.lower():
        return "pstate"
    if "WARNING" in line or " warning" in line.lower():
        return "warning"
    if "ERROR" in line or " error" in line.lower():
        return "error"
    if "[Fastllm]" in line or "server_start" in line:
        return "startup"
    return None


def category_for_line(line: str) -> str | None:
    """Return the event category used to color *line*, if one is known."""

    match = _EVENT_RE.search(line)
    if match:
        return match.group("category")
    return _legacy_category(line)


def project_user_line(line: str) -> str | None:
    """Select core fields from one debug event without deriving any values."""

    plain = strip_ansi(line)
    match = _EVENT_RE.search(plain)
    if not match:
        upper = plain.upper()
        if (_READY_RE.match(plain) or category_for_line(plain) in {"warning", "error"}
                or any(word in upper for word in ("FATAL", "TRACEBACK", "EXCEPTION"))):
            return plain
        return None
    body = plain[plain.index("[fastllm-event]") + len("[fastllm-event]"):].rstrip()
    fields = list(_FIELD_RE.finditer(body))
    values = {item[1]: item[2] for item in fields}
    category, event = values.get("category"), values.get("event")
    if category in {"warning", "error"}:
        return plain
    allowed = _USER_EVENT_FIELDS.get((category, event))
    if allowed is None:
        return None
    # Scheduler progress currently has no trustworthy native generation window;
    # Python callback/request-average rates are diagnostics, so neither is
    # selected as native decode. The raw stream retains all original fields.
    if category == "prefill" and values.get("source") != "native":
        return None
    # Native generation completion may precede tool parsing. Only the API
    # lifecycle event represents request completion in the user projection.
    if event == "request_finished" and values.get("source") != "python":
        return None
    identities = _USER_ID_FIELDS.copy()
    if any(values.get(key) for key in ("public_request_id", "public_request", "request_id")):
        identities.discard("request")
    selected = [item[0] for item in fields if item[1] in allowed or item[1] in identities]
    suffix = "\n" if plain.endswith("\n") else ""
    return "[fastllm-event] " + " ".join(selected) + suffix


def user_line_allowed(line: str) -> bool:
    """Return whether a ready-phase line has a user projection."""

    return project_user_line(line) is not None


def colorize_line(line: str) -> str:
    """Add ANSI color to one line, leaving unknown lines unchanged."""

    plain = strip_ansi(line)
    category = category_for_line(plain)
    color = None
    if category == "cache":
        source = _SOURCE_TIER_RE.search(plain)
        color = CACHE_COLORS.get(source.group("tier")) if source else None
    color = color or COLORS.get(category or "")
    if not color:
        return plain
    body = plain[:-1] if plain.endswith("\n") else plain
    suffix = "\n" if plain.endswith("\n") else ""
    return f"{color}{body}{ANSI_RESET}{suffix}"


# Translation applies to fixed user labels only, before raw values are inserted.
# Debug/raw/startup and message/model payloads never pass through this table.
_USER_ZH_LABELS = {
    "Request started": "请求开始", "L1 prefix candidate": "L1 可复用前缀",
    "L2 prefix candidate": "L2 可复用前缀", "L1 cache": "L1 缓存",
    "L2 cache": "L2 缓存", "SSD cache": "SSD 缓存", "Prefix cache": "前缀缓存",
    "reading": "正在读取", "restoring": "正在恢复", "hit": "命中",
    "MISS": "未命中", "HIT": "命中", "read": "读取速度", "restore": "恢复速度",
    "restore time": "恢复耗时", "total restore time": "恢复总耗时", "import time": "导入耗时", "Cache": "缓存", "Uncached": "未命中缓存",
    "Prefill": "预填充", "First token": "首 token 耗时",
    "Native decode": "原生解码", "Native decode average": "平均原生解码速度",
    "Preparing input": "准备输入", "Image processing": "图片处理",
    "Restoring prefix cache": "恢复前缀缓存", "Finishing": "正在收尾",
    "Request active": "请求处理中", "speed pending": "等待速度测量",
    "waiting for progress": "等待新的进度", "tokens completed": "tokens 已完成",
    "output tokens": "输出 tokens", "L1 used": "L1 已用",
    " / free ": " / 剩余 ", " / total ": " / 总量 ", " free ": " 剩余 ",
    "L1 capacity": "L1 容量", "Cache capacity": "缓存容量", "SSD quota": "SSD 配额",
    "capacity unavailable": "容量不可用", "unknown": "未知", "busy": "忙碌",
    "disabled": "已关闭", "unavailable": "不可用", "unsupported": "不支持",
    "initializing": "初始化中", "queued": "排队中", "stat_failed": "容量采样失败", "ready": "可用", "ok": "成功",
    "Image": "图片", "Video": "视频", "Vision": "视觉",
    "processing started": "开始处理", "processing": "处理耗时",
    "cache restored": "缓存已恢复", "Tool call": "工具调用", "count": "数量",
    "Finished": "完成", "Warning": "警告", "Error": "错误", "cache": "缓存",
}


_USER_TERMINAL_HISTORY_LIMIT = 4096


class UserTerminalView:
    """Readable labels over raw values; numbering is presentation identity only."""

    def __init__(self, language: str = "en"):
        if language not in {"en", "zh"}:
            raise ValueError("user language must be en or zh")
        self._language = language
        self._requests = {}
        self._handles = OrderedDict()
        # No timeout: native cleanup may legitimately outlive API cancellation.
        # Retain a fixed number of recent terminal public owners, not forever.
        self._terminal_requests = OrderedDict()

    def _text(self, label):
        return _USER_ZH_LABELS.get(label, label) if self._language == "zh" else label

    def _bind_handle(self, handle, public):
        if not handle or handle == "unknown" or not public or public in self._terminal_requests:
            return
        self._handles[handle] = public
        self._handles.move_to_end(handle)
        while len(self._handles) > _USER_TERMINAL_HISTORY_LIMIT:
            self._handles.popitem(last=False)

    def _remember_terminal(self, key, handle):
        self._terminal_requests[key] = handle
        self._terminal_requests.move_to_end(key)
        while len(self._terminal_requests) > _USER_TERMINAL_HISTORY_LIMIT:
            expired, old_handle = self._terminal_requests.popitem(last=False)
            if self._handles.get(old_handle) == expired:
                self._handles.pop(old_handle, None)

    def _request(self, fields):
        public = next((fields.get(key) for key in
                       ("public_request_id", "public_request", "request_id") if fields.get(key)), None)
        if not public and fields.get("request", "").isdigit() is False:
            public = fields.get("request")
        handle = fields.get("native_handle")
        native_key = "native:" + (handle or fields.get("request", "unknown"))
        key = public or self._handles.get(handle) or native_key
        if public and key not in self._requests and native_key in self._requests:
            self._requests[key] = self._requests.pop(native_key)
        if key not in self._requests:
            self._requests[key] = {"kind": "Vision", "uncached": False, "cache_seen": set()}
        # A raw ID substring is a display label, not a separately counted request.
        label = public[-8:] if public else (handle or fields.get("request", "unknown"))
        self._requests[key]["label"] = label
        if public and handle and handle != "unknown":
            self._bind_handle(handle, public)
        return key, self._requests[key], public

    def line(self, original):
        time_label = _PREFIX_TIME_RE.match(original)
        # A native owner binding is raw identity evidence even though it has no
        # user line. Observe it so a reused handle's new public owner is allowed.
        if "[fastllm-event]" in original:
            identity = {match[1]: match[2] for match in _FIELD_RE.finditer(original.rstrip())}
            if identity.get("event") == "native_observer_bound":
                public = next((identity.get(key) for key in
                    ("public_request_id", "public_request", "request_id") if identity.get(key)), None)
                self._bind_handle(identity.get("native_handle"), public)
        selected = project_user_line(original)
        if selected is None:
            return None
        if "[fastllm-event]" not in selected:
            return selected
        body = selected.split("[fastllm-event]", 1)[1].rstrip()
        fields = {match[1]: match[2] for match in _FIELD_RE.finditer(body)}
        category, event = fields.get("category"), fields.get("event")
        public = next((fields.get(name) for name in
            ("public_request_id", "public_request", "request_id") if fields.get(name)), None)
        if not public and fields.get("request", "").isdigit() is False:
            public = fields.get("request")
        owner = public or self._handles.get(fields.get("native_handle"))
        if owner in self._terminal_requests and category not in {"warning", "error"}:
            return None  # Late native facts remain intact in debug/raw only.
        # Callback first-token and first-chunk prefill are diagnostics. The
        # native events below carry actual committed output and complete counts.
        if event in {"first_token", "prefill"}:
            return None
        key, state, public = self._request(fields)
        prefix = f"[{state['label']}] "
        output = []
        if category in {"warning", "error"}:
            message = fields.get("message") or fields.get("reason") or (event or category).replace("_", " ")
            output.append(f"{self._text(category.title())}: {message}")
        elif event == "request_received":
            output.append("Request started" + (f" | ID: {public}" if public else ""))
        elif event == "phase_status":
            phase = fields.get("phase", "unknown")
            line = "Status: " + phase.replace("_", " ")
            if fields.get("detail"):
                line += f" | {fields['detail'].replace('_', ' ')}"
            if "phase_elapsed_ms" in fields:
                line += f" | elapsed: {fields['phase_elapsed_ms']} ms"
            if "idle_ms" in fields:
                line += f" | last update: {fields['idle_ms']} ms ago"
            if "completed_tokens" in fields:
                line += f" | completed: {fields['completed_tokens']} tokens"
            if "total_tokens" in fields:
                line += f" | total: {fields['total_tokens']} tokens"
            output.append(line)
            output.append(self._text("Request started") + (f" | ID: {public}" if public else ""))
        elif event == "cache_candidates":
            status = fields.get("l2_query_status", "unknown")
            l1 = fields.get("l1_resident_tokens")
            l2 = fields.get("l2_host_tokens")
            if l1 is not None:
                output.append(f"{self._text('L1 prefix candidate')}: {l1} tokens")
            if status == "ok" and l2 is not None:
                output.append(f"{self._text('L2 prefix candidate')}: {l2} tokens")
            elif status != "ok":
                output.append(f"{self._text('L2 cache')}: {self._text(status)}")
            elif l2 is None:
                output.append(self._text("L2 cache") + ": " + self._text("unknown"))
        elif event == "cache_restore_begin":
            tier = fields.get("tier", "")
            tokens = fields.get("candidate_tokens")
            if tier.startswith("L3"):
                output.append(self._text("SSD cache") + ": " + self._text("reading") + (f" {tokens} tokens" if tokens else ""))
            elif tier.startswith("L2"):
                output.append(self._text("L2 cache") + ": " + self._text("restoring") + (f" {tokens} tokens" if tokens else ""))
        elif event == "cache_result":
            fingerprint = tuple(fields.get(k) for k in (
                "source_tier", "promoted_to", "source_tokens", "effective_reused_tokens",
                "read_mib_s", "restore_tokens_s", "restore_ms"))
            tier = fields.get("source_tier", "unknown")
            if fingerprint in state["cache_seen"]:
                return None
            state["cache_seen"].add(fingerprint)
            label = {"L1": "L1 cache", "L2": "L2 cache", "L3": "SSD cache"}.get(tier)
            tokens = fields.get("effective_reused_tokens")
            if tier == "MISS":
                output.append(self._text("Prefix cache") + ": " + self._text("MISS"))
            elif label and tokens is not None:
                line = f"{self._text(label)}: {self._text('hit')} {tokens} tokens"
                if fields.get("promoted_to") and fields["promoted_to"] != "NONE":
                    line += f" -> {fields['promoted_to']}"
                output.append(line)
            speed = []
            timing_fields = (("read_mib_s", "read", "MiB/s"),
                             ("restore_tokens_s", "restore", "tokens/s"))
            if "restore_total_ms" in fields:
                timing_fields = timing_fields + (("restore_total_ms", "total restore time", "ms"),)
            elif "import_ms" in fields:
                timing_fields = timing_fields + (("import_ms", "import time", "ms"),)
            elif "restore_ms" in fields:
                # Historical records lack an explicit scope; label only the
                # actual native import timer, never the total-rate denominator.
                timing_fields = timing_fields + (("restore_ms", "import time", "ms"),)
            for name, title, unit in timing_fields:
                if name in fields and not fields[name].startswith("-"):
                    speed.append(f"{self._text(title)}: {fields[name]} {unit}")
            if speed:
                output.append(("SSD " if tier == "L3" else self._text("Cache") + " ") + " | ".join(speed))
        elif event == "prefill_summary":
            if "missed_input_tokens" in fields:
                output.append(f"{self._text('Uncached')}: {fields['missed_input_tokens']} tokens")
                state["uncached"] = True
        elif event == "prefill_speed":
            parts = []
            if "prefill_tokens" in fields:
                parts.append(f"{fields['prefill_tokens']} tokens")
            if "prefill_tokens_s" in fields:
                parts.append(f"{fields['prefill_tokens_s']} tokens/s")
            if "elapsed_ms" in fields:
                parts.append(f"{fields['elapsed_ms']} ms")
            if parts:
                output.append(self._text("Prefill") + ": " + " | ".join(parts))
        elif event == "native_first_token":
            if "ttft_ms" in fields:
                output.append(f"{self._text('First token')}: {fields['ttft_ms']} ms")
        elif event == "request_status":
            if fields.get("native_status_state") not in {None, "observed"}:
                return None
            phase = fields.get("phase", "unknown")
            labels = {"preparing": "Preparing input", "vision": "Image processing",
                      "cache_restore": "Restoring prefix cache", "prefill": "Prefill",
                      "decode": "Native decode", "finishing": "Finishing",
                      "unknown": "Request active", "queued": "queued",
                      "native_closed": "Finishing"}
            parts = [self._text(labels.get(phase, "Request active"))]
            if phase == "prefill":
                if "prefill_completed_tokens" in fields:
                    done = fields["prefill_completed_tokens"]
                    total = fields.get("prefill_total_tokens")
                    parts.append(done + ("/" + total if total else "") + " " + self._text("tokens completed"))
                if fields.get("native_rate_state") == "fresh" and "native_interval_tokens_s" in fields:
                    parts.append(fields["native_interval_tokens_s"] + " tokens/s")
                else:
                    parts.append(self._text("speed pending" if fields.get("native_rate_state") != "stale"
                                            else "waiting for progress"))
            elif phase in {"decode", "finishing"}:
                if fields.get("native_rate_state") == "fresh" and "native_interval_tokens_s" in fields:
                    parts.append(fields["native_interval_tokens_s"] + " tokens/s")
                else:
                    parts.append(self._text("speed pending" if fields.get("native_rate_state") != "stale"
                                            else "waiting for progress"))
                if "native_output_tokens" in fields:
                    parts.append(fields["native_output_tokens"] + " " + self._text("output tokens"))
            output.append(" | ".join(parts))
            # Capacity is a separate projection of this SAME raw event. Event
            # identity/phase selection changes presentation frequency only.
            capacity = []
            if fields.get("l1_capacity_state") == "ok" and all(
                    key in fields for key in ("l1_used_tokens", "l1_available_tokens", "l1_total_tokens")):
                capacity.append(self._text("L1 used") + " " + fields["l1_used_tokens"] + self._text(" / free ") +
                             fields["l1_available_tokens"] + self._text(" / total ") +
                             fields["l1_total_tokens"] + " tokens")
            else:
                capacity.append(self._text("L1 capacity") + " " + self._text(fields.get("l1_capacity_state", "unknown")))
            for tier, label in (("l2", "L2"), ("ssd", "SSD quota")):
                state_key = tier + "_capacity_state"
                free_key, limit_key = tier + "_available_gib", tier + "_limit_gib"
                if fields.get(state_key) in {"ok", "ready", "capacity_unavailable"} and all(
                        key in fields for key in (free_key, limit_key)):
                    capacity.append(self._text(label) + self._text(" free ") + fields[free_key] + " / " + fields[limit_key] + " GiB")
                    if fields[state_key] == "capacity_unavailable":
                        capacity.append(self._text(label) + " " + self._text("capacity unavailable"))
                else:
                    capacity.append(self._text(label) + " " + self._text(fields.get(state_key, "unknown")))

            capacity_states = tuple(fields.get(tier + "_capacity_state")
                                    for tier in ("l1", "l2", "ssd"))
            configuration = tuple(fields.get(name) for name in
                                  ("l1_total_tokens", "l2_limit_gib", "ssd_limit_gib"))
            configuration_changed = state.get("capacity_configuration") != configuration
            phase_changed = state.get("capacity_phase") != phase
            state_changed = state.get("capacity_states") != capacity_states
            if phase_changed or state_changed or configuration_changed or fields.get("snapshot_kind") == "native_release":
                output.append(self._text("Cache capacity") + " | " + " | ".join(capacity))
            state["capacity_phase"] = phase
            state["capacity_states"] = capacity_states
            state["capacity_configuration"] = configuration
        elif event == "native_generation_summary":
            if fields.get("native_rate_available") == "1" and "native_average_tokens_s" in fields:
                output.append(self._text("Native decode average") + ": " + fields["native_average_tokens_s"] + " tokens/s")
            elif fields.get("native_rate_available") == "0":
                output.append(self._text("Native decode") + ": " + self._text("unknown"))
        elif event == "vision_encode_begin":
            state["kind"] = {"0": "Image", "1": "Video"}.get(fields.get("video"), "Vision")
            output.append(self._text(state["kind"]) + " " + self._text("processing started"))
        elif event == "vision_cache_lookup_result":
            hit = {"0": "MISS", "1": "HIT"}.get(fields.get("hit"))
            if hit:
                output.append(f"{self._text(state['kind'])} {self._text('cache')}: {self._text(hit)}")
        elif event == "vision_cache_restore":
            line = self._text(state["kind"]) + " " + self._text("cache restored")
            if "vision_ssd_load_mib_s" in fields:
                line += f" | SSD: {fields['vision_ssd_load_mib_s']} MiB/s"
            output.append(line)
        elif event == "vision_encode_end":
            if "elapsed_ms" in fields:
                output.append(f"{self._text(state['kind'])} {self._text('processing')}: {fields['elapsed_ms']} ms")
        elif event == "tool_call":
            line = self._text("Tool call")
            if "count" in fields:
                line += f" | {self._text('count')}: {fields['count']}"
            output.append(line)
        elif event == "request_finished":
            if not state["uncached"] and "missed_input_tokens" in fields:
                output.append(f"{self._text('Uncached')}: {fields['missed_input_tokens']} tokens")
            parts = [self._text(fields["status"])] if "status" in fields else []
            if "output_tokens" in fields:
                parts.append(f"{fields['output_tokens']} {self._text('output tokens')}")
            if "elapsed_ms" in fields:
                parts.append(f"{fields['elapsed_ms']} ms")
            output.append(self._text("Finished") + (": " + " | ".join(parts) if parts else ""))
            self._remember_terminal(key, fields.get("native_handle"))
            self._requests.pop(key, None)
            # Keep the bounded owner binding until a new public owner replaces
            # it, so anonymous late records cannot revive this request either.
        if key in self._terminal_requests:
            self._requests.pop(key, None)  # Late errors must not recreate active state.
        # A logger prefix is emission time. Display the source event/sample
        # capture timestamp verbatim; do not restamp an older sample as current.
        observed_stamp = fields.get("snapshot_timestamp_utc") if event == "request_status" else None
        observed_stamp = observed_stamp or fields.get("timestamp_utc")
        if observed_stamp:
            stamp = f"[{observed_stamp}] "
        elif time_label:
            stamp = f"[{time_label.group(1)}] "
        else:
            stamp = ""
        return "".join(prefix + stamp + line + "\n" for line in output) or None


def _user_line_color(plain: str) -> str | None:
    """Pick the user-terminal color from the same raw event.

    ``request_status`` carries the live phase, so prefill and native decode use
    their reserved palette entries instead of the generic progress color.
    Every other event keeps its existing color.
    """

    category = category_for_line(plain)
    if category == "progress":
        fields = {match[1]: match[2] for match in _FIELD_RE.finditer(plain)}
        if fields.get("event") == "request_status":
            phase = fields.get("phase", "")
            if phase == "prefill":
                return COLORS["prefill"]
            if phase == "cache_restore":
                return CACHE_RESTORE_COLOR
            if phase in {"decode", "finishing"}:
                return COLORS["decode"]
    color = None
    if category == "cache":
        source = _SOURCE_TIER_RE.search(plain)
        color = CACHE_COLORS.get(source.group("tier")) if source else None
    return color or COLORS.get(category or "")


def _cache_tier_color(text: str) -> str | None:
    """Pick the cache shade named by one projected line (L1 < L2 < L3 lightness)."""

    if "SSD" in text or "L3" in text:
        return CACHE_COLORS["L3"]
    if "L2" in text:
        return CACHE_COLORS["L2"]
    if "L1" in text:
        return CACHE_COLORS["L1"]
    if "MISS" in text or "未命中" in text:
        return CACHE_COLORS["MISS"]
    return None


def colorize_user_line(original, visible):
    """Color each visible user line from its original event; change text only."""

    plain = strip_ansi(original)
    suffix = "\n" if plain.endswith("\n") else ""
    replacement = visible[:-1] if visible.endswith("\n") else visible
    base = _user_line_color(plain)
    category = category_for_line(plain)
    painted = []
    for part in replacement.split("\n"):
        if not part:
            painted.append(part)
            continue
        if "缓存容量" in part or "Cache capacity" in part:
            color = CACHE_CAPACITY_COLOR
        elif "平均原生解码速度" in part or "Native decode average" in part:
            color = DECODE_AVERAGE_COLOR
        elif "首 token 耗时" in part or "First token" in part:
            # TTFT closes the prefill phase on screen, so it shares its blue.
            color = COLORS["prefill"]
        elif category == "cache":
            color = _cache_tier_color(part) or base
        else:
            color = base
        painted.append(f"{color}{part}{ANSI_RESET}" if color else part)
    return "\n".join(painted) + suffix


def process_lines(
    lines: Iterable[str],
    log_stream: TextIO,
    terminal_stream: TextIO | None = None,
    mode: str | None = None,
    raw_stream: TextIO | None = None,
    user_stream: TextIO | None = None,
    user_language: str = "en",
) -> None:
    """Keep complete plain logs; project only the ready-phase user terminal."""

    mode = observability_mode(mode)
    ready = False
    traceback_active = False
    user_view = UserTerminalView(language=user_language)
    for line in lines:
        plain = strip_ansi(line)
        log_stream.write(plain)
        log_stream.flush()
        if raw_stream is not None:
            raw_stream.write(plain)
            raw_stream.flush()
        if mode == "off" or not ready:
            # The ready marker itself is part of unchanged startup output.
            if terminal_stream is not None:
                terminal_stream.write(plain)
                terminal_stream.flush()
            if user_stream is not None and mode == "user":
                user_stream.write(plain)
                user_stream.flush()
            if _READY_RE.match(plain):
                ready = True
            continue
        visible = plain
        if mode == "user":
            continuation = traceback_active and (
                not plain.strip() or _TRACEBACK_CONTINUATION_RE.match(plain))
            if not continuation:
                traceback_active = "Traceback (most recent call last):" in plain
                visible = user_view.line(plain)
            if visible is None:
                continue
        if user_stream is not None and mode == "user":
            # Persist exactly the plain user view; the terminal copy keeps color.
            user_stream.write(visible)
            user_stream.flush()
        if terminal_stream is not None:
            terminal_stream.write(colorize_user_line(plain, visible) if mode == "user" else colorize_line(visible))
            terminal_stream.flush()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="plain-text log path")
    parser.add_argument(
        "--raw-log",
        default=None,
        help="optional complete plain-text stream; never filtered by presentation mode",
    )
    parser.add_argument(
        "--user-log",
        default=None,
        help="optional plain user view; server/raw logs stay the complete debug stream",
    )
    parser.add_argument("--tty", default="/dev/tty", help="terminal device (default: /dev/tty)")
    parser.add_argument(
        "--mode",
        choices=("off", "user", "debug"),
        default=None,
        help="presentation mode (default: FASTLLM_OBSERVABILITY_MODE or off)",
    )
    parser.add_argument(
        "--user-language", choices=("en", "zh"), default="en",
        help="language of ready-phase user labels only; debug/raw/startup remain unchanged",
    )
    args = parser.parse_args(argv)

    terminal: TextIO | None = None
    try:
        terminal = open(args.tty, "w", encoding="utf-8", buffering=1)
    except OSError:
        # A detached service still gets a complete plain log.  Do not inject
        # ANSI into stdout, because callers may be collecting it as a file.
        terminal = None
    user_log = (
        open(args.user_log, "a", encoding="utf-8", buffering=1)
        if args.user_log else None
    )
    interrupted = False
    try:
        with open(args.log, "a", encoding="utf-8", buffering=1) as log:
            raw = (
                open(args.raw_log, "a", encoding="utf-8", buffering=1)
                if args.raw_log else None
            )
            try:
                process_lines(sys.stdin, log, terminal, args.mode, raw, user_log, args.user_language)
            except KeyboardInterrupt:
                # SIGINT is expected when the foreground server is stopped.
                # Return its conventional status without printing a traceback
                # into the user's terminal or plain-text server.log.
                interrupted = True
            finally:
                if raw is not None:
                    raw.close()
    finally:
        if user_log is not None:
            user_log.close()
        if terminal is not None:
            terminal.close()
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(_main())
