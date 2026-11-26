"""有界内部 JSON/NDJSON 控制连接；KV 数据不经过 HTTP。"""

import json
import urllib.request
import urllib.error

from .protocol import MAX_META_BYTES


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def call(url, path, value=None, *, token, timeout=30):
    """/** 打开内部控制连接；调用者负责关闭响应，流式结果逐行读取。 */"""
    data = None if value is None else dumps(value)
    if data is not None and len(data) > MAX_META_BYTES:
        raise ValueError("control request too large")
    request = urllib.request.Request(url.rstrip("/") + path, data=data,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def rpc(url, path, value=None, *, token, timeout=30):
    with call(url, path, value, token=token, timeout=timeout) as response:
        data = response.read(MAX_META_BYTES + 1)
        if len(data) > MAX_META_BYTES:
            raise ValueError("control response too large")
        return json.loads(data)


def events(response):
    """/** 解析持续内部结果流；缺少终态由上层判错，不能将断流记为完成。 */"""
    while True:
        line = response.readline(MAX_META_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_META_BYTES or not line.endswith(b"\n"):
            raise ValueError("invalid internal result frame")
        yield json.loads(line)
