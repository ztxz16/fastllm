"""Attention DP/EP 的 PD 接口；所有 rank 的计算仍由一个原生联合轮次驱动。"""

from . import _core

from collections import deque
from dataclasses import dataclass, field
import threading
import json

from ..qwen_moe_process_session import RoundTransport
from ..qwen3_process_chat import validate_request
from .protocol import new_id


@dataclass
class Call:
    rank: int
    request: dict
    done: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: object = None


class MoePDGroup:
    """/**
     * 一个 CPU 所有者串行处理整组控制流；可把不同 rank 的 Decode 合成一轮。
     * 存储命令也经同一队列，禁止服务线程直接跨过计算轮次写入 EP worker。
     */"""

    def __init__(self, *args, **kwargs):
        return _core.invoke('moe.MoePDGroup.__init__', locals())

    def __enter__(self):
        return _core.invoke('moe.MoePDGroup.__enter__', locals())

    def _run(self):
        return _core.invoke('moe.MoePDGroup._run', locals())

    def call(self, rank, request):
        return _core.invoke('moe.MoePDGroup.call', locals())

    def __exit__(self, exc_type, exc, traceback):
        return _core.invoke('moe.MoePDGroup.__exit__', locals())


class MoePDView:
    """/** 一个 Attention 副本的接口视图；内部始终经组协调器，不能独立进入 EP 通信。 */"""

    storage = "continuous"
    page_size = 0

    def __init__(self, owner, rank):
        return _core.invoke('moe.MoePDView.__init__', locals())

    @property
    def state(self):
        return _core.invoke('moe.MoePDView.state', locals())

    def pd_command(self, request):
        return _core.invoke('moe.MoePDView.pd_command', locals())

    def generate(self, request, *, cancel_event=None, on_token=None):
        return _core.invoke('moe.MoePDView.generate', locals())
