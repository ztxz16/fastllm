"""Attention DP/EP 的 PD 接口；所有 rank 的计算仍由一个原生联合轮次驱动。"""

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
        self.transport = RoundTransport(*args, **kwargs)
        self.config = json.loads((self.transport.model / "config.json").read_text())
        self.condition = threading.Condition()
        self.pending = deque()
        self.stopping = False
        self.failure = None
        self.views = []

    def __enter__(self):
        self.transport.__enter__()
        if self.transport.ready.get("pd_storage_protocol") != 1:
            self.transport.close()
            raise RuntimeError("MoE coordinator needs PD protocol; rebuild on server")
        self.views = [MoePDView(self, rank) for rank in range(len(self.transport.devices))]
        self.thread = threading.Thread(target=self._run, name="pd-ep-joint-rounds", daemon=True)
        self.thread.start()
        return self

    def _run(self):
        batch = []
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending or self.stopping)
                    if self.stopping and not self.pending:
                        return
                    batch, occupied = [], set()
                    rest = deque()
                    while self.pending:
                        call = self.pending.popleft()
                        if call.rank in occupied:
                            rest.append(call)
                        else:
                            batch.append(call)
                            occupied.add(call.rank)
                    self.pending = rest
                slots = [None] * len(self.views)
                for call in batch:
                    slots[call.rank] = call.request
                reply = self.transport.exchange("pd", requests=slots)
                results = reply.get("results")
                if not isinstance(results, list) or len(results) != len(self.views):
                    raise RuntimeError("invalid MoE PD joint reply")
                for call in batch:
                    call.result = results[call.rank]
                    call.done.set()
                batch = []
        except BaseException as error:
            with self.condition:
                self.failure = error
                for call in [*batch, *self.pending]:
                    call.error = error
                    call.done.set()
                self.pending.clear()
                self.condition.notify_all()

    def call(self, rank, request):
        call = Call(rank, request)
        with self.condition:
            if self.stopping or self.failure is not None:
                raise RuntimeError(f"EP coordinator unavailable: {self.failure}")
            self.pending.append(call)
            self.condition.notify_all()
        # 原生 exchange 有整体超时；不能在命令仍可能执行时提前宣告取消完成。
        call.done.wait()
        if call.error is not None:
            raise RuntimeError("EP joint operation failed") from call.error
        return call.result

    def __exit__(self, exc_type, exc, traceback):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.thread.join()
        if self.failure is not None:
            self.transport._dispose("failed")
        else:
            self.transport.close()


class MoePDView:
    """/** 一个 Attention 副本的接口视图；内部始终经组协调器，不能独立进入 EP 通信。 */"""

    storage = "continuous"
    page_size = 0

    def __init__(self, owner, rank):
        self.owner, self.rank = owner, rank
        self.devices = [owner.transport.devices[rank]]
        self.timeout = owner.transport.timeout
        self.ready = owner.transport.ready

    @property
    def state(self):
        return "failed" if self.owner.failure is not None else self.owner.transport.state

    def pd_command(self, request):
        return self.owner.call(self.rank, request)

    def generate(self, request, *, cancel_event=None, on_token=None):
        """/** 完整服务也走同一联合计算和采样入口，用于非 PD 对照及 Router 托管服务。 */"""
        validate_request(request, self.owner.config["vocab_size"])
        context = new_id()
        sampling = {key: request.get(key, default) for key, default in
                    (("temperature", 0.0), ("top_k", 0), ("top_p", 1.0), ("seed", 0))}
        output = []
        try:
            for index in range(request["max_new_tokens"]):
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("full MoE request cancelled")
                if index == 0:
                    command = dict(action="full_prefill", token_ids=request["token_ids"],
                        capacity=len(request["token_ids"]) + request["max_new_tokens"] - 1)
                else:
                    command = dict(action="step", input=output[-1])
                reply = self.pd_command(dict(command, first=index == 0, context=context, sampling=sampling))
                token = reply["token"]
                output.append(token)
                if on_token is not None:
                    on_token(token)
                if token in request.get("eos_token_ids", []):
                    break
            return {"token_ids": output}
        finally:
            if self.state == "ready":
                self.pd_command(dict(action="end"))
                self.pd_command(dict(action="sampling_end", context=context))
