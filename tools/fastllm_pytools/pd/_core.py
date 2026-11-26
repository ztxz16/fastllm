"""C++ PD 模块加载与参数适配；不包含调度、传输或资源决策。"""

import importlib
import importlib.util
import os
import sys
import queue
import threading
from pathlib import Path

_module = None
_load_lock = threading.RLock()


def load():
    with _load_lock:
        return _load_native()


def _load_native():
    global _module
    if _module is not None:
        return _module
    try:
        candidate = importlib.import_module(__package__ + "._pd_core")
    except ModuleNotFoundError as error:
        if error.name != __package__ + "._pd_core":
            raise
        configured = os.environ.get("FT_PD_CORE_LIBRARY")
        root = Path(__file__).resolve().parents[3]
        paths = [Path(configured)] if configured else [
            root / "build-process-tp" / "_pd_core.so",
            root / "build" / "_pd_core.so",
        ]
        path = next((p for p in paths if p.is_file()), None)
        if path is None:
            raise RuntimeError(
                "PD requires the C++ _pd_core module. Build target fastllm_pd_core on the server "
                "or set FT_PD_CORE_LIBRARY; Python core fallback is disabled."
            ) from error
        spec = importlib.util.spec_from_file_location(__package__ + "._pd_core", path)
        candidate = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = candidate
        try:
            spec.loader.exec_module(candidate)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
    if candidate.abi_version != 1:
        raise RuntimeError("incompatible native PD ABI; rebuild fastllm_pd_core")
    candidate.configure(__package__)
    _module = candidate
    return candidate


def invoke(name, arguments):
    return load().invoke(name, arguments)


def stream(name, arguments):
    """/**
     * 将原生回调适配成 Python 迭代器，逐条确认以保留 yield 背压。
     * 本层只搬运事件与取消信号；原生层决定重试、终止及资源释放。
     * 关闭迭代器时等待原生调用退出，不提前销毁仍被后台调用持有的对象。
     * @param name 原生入口名称。
     * @param arguments 原有方法参数；由绑定层补充事件回调和关闭信号。
     * @return 内部结果事件迭代器；原生错误在调用方重新抛出。
     */"""
    native = load()
    messages = queue.Queue(maxsize=1)
    closed = threading.Event()
    acknowledged = threading.Event()
    stopped = object()

    def emit(event):
        acknowledged.clear()
        while not closed.is_set():
            try:
                messages.put(event, timeout=0.05)
                break
            except queue.Full:
                continue
        while not closed.is_set():
            if acknowledged.wait(0.05):
                return
        raise InterruptedError("result stream closed")

    def produce():
        outcome = stopped
        try:
            native.invoke(name, dict(arguments, _emit=emit, _closed=closed))
        except BaseException as error:
            outcome = error
        finally:
            while not closed.is_set():
                try:
                    messages.put(outcome, timeout=0.05)
                    break
                except queue.Full:
                    continue

    thread = threading.Thread(target=produce, name="ft-pd-native-stream", daemon=True)
    thread.start()
    completed = False
    try:
        while True:
            event = messages.get()
            if event is stopped:
                completed = True
                break
            if isinstance(event, BaseException):
                raise event
            yield event
            acknowledged.set()
    finally:
        closed.set()
        if not completed:
            native.cancel_stream(name, arguments)
        acknowledged.set()
        # 原生调用执行既定截止时间和排空规则；适配层不得提前释放核心对象。
        thread.join()
