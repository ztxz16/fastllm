"""正式进程执行器的 PD 适配器；KV 不拷贝到 Python 或 CPU 内存。"""

from . import _core


class NativeReplica:
    """/**
     * 一个 TP 组的本地调度所有者；原生计算串行，已导出的页可独立驻留。
     * 各请求交接 key 与服务身份分别校验，所有原生调用复用会话序号机制。
     */"""

    def __init__(self, session, peer, config, model_id, *, growth=512):
        return _core.invoke('native.NativeReplica.__init__', locals())

    @property
    def layout(self):
        return _core.invoke('native.NativeReplica.layout', locals())

    def refresh_cache(self):
        return _core.invoke('native.NativeReplica.refresh_cache', locals())

    def _command(self, action, key=None, **fields):
        return _core.invoke('native.NativeReplica._command', locals())

    def score(self, waiting):
        return _core.invoke('native.NativeReplica.score', locals())

    def acquire(self, identity, deadline, cancelled):
        return _core.invoke('native.NativeReplica.acquire', locals())

    def wait_capacity(self, prompt, decode, remaining, deadline, cancelled):
        return _core.invoke('native.NativeReplica.wait_capacity', locals())

    def _manifest(self, identity, tokens, regions):
        return _core.invoke('native.NativeReplica._manifest', locals())

    def prefill(self, identity, tokens, *, deadline=None, cancelled=None):
        return _core.invoke('native.NativeReplica.prefill', locals())

    def prepared_prefill(self, identity, tokens, on_layer):
        return _core.invoke('native.NativeReplica.prepared_prefill', locals())

    def export(self, identity, tokens, *, deadline=None, cancelled=None, optional=False):
        return _core.invoke('native.NativeReplica.export', locals())

    def prepare(self, identity, tokens, output_tokens, *, deadline=None, cancelled=None):
        return _core.invoke('native.NativeReplica.prepare', locals())

    def commit(self, identity, tokens):
        return _core.invoke('native.NativeReplica.commit', locals())

    def restore_local(self, identity, tokens):
        return _core.invoke('native.NativeReplica.restore_local', locals())

    def release(self, identity):
        return _core.invoke('native.NativeReplica.release', locals())

    def finish(self, identity):
        return _core.invoke('native.NativeReplica.finish', locals())

    def generate(self, identity, request, cancelled, *, deadline=None):
        return _core.stream('native.NativeReplica.generate', locals())
