"""FastLLM 正式入口的进程 TP 适配层；权重、采样和前向仍由 C++/CUDA 执行。"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import sys
import threading
import uuid

from .qwen3_process_chat import prepare_request, validate_request
from .qwen3_process_session import BACKEND, GenerationCancelled, Qwen3ProcessSession


def process_devices(spec):
    """解析可见设备的逻辑编号；裸数字为卡数，不裁剪、不套用线程后端的比例分片。"""
    spec = str(spec or '').strip()
    if spec == 'auto':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is None:
            raise ValueError('--tp auto requires CUDA_VISIBLE_DEVICES; alternatively use --tp 2 or --tp 0,1')
        entries = visible.split(',')
        if not entries or any(not item.strip() or item.strip() == '-1' for item in entries):
            raise ValueError('CUDA_VISIBLE_DEVICES has no usable devices')
        ids = list(range(len(entries)))
    elif re.fullmatch(r'[0-9]+', spec):
        count = int(spec)
        if count > 64:
            raise ValueError('process TP supports at most 64 ranks')
        ids = list(range(count)) if count else [0]
    elif re.fullmatch(r'[0-9]+(?:,[0-9]+)+', spec):
        ids = list(map(int, spec.split(',')))
    else:
        raise ValueError('process TP requires --tp N, --tp 0,1 or --tp auto; weighted/device specs are unsupported')
    if not 1 <= len(ids) <= 64 or len(set(ids)) != len(ids) or any(i > 2147483647 for i in ids):
        raise ValueError('TP devices must be distinct nonnegative logical indices, at most 64 ranks')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and (not visible.strip() or any(i >= len(visible.split(',')) for i in ids) or
                                any(x.strip() in ('', '-1') for x in visible.split(','))):
        raise ValueError('--tp exceeds CUDA_VISIBLE_DEVICES; use logical indices, not physical GPU IDs')
    return ','.join(map(str, ids))


def native_binary(value, name):
    """只使用显式指定、包内或 PATH 中的已有产物，不触发编译或下载。"""
    if value:
        path = Path(value).expanduser().resolve()
    else:
        bundled = Path(__file__).resolve().parent / name
        found = str(bundled) if bundled.is_file() else shutil.which(name)
        if not found:
            raise ValueError(f'{name} not found; build BUILD_PROCESS_TP=ON with USE_CUDA=ON on the server, '
                             'then set --tp-binary/--tp-worker')
        path = Path(found).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f'native executable is missing or not executable: {path}')
    return path


def validate_options(args):
    """进程后端仅接受已实现的参数；拒绝线程后端专用配置，绝不静默忽略。"""
    from .util import make_normal_parser, add_server_args

    parser = make_normal_parser('process validation')
    add_server_args(parser)
    defaults = vars(parser.parse_args([]))
    allowed = {'model', 'path', 'tp', 'tp_backend', 'tp_binary', 'tp_worker', 'tp_output', 'tp_timeout',
               'tp_request_timeout', 'tp_queue_size', 'host', 'port', 'model_name', 'cuda_graph', 'custom_allreduce'}
    compatible = {'dtype': ('auto', 'float16'), 'atype': ('auto', 'float16'),
                  'kv_cache_dtype': ('auto', 'float16'), 'kv_cache_layout': ('auto', 'continuous'),
                  'device': (None, ''), 'FASTLLM_ACTIVATE_NUMA': ('OFF',), 'FASTLLM_NUMA_THREADS': (27,),
                  'max_batch': (-1, 1), 'enable_thinking': ('', 'false'),
                  'prefix_cache': ('', 'false', 'true'), 'cache_history': ('', 'false'), 'cache_fast': ('', 'false')}
    frontend = {'command', 'version', 'prompt', 'max_new_tokens', '_model_config_loaded'}
    for key, value in vars(args).items():
        if key in allowed or key in frontend:
            continue
        if key in compatible:
            valid = value in compatible[key]
        else:
            valid = key in defaults and value == defaults[key]
        if not valid:
            raise ValueError(f'process TP does not support {key}={value!r}; no thread fallback')
    for key in ('FASTLLM_PREFIX_CACHE', 'FASTLLM_CUDA_GRAPH', 'FASTLLM_CUDA_CUSTOM_ALLREDUCE',
                'FASTLLM_CUDA_TP2_P2P_ALLREDUCE'):
        if os.environ.get(key, '').strip().lower() not in ('', '0', 'false', 'off'):
            if key == 'FASTLLM_PREFIX_CACHE':
                raise ValueError('process TP rejects FASTLLM_PREFIX_CACHE; use --prefix-cache true')
            if key == 'FASTLLM_CUDA_GRAPH':
                raise ValueError('process TP rejects FASTLLM_CUDA_GRAPH; use --cuda-graph for Decode Graph')
            if key == 'FASTLLM_CUDA_CUSTOM_ALLREDUCE':
                raise ValueError('process TP rejects FASTLLM_CUDA_CUSTOM_ALLREDUCE; use --custom-allreduce')
            raise ValueError(f'process TP does not support {key} yet')
    if type(getattr(args, 'custom_allreduce', False)) is not bool:
        raise ValueError('custom_allreduce must be boolean')
    if type(getattr(args, 'cuda_graph', False)) is not bool:
        raise ValueError('cuda_graph must be boolean')
    if not sys.platform.startswith('linux'):
        raise ValueError('process TP currently requires Linux + NVIDIA CUDA')
    if hasattr(args, 'max_new_tokens') and (type(args.max_new_tokens) is not int or not 1 <= args.max_new_tokens <= 2048):
        raise ValueError('max-new-tokens must be in 1..2048')
    if getattr(args, 'prompt', None) is not None and (not isinstance(args.prompt, str) or not args.prompt.strip()):
        raise ValueError('prompt must be nonempty text')
    timeout = getattr(args, 'tp_timeout', 180)
    request_timeout = getattr(args, 'tp_request_timeout', 120)
    size = getattr(args, 'tp_queue_size', 8)
    if (type(timeout) not in (int, float) or type(request_timeout) not in (int, float) or
            not math.isfinite(timeout) or not math.isfinite(request_timeout) or
            not 0 < request_timeout <= timeout or type(size) is not int or size < 1):
        raise ValueError('require 0 < tp-request-timeout <= tp-timeout and positive tp-queue-size')
    if getattr(args, 'command', None) in ('serve', 'server'):
        port = getattr(args, 'port', 8080)
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError('port must be in 0..65535')


class ProcessTPModel:
    """模型工厂的进程后端：常驻权重、串行请求、可选前缀快照；提供显式清理与上下文管理。"""

    tp_backend = 'process'
    backend = BACKEND

    def __init__(self, args):
        validate_options(args)
        self.devices = process_devices(getattr(args, 'tp', ''))
        self.model_path = Path(getattr(args, 'path', '') or args.model or '').expanduser().resolve()
        if not self.model_path.is_dir() or not (self.model_path / 'config.json').is_file():
            raise ValueError('process TP requires a local HF Qwen3 model directory; no automatic download')
        self.config = json.loads((self.model_path / 'config.json').read_text())
        if self.config.get('model_type') != 'qwen3' or self.config.get('quantization_config'):
            raise ValueError('process TP only supports unquantized dense Qwen3')
        binary = native_binary(getattr(args, 'tp_binary', ''), 'fastllm_tp_generate')
        worker = native_binary(getattr(args, 'tp_worker', ''), 'fastllm_tp_worker')
        os.environ.setdefault('USE_TORCH', '0')
        from transformers import AutoTokenizer
        self.hf_tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True,
                                                        trust_remote_code=False)
        self.output = Path(getattr(args, 'tp_output', 'process-tp-results')).resolve() / (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
        self.output.mkdir(parents=True)
        self._text_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._pending_cancels = []
        self._closing = False
        self.session = Qwen3ProcessSession(binary, worker, self.model_path, self.devices,
                                          self.output / 'session', timeout=getattr(args, 'tp_timeout', 180),
                                          cuda_graph=getattr(args, 'cuda_graph', False),
                                          custom_allreduce=getattr(args, 'custom_allreduce', False),
                                          prefix_cache=getattr(args, 'prefix_cache', '') == 'true')
        self.session.__enter__()
        args.path = str(self.model_path)

    @property
    def state(self):
        return self.session.state

    @property
    def ready(self):
        return self.session.ready

    @property
    def timeout(self):
        return self.session.timeout

    def generate(self, request, *, cancel_event=None, on_token=None):
        """提交已编码请求；采样和循环留在 C++，on_token 不得重入生成或关闭模型。"""
        validate_request(request, self.config['vocab_size'])
        cancel_event = cancel_event if cancel_event is not None else threading.Event()
        with self._lifecycle_lock:
            if self._closing:
                raise RuntimeError('process TP model is closed')
            self._pending_cancels.append(cancel_event)
        try:
            return self.session.generate(request, cancel_event=cancel_event, on_token=on_token)
        finally:
            with self._lifecycle_lock:
                self._pending_cancels.remove(cancel_event)

    def _request(self, query, history, max_length, top_k, top_p, temperature, repeat_penalty, seed):
        if repeat_penalty != 1.0:
            raise ValueError('process TP currently requires repeat_penalty=1.0')
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be nonempty text')
        messages = []
        for pair in history or []:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2 or any(not isinstance(x, str) for x in pair):
                raise ValueError('history must contain (user text, assistant text) pairs')
            messages.extend([dict(role='user', content=pair[0]), dict(role='assistant', content=pair[1])])
        messages.append(dict(role='user', content=query))
        return prepare_request(self.model_path, self.hf_tokenizer, None, messages=messages,
                               max_new_tokens=max_length, top_k=top_k, top_p=top_p, temperature=temperature, seed=seed)

    def response(self, query, history=None, max_length=32, top_k=0, top_p=1.0,
                 temperature=0.0, repeat_penalty=1.0, seed=0):
        with self._text_lock:
            request = self._request(query, history, max_length, top_k, top_p, temperature, repeat_penalty, seed)
            report = self.generate(request)
            return self.hf_tokenizer.decode(report['token_ids'], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)

    def stream_response(self, query, history=None, max_length=32, top_k=0, top_p=1.0,
                        temperature=0.0, repeat_penalty=1.0, seed=0):
        """有界增量输出；提前关闭迭代器时取消、排空原生响应并释放 KV 后再返回。"""
        from .qwen3_process_server import IncrementalTextDecoder

        with self._text_lock:
            request = self._request(query, history, max_length, top_k, top_p, temperature, repeat_penalty, seed)
            events, cancel, done, errors = queue.Queue(maxsize=64), threading.Event(), threading.Event(), []
            decoder = IncrementalTextDecoder(self.hf_tokenizer)

            def emit(text):
                if not text or cancel.is_set():
                    return
                try:
                    events.put_nowait(text)
                except queue.Full:
                    raise RuntimeError('text stream consumer too slow')

            def run():
                try:
                    self.generate(request, cancel_event=cancel, on_token=lambda token: emit(decoder.push(token)))
                    emit(decoder.finish())
                except BaseException as error:
                    errors.append(error)
                finally:
                    done.set()

            thread = threading.Thread(target=run, name='process-tp-text')
            thread.start()
            try:
                while not done.is_set() or not events.empty():
                    try:
                        text = events.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    yield text
                if errors:
                    raise errors[0]
            finally:
                cancel.set()
                thread.join(self.timeout + 40)
                if thread.is_alive():
                    raise RuntimeError('text stream did not stop within session timeout')
                # 不能将取消时的协议/回收失败吞掉；正常协作取消允许复用模型。
                if errors and not isinstance(errors[0], GenerationCancelled) and self.state != 'ready':
                    raise errors[0]

    def clear_prefix_cache(self):
        """清空跨请求前缀快照；等待当前生成完成，不卸载权重。"""
        with self._lifecycle_lock:
            if self._closing:
                raise RuntimeError('process TP model is closed')
        return self.session.clear_prefix_cache()

    def release_memory(self):
        with self._lifecycle_lock:
            self._closing = True
            for cancel in self._pending_cancels:
                cancel.set()
        self.session.close()

    close = release_memory

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run_chat(args):
    """正式 chat/run 命令；沿用聊天历史语义，每次请求重算历史，不宣称前缀缓存。"""
    from .util import make_normal_llm_model

    with make_normal_llm_model(args) as model:
        print(f'backend={model.backend}, devices={model.devices}, logs={model.output}', file=sys.stderr)
        if args.prompt is not None:
            print(model.response(args.prompt, max_length=args.max_new_tokens))
            return
        history = []
        print("输入内容开始对话；clear 清空历史，stop 或 Ctrl-D 退出。", file=sys.stderr)
        while True:
            try:
                query = input('\nUser：')
            except EOFError:
                return
            if query.strip() == 'stop':
                return
            if query.strip() == 'clear':
                history.clear()
                continue
            if not query.strip():
                continue
            print('AI: ', end='', flush=True)
            stream = model.stream_response(query, history=history, max_length=args.max_new_tokens)
            text = ''
            try:
                for delta in stream:
                    text += delta
                    print(delta, end='', flush=True)
            finally:
                stream.close()
            print()
            history.append((query, text))


def run_server(args):
    """正式 server/serve 的进程后端；复用已有服务协议，模型创建统一走工厂。"""
    from .util import make_normal_llm_model
    from .qwen3_process_server import serve_model

    with make_normal_llm_model(args) as model:
        return serve_model(model, model.hf_tokenizer, model.model_path, args.model_name or str(model.model_path),
                           model.output, host=args.host, port=args.port, queue_size=args.tp_queue_size,
                           request_timeout=args.tp_request_timeout)
