import argparse
import asyncio
import concurrent.futures
import fastapi
import logging
import os
import sys
import threading
import time
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from .openai_server.protocal.openai_protocol import *
from .openai_server.protocal.anthropic_protocol import *
from .openai_server.fastllm_completion import FastLLmCompletion
from .openai_server.streaming_response import SSEStreamingResponse
from .openai_server.fastllm_embed import FastLLmEmbed
from .openai_server.fastllm_reranker import FastLLmReranker
from .openai_server.fastllm_model import FastLLmModel
from .util import make_normal_parser
from .util import add_server_args
from .util import apply_page_size_default
from .startup_progress import StartupProgressReporter
from .pstate_controller import PStateAccessLogMiddleware, install_pstate_access_log_filter
global fastllm_completion
global fastllm_embed
global fastllm_reranker
global fastllm_model
global dev_mode_enabled
global api_thread_pool_workers
global request_executor
global pstate_controller

# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630
def set_ulimit(target_soft_limit: int = 65535):
    if sys.platform.startswith("win"):
        print("Windows detected, skipping ulimit adjustment.")
        return

    import resource

    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        new_soft = min(target_soft_limit, current_hard)
        if new_soft > current_soft:
            try:
                resource.setrlimit(resource_type, (new_soft, current_hard))
                print(f"Increased ulimit from {current_soft} to {new_soft}.")
            except ValueError as e:
                print(
                    f"Found ulimit of {current_soft} and failed to automatically increase "
                    f"with error {e}. This can cause fd limit errors like "
                    "`OSError: [Errno 24] Too many open files`. Consider "
                    "increasing with ulimit -n"
                )
        else:
            print(
                f"Current ulimit soft={current_soft}, hard={current_hard}. "
                f"Cannot increase to {target_soft_limit}. Consider running "
                "`ulimit -n 65535` as root or updating /etc/security/limits.conf."
            )

def parse_args():
    parser = make_normal_parser("OpenAI-compatible API server")
    add_server_args(parser)
    return apply_page_size_default(parser.parse_args())

app = fastapi.FastAPI()
# 设置允许的请求来源, 生产环境请做对应变更
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastllm_completion:FastLLmCompletion
fastllm_embed:FastLLmEmbed
fastllm_model:FastLLmModel
dev_mode_enabled:bool = False
api_thread_pool_workers:int = 32
request_executor = None
pstate_controller = None

_PSTATE_INFERENCE_PATHS = frozenset({
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/response",
    "/v1/messages",
    "/v1/embed",
    "/v1/rerank",
})
_request_activity = None


def _get_request_activity():
    global _request_activity
    if _request_activity is None:
        from .pstate_controller import RequestActivity
        _request_activity = RequestActivity()
    return _request_activity


@app.middleware("http")
async def pstate_activity_middleware(request: Request, call_next):
    tracked = request.method == "POST" and request.url.path in _PSTATE_INFERENCE_PATHS
    activity = _get_request_activity() if tracked else None
    if activity is not None:
        activity.begin()
        controller = globals().get("pstate_controller")
        if controller is not None:
            controller.notify_request()
    try:
        response = await call_next(request)
    except BaseException:
        if activity is not None:
            activity.end()
        raise
    if activity is None:
        return response

    # StreamingResponse returns before the body is drained. Keep the active
    # count until the final chunk so a long stream cannot be classified idle.
    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:
        activity.end()
        return response

    async def tracked_body():
        try:
            async for chunk in body_iterator:
                yield chunk
        finally:
            activity.end()

    response.body_iterator = tracked_body()
    return response


@app.get("/metrics")
async def pstate_metrics():
    return PlainTextResponse(_get_request_activity().render_prometheus())

def _prewarm_request_executor(executor, workers: int) -> int:
    started = 0
    ready = threading.Condition()
    release = threading.Event()

    def warm_worker():
        nonlocal started
        with ready:
            started += 1
            ready.notify_all()
        release.wait()

    futures = [executor.submit(warm_worker) for _ in range(workers)]
    deadline = time.monotonic() + 5.0
    with ready:
        while started < workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready.wait(timeout = remaining)
    release.set()
    for future in futures:
        future.result()
    return started

class FastLLMUvicornServer(uvicorn.Server):
    def __init__(self, config, startup_progress = None):
        super().__init__(config)
        self.startup_progress = startup_progress

    async def shutdown(self, sockets=None):
        # SSD 持久前缀缓存：关停前等待后台提交完成（默认 300 秒，可配置）。
        await super().shutdown(sockets)
        completion = globals().get("fastllm_completion")
        model = getattr(completion, "model", None)
        # §13 第 2 条：关闭准入必须在排空之前（第 1 条已由 super().shutdown() 完成）。
        if model is not None and hasattr(model, "close_persistent_prefix_admission"):
            model.close_persistent_prefix_admission()
        if model is not None and hasattr(model, "wait_persistent_prefix_cache"):
            # Large, valid workloads can have hundreds of accepted v4 writes
            # still in the store pipeline when the API receives SIGTERM.  A
            # fixed 30-second budget turns that normal backlog into a false
            # loss report and exits before the accepted tickets are durable.
            # Keep the timeout bounded and configurable for deployment; the
            # default covers the production-sized drain while preserving the
            # fail-closed exit on a genuinely stuck pipeline.
            try:
                drain_timeout_s = max(
                    1.0, float(os.environ.get("FASTLLM_PREFIX_CACHE_DRAIN_TIMEOUT_S", "300")))
            except ValueError:
                drain_timeout_s = 300.0
            logging.info("SSD prefix cache drain started (timeout=%ss)", drain_timeout_s)
            finished = await asyncio.to_thread(model.wait_persistent_prefix_cache,
                                                drain_timeout_s)
            if not finished:
                # S1-3 修复（2026-10-02，对标审计 P0-C）：关停排空超时**不得静默成功**。
                # 旧代码只打 warning，且进程照常以 0 退出 ⇒ 编排层看不出"缓存里还有已受理未落盘的数据"。
                # 现在：打 ERROR 日志 + 注册退出码 3（atexit 里 os._exit，避免与 uvicorn 的退出流程打架），
                # 让容器/编排层能识别"这次关停不完整"。
                logging.error("SSD prefix cache drain timed out; only committed checkpoints survive restart "
                              "(process will exit with code 3)")
                import atexit
                import os as _os
                atexit.register(lambda: _os._exit(3))
            else:
                logging.info("SSD prefix cache drain complete")

    async def startup(self, sockets = None):
        global request_executor
        loop = asyncio.get_running_loop()
        if request_executor is None:
            request_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers = api_thread_pool_workers,
                thread_name_prefix = "fastllm-api")
            loop.set_default_executor(request_executor)
            started = _prewarm_request_executor(
                request_executor, api_thread_pool_workers)
            logging.info(
                "FastLLM API request thread pool: workers=%d, prewarmed=%d",
                api_thread_pool_workers, started)
        await super().startup(sockets)
        if self.started and not self.should_exit:
            from .llm import disable_cuda_malloc
            disable_cuda_malloc()
            controller = globals().get("pstate_controller")
            if controller is not None:
                controller.mark_ready()
            if self.startup_progress is not None:
                self.startup_progress.ready()

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await fastllm_completion.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content = generator.model_dump(),
                            status_code = generator.code)
    if request.stream:
        return SSEStreamingResponse(content = generator[0],
                                    background = generator[1])
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content = generator.model_dump())

@app.post("/v1/responses")
@app.post("/v1/response")
async def create_response(request: ResponsesRequest,
                          raw_request: Request):
    generator = await fastllm_completion.create_response(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content = generator.model_dump(),
                            status_code = generator.code)
    if request.stream:
        return SSEStreamingResponse(content = generator[0],
                                    background = generator[1])
    else:
        assert isinstance(generator, ResponsesResponse)
        return JSONResponse(content = generator.model_dump())

@app.post("/v1/messages")
async def create_anthropic_message(request: AnthropicMessageRequest,
                                   raw_request: Request):
    generator = await fastllm_completion.create_anthropic_message(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        error_type = {400: "invalid_request_error", 404: "not_found_error"}.get(
            generator.code, "api_error")
        return JSONResponse(content = {"type": "error", "error": {
                                "type": error_type, "message": generator.message}},
                            status_code = generator.code)
    if request.stream:
        return SSEStreamingResponse(content = generator[0],
                                    background = generator[1])
    else:
        assert isinstance(generator, AnthropicMessageResponse)
        return JSONResponse(content = generator.model_dump(exclude_none = True))

@app.post("/v1/embed")
async def create_embed(request: EmbedRequest,
                       raw_request: Request):
    embedding = fastllm_embed.embedding_sentence(request, raw_request)
    return JSONResponse(embedding)

@app.post("/v1/rerank")
async def create_rerank(request: RerankRequest,
                       raw_request: Request):
    print(request)
    scores = fastllm_reranker.rerank(request, raw_request)    
    return JSONResponse(scores)


@app.get("/v1/models")
async def list_models():
    model_response = fastllm_model.response
    return JSONResponse(content = model_response)

@app.post("/v1/cancel")
async def cancel_generation(request: Request):
    # Check if development mode is enabled
    if not dev_mode_enabled:
        return JSONResponse(content = {"error": "This API is only available in development mode"}, 
                            status_code = 403)
    
    try:
        json_data = await request.json()
        if 'conversation_id' not in json_data:
            return JSONResponse(content = {"error": "Missing required parameter: conversation_id"},
                                status_code = 400)
        
        conversation_id = json_data['conversation_id']
        success = fastllm_completion.abort_conversation(conversation_id)
        
        if success:
            return JSONResponse(content = {"message": f"Conversation {conversation_id} cancelled successfully"})
        else:
            return JSONResponse(content = {"error": f"Failed to cancel conversation {conversation_id}. Conversation not found or already finished."},
                                status_code = 404)
    except Exception as e:
        logging.error(f"Error cancelling conversation: {e}")
        return JSONResponse(content = {"error": f"Internal server error: {str(e)}"},
                            status_code = 500)

@app.get("/v1/active_conversations")
async def get_active_conversations():
    # Check if development mode is enabled
    if not dev_mode_enabled:
        return JSONResponse(content = {"error": "This API is only available in development mode"}, 
                            status_code = 403)
        
    try:
        conversations = fastllm_completion.get_active_conversations()
        return JSONResponse(content = {
            "active_conversations": conversations,
            "count": len(conversations)
        })
    except Exception as e:
        logging.error(f"Error getting active conversations: {e}")
        return JSONResponse(content = {"error": f"Internal server error: {str(e)}"},
                            status_code = 500)

def init_logging(log_level = logging.INFO, log_file:str = None):
    logging_format = '%(asctime)s %(process)d %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    root = logging.getLogger()
    root.setLevel(log_level)
    if log_file is not None:
        logging.basicConfig(level=log_level, filemode='a', filename=log_file, format=logging_format)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(logging_format))
    root.addHandler(stdout_handler)

def apply_default_generation_config_overrides(model, args):
    overrides = {}
    for arg_name, config_name in [
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("repeat_penalty", "repetition_penalty"),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            model.default_generation_config[config_name] = value
            overrides[config_name] = value
    if overrides:
        logging.info("Override default generation config from cli: %s", overrides)


def _reset_pstate_before_load():
    """FASTLLM_PSTATE_RESET_AT_START=1 (default off): return the GPUs to driver-auto
    (the controller's own "active" P-State) before the model loads.

    The controller starts only after the model is ready, and nvidia-pstated (the
    design it follows) restores P-State 16 only on a clean exit. A previous process
    that ended while idle can leave the GPUs forced to P8 (645 MHz on Tesla T10);
    load and warmup then run at that clock: start 205-221 s instead of 96-104 s and
    a different greedy output for the whole process (chain73 P1 vs P2)."""

    if os.environ.get("FASTLLM_PSTATE_RESET_AT_START", "") != "1":
        return
    from .pstate_controller import NvApiPStateBackend, PStateConfig

    config = PStateConfig.from_env()
    if config is None:
        return
    backend = NvApiPStateBackend(config.gpu_ids)
    try:
        backend.set_pstate(config.high)
    finally:
        backend.close()
    logging.info("FastLLM P-State reset before model load: gpus=%s pstate=%d",
                 config.gpu_ids, config.high)


def _start_pstate_controller(port: int):
    """Start the opt-in driver controller after the model is ready."""

    from .pstate_controller import PStateConfig, PStateController

    config = PStateConfig.from_env(f"http://127.0.0.1:{port}/metrics")
    if config is None:
        return None
    completion = globals().get("fastllm_completion")
    native_model = getattr(completion, "model", None)
    provider = getattr(native_model, "next_observation_sequence", None)
    controller = PStateController(
        config, sequence_provider=provider if callable(provider) else None)
    if controller.start():
        return controller
    return None


def _fastllm_server(args, startup_progress):
    if args.api_key:
        @app.middleware("http")
        async def authentication(request: Request, call_next):
            print("auth")
            if request.method == "OPTIONS":
                return await call_next(request)
            url_path = request.url.path            
            if not url_path.startswith("/v1"):
                return await call_next(request)
            auth_header = request.headers.get("Authorization")
            anthropic_api_key = request.headers.get("x-api-key")
            if (auth_header != "Bearer " + args.api_key
                    and anthropic_api_key != args.api_key):
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)
        
    global fastllm_completion
    global fastllm_embed
    global fastllm_reranker
    global fastllm_model
    global dev_mode_enabled
    global api_thread_pool_workers
    
    # Set development mode from args
    dev_mode_enabled = args.dev_mode
    if dev_mode_enabled:
        logging.info("Development mode enabled - conversation management APIs are active")
    
    apply_page_size_default(args)
    init_logging()
    logging.info(args)
    # Materialize API serving high-water scratch before automatic KV sizing.
    # Real CUDA allocations are frozen only when FASTLLM_CUDA_MEM_CHECK is
    # explicitly enabled; ordinary serving must not pay the fixed frozen-pool
    # reserve merely because it uses the API frontend.
    os.environ["FASTLLM_CUDA_SERVING_WARMUP"] = "1"
    _reset_pstate_before_load()
    from .util import make_normal_llm_model
    model = make_normal_llm_model(args, startup_progress = startup_progress)
    apply_default_generation_config_overrides(model, args)
    model.set_verbose(True)
    if (args.model_name is None or args.model_name == ''):
        args.model_name = args.path
        if (args.model_name is None or args.model_name == ''):
            args.model_name = args.model
    fastllm_completion = FastLLmCompletion(model_name = args.model_name, model = model,
                                           think = (args.think.lower() != "false"),
                                           enable_thinking = getattr(model, "enable_thinking", True),
                                           hide_input = args.hide_input)
    fastllm_embed = FastLLmEmbed(model_name = args.model_name, model = model)
    fastllm_reranker = FastLLmReranker(model_name = args.model_name, model = model)
    fastllm_model = FastLLmModel(model_name = args.model_name, model = model)
    logging.info(
        "Model context window: %d tokens per session "
        "(model=%s, shared KV cache=%s, configured limit=%s)",
        fastllm_model.context_window,
        fastllm_model.model_context_window,
        fastllm_model.kv_cache_token_limit,
        fastllm_model.configured_context_window_limit,
    )
    default_workers = args.max_batch if args.max_batch > 0 else 64
    default_workers = max(32, min(128, default_workers))
    workers_env = os.getenv("FASTLLM_API_THREADPOOL_WORKERS", "")
    try:
        api_thread_pool_workers = int(workers_env) if workers_env else default_workers
    except ValueError:
        api_thread_pool_workers = default_workers
    api_thread_pool_workers = max(1, min(256, api_thread_pool_workers))
    set_ulimit()
    # RCA 2（2026-10-03，真机 merge-verify-a1 实证）：uvicorn 默认 `timeout_graceful_shutdown=None`
    # ⇒ `await super().shutdown()` 会**无限期**等在途请求连接；而"客户端是否断开"是**逐 token**
    # 检查的（fastllm_completion.py 的 is_disconnected），一旦生成长时间不出 token，
    # 关服就会一直挂住并继续占显存。实测：SIGTERM 14:49:00 → Finished 14:52:37 = **217s**。
    # 业界做法就是给 uvicorn 设一个**有界的优雅关服超时**：到点取消在途任务，继续 lifespan
    # shutdown 并退出（客户端会收到中断，但进程必然释放显存）。
    # 可用环境变量覆盖（秒），便于现场调配；默认 20s。
    try:
        graceful_shutdown_timeout = int(os.getenv("FASTLLM_SHUTDOWN_GRACEFUL_TIMEOUT_S", "20"))
    except ValueError:
        graceful_shutdown_timeout = 20
    # Wrap outside FastAPI's middleware stack: Uvicorn emits access logs
    # synchronously when this outer ASGI send forwards response headers.
    config = uvicorn.Config(PStateAccessLogMiddleware(app), host = args.host, port = args.port,
                            timeout_graceful_shutdown=max(1, graceful_shutdown_timeout))
    install_pstate_access_log_filter()
    startup_progress.progress("server_starting", 1, 1)
    global pstate_controller
    pstate_controller = None
    try:
        pstate_controller = _start_pstate_controller(args.port)
        FastLLMUvicornServer(config, startup_progress = startup_progress).run()
    finally:
        controller = pstate_controller
        pstate_controller = None
        if controller is not None:
            controller.stop()

def fastllm_server(args):
    startup_progress = StartupProgressReporter(
        getattr(args, "startup_progress", "off")
    )
    startup_progress.progress("initializing", 0, 1)
    try:
        return _fastllm_server(args, startup_progress)
    except BaseException as error:
        startup_progress.fail(error)
        raise
    finally:
        startup_progress.close()

if __name__ == "__main__":
    args = parse_args()
    fastllm_server(args)
    
