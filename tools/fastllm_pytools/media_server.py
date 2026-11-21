"""Public ftllm server arguments shared by the launcher and the CLI media dispatcher."""
import os

from .service_backends import BACKENDS, launcher_host, logical_devices, visible_gpu_environment

H3 = ("minimax_h3", "minimax_h3_image")
OPTIONS = {
    "base_model": ("--base-model", ("minimax_h3",)),
    "video_gpu_memory_limit": ("--gpu-memory-limit", H3),
    "video_host_memory_limit": ("--host-memory-limit", H3),
    "video_parallel_mode": ("--parallel-mode", H3),
    "nomas_text_encoder": ("--text-encoder", ("hy_nomas",)),
    "nomas_pe_url": ("--prompt-enhancer-url", ("hy_nomas",)),
    "nomas_pe_model": ("--prompt-enhancer-model", ("hy_nomas",)),
    "nomas_pe_prompts": ("--prompt-enhancer-prompts", ("hy_nomas",)),
    "duo_assets": ("--assets", ("duo",)),
    "duo_intent_cache": ("--intent-cache", ("duo",)),
}


def add_media_args(parser):
    group = parser.add_argument_group("Local DiT services (detected from model path)")
    group.add_argument("--devices", dest="visible_gpus", help="CUDA devices within the current visibility mask")
    group.add_argument("--output", dest="output_dir", help="Generated media directory")
    for field, (flag, families) in OPTIONS.items():
        group.add_argument(flag, dest=field, help=", ".join(families))


def server_argv(config, backend):
    host = launcher_host(config, backend)
    argv = ["ftllm", "server", os.path.expanduser(config.model.strip()), "--host", host, "--port", config.port]
    if config.visible_gpus.strip():
        # The parent set CUDA_VISIBLE_DEVICES; the child sees logical devices.
        argv.extend(["--devices", ",".join(logical_devices(config))])
    for flag, value in (("--output", config.output_dir), ("--api_key", config.api_key)):
        if value.strip():
            argv.extend([flag, value.strip()])
    for field, (flag, families) in OPTIONS.items():
        value = getattr(config, field).strip()
        if backend.id in families and value and not (field == "video_parallel_mode" and value == "auto"):
            argv.extend([flag, value])
    return argv


def dispatch(args, parser):
    from .model_service import identify_model
    from .tui import DeployConfig
    model = args.path or args.model or ""
    family = identify_model(model)
    if family.error:
        parser.error(family.error)
    fields = {"visible_gpus", "output_dir", *OPTIONS}
    supplied = set(args._auto_explicit)
    if family.backend == "llm":
        unexpected = supplied & fields
        if unexpected:
            parser.error("DiT service options require a recognized local DiT model: " + ", ".join(sorted(unexpected)))
        return False
    backend = BACKENDS[family.backend]
    allowed = {field for field, (_, families) in OPTIONS.items() if backend.id in families}
    if backend.id == "duo":
        allowed.add("tp")
    unexpected = supplied - allowed - {"command", "model", "path", "host", "port", "api_key", "visible_gpus", "output_dir"}
    if unexpected:
        parser.error(f"{backend.label} does not accept: " + ", ".join(sorted(unexpected)))
    config = DeployConfig(model=os.path.expanduser(model), host=args.host, port=str(args.port), api_key=args.api_key)
    for field in fields:
        if getattr(args, field, None) is not None:
            setattr(config, field, getattr(args, field))
    if backend.kind == "interactive" and "host" not in supplied:
        config.host = "127.0.0.1"
    if backend.id == "duo" and config.host not in {"127.0.0.1", "localhost"}:
        parser.error(f"{backend.label} listens on 127.0.0.1")
    try:
        if backend.id == "duo" and "tp" in supplied:
            config.visible_gpus = backend.tp_devices(args.tp, config.visible_gpus)
        env = {**os.environ, **visible_gpu_environment(config.visible_gpus)}
    except ValueError as error:
        parser.error(str(error))
    errors = backend.validate(config)
    if errors:
        parser.error("\n".join(item["message"] for item in errors))
    argv = backend.argv(config)
    os.execvpe(argv[0], argv, env)
    return True
