"""PD 服务与独立 Router 启动配置；不会自动编译原生程序。"""

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from .protocol import Peer, new_id
from .routing import POLICIES, Registry


def add_service_args(parser):
    parser.add_argument("--role", choices=("full", "p", "d"), default="full")
    parser.add_argument("--router-managed", action="store_true", help="通过独立 Router 暴露完整推理服务")
    parser.add_argument("--instance", default="")
    parser.add_argument("--pd-token", default=os.environ.get("FT_PD_TOKEN", ""))
    parser.add_argument("--pd-growth-budget", type=int, default=512)
    parser.add_argument("--pd-long-threshold", type=int, default=512)
    parser.add_argument("--pd-model-id", default="")
    parser.add_argument("--pd-return-kv", action="store_true")
    parser.add_argument("--pd-retention", type=float, default=480)


def add_router(subparsers):
    parser = subparsers.add_parser("router", help="独立 P/D 服务路由入口")
    parser.add_argument("model", type=Path)
    parser.add_argument("--prefill", action="append", default=[])
    parser.add_argument("--decode", action="append", default=[])
    parser.add_argument("--worker", action="append", default=[], help="完整推理服务地址；不能与 P/D 地址混用")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pd-token", default=os.environ.get("FT_PD_TOKEN", ""))
    parser.add_argument("--pd-model-id", default="")
    parser.add_argument("--kv-mode", choices=("pull", "push"), default="pull")
    parser.add_argument("--policy", choices=POLICIES, default="cache_aware")
    parser.add_argument("--prefill-policy", choices=POLICIES)
    parser.add_argument("--decode-policy", choices=POLICIES)
    parser.add_argument("--cache-threshold", type=float, default=0.3)
    parser.add_argument("--balance-absolute", type=int, default=64)
    parser.add_argument("--balance-relative", type=float, default=1.5)
    parser.add_argument("--health-interval", type=float, default=60)
    parser.add_argument("--health-timeout", type=float, default=5)
    parser.add_argument("--health-failures", type=int, default=3)
    parser.add_argument("--health-successes", type=int, default=2)
    parser.add_argument("--session-ttl", type=float, default=7200)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--retry-max-attempts", type=int, default=1)
    parser.add_argument("--pd-long-threshold", type=int, default=512)
    parser.add_argument("--pd-return-kv", action="store_true")


def fingerprint(path, override=""):
    """/** 默认对配置和实际权重计算指纹；显式模型版本由部署者保证两端权重一致。 */"""
    if override:
        from .protocol import identifier
        return identifier(override, "model version")
    digest = hashlib.sha256()
    files = [path / "config.json"] + sorted(path.glob("*.safetensors"))
    if len(files) < 2:
        raise ValueError("local safetensors model required")
    for file in files:
        digest.update(file.name.encode() + b"\0")
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def run_service(args):
    from ..process_tp import native_binary, validate_options, process_termination_scope
    from ..util import load_model_args
    from ..qwen3_process_session import Qwen3ProcessSession
    from .native import NativeReplica
    from .service import Service, serve
    from .protocol import integer
    load_model_args(args)
    validate_options(args, allow_dp=True, allow_pd=True)
    integer(args.pd_growth_budget, "decode growth budget")
    integer(args.pd_long_threshold, "long Prompt threshold", 1)
    if args.tp_backend != "process":
        raise ValueError("P/D service requires --tp_backend process")
    if not args.pd_token:
        raise ValueError("set FT_PD_TOKEN or --pd-token for internal control authentication")
    path = Path(getattr(args, "path", "") or args.model).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") not in ("qwen3", "qwen2_moe") or config.get("quantization_config"):
        raise ValueError("P/D native service requires unquantized Qwen3 or Qwen2 MoE")
    model_id = fingerprint(path, args.pd_model_id)
    host = hashlib.sha256(Path("/proc/sys/kernel/random/boot_id").read_bytes()).hexdigest()
    peer = Peer(args.instance or f"{args.role}-{args.port}", new_id(), host)
    from ..parallel_config import build_parallel_plan, is_attention_ep
    attention_ep = is_attention_ep(args)
    if attention_ep and config.get("model_type") != "qwen2_moe":
        raise ValueError("Attention DP/EP requires Qwen2 MoE")
    plan = build_parallel_plan(args)
    assignments = [[device["logical_id"] for device in replica["devices"]] for replica in plan["replicas"]]
    directory = Path(getattr(args, "tp_output", "process-tp-results")) / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_pd_" + new_id()[:8])
    binary = native_binary(getattr(args, "tp_binary", ""), "fastllm_moe_generate" if attention_ep else "fastllm_tp_generate")
    worker = native_binary(getattr(args, "tp_worker", ""), "fastllm_tp_worker")
    with process_termination_scope(), ExitStack() as stack:
        replicas = []
        if attention_ep:
            from .moe import MoePDGroup
            group = stack.enter_context(MoePDGroup(binary, worker, path,
                ",".join(map(str, assignments[0])), directory / "ep",
                timeout=args.tp_timeout, cuda_graph=args.cuda_graph,
                prefix_cache=getattr(args, "prefix_cache", "false") == "true",
                batch_invariant=getattr(args, "batch_invariant", False)))
            replicas = [NativeReplica(view, peer, config, model_id, growth=args.pd_growth_budget)
                        for view in group.views]
        for rank, devices in enumerate(assignments):
            if attention_ep:
                break
            session = stack.enter_context(Qwen3ProcessSession(binary, worker, path,
                ",".join(map(str, devices)), directory / str(rank),
                timeout=args.tp_timeout, cuda_graph=args.cuda_graph,
                custom_allreduce=args.custom_allreduce,
                prefix_cache=getattr(args, "prefix_cache", "false") == "true",
                page_size=args.page_size if args.page_size > 0 else 128,
                cache_tokens=getattr(args, "tokens", -1)))
            if session.ready.get("pd_storage_protocol") != 1:
                raise RuntimeError("native binaries need the PD storage protocol; build on the server first")
            replicas.append(NativeReplica(session, peer, config, model_id, growth=args.pd_growth_budget))
        service = Service(args.role, peer, replicas, token=args.pd_token, timeout=args.tp_timeout,
                          long_threshold=args.pd_long_threshold, return_kv=args.pd_return_kv, retention=args.pd_retention,
                          routing=plan['dp_routing'])
        serve(service, args.host, args.port)


def run_router(args):
    from transformers import AutoTokenizer
    from .router import Router, serve
    path = args.model.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    registry = Registry(policy=args.policy, prefill_policy=args.prefill_policy, decode_policy=args.decode_policy,
        cache_threshold=args.cache_threshold, balance_absolute=args.balance_absolute,
        balance_relative=args.balance_relative, failure_threshold=args.health_failures,
        success_threshold=args.health_successes, session_ttl=args.session_ttl)
    services = [("p", url.rstrip("/")) for url in args.prefill] + [("d", url.rstrip("/")) for url in args.decode]
    if args.worker:
        if services:
            raise ValueError("use either complete workers or a P/D deployment")
        services = [("full", url.rstrip("/")) for url in args.worker]
    elif not args.prefill or not args.decode:
        raise ValueError("P/D deployment requires both --prefill and --decode")
    router = Router(registry, services, token=args.pd_token, model_id=fingerprint(path, args.pd_model_id),
        tokenizer=tokenizer, model_path=path, mode=args.kv_mode, timeout=args.request_timeout,
        attempts=args.retry_max_attempts, health_interval=args.health_interval, health_timeout=args.health_timeout,
        long_threshold=args.pd_long_threshold, return_kv=args.pd_return_kv)
    serve(router, args.host, args.port)
