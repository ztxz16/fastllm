"""V5 deploy flags: every V5 tunable has a built-in default and a deploy-command override."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "fastllm_pytools"))
import util  # noqa: E402

ENV = ["FASTLLM_PREFIX_V5", "FASTLLM_PREFIX_V5_TRIM", "FASTLLM_PREFIX_V5_REQUEST_HOLD", "FASTLLM_QWEN35_RESTORE_DRAIN",
       "FASTLLM_QWEN35_MM_TAIL_ALIGN", "FASTLLM_PREFIX_V5_READ_THREADS", "FASTLLM_PREFIX_V5_DRAIN_MS",
       "FASTLLM_PREFIX_V5_INTERVAL_TOKENS"]
NEW = {  # flag -> (value, env, exported)
    "--prefix_cache_ledger_bound": ("off", "FASTLLM_PREFIX_LEDGER_BOUND", "0"),
    "--prefix_cache_ledger_max_page_refs": ("4096", "FASTLLM_PREFIX_LEDGER_MAX_PAGE_REFS", "4096"),
    "--prefix_cache_ledger_max_records": ("4096", "FASTLLM_PREFIX_LEDGER_MAX_RECORDS", "4096"),
    "--host_malloc_trim_seconds": ("0", "FASTLLM_HOST_MALLOC_TRIM_SECONDS", "0"),
    "--prefix_cache_v4_segments": ("on", "FASTLLM_PREFIX_V4_SEGMENTS", "1"),
    "--prefix_cache_min_context_tokens": ("2048", "FASTLLM_PREFIX_CACHE_MIN_CONTEXT_TOKENS", "2048"),
    "--prefix_cache_write_min_reuse": ("2", "FASTLLM_PREFIX_CACHE_WRITE_MIN_REUSE", "2"),
    "--prefix_cache_drain_timeout_s": ("120", "FASTLLM_PREFIX_CACHE_DRAIN_TIMEOUT_S", "120"),
    "--prefix_cache_write_workers": ("2", "FASTLLM_PREFIX_CACHE_WRITE_WORKERS", "2"),
    "--prefix_cache_write_mode": ("serial", "FASTLLM_PREFIX_CACHE_WRITE_MODE", "serial"),
    "--prefix_cache_pack_workers": ("2", "FASTLLM_PREFIX_CACHE_PACK_WORKERS", "2"),
    "--prefix_cache_pinned_slots": ("8", "FASTLLM_PREFIX_CACHE_PINNED_SLOTS", "8"),
    "--prefix_cache_d2_staging_mb": ("512", "FASTLLM_PREFIX_CACHE_D2_STAGING_MB", "512"),
    "--prefix_cache_gpu_pin_limit_mb": ("128", "FASTLLM_PREFIX_CACHE_GPU_PIN_LIMIT_MB", "128"),
    "--prefix_cache_host_limit_max_mb": ("4096", "FASTLLM_PREFIX_CACHE_HOST_LIMIT_MAX_MB", "4096"),
    "--prefix_cache_hash_mode": ("pool1", "FASTLLM_PREFIX_CACHE_HASH_MODE", "pool1"),
    "--prefix_cache_hash_pool_mb": ("256", "FASTLLM_PREFIX_CACHE_HASH_POOL_MB", "256"),
}
ENV = ENV + [v[1] for v in NEW.values()]
fails = 0
def check(ok, msg):
    global fails
    print(("OK   " if ok else "FAIL ") + msg); fails += 0 if ok else 1
def run(argv, preset=None):
    for k in ENV: os.environ.pop(k, None)
    os.environ.update(preset or {})
    args = util.make_normal_parser("t").parse_args(argv)
    util.apply_prefix_cache_env(args)
    return args, {k: os.environ.get(k) for k in ENV}
args, env = run([])
check(args.prefix_cache_disk_gb == 256.0, "SSD quota default 256 GiB")
check(all(v is None for v in env.values()), "no flag given -> no env exported (built-in defaults apply)")
args, env = run(["--prefix_cache_disk_gb", "16"])
check(args.prefix_cache_disk_gb == 16.0, "quota override 16 GiB")
args, env = run(["--prefix_cache_v5", "off", "--prefix_cache_v5_trim", "off", "--prefix_cache_v5_request_hold", "off",
                 "--prefix_cache_restore_drain", "off", "--prefix_cache_mm_tail_align", "on",
                 "--prefix_cache_v5_read_threads", "4", "--prefix_cache_v5_drain_ms", "5000",
                 "--prefix_cache_v5_interval_tokens", "2048"])
check({k: env[k] for k in ENV[:8]} == {"FASTLLM_PREFIX_V5": "0", "FASTLLM_PREFIX_V5_TRIM": "0", "FASTLLM_PREFIX_V5_REQUEST_HOLD": "0",
              "FASTLLM_QWEN35_RESTORE_DRAIN": "0", "FASTLLM_QWEN35_MM_TAIL_ALIGN": "1",
              "FASTLLM_PREFIX_V5_READ_THREADS": "4", "FASTLLM_PREFIX_V5_DRAIN_MS": "5000",
              "FASTLLM_PREFIX_V5_INTERVAL_TOKENS": "2048"}, f"all overrides exported: {env}")
args, env = run([], {"FASTLLM_PREFIX_V5_TRIM": "0"})
check(env["FASTLLM_PREFIX_V5_TRIM"] == "0", "env set, flag unset -> env kept")
args, env = run(["--prefix_cache_v5_trim", "on"], {"FASTLLM_PREFIX_V5_TRIM": "0"})
check(env["FASTLLM_PREFIX_V5_TRIM"] == "1", "flag beats env")
argv = [x for k, v in NEW.items() for x in (k, v[0])]
args, env = run(argv)
check(all(env[v[1]] == v[2] for v in NEW.values()), f"17 new overrides exported: {[(v[1], env[v[1]]) for v in NEW.values() if env[v[1]] != v[2]]}")
try:
    pp_fields = __import__("persistent_prefix")._classified
    sp0 = util.make_normal_parser("fastllm", add_help=False); util.add_server_args(sp0)
    pp_fields(vars(sp0.parse_args(argv)), "execution"); check(True, "new server fields classified")
except ValueError as e:
    check(False, f"new server fields classified: {e}")
for bad in (["--prefix_cache_write_workers", "17"], ["--prefix_cache_pinned_slots", "0"], ["--host_malloc_trim_seconds", "-1"]):
    try: run(bad); check(False, f"rejects {bad}")
    except ValueError: check(True, f"rejects {bad}")
for bad in (["--prefix_cache_v5_read_threads", "0"], ["--prefix_cache_v5_read_threads", "65"], ["--prefix_cache_v5_drain_ms", "0"]):
    try: run(bad); check(False, f"rejects {bad}")
    except ValueError: check(True, f"rejects {bad}")
# The server path classifies every parsed field (persistent_prefix._execution_record); a new flag that is
# not classified stops the server at startup (def1, 2026-11-08). Use the real server parser.
import persistent_prefix as pp  # noqa: E402
sp = util.make_normal_parser("fastllm", add_help=False); util.add_server_args(sp)
for argv in ([], ["--prefix_cache_v5", "off", "--prefix_cache_v5_trim", "off", "--prefix_cache_v5_read_threads", "4"]):
    try:
        pp._classified(vars(sp.parse_args(["server", "/m"] + argv if False else argv)), "execution"); check(True, f"server fields classified {argv}")
    except ValueError as e:
        check(False, f"server fields classified {argv}: {e}")
print(f"FAILS {fails}")
sys.exit(1 if fails else 0)
