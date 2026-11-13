"""Kernel catalog and bounded, per-signature warmup tuning.

Plans are immutable after their first selection. New graph shapes may be prepared
lazily, before capture; freeze() closes this preparation window globally.
"""
import atexit
import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile

_active = None


def _library():
    from .llm import fastllm_lib
    lib = fastllm_lib
    lib.fastllm_kernel_configure.argtypes = [ctypes.c_char_p]
    lib.fastllm_kernel_configure.restype = ctypes.c_int
    lib.fastllm_kernel_freeze.argtypes = []
    lib.fastllm_kernel_freeze.restype = None
    for name in ("report", "catalog", "error"):
        fn = getattr(lib, "fastllm_kernel_" + name)
        fn.argtypes = []
        fn.restype = ctypes.c_char_p
    return lib


def catalog():
    return json.loads(_library().fastllm_kernel_catalog())


class KernelPolicy:
    def __init__(self, mode, policy_path="", profile_path=""):
        self.lib = _library()
        self.path = Path(profile_path).expanduser() if profile_path else None
        policy = json.loads(Path(policy_path).read_text()) if policy_path else {}
        if not isinstance(policy, dict):
            raise ValueError("kernel policy must be a JSON object")
        # The cache identity is owned by the runtime, not a user-supplied label.
        if any(k in policy for k in ("profile", "build_id", "mode")):
            raise ValueError("use --kernel-tune and --kernel-profile for mode/cache")
        identity = {"library": hashlib.sha256(Path(self.lib._name).read_bytes()).hexdigest(),
                    "kernel_environment": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith("FASTLLM_CUDA_") and not any(x in k for x in
                            ("SERVER_", "CACHE_DIR", "PYTHON"))}}
        compiler = os.environ.get("FASTLLM_CUDA_TRITON_SERVER_SCRIPT")
        if compiler and Path(compiler).is_file():
            identity["compiler"] = hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
        policy.update(mode=mode, build_id=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())
        if self.path and self.path.exists():
            profile = json.loads(self.path.read_text())
            if profile.get("schema") != 1:
                raise ValueError("unsupported kernel profile schema")
            policy["profile"] = profile
        if not self.lib.fastllm_kernel_configure(json.dumps(policy).encode()):
            raise ValueError(self.lib.fastllm_kernel_error().decode())
        self.mode = mode
        atexit.register(self.save)
        print("[kernels] mode=%s; budget=%s ms; max_l2=%s; new signatures are prepared before capture." %
              (mode, policy.get("budget_ms", 30000), policy.get("max_l2", .0032)), flush=True)
        for family in catalog()["families"]:
            print("[kernels] %s candidates: %s" % (family["family"],
                  ", ".join(item["id"] for item in family["implementations"])), flush=True)
        print("[kernels] inspect: ftllm ops; adjust: --kernel-policy JSON (budget_ms/max_l2/force/deny); "
              "cache/report: --kernel-profile FILE; disable: --kernel-tune off", flush=True)

    def report(self):
        return json.loads(self.lib.fastllm_kernel_report())

    def save(self):
        report = self.report()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            name = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=self.path.name + ".", delete=False) as f:
                    name = f.name
                    json.dump(report, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                os.replace(name, self.path)
            finally:
                if name and os.path.exists(name):
                    os.unlink(name)
        return report

    def freeze(self):
        self.lib.fastllm_kernel_freeze()
        report = self.save()
        print("[kernels] frozen %d signatures; preparation %.1f ms; profile=%s" %
              (len(report["records"]), report["tuning_ms"], self.path or "in memory"), flush=True)
        return report


def resolve_mode(args):
    """Resolve an omitted mode without overriding explicit CLI/JSON choices."""
    mode = getattr(args, "kernel_tune", None)
    if mode is None:
        mode = "quick" if getattr(args, "auto", False) else "off"
    args.kernel_tune = mode
    return mode


def configure(args):
    global _active
    mode = resolve_mode(args)
    policy = getattr(args, "kernel_policy", "")
    profile = getattr(args, "kernel_profile", "")
    if mode == "off":
        if policy or profile:
            raise ValueError("--kernel-policy/--kernel-profile require --kernel-tune quick or cache")
        if _active:
            raise RuntimeError("kernel policy is process-wide; use a separate process for another configuration")
        return None
    if _active:
        raise RuntimeError("configure one kernel policy per process before loading a model")
    _active = KernelPolicy(mode, policy, profile)
    return _active


def print_catalog():
    print(json.dumps(catalog(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    print_catalog()
