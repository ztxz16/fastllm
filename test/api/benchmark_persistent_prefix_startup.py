#!/usr/bin/env python3
"""CPU-only serial/parallel identity comparison against an existing manifest.

Does not start a model, write a cache, clear the OS page cache, or use a GPU.
Both implementations use the installed runtime files and identical settings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from fastllm_pytools import persistent_prefix as candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--installed-module", type=Path, required=True)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--workers", help="Measure comma-separated worker counts; repeats are allowed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Preserve prior evidence; choose a new output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    library = args.installed_module.parent / "libfastllm_tools.so"
    previous = json.loads(args.manifest.read_text())
    expected = candidate.identity_keys(previous)
    if bool(args.workers) == bool(args.baseline_ref):
        parser.error("Provide exactly one of --baseline-ref and --workers")
    if args.workers:
        counts = [int(value) for value in args.workers.split(",")]
        if not counts or any(count < 1 or count > 32 for count in counts):
            parser.error("Worker counts must be between 1 and 32")
        runs = [("workers-%d-run-%d" % (count, index + 1), candidate, count)
                for index, count in enumerate(counts)]
    else:
        source = subprocess.check_output(
            ["git", "show", args.baseline_ref + ":tools/fastllm_pytools/persistent_prefix.py"],
            cwd=ROOT, text=True)
        baseline = types.ModuleType("baseline_persistent_prefix")
        baseline.__file__ = str(args.installed_module)
        exec(compile(source, "baseline_persistent_prefix.py", "exec"), baseline.__dict__)
        runs = [("baseline", baseline, None), ("candidate", candidate, None)]
    # _runtime_record resolves the neighboring installed Python files. Compare
    # exactly the same runtime as the user's existing manifest, for both runs.
    candidate.__file__ = str(args.installed_module)
    result = {"status": "running", "baseline_ref": args.baseline_ref,
              "worker_sequence": args.workers,
              "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "expected_keys": expected, "runs": [],
              "limitations": ["Ordered runs as recorded; OS page cache not cleared",
                              "Identity hashing only; no GPU server restart or inference benchmark"]}

    def save():
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    save()
    try:
        for label, module, workers in runs:
            if workers is not None:
                module._HASH_WORKERS = workers
            print("Starting " + label + " identity scan", flush=True)
            started = time.perf_counter()
            cpu_before = resource.getrusage(resource.RUSAGE_SELF)
            options = {"progress": True} if module is candidate else {}
            manifest = module.build_identity(args.model, previous["execution"], library, args.draft, **options)
            keys = module.identity_keys(manifest)
            cpu_after = resource.getrusage(resource.RUSAGE_SELF)
            record = {"label": label, "seconds": time.perf_counter() - started,
                      "workers": workers if workers is not None else getattr(module, "_HASH_WORKERS", 1),
                      "cpu_seconds": cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime,
                      "keys": keys, "manifest_equal": manifest == previous}
            result["runs"].append(record)
            save()
            if keys != expected or manifest != previous:
                raise AssertionError(label + " changed the existing model/cache identity")
            print(json.dumps(record), flush=True)
        if not args.workers:
            result["speedup"] = result["runs"][0]["seconds"] / result["runs"][1]["seconds"]
        result["status"] = "ok"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    print(json.dumps({"status": result["status"], "speedup": result.get("speedup"),
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
