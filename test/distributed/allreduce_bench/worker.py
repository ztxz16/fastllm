"""服务器每卡一个 spawn worker，只调用已有算子，不在运行时构建。"""

from datetime import timedelta
import json
from pathlib import Path
import sys
import traceback


def rank_main(rank, job):
    import torch
    import torch.distributed as dist
    from .backends import Native, Vllm
    from .operation import validate

    directory = Path(job["directory"])
    result_file = directory / f"rank{rank}.jsonl"
    # 每个组共同可见所选卡，组内可见序号为 0..world-1；vLLM 的 P2P 检查按 rank 查询。
    device = rank
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", init_method=(directory / "gloo.store").as_uri(),
                            rank=rank, world_size=len(job["devices"]),
                            timeout=timedelta(seconds=job["timeout"]))
    group = dist.group.WORLD
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        try:
            if job["backend"] == "vllm":
                backend = Vllm(device, group, job["mode"])
            else:
                backend = Native(job["native"], device, stream, rank, group,
                                 mode=1 if job["backend"] == "ft" else 0)
        except Exception as exc:
            if "expected_init_error" not in job:
                raise
            passed = job["expected_init_error"] in str(exc)
            row = dict(stage=job["stage"], case=job["cases"][0]["case"], n=0, rank=rank,
                       devices=job["devices"], backend=job["backend"], mode=job["mode"],
                       status="PASS" if passed else "FAIL", reason=str(exc), path="rejected")
            result_file.write_text(json.dumps(row) + "\n")
            if not passed:
                raise
            dist.destroy_process_group()
            return
        if "expected_init_error" in job:
            raise RuntimeError("非法配置未按约定拒绝")
        meta = {"rank": rank, "device": job["devices"][rank], "visible_device": device,
                "device_name": torch.cuda.get_device_name(device),
                "device_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unknown")),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "torch_nccl": list(torch.cuda.nccl.version()), "backend": backend.metadata(),
                "backend_name": job["backend"], "mode": job["mode"]}
        major, minor, patch = torch.cuda.nccl.version()
        if meta["backend"]["nccl_version"] != major * 10000 + minor * 100 + patch:
            raise RuntimeError("算子与 PyTorch 的 NCCL 版本不一致")
        if job.get("expected_runtime"):
            def software(item):
                return {key: value for key, value in item.items()
                        if key not in ("rank", "device", "device_name", "device_uuid")}
            matched = meta in job["expected_runtime"] if job.get("runtime_device_match", True) else (
                software(meta) in [software(item) for item in job["expected_runtime"]])
            if not matched:
                raise RuntimeError("软件/算子扩展或设备信息与准确性验证时不同，需要重新验证")
        (directory / f"rank{rank}.metadata.json").write_text(json.dumps(meta, indent=2))
        cache = {}
        with result_file.open("w", buffering=1) as results:
            for case in job["cases"]:
                row = dict(stage=job["stage"], case=case["case"], n=case["n"], rank=rank,
                           devices=job["devices"], backend=job["backend"], mode=job["mode"],
                           scenario=job.get("scenario", "single"))
                try:
                    if case.get("action") == "reject_zero":
                        before = backend.info()
                        try:
                            backend(torch.empty(0, dtype=torch.float32, device=device))
                        except RuntimeError as exc:
                            if "invalid AllReduce input" not in str(exc):
                                raise
                        else:
                            raise RuntimeError("零长度没有被拒绝")
                        if backend.info() != before:
                            raise RuntimeError("拒绝路径修改了 GPU 通信计数")
                        row.update(status="PASS", path="rejected", max_abs_error=0.0,
                                   mean_abs_error=0.0, failed_elements=0)
                    else:
                        row.update(validate(case, job, backend, cache, device, rank, stream))
                        if job.get("measure") and row["status"] == "PASS":
                            from .timing import measure
                            row.update(measure(case, job, backend, cache[case["n"]], stream, group))
                except Exception as exc:
                    row.update(status="FAIL", reason=str(exc))
                    results.write(json.dumps(row, allow_nan=False) + "\n")
                    raise
                results.write(json.dumps(row, allow_nan=False) + "\n")
                if row["status"] != "PASS":
                    raise RuntimeError("AllReduce output mismatch")
        stream.synchronize()
        dist.barrier(group=group)
        for operation in cache.values():
            operation.close()
        cache.clear()
        backend.close()
    if job["backend"] == "vllm":
        from vllm.distributed.parallel_state import destroy_distributed_environment
        destroy_distributed_environment()
    else:
        dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp
    job = json.loads(Path(sys.argv[1]).read_text())
    try:
        mp.spawn(rank_main, args=(job,), nprocs=len(job["devices"]), join=True)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
