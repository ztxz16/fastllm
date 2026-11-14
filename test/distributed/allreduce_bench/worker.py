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

    directory = Path(job["directory"])
    result_file = directory / f"rank{rank}.jsonl"
    device = job["devices"][rank]
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", init_method=(directory / "gloo.store").as_uri(),
                            rank=rank, world_size=len(job["devices"]),
                            timeout=timedelta(seconds=job["timeout"]))
    group = dist.group.WORLD
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        if job["backend"] == "vllm":
            backend = Vllm(device, group, job["mode"])
        else:
            backend = Native(job["native"], device, stream, rank, group,
                             mode=1 if job["backend"] == "ft" else 0)
        meta = {"rank": rank, "device": device, "device_name": torch.cuda.get_device_name(device),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "torch_nccl": torch.cuda.nccl.version(), "backend": backend.metadata()}
        if job["backend"] != "vllm":
            major, minor, patch = torch.cuda.nccl.version()
            if meta["backend"]["nccl_version"] != major * 10000 + minor * 100 + patch:
                raise RuntimeError("native adapter 与 PyTorch 的 NCCL 版本不一致")
        (directory / f"rank{rank}.metadata.json").write_text(json.dumps(meta, indent=2))
        with result_file.open("w", buffering=1) as results:
            for case in job["cases"]:
                row = dict(stage=job["stage"], case=case["case"], n=case["n"], rank=rank,
                           devices=job["devices"], backend=job["backend"], mode=job["mode"])
                try:
                    data = torch.full((case["n"],), rank + 1, dtype=torch.float32, device=device)
                    before = backend.info()
                    output = backend(data)
                    actual = output.cpu()
                    expected = len(job["devices"]) * (len(job["devices"]) + 1) / 2
                    error = (actual.double() - expected).abs()
                    row.update(max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                               failed_elements=int((error != 0).sum()), path=backend.path(case["n"]),
                               counters_before=before, counters_after=backend.info())
                    if before is not None:
                        after = row["counters_after"]
                        if after["custom"] - before["custom"] != 1 or after["nccl"] != before["nccl"]:
                            raise RuntimeError("FT GPU 计数没有证明一次自定义调用")
                    row["status"] = "PASS" if not row["failed_elements"] else "FAIL"
                    if row["path"] != ("nccl" if job["backend"] == "nccl" else job["backend"] + "_custom"):
                        raise RuntimeError("smoke 没有执行要求的自定义路径")
                except Exception as exc:
                    row.update(status="FAIL", reason=str(exc))
                    results.write(json.dumps(row, allow_nan=False) + "\n")
                    raise
                results.write(json.dumps(row, allow_nan=False) + "\n")
                if row["status"] != "PASS":
                    raise RuntimeError("AllReduce output mismatch")
        stream.synchronize()
        dist.barrier(group=group)
        backend.close()
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
