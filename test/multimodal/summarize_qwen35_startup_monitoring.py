#!/usr/bin/env python3
"""Summarize recorded sysstat/NVIDIA telemetry against server startup stages.

Parses the LC_ALL=C UTC one-second logs from benchmark_qwen35_server_startup.py.
No sampling, inference or GPU work is performed here.
"""
import argparse
import collections
import datetime as dt
import json
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    root = args.evidence
    result = json.loads((root / "result.json").read_text())
    if result["status"] != "ok":
        raise RuntimeError("Summarize a completed successful monitored startup")
    started = dt.datetime.fromisoformat(result["started_utc"])
    epoch = started.timestamp()

    def lines(name):
        return (root / (name + ".log")).read_text().splitlines()

    process, threads = [], []
    for line in lines("pidstat"):
        fields = line.split()
        if len(fields) < 22 or not fields[0].isdigit():
            continue
        row = {"t": int(fields[0]) - epoch, "cpu_percent": float(fields[8]),
               "usr_percent": float(fields[4]), "system_percent": float(fields[5]),
               "wait_percent": float(fields[7]), "rss_KiB": int(fields[13]),
               "read_KiB_s": float(fields[15]), "write_KiB_s": float(fields[16]),
               "major_fault_s": float(fields[11])}
        if fields[2] == str(result["pid"]) and fields[3] == "-":
            process.append(row)
        elif fields[2] == "-":
            threads.append({**row, "tid": int(fields[3])})

    disk, timestamp, columns = [], None, None
    for line in lines("iostat"):
        if re.fullmatch(r"\d{4}-\d\d-\d\dT.*", line):
            timestamp = dt.datetime.strptime(line, "%Y-%m-%dT%H:%M:%S%z").timestamp() - epoch
        fields = line.split()
        if fields and fields[0] == "Device":
            columns = fields[1:]
        elif fields and fields[0] == "nvme0n1" and columns:
            disk.append({"t": timestamp, **dict(zip(columns, map(float, fields[1:])))})

    cpu = []
    for line in lines("mpstat"):
        fields = line.split()
        if len(fields) >= 12 and re.fullmatch(r"\d\d:\d\d:\d\d", fields[0]) and fields[1] == "all":
            timestamp = dt.datetime.combine(started.date(), dt.time.fromisoformat(fields[0]),
                                            tzinfo=dt.timezone.utc).timestamp()
            cpu.append({"t": timestamp - epoch, "idle": float(fields[-1]), "iowait": float(fields[5])})

    memory = []
    for line in lines("vmstat"):
        fields = line.split()
        if len(fields) >= 20 and fields[0].isdigit():
            timestamp = dt.datetime.fromisoformat(fields[-2] + "T" + fields[-1] + "+00:00").timestamp()
            memory.append({"t": timestamp - epoch, "swap_in": int(fields[6]), "swap_out": int(fields[7]),
                           "free_KiB": int(fields[3]), "cache_KiB": int(fields[5])})

    gpu = []
    for line in lines("gpu"):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 8:
            timestamp = dt.datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc)
            gpu.append({"t": timestamp.timestamp() - epoch, "device": int(fields[1]),
                        "gpu_percent": float(fields[2]), "memory_percent": float(fields[3]),
                        "memory_MiB": float(fields[4]), "power_W": float(fields[5])})
    for name, rows in (("pidstat", process), ("iostat", disk), ("mpstat", cpu), ("vmstat", memory), ("gpu", gpu)):
        if not rows:
            raise RuntimeError("No usable samples: " + name)

    phases = {"startup_to_api": (0, result["http_ready_seconds"]),
              "identity": (result["markers"]["hash_start"]["seconds"], result["markers"]["hash_end"]["seconds"])}
    for stage in ("weights_load", "warmup"):
        events = [event for event in result["events"] if event["stage"] == stage]
        phases[stage] = (events[0]["observed_seconds"], events[-1]["observed_seconds"])
    for label, start_marker, end_marker in (
        ("target_identity", "[Prefix SSD] hashing Qwen3.8-27B-FP8: files=", "files=66/66"),
        ("draft_identity", "[Prefix SSD] hashing Qwen3.8-27B-DFlash2: files=", "files=1/1"),
        ("vision_max_warmup", "[Vision] Multimodal warmup before KV cache:", "[Vision] Multimodal workspace ready:")):
        starts = [event["observed_seconds"] for event in result["stage_logs"] if start_marker in event["line"]]
        ends = [event["observed_seconds"] for event in result["stage_logs"] if end_marker in event["line"]]
        if starts and ends:
            phases[label] = (min(starts), max(ends))

    def subset(rows, lo, hi):
        # Drop boundary samples that can span two different startup stages.
        return [row for row in rows if lo + 1 <= row["t"] <= hi]

    def stats(rows, key):
        values = [row[key] for row in rows]
        return {"mean": round(statistics.mean(values), 3), "max": round(max(values), 3)} if values else None

    summary = {"evidence": str(root.resolve()), "server_pid": result["pid"],
               "http_ready_seconds": result["http_ready_seconds"], "phases": {},
               "sample_counts": {"process": len(process), "disk": len(disk), "cpu": len(cpu),
                                 "memory": len(memory), "gpu": len(gpu)},
               "limitations": ["One-second interval samples; boundary samples excluded",
                               "Process CPU 100 percent equals one logical CPU; it includes all service threads",
                               "NVMe samples are machine-wide; process IO is separate",
                               "Physical reads do not include OS page-cache reads; cache was not cleared",
                               "NVIDIA utilization does not measure exact CUDA compute/memory bottlenecks"]}
    for label, (lo, hi) in phases.items():
        p, d, c, m = [subset(rows, lo, hi) for rows in (process, disk, cpu, memory)]
        costs = collections.defaultdict(float)
        for row in subset(threads, lo, hi):
            costs[row["tid"]] += row["cpu_percent"] / 100
        summary["phases"][label] = {
            "start_s": lo, "end_s": hi, "duration_s": hi - lo, "process_samples": len(p),
            "process_cpu_percent": stats(p, "cpu_percent"), "process_user_percent": stats(p, "usr_percent"),
            "process_system_percent": stats(p, "system_percent"), "rss_KiB": stats(p, "rss_KiB"),
            "process_read_KiB_s": stats(p, "read_KiB_s"), "major_fault_s": stats(p, "major_fault_s"),
            "nvme_read_MiB_s": stats(d, "rMB/s"), "nvme_util_percent": stats(d, "%util"),
            "nvme_read_await_ms": stats(d, "r_await"), "system_idle_percent": stats(c, "idle"),
            "system_iowait_percent": stats(c, "iowait"),
            "swap_in": stats(m, "swap_in"), "swap_out": stats(m, "swap_out"),
            "busiest_threads_approx_cpu_seconds": sorted(costs.items(), key=lambda item: item[1], reverse=True)[:8],
            "gpu": {str(device): {key: stats([row for row in subset(gpu, lo, hi) if row["device"] == device], key)
                                  for key in ("gpu_percent", "memory_percent", "memory_MiB", "power_W")}
                    for device in range(4)}}
    path = root / "monitor-summary.json"
    if path.exists():
        raise RuntimeError("Preserve previous summary")
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(path)
    for label, phase in summary["phases"].items():
        print(label, "seconds", round(phase["duration_s"], 2), "process CPU", phase["process_cpu_percent"],
              "NVMe MiB/s", phase["nvme_read_MiB_s"], "GPU avg", [phase["gpu"][str(i)]["gpu_percent"] for i in range(4)])


if __name__ == "__main__":
    main()
