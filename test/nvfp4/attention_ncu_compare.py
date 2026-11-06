#!/usr/bin/env python3
"""用Nsight Compute对比FastLLM与vLLM的Attention合并Kernel。"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "test" / "nvfp4"))

import decode_nsys_compare as decode_nsys  # noqa: E402


BACKEND_KERNELS = {
    "fastllm": "PersistentVariableLengthMergeStatesKernel",
    "vllm": "flash_fwd_splitkv_combine_kernel",
}

SUMMARY_METRICS = {
    "Duration": ("gpu__time_duration.sum",),
    "DRAM Read Bytes": ("dram__bytes_read.sum",),
    "DRAM Write Bytes": ("dram__bytes_write.sum",),
    "DRAM Throughput": (
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",),
    "Achieved Occupancy": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",),
    "Registers/Thread": ("launch__registers_per_thread",),
    "Shared Memory/Block": (
        "launch__shared_mem_per_block", "launch__shared_mem_per_block_allocated"),
    "Waves/SM": ("launch__waves_per_multiprocessor",),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="对比相同Decode形状下两个Attention合并Kernel的NCU指标")
    parser.add_argument("--model", required=True)
    parser.add_argument("--nsys-result-dir", required=True,
                        help="同版本decode_nsys_compare.py生成的校准结果目录")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--quantization", choices=("nvfp4", "w8a8"),
                        default="w8a8")
    parser.add_argument("--backends", default="fastllm,vllm")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--prefix-cache-padding-tokens", type=int, default=128)
    parser.add_argument("--vllm-python", default=os.environ.get(
        "NVFP4_VLLM_PYTHON", sys.executable))
    parser.add_argument("--fastllm-python", default=sys.executable)
    parser.add_argument("--flm-dtype", default="auto")
    parser.add_argument("--flm-atype", default="bfloat16")
    parser.add_argument("--flm-device", default="cuda")
    parser.add_argument("--flm-kv-cache-layout", default="auto",
                        choices=("auto", "continuous", "paged"))
    parser.add_argument("--flm-attention-backend", default="auto")
    parser.add_argument("--flm-attention-backend-strict", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--profile-timeout", type=int, default=1800)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--set", dest="section_set", default="full")
    parser.add_argument(
        "--skip-discovery", action="store_true",
        help="跳过目标Kernel启动期发现；仅用于已验证环境的重复测试")
    parser.add_argument(
        "--vllm-extra-arg", action="append", default=[],
        help="追加一个vLLM服务参数；需要多个参数时重复使用")
    return parser.parse_args()


def selected_backends(value):
    result = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not result or any(item not in BACKEND_KERNELS for item in result):
        raise ValueError("backends只能包含fastllm和vllm")
    return list(dict.fromkeys(result))


def read_calibration(result_dir, backend, batch, marker):
    """读取Nsight Systems稳定Decode边界并计算NCU应跳过的目标调用数。"""
    result_path = result_dir / backend / f"batch-{batch}" / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"缺少Nsight校准结果：{result_path}")
    row = json.loads(result_path.read_text(encoding="utf-8"))
    timeline_path = Path(row["nsys"]["timeline_csv"])
    if not timeline_path.exists():
        raise FileNotFoundError(f"缺少Nsight逐事件GPU Trace：{timeline_path}")
    boundary = row["nsys"]["stable_decode_filter"]["boundary_ns"]
    before = 0
    stable = 0
    for event in decode_nsys.read_stats_csv(timeline_path):
        name = decode_nsys.trace_name(event)
        if marker.lower() not in name.lower():
            continue
        start = decode_nsys.numeric(event, ("Start (ns)", "Start"))
        if start < boundary:
            before += 1
        else:
            stable += 1
    if stable <= 0:
        raise RuntimeError(
            f"{backend}稳定Decode区间未找到目标Kernel：{marker}")
    return {
        "result": row,
        "timeline_csv": str(timeline_path),
        "launch_skip": before,
        "stable_matches": stable,
        "boundary_ns": boundary,
    }


def wait_pid_files(path, process, timeout, log_path):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pid_files = list(path.glob("*.pid")) if path.exists() else []
        if pid_files:
            return pid_files
        if process.poll() is not None:
            tail = log_path.read_text(
                encoding="utf-8", errors="replace")[-12000:]
            raise RuntimeError(f"NCU目标服务提前退出：\n{tail}")
        time.sleep(0.1)
    raise TimeoutError("等待NCU目标PID超时")


def signal_target_processes(pid_dir, signum):
    roots = []
    for pid_file in sorted(pid_dir.glob("*.pid")):
        try:
            roots.append(int(pid_file.read_text(encoding="utf-8").strip()))
        except ValueError:
            continue
    # multiprocessing使用spawn时子进程会自行登记；使用fork时只继承信号
    # handler而不会重新执行sitecustomize。递归读取/proc补齐后一种子进程。
    candidates = set(roots)
    pending = list(roots)
    while pending:
        parent = pending.pop()
        children_path = Path(
            f"/proc/{parent}/task/{parent}/children")
        try:
            children = [int(value) for value in
                        children_path.read_text().split()]
        except (OSError, ValueError):
            continue
        for child in children:
            if child not in candidates:
                candidates.add(child)
                pending.append(child)

    signaled = []
    for pid in sorted(candidates):
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().lower()
            # 只通知已登记进程或其Python子进程，避免误杀外部辅助程序。
            if pid not in roots and b"python" not in command:
                continue
            os.kill(pid, signum)
            signaled.append(pid)
        except OSError:
            continue
    if not signaled:
        raise RuntimeError("没有找到可控制的NCU Python目标进程")
    return signaled


def prepare_target(args, backend, backend_dir):
    """准备启用CUDA Graph的服务命令、进程登记目录和隔离后的环境。"""
    server_command, target_env, served_name = decode_nsys.server_spec(
        args, backend, args.batch)
    pid_dir = backend_dir / "target-pids"
    pid_dir.mkdir(parents=True, exist_ok=True)
    for stale_pid in pid_dir.glob("*.pid"):
        stale_pid.unlink()
    target_env["FASTLLM_NCU_TARGET_PID_DIR"] = str(pid_dir)
    sitecustomize_dir = str(
        REPO_DIR / "test" / "nvfp4" / "ncu_sitecustomize")
    current_pythonpath = target_env.get("PYTHONPATH", "")
    target_env["PYTHONPATH"] = (
        sitecustomize_dir if not current_pythonpath else
        f"{sitecustomize_dir}{os.pathsep}{current_pythonpath}")
    target = decode_nsys.target_command(server_command)
    control_env = decode_nsys.nsys_control_env()
    control_env.update({key: value for key, value in target_env.items()
                        if key != "LD_PRELOAD"})
    server = {
        "base_url": f"http://127.0.0.1:{args.port}",
        "served_name": served_name,
        "backend": backend,
    }
    return target, control_env, server, pid_dir


def find_report(output_base):
    direct = output_base.with_suffix(".ncu-rep")
    if direct.exists():
        return direct
    matches = sorted(output_base.parent.glob(output_base.name + "*.ncu-rep"))
    if matches:
        return matches[-1]
    raise RuntimeError(f"Nsight Compute没有生成报告：{output_base}.ncu-rep")


def stop_process_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def target_match_count(timeline_path, marker):
    count = 0
    names = {}
    for event in decode_nsys.read_stats_csv(timeline_path):
        name = decode_nsys.trace_name(event)
        if marker.lower() not in name.lower():
            continue
        count += 1
        names[name] = names.get(name, 0) + 1
    return count, names


def calculate_full_lifecycle_skip(total_matches, request_calibration):
    """计算从进程启动到首个稳定Decode目标Kernel前的跳过次数。"""
    formal_total = (request_calibration["launch_skip"] +
                    request_calibration["stable_matches"])
    if total_matches < formal_total:
        raise RuntimeError(
            f"启动期校准目标调用不足：total={total_matches}, "
            f"formal={formal_total}")
    return {
        "startup_and_warmup_matches": total_matches - formal_total,
        "launch_skip": total_matches - request_calibration["stable_matches"],
    }


def wait_profile_result(process, output_base, timeout, log_path):
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        stop_process_group(process)
        raise TimeoutError(f"等待Profiler输出超时：{log_path}") from error
    return find_report(output_base)


def completed_profile_report(process, output_base, timeout, log_path):
    """等待NCU因命中目标而结束，并返回生成的报告。"""
    return wait_profile_result(process, output_base, timeout, log_path)


def run_discovery(args, backend, prompt, backend_dir):
    """从进程启动开始验证NCU能够识别目标Kernel及其实际名称。"""
    discovery_dir = backend_dir / "01-discovery"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    target, control_env, server, pid_dir = prepare_target(
        args, backend, discovery_dir)
    marker = BACKEND_KERNELS[backend]
    output_base = discovery_dir / "target-kernel"
    command = [
        args.ncu, "--target-processes", "all",
        "--profile-from-start", "on", "--kill", "yes",
        "--graph-profiling", "node", "--replay-mode", "kernel",
        "--cache-control", "none", "--clock-control", "none",
        "--kernel-name-base", "demangled",
        "--kernel-name", f"regex:.*{re.escape(marker)}.*",
        "--launch-skip", "0", "--launch-count", "1", "--set", "basic",
        "--export", str(output_base), "--force-overwrite", *target,
    ]
    log_path = discovery_dir / "ncu-discovery.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"COMMAND: {shlex.join(command)}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command, cwd=REPO_DIR, env=control_env, stdout=log_handle,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            wait_pid_files(pid_dir, process, args.startup_timeout, log_path)
            try:
                decode_nsys.wait_server(
                    server["base_url"], process, args.startup_timeout,
                    log_path, backend)
            except RuntimeError:
                # 目标也可能在模型启动或CUDA Graph warmup期间出现。此时NCU
                # 已按launch-count=1结束服务，直接读取报告就是发现成功。
                if process.poll() is None:
                    raise
                report = completed_profile_report(
                    process, output_base, args.profile_timeout, log_path)
            else:
                warmup_prompt = decode_nsys.make_cache_warmup_prompt(
                    prompt, args.prefix_cache_padding_tokens)
                try:
                    decode_nsys.run_decode_batch(
                        server, warmup_prompt, args.warmup_output_tokens,
                        args.batch, args, "ncu-discovery")
                except Exception as error:
                    log_handle.write(f"EXPECTED DISCOVERY END: {error}\n")
                    log_handle.flush()

                # --launch-count只限制NCU采集的Kernel数量，不保证常驻HTTP
                # 服务随采集结束自动退出。请求结束后主动关闭已登记的Python
                # 目标进程，使NCU完成报告落盘，避免一直等待到profile超时。
                if process.poll() is None:
                    signaled = signal_target_processes(
                        pid_dir, signal.SIGTERM)
                    log_handle.write(
                        "DISCOVERY TARGET SIGTERM: "
                        f"{','.join(map(str, signaled))}\n")
                    log_handle.flush()

                report = completed_profile_report(
                    process, output_base, args.profile_timeout, log_path)
        except Exception:
            stop_process_group(process)
            raise
        finally:
            stop_process_group(process)
    raw_csv = discovery_dir / "target-kernel-raw.csv"
    rows = parse_ncu_raw(args, report, raw_csv)
    _, launch = normalize_metrics(rows)
    return {
        "report": str(report), "raw_csv": str(raw_csv),
        "kernel": launch["kernel"], "grid": launch["grid"],
        "block": launch["block"],
    }


def run_startup_calibration(args, backend, prompt, request_calibration,
                            backend_dir):
    """用完整生命周期Nsys统计正式稳定Decode前的全部目标调用。"""
    calibration_dir = backend_dir / "02-startup-calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    target, control_env, server, pid_dir = prepare_target(
        args, backend, calibration_dir)
    output_base = calibration_dir / "startup"
    command = [
        args.nsys, "profile", "--trace=cuda,nvtx",
        "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
        f"--output={output_base}", "--force-overwrite=true", *target,
    ]
    log_path = calibration_dir / "nsys-startup.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"COMMAND: {shlex.join(command)}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command, cwd=REPO_DIR, env=control_env, stdout=log_handle,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            wait_pid_files(pid_dir, process, args.startup_timeout, log_path)
            decode_nsys.wait_server(
                server["base_url"], process, args.startup_timeout,
                log_path, backend)
            warmup_prompt = decode_nsys.make_cache_warmup_prompt(
                prompt, args.prefix_cache_padding_tokens)
            decode_nsys.run_decode_batch(
                server, warmup_prompt, args.warmup_output_tokens,
                args.batch, args, "startup-calibration-warmup")
            decode_nsys.run_decode_batch(
                server, prompt, args.output_tokens,
                args.batch, args, "startup-calibration-formal")
            signal_target_processes(pid_dir, signal.SIGTERM)
            report = wait_profile_result(
                process, output_base, args.profile_timeout, log_path)
        except Exception:
            stop_process_group(process)
            raise
        finally:
            stop_process_group(process)

    timeline_path = decode_nsys.export_stats(
        args, report, "cuda_gpu_trace",
        calibration_dir / "startup-gpu-trace")
    marker = BACKEND_KERNELS[backend]
    total_matches, names = target_match_count(timeline_path, marker)
    skip = calculate_full_lifecycle_skip(total_matches, request_calibration)
    result = {
        "report": str(report), "timeline_csv": str(timeline_path),
        "target_names": names, "total_matches": total_matches,
        "startup_and_warmup_matches": skip["startup_and_warmup_matches"],
        "formal_before_stable_matches": request_calibration["launch_skip"],
        "stable_matches": request_calibration["stable_matches"],
        "launch_skip": skip["launch_skip"],
    }
    (calibration_dir / "startup-calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_profile(args, backend, prompt, calibration, backend_dir):
    """从进程启动观察Graph，并只剖析首个稳定Decode目标Kernel。"""
    profile_dir = backend_dir / "03-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    target, control_env, server, pid_dir = prepare_target(
        args, backend, profile_dir)
    marker = BACKEND_KERNELS[backend]
    output_base = profile_dir / "attention-merge"
    command = [
        args.ncu, "--target-processes", "all",
        "--profile-from-start", "on", "--kill", "yes",
        "--graph-profiling", "node", "--replay-mode", "kernel",
        "--cache-control", "none", "--clock-control", "none",
        "--kernel-name-base", "demangled",
        "--kernel-name", f"regex:.*{re.escape(marker)}.*",
        "--launch-skip", str(calibration["launch_skip"]),
        "--launch-count", "1", "--set", args.section_set,
        "--export", str(output_base), "--force-overwrite", *target,
    ]
    log_path = profile_dir / "ncu-run.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"COMMAND: {shlex.join(command)}\n\n")
        log_handle.write(
            f"CALIBRATED LAUNCH SKIP: {calibration['launch_skip']}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command, cwd=REPO_DIR, env=control_env, stdout=log_handle,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            wait_pid_files(pid_dir, process, args.startup_timeout, log_path)
            decode_nsys.wait_server(
                server["base_url"], process, args.startup_timeout,
                log_path, backend)
            warmup_prompt = decode_nsys.make_cache_warmup_prompt(
                prompt, args.prefix_cache_padding_tokens)
            decode_nsys.run_decode_batch(
                server, warmup_prompt, args.warmup_output_tokens,
                args.batch, args, "ncu-warmup")
            request_completed = False
            try:
                decode_nsys.run_decode_batch(
                    server, prompt, args.output_tokens,
                    args.batch, args, "ncu-profile")
                request_completed = True
            except Exception as error:
                log_handle.write(f"EXPECTED PROFILE END: {error}\n")
                log_handle.flush()

            # NCU达到launch-count后仍可能等待常驻服务退出。不能以目标进程
            # 仍存活判断Kernel未命中；应先正常终止服务，再由报告是否生成及
            # 报告内容确认目标Kernel是否被采集。
            if process.poll() is None:
                signaled = signal_target_processes(
                    pid_dir, signal.SIGTERM)
                log_handle.write(
                    "PROFILE TARGET SIGTERM: "
                    f"{','.join(map(str, signaled))}; "
                    f"request_completed={request_completed}\n")
                log_handle.flush()

            report = wait_profile_result(
                process, output_base, args.profile_timeout, log_path)
        except Exception:
            stop_process_group(process)
            raise
        finally:
            stop_process_group(process)
    return report, log_path


def parse_ncu_raw(args, report, output_csv):
    command = [
        args.ncu, "--import", str(report), "--csv", "--page", "raw",
        "--print-metric-name", "name", "--print-units", "base",
    ]
    result = subprocess.run(
        command, cwd=REPO_DIR, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_csv.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"导出NCU raw指标失败：\n{result.stdout[-12000:]}")
    lines = result.stdout.splitlines()
    header = next((index for index, line in enumerate(lines)
                   if re.match(r'^"?ID"?,', line)), None)
    if header is None:
        raise RuntimeError("NCU raw输出中没有找到CSV表头")
    rows = list(csv.DictReader(lines[header:]))
    if not rows:
        raise RuntimeError("NCU raw输出没有指标记录")
    return rows


def metric_number(value):
    if value is None:
        return None
    cleaned = str(value).replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_metrics(rows):
    metrics = []
    for row in rows:
        name = row.get("Metric Name", "").strip()
        if not name:
            continue
        metrics.append({
            "name": name,
            "value": metric_number(row.get("Metric Value")),
            "value_text": row.get("Metric Value", "").strip(),
            "unit": row.get("Metric Unit", "").strip(),
        })
    first = rows[0]
    return metrics, {
        "kernel": first.get("Kernel Name", ""),
        "grid": first.get("Grid Size", first.get("GridXYZ", "")),
        "block": first.get("Block Size", first.get("BlockXYZ", "")),
        "device": first.get("Device Name", first.get("Device", "")),
    }


def select_metric(metrics, candidates):
    by_name = {item["name"]: item for item in metrics}
    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]
    return None


def duration_ns(metric):
    if metric is None or metric["value"] is None:
        return None
    unit = metric["unit"].lower()
    factors = {
        "ns": 1.0, "nsecond": 1.0, "nanosecond": 1.0,
        "us": 1e3, "usecond": 1e3, "microsecond": 1e3,
        "ms": 1e6, "msecond": 1e6, "millisecond": 1e6,
        "s": 1e9, "second": 1e9,
    }
    return metric["value"] * factors.get(unit, 1.0)


def summarize_backend(backend, report, raw_csv, request_calibration,
                      startup_calibration, discovery, rows):
    metrics, launch = normalize_metrics(rows)
    selected = {
        label: select_metric(metrics, candidates)
        for label, candidates in SUMMARY_METRICS.items()
    }
    stalls = [item for item in metrics
              if "warp_issue_stalled" in item["name"] and
              item["value"] is not None]
    stalls.sort(key=lambda item: item["value"], reverse=True)
    return {
        "result": "PASS",
        "backend": backend,
        "kernel_marker": BACKEND_KERNELS[backend],
        "report": str(report),
        "raw_csv": str(raw_csv),
        "launch_skip": startup_calibration["launch_skip"],
        "request_before_stable_matches": request_calibration["launch_skip"],
        "startup_and_warmup_matches":
            startup_calibration["startup_and_warmup_matches"],
        "stable_matches_in_calibration": request_calibration["stable_matches"],
        "discovery": discovery,
        "startup_calibration": startup_calibration,
        "grid": launch["grid"], "block": launch["block"],
        "device": launch["device"], "kernel": launch["kernel"],
        "selected_metrics": selected,
        "top_warp_stalls": stalls[:10],
        "all_metrics": metrics,
        "split_count": None,
        "split_count_note": "NCU不直接暴露逻辑Split数量，不能仅凭Grid臆测",
        "profiling_mode": "cuda_graph_full_lifecycle_calibrated",
    }


def metric_text(metric):
    if metric is None:
        return "n/a"
    return f"{metric['value_text']} {metric['unit']}".strip()


def make_reports(result_dir, summaries):
    lines = [
        "# FastLLM / vLLM Attention合并Kernel NCU对比", "",
        "> 双方保持BEST模式CUDA Graph。脚本先用NCU确认目标Kernel，再用",
        "> Nsight Systems统计进程启动、Graph warmup和正式请求的完整生命周期，",
        "> 最后从进程启动采集首个稳定Decode目标Kernel。NCU会重放Kernel，",
        "> 本报告用于微观归因，不能替代未使用Profiler的TPOT。", "",
        "| 状态 | 后端 | 请求Attention后端 | 实际Attention后端 | 后端确认 | Kernel | "
        "完整跳过数 | 启动/预热调用 | Grid | Block | Duration | DRAM Read | DRAM Write | "
        "DRAM吞吐 | Occupancy | Registers/Thread | Shared Memory/Block |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        selected = item["selected_metrics"]
        lines.append(
            f"| {item['result']} | {item['backend']} | "
            f"{item.get('requested_attention_backend', 'n/a')} | "
            f"{','.join(item.get('actual_attention_backends', [])) or 'n/a'} | "
            f"{item.get('attention_backend_confirmed', 'n/a')} | "
            f"`{item['kernel_marker']}` | "
            f"{item['launch_skip']} | {item['startup_and_warmup_matches']} | "
            f"{item['grid'] or 'n/a'} | {item['block'] or 'n/a'} | "
            f"{metric_text(selected['Duration'])} | "
            f"{metric_text(selected['DRAM Read Bytes'])} | "
            f"{metric_text(selected['DRAM Write Bytes'])} | "
            f"{metric_text(selected['DRAM Throughput'])} | "
            f"{metric_text(selected['Achieved Occupancy'])} | "
            f"{metric_text(selected['Registers/Thread'])} | "
            f"{metric_text(selected['Shared Memory/Block'])} |")

    by_backend = {item["backend"]: item for item in summaries}
    if "fastllm" in by_backend and "vllm" in by_backend:
        fast_duration = duration_ns(
            by_backend["fastllm"]["selected_metrics"]["Duration"])
        vllm_duration = duration_ns(
            by_backend["vllm"]["selected_metrics"]["Duration"])
        ratio = (None if not fast_duration or vllm_duration is None else
                 vllm_duration / fast_duration)
        lines.extend([
            "", "## 时长结论", "",
            f"- vLLM/FastLLM Kernel时长比："
            f"{'n/a' if ratio is None else f'{ratio:.3f}x'}",
            "- Split数量：NCU不直接暴露，不能仅凭Grid判定工作量相同。",
        ])

    for item in summaries:
        lines.extend([
            "", f"## {item['backend']} Warp Stall", "",
            "| Metric | Value | Unit |", "| --- | ---: | --- |",
        ])
        for metric in item["top_warp_stalls"]:
            lines.append(
                f"| `{metric['name']}` | {metric['value_text']} | {metric['unit']} |")

    md_path = result_dir / "attention-ncu-compare.md"
    json_path = result_dir / "attention-ncu-compare.json"
    xlsx_path = result_dir / "attention-ncu-compare.xlsx"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = []
    all_metric_rows = []
    stall_rows = []
    for item in summaries:
        selected = item["selected_metrics"]
        summary_rows.append([
            item["result"], item["backend"],
            item.get("requested_attention_backend", "n/a"),
            ",".join(item.get("actual_attention_backends", [])) or "n/a",
            item.get("attention_backend_confirmed", "n/a"),
            item["kernel_marker"],
            item["grid"], item["block"],
            metric_text(selected["Duration"]),
            metric_text(selected["DRAM Read Bytes"]),
            metric_text(selected["DRAM Write Bytes"]),
            metric_text(selected["DRAM Throughput"]),
            metric_text(selected["Achieved Occupancy"]),
            metric_text(selected["Registers/Thread"]),
            metric_text(selected["Shared Memory/Block"]),
            item["launch_skip"], item["startup_and_warmup_matches"],
            item["request_before_stable_matches"],
            item["stable_matches_in_calibration"],
        ])
        for metric in item["all_metrics"]:
            all_metric_rows.append([
                item["backend"], metric["name"], metric["value"],
                metric["value_text"], metric["unit"],
            ])
        for metric in item["top_warp_stalls"]:
            stall_rows.append([
                item["backend"], metric["name"], metric["value"],
                metric["value_text"], metric["unit"],
            ])
    decode_nsys.write_xlsx(xlsx_path, [
        ("关键指标", [
            "状态", "后端", "请求Attention后端", "实际Attention后端",
            "Attention后端确认", "Kernel", "Grid", "Block", "Duration", "DRAM Read",
            "DRAM Write", "DRAM吞吐", "Occupancy", "Registers/Thread",
            "Shared Memory/Block", "完整launch-skip", "启动及预热调用数",
            "正式请求稳定边界前调用数", "校准区间稳定调用数"],
         summary_rows),
        ("Warp Stall", ["后端", "Metric", "数值", "原始值", "单位"],
         stall_rows),
        ("全部NCU指标", ["后端", "Metric", "数值", "原始值", "单位"],
         all_metric_rows),
    ])
    return md_path, json_path, xlsx_path


def main():
    args = parse_args()
    if args.batch <= 0 or args.prompt_tokens <= 0 or args.output_tokens <= 1:
        raise ValueError("batch和prompt-tokens必须大于0，output-tokens必须大于1")
    backends = selected_backends(args.backends)
    result_dir = Path(args.result_dir).resolve()
    calibration_dir = Path(args.nsys_result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = decode_nsys.make_prompt_tokens(
        tokenizer, args.prompt_tokens, args.batch)
    calibration_prompts = calibration_dir / "prompt-token-ids.json"
    if calibration_prompts.exists():
        saved = json.loads(calibration_prompts.read_text(encoding="utf-8"))
        if saved.get(str(args.batch)) != prompt:
            raise RuntimeError("当前Prompt与Nsight Systems校准Prompt不一致")

    summaries = []
    for index, backend in enumerate(backends):
        marker = BACKEND_KERNELS[backend]
        request_calibration = read_calibration(
            calibration_dir, backend, args.batch, marker)
        backend_dir = result_dir / backend
        original_port = args.port
        args.port = original_port + index
        try:
            if args.skip_discovery:
                discovery = {"skipped": True}
                print(f"[{backend}] 1/3 跳过目标Kernel发现", flush=True)
            else:
                print(f"[{backend}] 1/3 发现目标Kernel", flush=True)
                discovery = run_discovery(
                    args, backend, prompt, backend_dir)
            print(f"[{backend}] 2/3 校准完整生命周期调用数", flush=True)
            startup_calibration = run_startup_calibration(
                args, backend, prompt, request_calibration, backend_dir)
            print(
                f"[{backend}] launch-skip="
                f"{startup_calibration['launch_skip']}", flush=True)
            print(f"[{backend}] 3/3 正式采集稳定Decode Kernel", flush=True)
            report, profile_log = run_profile(
                args, backend, prompt, startup_calibration, backend_dir)
        finally:
            args.port = original_port
        raw_csv = backend_dir / "03-profile" / "attention-merge-raw.csv"
        raw_rows = parse_ncu_raw(args, report, raw_csv)
        summary = summarize_backend(
            backend, report, raw_csv, request_calibration,
            startup_calibration, discovery, raw_rows)
        if backend == "fastllm":
            actual_backends = decode_nsys.read_attention_backend_selection(
                profile_log)
            confirmed = decode_nsys.attention_backend_confirmed(
                args.flm_attention_backend, actual_backends)
            if args.flm_attention_backend_strict and not confirmed:
                raise RuntimeError(
                    "FastLLM请求的Attention Backend未在NCU目标日志中确认："
                    f"requested={args.flm_attention_backend}, "
                    f"actual={actual_backends}")
            summary["requested_attention_backend"] = args.flm_attention_backend
            summary["actual_attention_backends"] = actual_backends
            summary["attention_backend_confirmed"] = confirmed
            actual_layouts = decode_nsys.read_kv_cache_layout_selection(
                profile_log)
            layout_confirmed = decode_nsys.kv_cache_layout_confirmed(
                args.flm_kv_cache_layout, actual_layouts)
            if not layout_confirmed:
                raise RuntimeError(
                    "FastLLM请求的KV Cache布局未在NCU目标日志中确认："
                    f"requested={args.flm_kv_cache_layout}, "
                    f"actual={actual_layouts}")
            summary["requested_kv_cache_layout"] = args.flm_kv_cache_layout
            summary["actual_kv_cache_layouts"] = actual_layouts
            summary["kv_cache_layout_confirmed"] = layout_confirmed
        summaries.append(summary)

    md_path, json_path, xlsx_path = make_reports(result_dir, summaries)
    print(f"Markdown: {md_path}")
    print(f"JSON: {json_path}")
    print(f"Excel: {xlsx_path}")
    print(f"Summary: PASS ({len(summaries)} Attention NCU cases)")


if __name__ == "__main__":
    main()
