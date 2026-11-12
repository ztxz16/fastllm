"""独立进程中的只读GPU/CPU采样；不导入模型、不创建CUDA上下文、不修改设备设置。"""

from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


GPU_FIELDS = ['uuid', 'index', 'pstate', 'clocks.current.sm', 'clocks.current.memory',
              'utilization.gpu', 'utilization.memory', 'temperature.gpu', 'power.draw', 'power.limit']
GPU_REASONS = 'clocks_throttle_reasons.active'


def process_tree(root, excluded, proc=Path('/proc')):
    """/** 读取父进程及后代的累计CPU时间和亲和性；不采集命令行或环境变量，排除观察进程。 */"""
    records = {}
    ticks = os.sysconf('SC_CLK_TCK')
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) == excluded:
            continue
        try:
            stat = (directory / 'stat').read_text()
            end = stat.rindex(')')
            fields = stat[end + 2:].split()
            status = dict(line.split(':', 1) for line in (directory / 'status').read_text().splitlines() if ':' in line)
            pid = int(directory.name)
            records[pid] = dict(pid=pid, ppid=int(fields[1]), name=stat[stat.index('(') + 1:end],
                                start_ticks=int(fields[19]), cpu_seconds=(int(fields[11]) + int(fields[12])) / ticks,
                                threads=int(fields[17]), allowed_cpus=status.get('Cpus_allowed_list', '').strip())
        except (OSError, ValueError, IndexError):
            continue  # 进程可在读取期间退出。
    selected = {root} if root in records else set()
    while True:
        children = {pid for pid, row in records.items() if row['ppid'] in selected}
        if children <= selected:
            break
        selected.update(children)
    return [records[pid] for pid in sorted(selected)]


def cgroup_cpu():
    """/** 保存实际读取路径及cgroup v2 CPU计数；不支持时明确标记，不能据缺失值推断无限制。 */"""
    base = Path('/sys/fs/cgroup')
    try:
        membership = Path('/proc/self/cgroup').read_text().strip()
        suffix = next(line[3:] for line in membership.splitlines() if line.startswith('0::'))
        candidate = base / suffix.lstrip('/')
        if (candidate / 'cpu.stat').is_file():
            base = candidate
        counters = dict(line.split() for line in (base / 'cpu.stat').read_text().splitlines())
        return dict(path=str(base), membership=membership, cpu_max=(base / 'cpu.max').read_text().strip(),
                    counters={key: int(value) for key, value in counters.items()})
    except (OSError, StopIteration, ValueError) as error:
        return dict(unavailable=f'{type(error).__name__}: {error}')


def gpu_sample(fields):
    """/** 执行一次有超时的只读nvidia-smi查询；按UUID识别设备，保留N/A而不伪造数值。 */"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=3)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f'nvidia-smi exit {result.returncode}')
    rows = []
    for values in csv.reader(result.stdout.splitlines()):
        if len(values) != len(fields):
            raise RuntimeError('nvidia-smi column count mismatch')
        row = {}
        for key, value in zip(fields, values):
            value = value.strip()
            try:
                row[key] = int(value) if key == 'index' else (float(value) if key not in ('uuid', 'pstate', GPU_REASONS) else value)
            except ValueError:
                row[key] = value
        rows.append(row)
    if not rows:
        raise RuntimeError('nvidia-smi returned no GPUs')
    return rows


def observe(output, root, interval):
    """/** 持续记录GPU查询时间窗和CPU快照；SIGTERM停止，失败保留日志并返回非零。 */"""
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous = {}
    fields = [*GPU_FIELDS, GPU_REASONS]
    errors, warnings = [], []
    count = 0
    try:
        with (output / 'telemetry.jsonl').open('w') as handle:
            while not stopped and os.getppid() == root:
                start = time.perf_counter()
                record = dict(start=start, wall_time=time.time())
                try:
                    try:
                        record['gpu'] = gpu_sample(fields)
                    except RuntimeError:
                        if count or GPU_REASONS not in fields:
                            raise
                        fields = GPU_FIELDS[:]
                        record['gpu'] = gpu_sample(fields)
                        warnings.append('clock event reasons query unavailable; base GPU fields retained')
                    record['gpu_end'] = time.perf_counter()
                    cpu_time = time.perf_counter()
                    processes = process_tree(root, os.getpid())
                    if not any(row['pid'] == root for row in processes):
                        raise RuntimeError('benchmark process CPU counters unavailable')
                    for row in processes:
                        key = (row['pid'], row['start_ticks'])
                        old = previous.get(key)
                        row['cpu_percent_one_core'] = (100 * (row['cpu_seconds'] - old[1]) / (cpu_time - old[0])
                                                       if old is not None else None)
                        row.update(cpu_interval_start=old[0] if old is not None else None, cpu_interval_end=cpu_time)
                        previous[key] = (cpu_time, row['cpu_seconds'])
                    record.update(cpu_time=cpu_time, processes=processes, cgroup=cgroup_cpu())
                    for name, path in [('cpu_pressure', '/proc/pressure/cpu'), ('system_cpu_stat', '/proc/stat')]:
                        try:
                            record[name] = Path(path).read_text().splitlines()[0]
                        except OSError as error:
                            record[name] = f'unavailable: {error}'
                except Exception as error:
                    record['error'] = f'{type(error).__name__}: {error}'
                    errors.append(record['error'])
                record['end'] = time.perf_counter()
                handle.write(json.dumps(record) + '\n')
                handle.flush()
                count += 1
                if count == 1:
                    temporary = output / 'observer.ready.tmp'
                    temporary.write_text(json.dumps(dict(error=record.get('error'), warnings=warnings)))
                    temporary.replace(output / 'observer.ready.json')
                if record.get('error'):
                    break
                while not stopped and time.perf_counter() < start + interval:
                    time.sleep(min(.05, max(0, start + interval - time.perf_counter())))
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')
    (output / 'observer.status.json').write_text(json.dumps(dict(samples=count, errors=errors, warnings=warnings,
        end_monotonic=time.perf_counter(),
        fields=fields, cpu_percent_definition='100% = one logical CPU; process total can exceed 100%')))
    return int(bool(errors))


@contextmanager
def observer(output, interval=1.0, enabled=True):
    """/** 在独立解释器中采样；启动失败、测试异常或中断时均回收观察进程，不遗留后台任务。 */"""
    if not enabled:
        yield
        return
    with (output / 'observer.stderr.log').open('w') as stderr:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(output), str(os.getpid()), str(interval)],
                                   stderr=stderr, stdout=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10
            ready = output / 'observer.ready.json'
            while not ready.exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f'observer startup failed; see {output / "observer.stderr.log"}')
                time.sleep(.05)
            error = json.loads(ready.read_text()).get('error')
            if error:
                raise RuntimeError('observer startup: ' + error)
            yield process.pid
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == '__main__':
    raise SystemExit(observe(Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])))
