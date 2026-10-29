"""Linux process ownership and scoped termination; no model imports."""
import contextlib
import json
import os
import signal
import time
from pathlib import Path


def configure_nofile(config, apply=False):
    """Budget proxy sockets before loading workers; change only the soft limit."""
    import resource
    active = sum(replica['max_inflight'] for replica in config['replicas'])
    queued = len(config['replicas']) * config['router']['queue_limit_per_replica']
    # An active proxy owns downstream + upstream sockets; queued requests own
    # only downstream sockets. Reserve descriptors for health, logs and idle
    # connections. This is a capacity estimate, not a global connection cap.
    required = 2 * active + queued + 256
    requested = config.get('nofile')
    if requested is not None and requested < required:
        raise ValueError(f'nofile={requested} is below the deployment capacity requirement {required}')
    target = requested if requested is not None else required
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and target > hard:
        raise RuntimeError(f'RLIMIT_NOFILE hard limit {hard} is below required {target}; '
                           'raise the launcher/container open-file limit before starting deploy')
    effective = soft if soft == resource.RLIM_INFINITY else max(soft, target)
    if apply and effective != soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (effective, hard))
        except (OSError, ValueError) as error:
            raise RuntimeError(f'Cannot raise RLIMIT_NOFILE soft limit from {soft} to {effective}: {error}') from error
    return {'required': required, 'requested': requested, 'soft_before': soft,
            'soft': effective, 'hard': hard}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def has_live_threads(fields):
    # /proc/<pid>/stat can report Z as soon as the main thread exits, while
    # other threads still run or release CUDA allocations. A final zombie has
    # only its leader left; stat field 20 counts the remaining threads.
    return fields[0] != "Z" or int(fields[17]) > 1


def process_identity(pid, include_zombie=False):
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if not include_zombie and not has_live_threads(parts):
            return None
        return {"pid": int(pid), "start_time": parts[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except FileNotFoundError:
        return None


def owned(record):
    current = process_identity(record["pid"])
    return current is not None and all(current[k] == record[k] for k in current)


def signal_owned(record, sig):
    if owned(record):
        # Only independent process groups recorded at start/adoption are owned.
        with contextlib.suppress(ProcessLookupError):
            if os.getpgid(record["pid"]) != record["pid"]:
                raise RuntimeError("Refusing to signal a process not leading its recorded group")
            os.killpg(record["pid"], sig)


def group_members(record):
    current = process_identity(record["pid"])
    if current is not None and not owned(record):
        return []
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != record["boot_id"]:
        return []
    members = []
    for p in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = p.read_text().rsplit(")", 1)[1].split()
            if (has_live_threads(fields) and int(fields[2]) == record["pid"]
                    and int(fields[3]) == record["pid"]
                    and int(fields[19]) >= int(record["start_time"])):
                members.append(int(p.parent.name))
        except FileNotFoundError:
            pass
    return members


def stop_records(records, timeout=30):
    for record in reversed(records):
        if owned(record):
            signal_owned(record, signal.SIGTERM)
        elif group_members(record):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(record["pid"], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(group_members(r) for r in records):
        time.sleep(.2)
    for record in records:
        # The original group can outlive its leader; PID reuse checks above
        # still prevent signalling an unrelated replacement process.
        if group_members(record):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(record["pid"], signal.SIGKILL)
    # SIGKILL is asynchronous, particularly while CUDA/pinned host allocations
    # are being released. A successful stop must leave no owned group behind.
    deadline = time.monotonic() + max(10, timeout)
    while True:
        remaining = [pid for record in records for pid in group_members(record)]
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Stopped service processes have not exited: {remaining}")
        time.sleep(.2)
