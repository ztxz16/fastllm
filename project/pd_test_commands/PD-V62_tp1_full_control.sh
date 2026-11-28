(
set -euo pipefail
cd /home/project/vllm_fastllm
umask 077
exec 9>test/distributed/results/.pd_v62_full_control_launch.lock
flock -n 9 || { echo '已有启动操作，未重复提交。'; exit 1; }
shopt -s nullglob
pd_existing=(test/distributed/results/*_pd_v62_full_control_*)
if ((${#pd_existing[@]})); then
    printf '已有诊断目录，先查看结果，未重复提交：\n%s\n' "${pd_existing[@]}"
    exit 0
fi
flock -n test/distributed/results/.pd_tp2_tp1_pull.lock true || {
    echo '已有 PD 测试持有锁，未提交新请求。'; exit 1;
}
test -x /venv/main/bin/python3
test -f build-process-tp/_pd_core.so
pd_run="$PWD/test/distributed/results/$(date -u +%Y%m%dT%H%M%SZ)_pd_v62_full_control_$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
mkdir "$pd_run"
cat > "$pd_run/run.py" <<'PY'
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/home/project/vllm_fastllm')
SOURCE = ROOT / 'test/distributed/results/20261128T110052Z_pd_tp1_ep2_pull_5ac24694'
GEN = ROOT / 'test/distributed/results/20261128T110401Z_pd_generation_d98fff95'
OUT = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8001'
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'test/distributed/pd')]
from check_generation import request
from generation_reference import deployment
from report import run


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read(path):
    return json.loads(path.read_text())


def router_identity():
    proc = Path(f'/proc/{router_pid}')
    args = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
    require(args == router_command, '8001 Router 命令已变化，停止提交')
    return proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]


def snapshot(idle):
    require(router_identity() == router_start, '8001 Router 启动身份已变化')
    query = urllib.request.Request(URL + '/health', headers={'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(query, timeout=10) as response:
        workers = json.load(response)['workers']
    require(len(workers) == 1 and workers[0]['url'].rstrip('/') == 'http://127.0.0.1:8300'
            and workers[0]['role'] == 'full' and workers[0]['healthy'], '8001 不是原完整服务入口')
    if idle:
        require(workers[0]['inflight'] == 0, '8001 仍有在途请求')

    values = {role: deployment(URL if role == 'full' else 'http://127.0.0.1:8000', token, role)
              for role in ('p', 'd', 'full')}
    for role, value in values.items():
        old = expected[role]
        require(value['config'] == old['config'] and value['peers'] == old['peers'], role + ' 配置/boot 已变化')
        require(set(value['evidence']) == set(old['evidence']) and len(value['evidence']) == 1,
                role + ' 服务集合不符')
        current = next(iter(value['evidence'].values()))
        previous = next(iter(old['evidence'].values()))
        require(len(current['replicas']) == len(previous['replicas']) and all(
            a['workers'] == b['workers'] and a.get('group_workers') == b.get('group_workers')
            for a, b in zip(current['replicas'], previous['replicas'])), role + ' 原生 worker 已变化')
        health = current['health']
        require(health['healthy'], role + ' 不健康')
        if idle:
            require(health['load'] == 0 and health['pending_control'] == 0, role + ' 仍忙碌')
    return values


def service(value, role):
    return next(iter(value[role]['evidence'].values()))


def new_records(before, after, role):
    known = {(r['request'], r['transfer']) for r in service(before, role)['requests']}
    return [r for r in service(after, role)['requests'] if (r['request'], r['transfer']) not in known]


def execute(name, key):
    global blocked
    require(not blocked, blocked or '')
    detail = dict(source_case=key, request=evidence[key]['request'],
                  reference_attention=evidence[key]['reference'], original_pd=evidence[key]['actual'])
    outputs[name] = detail
    try:
        before = snapshot(True)
        save(name + '.before.json', before)
        actual = request(URL, token, detail['request'])
        detail['actual_full_tp1'] = actual
        save(name + '.evidence.json', detail)
        ids = actual['token_ids']
        require(isinstance(ids, list) and len(ids) == detail['request']['max_new_tokens']
                and all(type(t) is int and t >= 0 for t in ids), '输出 Token 数量或类型非法')
        detail['tokens_valid'] = True

        deadline = time.monotonic() + 30
        while True:
            after = snapshot(False)
            save(name + '.after.json', after)
            fresh = {role: new_records(before, after, role) for role in ('p', 'd', 'full')}
            detail['service_records'] = fresh
            save(name + '.evidence.json', detail)
            require(not fresh['p'] and not fresh['d'], 'P/D 出现其他请求，不能归属本轮证据')
            require(len(fresh['full']) <= 1 and not any(r['error'] for r in fresh['full']),
                    '完整服务有额外请求或执行错误')
            if len(fresh['full']) == 1 and fresh['full'][0]['released']:
                break
            require(time.monotonic() < deadline, '完整服务证据缺失或释放超时')
            time.sleep(1)
        response_ids = {c['id'] for c in [actual.get('response', {})] + actual.get('deltas', []) if c.get('id')}
        require(not response_ids or response_ids == {fresh['full'][0]['request']}, '响应请求 ID 不匹配')
        a = service(before, 'full')['replicas'][0]['states'][0]
        b = service(after, 'full')['replicas'][0]['states'][0]
        require(b['eager_decodes'] > a['eager_decodes'], '完整服务没有实际 Eager Decode 增量')
        snapshot(True)
        detail['execution'] = 'PASS'
        save(name + '.evidence.json', detail)
        print(json.dumps(dict(case=name, token_ids=ids, execution='PASS'), ensure_ascii=False), flush=True)
        return dict(requests=1, prompt_tokens=len(detail['request']['token_ids']), output_tokens=len(ids),
                    eager_decode_increase=b['eager_decodes'] - a['eager_decodes'], released=True)
    except Exception as error:
        blocked = f'执行核查失败，停止后续生成：{type(error).__name__}: {error}'
        detail['execution_error'] = blocked
        save(name + '.evidence.json', detail)
        raise


def compare(name, side):
    detail = outputs.get(name, {})
    actual = detail.get('actual_full_tp1')
    require(actual is not None and detail.get('tokens_valid'), '有效生成证据缺失，不能完成 exact 比较')
    got, want = actual['token_ids'], detail[side]['token_ids']
    first = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b),
                 min(len(want), len(got)) if len(want) != len(got) else None)
    result = dict(exact='PASS' if got == want else 'FAIL', first_divergence=first,
                  expected_token_ids=want, actual_token_ids=got)
    detail.setdefault('comparisons', {})[side] = result
    save(name + '.evidence.json', detail)
    print(json.dumps(dict(case=name, against=side, **result), ensure_ascii=False), flush=True)
    require(got == want, f'exact FAIL against {side} at output {first}; evidence retained')
    return dict(exact='PASS', against=side)


if __name__ == '__main__':
    (OUT / 'test.pid').write_text(str(os.getpid()) + '\n')
    with (ROOT / 'test/distributed/results/.pd_tp2_tp1_pull.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        outputs, evidence, blocked = {}, {}, None
        metadata = dict(source_run=str(SOURCE), source_generation=str(GEN),
                        planned_http_requests=4, warmup=0, iterations=1, repeats=2, evidence_sha256={})
        try:
            require((SOURCE / 'exit_code').read_text().strip() == '1'
                    and (GEN / 'exit_code').read_text().strip() == '1', '原任务退出状态不符')
            expected = read(SOURCE / 'deployment.before.json')
            router_pid = read(SOURCE / 'processes.before.json')['reference_router']['pid']
            router_command = read(SOURCE / 'reference_router.command.json')
            router_start = router_identity()
            env = dict(s.split('=', 1) for s in Path(f'/proc/{router_pid}/environ').read_bytes().decode().split('\0') if '=' in s)
            token = (router_command[router_command.index('--pd-token') + 1]
                     if '--pd-token' in router_command else env.get('FT_PD_TOKEN', ''))
            require(bool(token), '未取得原完整服务认证')

            summary = read(GEN / 'summary.json')
            profile = summary['metadata']['profile']
            for key in ('sampling_fixed_seed', 'concurrent4_request3'):
                path = GEN / (profile + '_' + key + '.evidence.json')
                raw = path.read_bytes()
                (OUT / ('source.' + key + '.json')).write_bytes(raw)
                metadata['evidence_sha256'][key] = hashlib.sha256(raw).hexdigest()
                evidence[key] = json.loads(raw)
                require(evidence[key]['reference']['token_ids'] != evidence[key]['actual']['token_ids'], '原失败证据不符')
            full = expected['full']['config']
            require(len(full) == 1 and len(full[0]['replicas']) == 1, '完整服务不是单副本')
            replica = full[0]['replicas'][0]
            require(replica['model_type'] == 'qwen2_moe' and replica['storage'] == 'paged' and replica['tp'] == 1
                    and len(replica['ranks']) == 1 and not replica['ranks'][0]['graph']
                    and not replica['ranks'][0]['prefix_cache'], '完整服务不是预期普通 TP1 Eager/Prefix 关闭配置')
            require(full[0]['model'] == expected['d']['config'][0]['model'], '完整服务与 D 模型标识不同')
            save('deployment.before.json', snapshot(True))

            metadata['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,compute_cap',
                                                       '--format=csv,noheader'], text=True).strip()
            cache = ROOT / 'build-process-tp/CMakeCache.txt'
            arch = [s for s in cache.read_text().splitlines() if not s.startswith(('#', '//'))
                    and any(k in s.split('=', 1)[0] for k in ('CUDA_ARCH', 'CUDA_FLAGS'))] if cache.is_file() else []
            metadata['architecture'] = ('configured in existing CMakeCache.txt:\n' + '\n'.join(arch)
                                        if arch else 'not available from existing CMakeCache.txt')
            save('source.generation.summary.json', summary)
        except Exception as error:
            blocked = f'preflight: {type(error).__name__}: {error}'
            (OUT / 'preflight.error.txt').write_text(blocked + '\n')

        cases = []
        for key in ('sampling_fixed_seed', 'concurrent4_request3'):
            for trial in (1, 2):
                name = f'tp1_full_{key}_trial{trial}'
                cases.append((name + '_execution', lambda n=name, k=key: execute(n, k)))
                for side in ('reference_attention', 'original_pd'):
                    cases.append((name + '_vs_' + side, lambda n=name, s=side: compare(n, s)))
        raise SystemExit(run(cases, OUT, backend='formal HTTP 8001 -> existing ordinary TP1 full service',
            metadata_extra=metadata,
            scope='4 次生成请求；4 项执行/释放核查及 8 项独立 exact 比较。跨配置定位，不替换 Attention 验收基线；任一 exact FAIL 保留并返回非零'))
PY
cat > "$pd_run/command.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
pd_directory=$(cd -- "$(dirname -- "$0")" && pwd)
trap 'pd_code=$?; echo "$pd_code" > "$pd_directory/exit_code"' EXIT
cd /home/project/vllm_fastllm
export FT_PD_CORE_LIBRARY="$PWD/build-process-tp/_pd_core.so"
export PYTHONPATH="$PWD/tools${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/build-process-tp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
/venv/main/bin/python3 -u "$pd_directory/run.py"
SH
nohup bash "$pd_run/command.sh" > "$pd_run/stdout.log" 2> "$pd_run/stderr.log" < /dev/null 9>&- &
echo "$!" > "$pd_run/supervisor.pid"
echo "本次目录：$pd_run"
echo '已后台提交；查看 exit_code、summary.md 和 stdout.log，不要重复启动。'
)
