"""仅复验 PD-V62 的采样、失败请求单独执行与原并发 4；不切换部署。"""

import fcntl
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path('/home/project/vllm_fastllm')
SOURCE = ROOT / 'test/distributed/results/20261128T110052Z_pd_tp1_ep2_pull_5ac24694'
GEN = ROOT / 'test/distributed/results/20261128T110401Z_pd_generation_d98fff95'
OUT = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8000'
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'test/distributed/pd')]
from check_generation import request
from generation_reference import deployment
from report import run


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def setup():
    pid = int((SOURCE / 'router.pid').read_text())
    command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    require(command == json.loads((SOURCE / 'router.command.json').read_text()), 'Router 已变化；不提交请求')
    env = dict(entry.split('=', 1) for entry in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in entry)
    token = command[command.index('--pd-token') + 1] if '--pd-token' in command else env.get('FT_PD_TOKEN', '')
    require(bool(token), '未取得当前服务认证')
    old = json.loads((SOURCE / 'deployment.before.json').read_text())
    summary = json.loads((GEN / 'summary.json').read_text())
    profile = summary['metadata']['profile']
    names = ['sampling_fixed_seed'] + [f'concurrent4_request{i}' for i in range(4)]
    evidence = {}
    for name in names:
        path = GEN / (profile + '_' + name + '.evidence.json')
        raw = path.read_bytes()
        (OUT / ('source.' + name + '.json')).write_bytes(raw)
        evidence[name] = json.loads(raw)
    return token, old, evidence


def snapshot(token, expected):
    values = {r: deployment(URL, token, r) for r in ('p', 'd')}
    for role, value in values.items():
        require(value['config'] == expected[role]['config'] and value['peers'] == expected[role]['peers'],
                role + ' 配置或 boot 已变化；不使用旧部署继续复验')
        url = next(iter(value['evidence']))
        current = value['evidence'][url]['replicas']
        previous = expected[role]['evidence'][url]['replicas']
        require(len(current) == len(previous) and all(
            a['workers'] == b['workers'] and a.get('group_workers') == b.get('group_workers')
            for a, b in zip(current, previous)), role + ' 原生 worker 已变化')
        health = next(iter(value['evidence'].values()))['health']
        require(health['healthy'] and health['load'] == 0 and health['pending_control'] == 0,
                role + ' 不健康或仍忙碌；勿与其他客户端并行运行')
    return values


def records(snapshot_value, role):
    return next(iter(snapshot_value[role]['evidence'].values()))['requests']


def check(name, keys):
    if setup_error:
        raise RuntimeError(setup_error)
    before = snapshot(token, expected)
    save(name + '.before.json', before)

    def send(key):
        original = evidence[key]
        payload = original['request']
        try:
            actual = request(URL, token, payload)
            want, got = original['reference']['token_ids'], actual['token_ids']
            first = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b),
                         min(len(want), len(got)) if len(want) != len(got) else None)
            ids = sorted({chunk['id'] for chunk in [actual.get('response', {})] + actual.get('deltas', []) if chunk.get('id')})
            return dict(source_case=key, request=payload, actual=actual, reference=original['reference'],
                        original_actual_token_ids=original['actual']['token_ids'], exact=want == got,
                        matches_original_actual=got == original['actual']['token_ids'],
                        first_divergence=first, response_request_ids=ids,
                        candidate_comparison='not available: public endpoint does not export logits')
        except Exception as error:
            return dict(source_case=key, request=payload, error=f'{type(error).__name__}: {error}', exact=False)

    if len(keys) == 1:
        outputs = [send(keys[0])]
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(send, keys))
    detail = dict(outputs=outputs, scope='定位复验；不替代原 15 个生成 case 与完整运行核查')
    save(name + '.evidence.json', detail)

    deadline = time.monotonic() + 30
    while True:
        # 释放可能稍晚于 HTTP 完成；只轮询证据，不重复请求。
        after = {r: deployment(URL, token, r) for r in ('p', 'd')}
        save(name + '.after.json', after)
        fresh = {}
        for role in ('p', 'd'):
            require(after[role]['config'] == before[role]['config'] and after[role]['peers'] == before[role]['peers'],
                    role + ' 在复验中重启或变更配置')
            known = {(r['request'], r['transfer']) for r in records(before, role)}
            fresh[role] = [r for r in records(after, role) if (r['request'], r['transfer']) not in known]
        detail['service_records'] = fresh
        save(name + '.evidence.json', detail)
        require(not any(r['error'] for rows in fresh.values() for r in rows), '复验请求存在服务错误，见 evidence')
        require(all(len(rows) <= len(keys) for rows in fresh.values()), '出现额外请求，不能归属本轮证据')
        if all(len(rows) == len(keys) and all(r['released'] and r['prefill_done' if role == 'p' else 'completed']
                                            for r in rows) for role, rows in fresh.items()):
            break
        require(time.monotonic() < deadline, '请求证据缺失或尚未完成释放，见 evidence')
        time.sleep(1)
    require({(r['request'], r['transfer']) for r in fresh['p']} ==
            {(r['request'], r['transfer']) for r in fresh['d']}, 'P/D 请求集合不一致')
    for output in outputs:
        ids = output.get('response_request_ids', [])
        matches = [r for r in fresh['d'] if r['request'] in ids] if ids else (fresh['d'] if len(keys) == 1 else [])
        output['target_replicas'] = [(r.get('pair') or {}).get('target', {}).get('index') for r in matches]
        output['association'] = 'response_id' if ids else '唯一新增请求记录' if len(keys) == 1 else 'unavailable'
    save(name + '.evidence.json', detail)
    compact = [dict(case=o['source_case'], exact=o['exact'], first_divergence=o.get('first_divergence'),
                    matches_original_actual=o.get('matches_original_actual'), target_replicas=o.get('target_replicas'),
                    error=o.get('error')) for o in outputs]
    print(json.dumps(dict(case=name, requests=compact), ensure_ascii=False), flush=True)
    failures = [o for o in outputs if not o['exact']]
    require(not failures, '; '.join(f"{o['source_case']}: {o.get('error') or 'exact FAIL at output ' + str(o['first_divergence'])}" for o in failures))
    return dict(requests=len(keys), exact='PASS')


setup_error = None
try:
    token, expected, evidence = setup()
except Exception as error:
    setup_error = f'preflight: {type(error).__name__}: {error}'
    (OUT / 'preflight.error.txt').write_text(setup_error + '\n')
cases = []
for scene, keys in (
    ('sampling', ['sampling_fixed_seed']),
    ('request3_alone', ['concurrent4_request3']),
    ('concurrent4', [f'concurrent4_request{i}' for i in range(4)]),
):
    for trial in (1, 2):
        name = f'v62_{scene}_trial{trial}'
        cases.append((name, lambda n=name, k=keys: check(n, k)))

if __name__ == '__main__':
    (OUT / 'test.pid').write_text(str(os.getpid()) + '\n')
    with (ROOT / 'test/distributed/results/.pd_tp2_tp1_pull.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run(cases, OUT, backend='formal HTTP Router -> existing ordinary TP P / Attention EP D; CUDA Eager/IPC',
            metadata_extra=dict(source_run=str(SOURCE), source_generation=str(GEN),
                                gpu='see original generation summary.json', cuda='existing workers: see original deployment.before.json',
                                architecture='existing server artifacts; see original generation metadata'),
            scope='第 7 步 PD-V62 定位：三种场景各两轮，共 6 次场景执行/12 个请求；保留旧 exact FAIL，不代表完整验收'))
