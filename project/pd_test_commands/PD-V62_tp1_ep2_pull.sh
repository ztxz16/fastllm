(
set -euo pipefail
cd /home/project/vllm_fastllm
umask 077
pd_next="$PWD/test/distributed/results/$(date -u +%Y%m%dT%H%M%SZ)_pd_tp1_ep2_pull_$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
mkdir "$pd_next"
export FT_PD_CORE_LIBRARY="$PWD/build-process-tp/_pd_core.so"
export PYTHONPATH="$PWD/tools${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/build-process-tp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

cat > "$pd_next/run.py" <<'PY_RUN'
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/project/vllm_fastllm')
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'test/distributed/results/20261128T105105Z_pd_ep2_tp1_push_be393099'
BASE = ROOT / 'test/distributed/results/20261128T033907Z_pd_generation_536f636f/reference.json'
FULL, REF = 'http://127.0.0.1:8300', 'http://127.0.0.1:8001'
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'test/distributed/pd')]
from report import run


def check():
    source = OLD / 'helpers.source.py'
    spec = importlib.util.spec_from_file_location('tp_helpers', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.OUT, m.__file__ = OUT, str(Path(__file__).resolve())
    m.load_helpers()
    require, save = m.require, m.save
    (OUT / 'helpers.source.py').write_bytes(source.read_bytes())
    require((OLD / 'exit_code').read_text().strip() == '0'
            and json.loads((OLD / 'runtime_check.json').read_text())['passed'], 'PD-V61 尚未通过')
    previous = json.loads((OLD / 'deployment.after.json').read_text())

    raw = BASE.read_bytes()
    archive = json.loads(raw)
    summary_raw = BASE.with_name('summary.json').read_bytes()
    summary = json.loads(summary_raw)
    require(archive['version'] == 1 and archive['complete'] and len(archive['records']) == 21
            and archive['summary_sha256'] == hashlib.sha256(summary_raw).hexdigest()
            and summary['status'] == 'PASS' and len(summary['cases']) == 15
            and all(c['status'] == 'PASS' for c in summary['cases']), 'Attention 基线归档不完整，未停止服务')
    ids = archive['records']['first_token_only']['request']['token_ids']
    require(isinstance(ids, list) and ids and all(type(t) is int and t >= 0 for t in ids), '基线输入非法')
    save('prompt_token_ids.json', ids)
    save('reference.source.json', dict(path=str(BASE), sha256=hashlib.sha256(raw).hexdigest()))
    (OUT / 'reference.path').write_text(str(BASE) + '\n')

    specs, tokens = {}, set()
    for name, port in [('p', 8100), ('d', 8200), ('full', 8300),
                       ('router', 8000), ('reference_router', 8001)]:
        pid = m.listener(port)
        require(pid is not None, f'{port} 缺少预期服务，未切换')
        mark = m.stamp(pid)
        command, env = m.process(pid)
        require(mark is not None and command == json.loads((OLD / (name + '.command.json')).read_text()),
                name + ' 启动参数已变化，未切换')
        tokens.add(m.token_from(command, env))
        specs[name] = dict(pid=pid, mark=mark, command=command, env=env)
        save(name + '.before.command.json', command)
        if name in ('full', 'reference_router'):
            save(name + '.command.json', command)
    require(len(tokens) == 1 and all(tokens), '认证不一致')
    token = tokens.pop()
    os.environ['FT_PD_TOKEN'] = token
    save('processes.before.json', {k: dict(pid=v['pid'], start=v['mark']) for k, v in specs.items()})
    urls = dict(p=m.P, d=m.D, full=FULL)

    def snapshot():
        return {r: m.deployment(REF if r == 'full' else m.ROUTER, token, r) for r in urls}

    def unchanged(old, new, roles=('p', 'd', 'full')):
        for role in roles:
            url = urls[role]
            a, b = old[role], new[role]
            require(a['config'] == b['config'] and a['peers'] == b['peers'], role + ' 配置/启动身份变化')
            value = b['evidence'][url]
            m.idle(value)
            x, y = a['evidence'][url]['replicas'], value['replicas']
            require(len(x) == len(y) and all(u['workers'] == v['workers']
                    and u.get('group_workers') == v.get('group_workers') for u, v in zip(x, y)),
                    role + ' 原生 worker 变化')
            require(all(r['released'] and not r['error'] for r in value['requests']), role + ' 请求未释放或失败')
        for name in (*roles, 'reference_router'):
            require(m.stamp(specs[name]['pid']) == specs[name]['mark'], name + ' 退出或重启')

    for url, expected in [(m.ROUTER, {m.P, m.D}), (REF, {FULL})]:
        workers = m.get(url, '/health')['workers']
        require(len(workers) == len(expected) and {w['url'] for w in workers} == expected
                and all(w['healthy'] and w['inflight'] == 0 for w in workers), 'Router 未就绪或仍有请求')
    original = snapshot()
    save('deployment.original.json', original)
    unchanged(previous, original)
    require(original['p']['config'] == archive['config'], '旧 P 的 Attention 执行配置与基线不符，未切换')
    model_id = archive['config'][0]['model']
    require(all(v['config'][0]['model'] == model_id for v in original.values()), '模型版本不一致')

    def attention(value, role):
        m.idle(value)
        require(value['health']['role'] == role and len(value['replicas']) == 2, 'Attention 副本数/角色错误')
        group = value['replicas'][0]['group_workers']
        require(len(group) == 2 and [w['rank'] for w in group] == [0, 1], 'EP 组 rank 不完整')
        for i, replica in enumerate(value['replicas']):
            require(replica['parallel'] == dict(mode='attention_dp_ep', attention_dp=2, ep=2, rank=i)
                    and replica['storage'] == 'continuous' and len(replica['states']) == 1
                    and replica['group_workers'] == group and replica['workers'] == [group[i]]
                    and replica['devices'] == [group[i]['device']], 'Attention 副本/存储/worker 不符')
            state = replica['states'][0]
            require(not state['enabled'] and not state['prefix_cache']['enabled']
                    and state['captures'] == state['replays'] == 0, '应为 Eager/Prefix 关闭')

    attention(original['p']['evidence'][m.P], 'p')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,compute_cap',
                                   '--format=csv,noheader'], text=True)
    (OUT / 'gpu.csv').write_text(gpu)
    physical = {m.identity(row.split(',')[1]): int(row.split(',')[0]) for row in gpu.splitlines()}
    uuids = {int(row.split(',')[0]): row.split(',')[1].strip() for row in gpu.splitlines()}
    require(set(uuids) == {0, 1, 2, 3}, '预期四张 GPU')
    for role, devices in [('p', [0, 2]), ('d', [3]), ('full', [1])]:
        value = original[role]['evidence'][urls[role]]
        if role != 'p':
            m.layout(value, role, 1)
        workers = m.mapping(specs[role]['env'], value, physical, devices, role + '.old')
        specs[role]['workers'] = [(w['pid'], m.stamp(w['pid'])) for w in workers]
        require(all(mark for _, mark in specs[role]['workers']), role + ' 原生 worker 已退出')
    config = json.loads((m.MODEL / 'config.json').read_text())
    heads = config.get('num_key_value_heads', config['num_attention_heads'])
    cache = ROOT / 'build-process-tp/CMakeCache.txt'
    if cache.exists():
        (OUT / 'CMakeCache.txt').write_bytes(cache.read_bytes())

    def changed(command, values):
        command = list(command)
        command[0] = sys.executable
        for key, value in values.items():
            require(command.count(key) == 1, '启动参数缺失或重复：' + key)
            command[command.index(key) + 1] = str(value)
        return command

    from fastllm_pytools.cli import args_parser
    commands, envs = {}, {}
    preview = """import json, sys
from fastllm_pytools.cli import args_parser
from fastllm_pytools.util import load_model_args
from fastllm_pytools.process_tp import validate_options
from fastllm_pytools.parallel_config import build_parallel_plan
args = args_parser().parse_args(json.loads(sys.argv[1]))
load_model_args(args)
validate_options(args, allow_dp=True, allow_pd=True)
print(json.dumps(build_parallel_plan(args)))
"""
    for role, donor, port, order in [('p', 'd', 8100, [3, 0, 1, 2]), ('d', 'p', 8200, [0, 2, 1, 3])]:
        commands[role] = changed(specs[donor]['command'], {
            '--role': role, '--port': port, '--instance': role + '_tp1_ep2', '--tp-output': OUT / role})
        envs[role] = dict(specs[donor]['env'], FT_PD_TOKEN=token, PYTHONUNBUFFERED='1',
            FT_PD_CORE_LIBRARY=os.environ['FT_PD_CORE_LIBRARY'],
            CUDA_VISIBLE_DEVICES=','.join(uuids[i] for i in order))
        command = commands[role]
        cli_args = command[command.index('fastllm_pytools.cli') + 1:]
        args = args_parser().parse_args(cli_args)
        require(args.role == role and args.port == port and args.router_managed
                and args.tp_backend == 'process' and args.tp == '1' and args.dp == 1
                and not args.cuda_graph and args.prefix_cache == 'false' and not args.pd_return_kv
                and not args.batch_invariant and args.pd_model_id == model_id, role + ' 配置不符')
        require(args.attention_dp == args.ep == (2 if role == 'd' else 1), 'Attention 规模不符')
        if role == 'd':
            require(args.moe_parallel_mode == 'ep' and args.dp_routing == 'load', 'D 必须为 EP2/load')
        for binary in (args.tp_binary, args.tp_worker):
            require(os.access(binary, os.X_OK), '原生产物缺失：' + binary)
        probe = subprocess.run([sys.executable, '-c', preview, json.dumps(cli_args)],
            cwd=ROOT, env=envs[role], capture_output=True, text=True, timeout=60)
        (OUT / (role + '.plan.stdout.log')).write_text(probe.stdout)
        (OUT / (role + '.plan.stderr.log')).write_text(probe.stderr)
        require(probe.returncode == 0, role + ' 参数/设备规划失败，未停止服务')
        plan = json.loads(probe.stdout)
        save(role + '.plan.json', plan)
        expected = [3] if role == 'p' else [0, 2]
        require(plan['mode'] == ('replica_dp_tp' if role == 'p' else 'attention_dp_ep')
                and plan['tp_size'] == plan['dp_size'] == 1 and plan['visible_device_count'] == 4
                and [physical[m.identity(x['uuid'])] for x in plan['replicas'][0]['devices']] == expected,
                role + ' 实际规划或物理 GPU 不符')
        save(role + '.visibility.json', dict(order=order, uuids=envs[role]['CUDA_VISIBLE_DEVICES']))

    command = specs['router']['command']
    require(command.count('--kv-mode') == 1 and command[command.index('--kv-mode') + 1] == 'push',
            '8000 当前不是上轮 push Router')
    commands['router'] = changed(command, {'--kv-mode': 'pull'})
    args = args_parser().parse_args(commands['router'][commands['router'].index('fastllm_pytools.cli') + 1:])
    require(args.command == 'router' and args.port == 8000 and args.prefill == [m.P]
            and args.decode == [m.D] and not args.worker and not args.pd_return_kv
            and args.pd_model_id == model_id, 'Router 参数不符')
    envs['router'] = dict(specs['router']['env'], FT_PD_TOKEN=token, PYTHONUNBUFFERED='1',
                         FT_PD_CORE_LIBRARY=os.environ['FT_PD_CORE_LIBRARY'])
    unchanged(original, snapshot())
    print('复用 Attention 离线基线；切换 P=GPU3 TP1、D=GPU0/2 Attention DP2/EP2。', flush=True)
    m.stop(specs['router'], 8000)
    m.stop(specs['p'], 8100, specs['p']['workers'])
    m.stop(specs['d'], 8200, specs['d']['workers'])
    for name in ('p', 'd', 'router'):
        m.start(name, commands[name], envs[name])
    print('P、D 加载中，8000 为 pull；GPU1 的旧完整服务保留，本轮不调用。', flush=True)

    deadline, last = time.monotonic() + 2400, 0
    while True:
        require(not any((OUT / (name + '.exit_code')).exists() for name in ('p', 'd', 'router')),
                '新服务已退出，查看对应 stderr/native 日志')
        try:
            before = snapshot()
            break
        except Exception as error:
            (OUT / 'waiting.txt').write_text(str(error))
            require(time.monotonic() < deadline, '服务就绪超时：' + str(error))
            if time.monotonic() - last >= 30:
                print('等待就绪：' + str(error), flush=True)
                last = time.monotonic()
            time.sleep(3)
    save('deployment.before.json', before)
    unchanged(original, before, ('full',))
    for role, port, devices in [('p', 8100, [3]), ('d', 8200, [0, 2])]:
        pid = m.listener(port)
        require(pid == int((OUT / (role + '.pid')).read_text()) and m.process(pid)[0] == commands[role],
                role + ' 新监听进程不符')
        specs[role] = dict(pid=pid, mark=m.stamp(pid))
        require(specs[role]['mark'] is not None, role + ' 已退出')
        value = before[role]['evidence'][urls[role]]
        m.mapping(envs[role], value, physical, devices, role + '.new')
        require(not value['requests'], role + ' 存在外部请求，未提交测试')
    m.layout(before['p']['evidence'][m.P], 'p', 1)
    attention(before['d']['evidence'][m.D], 'd')
    require(before['d']['config'] == archive['config'], 'D 与 Attention 基线配置不符')
    require(hashlib.sha256(BASE.read_bytes()).hexdigest() == hashlib.sha256(raw).hexdigest(), '基线文件发生变化')

    launch = json.loads(subprocess.check_output([
        sys.executable, 'test/distributed/pd/launch.py', 'generation', '--',
        '--router', m.ROUTER, '--reference-record', str(BASE),
        '--profile', 'pull_p_tp1_d_attention_dp2_ep2_graph0_cache0',
        '--prompt-token-ids', str(OUT / 'prompt_token_ids.json')], cwd=ROOT, text=True))
    save('generation.launch.json', launch)
    generation = Path(launch['directory'])
    print('生成对照目录：', generation, flush=True)
    deadline = time.monotonic() + 7200
    while not (generation / 'exit_code').exists():
        require(time.monotonic() < deadline, '生成测试超时；勿重复提交')
        time.sleep(3)
    code = int((generation / 'exit_code').read_text())
    result = dict(generation_exit_code=code)
    save('runtime_check.json', result)
    save('services.after_generation.json', {r: m.get(url) for r, url in urls.items()})
    summary = json.loads((generation / 'summary.json').read_text())
    require(code == 0 and summary['status'] == 'PASS' and len(summary['cases']) == 15
            and all(c['status'] == 'PASS' for c in summary['cases']), '15 个 exact 用例未全部通过')

    deadline = time.monotonic() + 120
    while True:
        after, records = snapshot(), {}
        save('deployment.after.json', after)
        for role in ('p', 'd'):
            url = urls[role]
            keys = {(r['request'], r['transfer']) for r in before[role]['evidence'][url]['requests']}
            fresh = [r for r in after[role]['evidence'][url]['requests'] if (r['request'], r['transfer']) not in keys]
            records[role] = {(r['request'], r['transfer']): r for r in fresh}
            require(len(records[role]) == len(fresh) and not any(r['error'] for r in fresh), role + ' 请求重复或失败')
        if all(len(v) == 21 and all(r['released'] and r['prefill_done' if role == 'p' else 'completed']
                                   for r in v.values()) for role, v in records.items()):
            break
        require(time.monotonic() < deadline, '本轮请求未齐、未完成或未释放')
        time.sleep(2)
    unchanged(before, after)
    m.layout(after['p']['evidence'][m.P], 'p', 1)
    attention(after['d']['evidence'][m.D], 'd')
    require(records['p'].keys() == records['d'].keys()
            and all(r['pair'] == records['d'][k]['pair'] for k, r in records['p'].items()), '两端配对不一致')
    counts = [0, 0]
    for request in records['d'].values():
        pair = request['pair']
        selected = pair['target']['index']
        require(selected in (0, 1) and pair['source']['index'] == 0, '源/目标副本索引错误')
        for side, device, storage in [('source', 3, 'paged'), ('target', [0, 2][selected], 'continuous')]:
            replica = pair[side]
            require(replica['layout']['tp_size'] == 1 and replica['layout']['storage'] == storage
                    and len(replica['ranks']) == 1, '配对 TP 或分页到连续布局错误')
            rank = replica['ranks'][0]
            require(rank['rank'] == 0 and m.identity(rank['device']) == m.identity(uuids[device])
                    and rank['head_begin'] == 0 and rank['head_end'] == heads, '配对 GPU/head 覆盖错误')
        counts[selected] += 1
    require(all(counts), 'D 未覆盖两个 Attention 副本')
    a = before['p']['evidence'][m.P]['replicas'][0]['states'][0]
    b = after['p']['evidence'][m.P]['replicas'][0]['states'][0]
    nccl = b['allreduce']['nccl_calls'] - a['allreduce']['nccl_calls']
    eager = b['eager_decodes'] - a['eager_decodes']
    require(nccl > 0 and eager == 0, 'P Prefill 实际执行核查失败')
    result['p'] = dict(requests=len(records['p']), nccl_calls_increase=nccl, eager_decode_increase=eager)
    eager = [b['states'][0]['eager_decodes'] - a['states'][0]['eager_decodes'] for a, b in
             zip(before['d']['evidence'][m.D]['replicas'], after['d']['evidence'][m.D]['replicas'])]
    require(all(x > 0 for x in eager), 'D 两个 Attention rank 未实际 Decode')
    result['d'] = dict(requests_by_replica=counts, eager_decode_increase=eager)
    a = before['d']['evidence'][m.D]['health']['transfer_statistics']
    b = after['d']['evidence'][m.D]['health']['transfer_statistics']
    transfer = {k: b[k] - a[k] for k in ('completed_batches', 'copied_bytes')}
    require(min(transfer.values()) > 0, 'D 没有本轮实际拉取增量')
    result.update(transfer=dict(side='d', **transfer), reference_matches_d=True,
                  full_unchanged=True, kv_layout='paged_to_continuous', passed=True)
    save('runtime_check.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return dict(generation_exit_code=code, passed=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--supervise']:
        name, code = sys.argv[2], 1
        try:
            child = subprocess.Popen(json.loads((OUT / (name + '.command.json')).read_text()), cwd=ROOT)
            (OUT / (name + '.pid')).write_text(str(child.pid) + '\n')
            code = child.wait()
        finally:
            (OUT / (name + '.exit_code')).write_text(str(code) + '\n')
        raise SystemExit(code)
    with (ROOT / 'test/distributed/results/.pd_tp2_tp1_pull.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'test.pid').write_text(str(os.getpid()) + '\n')
        raise SystemExit(run([('tp1_ep2_pull_runtime', check)], OUT,
            backend='C++ ordinary TP and Attention DP/EP + CUDA Eager/IPC; Python orchestration',
            metadata_extra=dict(gpu='gpu.csv', architecture='CMakeCache.txt', cuda='deployment.before.json'),
            scope='第 7 步 PD-V62：普通 TP1 → Attention DP2/EP2 pull、Eager/Prefix 关闭；双目标副本、分页到连续 KV、释放及 D 拉取；15 个 exact 用例见独立生成报告'))
PY_RUN

cat > "$pd_next/command.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
trap 'code=$?; echo "$code" > exit_code' EXIT
python3 run.py
SH
nohup bash "$pd_next/command.sh" > "$pd_next/stdout.log" 2> "$pd_next/stderr.log" < /dev/null &
echo "$!" > "$pd_next/supervisor.pid"
echo "本次目录：$pd_next"
)
