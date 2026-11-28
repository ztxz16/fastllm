(
set -euo pipefail
cd /home/project/vllm_fastllm
umask 077
pd_diag="$PWD/test/distributed/results/$(date -u +%Y%m%dT%H%M%SZ)_pd_tp1_ep2_exact_diag_$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
mkdir "$pd_diag"
cat > "$pd_diag/collect.py" <<'PY'
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/home/project/vllm_fastllm/test/distributed/results')
SOURCE = ROOT / '20261128T110052Z_pd_tp1_ep2_pull_5ac24694'
GEN = ROOT / '20261128T110401Z_pd_generation_d98fff95'
OUT = Path(__file__).resolve().parent
(OUT / 'pid').write_text(str(os.getpid()) + '\n')

def read(path):
    return json.loads(path.read_text())

def copy_file(path, name):
    raw = path.read_bytes()
    (OUT / name).write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())

def first_difference(a, b):
    return next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]),
                min(len(a), len(b)) if len(a) != len(b) else None)

result = dict(source_run=str(SOURCE), generation_run=str(GEN),
              scope='仅提取已保存的失败证据；不运行推理、不更改原验收结论', files={})
for label, directory in [('outer', SOURCE), ('generation', GEN)]:
    result[label + '_exit_code'] = (directory / 'exit_code').read_text().strip()
    for name in ('summary.json', 'summary.md', 'runtime_check.json', 'deployment.before.json',
                 'services.after_generation.json', 'reference.source.json', 'prompt_token_ids.json'):
        path = directory / name
        if path.is_file():
            target = label + '.' + name
            result['files'][target] = copy_file(path, target)

summary = read(GEN / 'summary.json')
result['failed_cases'] = [dict(case=c['case'], reason=c['reason'])
                          for c in summary['cases'] if c['status'] != 'PASS']
result['metadata'] = summary.get('metadata', {})
after = read(SOURCE / 'services.after_generation.json')
before = read(SOURCE / 'deployment.before.json')
result['services'] = {}
for role, value in after.items():
    requests = value.get('requests', [])
    info = dict(health=value['health'], saved_requests=len(requests),
                released=sum(bool(r.get('released')) for r in requests),
                errors=[r for r in requests if r.get('error')])
    url = next(iter(before[role]['evidence']))
    old = before[role]['evidence'][url]['health']['transfer_statistics']
    new = value['health']['transfer_statistics']
    info['transfer_increase'] = {k: new[k] - old[k] for k in ('completed_batches', 'copied_bytes')}
    result['services'][role] = info

profile = summary['metadata']['profile']
names = [profile + '_sampling_fixed_seed'] + [profile + f'_concurrent4_request{i}' for i in range(4)]
result['requests'] = []
for name in names:
    path = GEN / (name + '.evidence.json')
    result['files'][path.name] = copy_file(path, path.name)
    evidence = read(path)
    payload = evidence['request']
    actual, reference = evidence['actual'], evidence['reference']
    a, b = actual['token_ids'], reference['token_ids']
    index = first_difference(b, a)
    messages = [actual.get('response', {})] + actual.get('deltas', [])
    ids = sorted({m['id'] for m in messages if m.get('id')})
    matches = {role: [r for r in value.get('requests', []) if r['request'] in ids]
               for role, value in after.items() if role in ('p', 'd')}
    item = dict(case=name, prompt_tokens=len(payload['token_ids']),
                prompt_sha256=hashlib.sha256(json.dumps(payload['token_ids']).encode()).hexdigest(),
                parameters={k: v for k, v in payload.items() if k != 'token_ids'},
                reference_token_ids=b, actual_token_ids=a, exact=a == b,
                first_divergence_zero_based=index,
                first_divergence_one_based=None if index is None else index + 1,
                saved_first_divergence=evidence.get('first_divergence'),
                request_ids=ids, service_records=matches,
                candidate_comparison=evidence.get('candidate_comparison'))
    if not ids:
        item['association_note'] = '非流式响应未提供请求 ID，不能按记录顺序猜测对应副本'
    result['requests'].append(item)

ref_source = GEN / 'reference.source.json'
if ref_source.is_file():
    archive = read(ref_source)
    path = Path(archive['path'])
    if path.is_file():
        result['reference_sha256_matches'] = hashlib.sha256(path.read_bytes()).hexdigest() == archive['sha256']
        report = read(path.with_name('summary.json'))
        result['reference_metadata'] = report.get('metadata', {})
    else:
        result['reference_note'] = '原基线文件缺失；每个请求保存的 reference 仍已提取'

(OUT / 'diagnostic.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print('诊断目录：', OUT)
print('原生成退出码：', result['generation_exit_code'], '；原总退出码：', result['outer_exit_code'])
for item in result['requests']:
    display = dict(item)
    display['service_records'] = {role: [dict(
        request=r['request'], transfer=r['transfer'],
        completed=r.get('completed'), released=r.get('released'), error=r.get('error'),
        source_replica=(r.get('pair') or {}).get('source', {}).get('index'),
        target_replica=(r.get('pair') or {}).get('target', {}).get('index')) for r in records]
        for role, records in item['service_records'].items()}
    print('\n' + json.dumps(display, ensure_ascii=False, indent=2))
print('\n服务快照统计（不是独立运行验收）：')
print(json.dumps(result['services'], ensure_ascii=False, indent=2))
for label in ('metadata', 'reference_metadata'):
    meta = result.get(label, {})
    print(label, json.dumps({k: meta.get(k) for k in ('git', 'profile', 'native_core')}, ensure_ascii=False))
print('证据已提取，原 FAIL 保留；完整记录：diagnostic.json')
PY
cat > "$pd_diag/command.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
trap 'code=$?; echo "$code" > exit_code' EXIT
python3 collect.py
SH
bash "$pd_diag/command.sh" 2>&1 | tee "$pd_diag/collect.log"
)
