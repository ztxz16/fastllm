(
set -euo pipefail
cd /home/project/vllm_fastllm
umask 077
pd_read="$PWD/test/distributed/results/$(date -u +%Y%m%dT%H%M%SZ)_pd_v62_status_text_$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
mkdir "$pd_read"
cat > "$pd_read/read.py" <<'PY'
import json
import os
from pathlib import Path
import socket

ROOT = Path('/home/project/vllm_fastllm/test/distributed/results')
SOURCE = ROOT / '20261128T110052Z_pd_tp1_ep2_pull_5ac24694'
GEN = ROOT / '20261128T110401Z_pd_generation_d98fff95'
OUT = Path(__file__).resolve().parent
result = dict(host=socket.gethostname(), source=str(SOURCE), generation=str(GEN),
              scope='只读进程及历史文件；不发推理请求，不改变 exact 结论', errors=[])

for label, directory in [('outer', SOURCE), ('generation', GEN)]:
    try:
        summary = json.loads((directory / 'summary.json').read_text())
        result[label] = dict(exit_code=(directory / 'exit_code').read_text().strip(),
            status=summary.get('status'), cases=[{k: c.get(k) for k in ('case', 'status', 'reason')}
                                              for c in summary['cases']])
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['errors'].append(f'{label}: {type(error).__name__}: {error}')

result['processes'] = []
for proc in sorted(Path('/proc').glob('[0-9]*'), key=lambda p: int(p.name)):
    if int(proc.name) == os.getpid():
        continue
    try:
        args = proc.joinpath('cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
        if not any('fastllm' in a or '/test/distributed/' in a for a in args):
            continue
        stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        result['processes'].append(dict(pid=int(proc.name), state=stat[0], start_ticks=int(stat[19]),
            cwd=os.readlink(proc / 'cwd'), executable=Path(args[0]).name,
            scripts=[a for a in args[1:] if '/' in a and a.endswith(('.py', '.sh'))
                     and not any(c.isspace() for c in a)]))
    except (FileNotFoundError, ProcessLookupError):
        continue
    except (OSError, ValueError, IndexError) as error:
        result['errors'].append(f'process {proc.name}: {type(error).__name__}: {error}')

result['recent_runs'] = [p.name for p in sorted(ROOT.iterdir())
    if p.is_dir() and p.name >= SOURCE.name and '_pd_' in p.name and p != OUT]
result['recorded_pids'] = {}
for directory in (SOURCE, GEN):
    for path in sorted(directory.glob('*.pid')):
        result['recorded_pids'][str(path)] = path.read_text().strip()
result['outputs'] = []
for suffix in ('sampling_fixed_seed', 'concurrent4_request3'):
    try:
        paths = list(GEN.glob('*_' + suffix + '.evidence.json'))
        if len(paths) != 1:
            raise ValueError(f'expected one evidence file, found {len(paths)}')
        evidence = json.loads(paths[0].read_text())
        item = dict(case=suffix, evidence=str(paths[0]), first_divergence=evidence.get('first_divergence'))
        for side in ('reference', 'actual'):
            value = evidence[side]
            chunks = [value['response']] if 'response' in value else value['deltas']
            texts = []
            for chunk in chunks:
                for choice in chunk.get('choices', []):
                    if choice.get('index', 0) != 0:
                        continue
                    text = choice.get('text')
                    if text is None:
                        text = choice.get('delta', choice.get('message', {})).get('content')
                    if isinstance(text, str):
                        texts.append(text)
            if not texts:
                raise ValueError(side + ' has no saved response text')
            item[side] = dict(token_ids=value['token_ids'], text=''.join(texts))
        result['outputs'].append(item)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['errors'].append(f'{suffix}: {type(error).__name__}: {error}')

(OUT / 'snapshot.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print('只读核查目录：', OUT)
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(1 if result['errors'] else 0)
PY
cat > "$pd_read/command.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
trap 'code=$?; echo "$code" > exit_code' EXIT
echo "$$" > pid
python3 read.py
SH
pd_read_code=0
bash "$pd_read/command.sh" > "$pd_read/stdout.log" 2> "$pd_read/stderr.log" || pd_read_code=$?
cat "$pd_read/stdout.log"
cat "$pd_read/stderr.log" >&2
exit "$pd_read_code"
)
