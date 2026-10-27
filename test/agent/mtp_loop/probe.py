"""Replay thinking-loop scenarios against an existing OpenAI-compatible API."""
import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from analyze import analyze_text, load_tokenizer

HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sse_data(lines):
    """Join SSE data fields and ignore comments; transport chunk boundaries do not matter."""
    fields = []
    for raw in lines:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if fields:
                yield '\n'.join(fields)
                fields = []
        elif line.startswith('data:'):
            value = line[5:]
            fields.append(value[1:] if value.startswith(' ') else value)
    if fields:
        yield '\n'.join(fields)


def make_payload(case, tools, *, model, temperature, top_k, top_p, max_tokens, effort=None):
    payload = dict(model=model, messages=copy.deepcopy(case['messages']), tools=copy.deepcopy(tools),
                   tool_choice='none', temperature=temperature, top_k=top_k, top_p=top_p,
                   max_tokens=max_tokens, stream=True,
                   chat_template_kwargs={'enable_thinking': True})
    if effort:
        payload['reasoning_effort'] = effort
    return payload


def probe(url, payload, out, *, tokenizer=None, api_key=None, timeout=120, deadline=600):
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'request.json', payload)
    start = time.monotonic()
    done, finish, error, usage = False, None, None, {}
    headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = urllib.request.Request(url, json.dumps(payload).encode('utf-8'), headers)
    calls = []
    with (out / 'events.jsonl').open('w', encoding='utf-8') as events, \
            (out / 'reasoning.txt').open('w', encoding='utf-8') as reasoning, \
            (out / 'content.txt').open('w', encoding='utf-8') as content:
        try:
            with urllib.request.urlopen(request, timeout=min(timeout, deadline)) as response:
                def lines():
                    for line in response:
                        if time.monotonic() - start > deadline:
                            raise TimeoutError('Request exceeded total deadline')
                        yield line
                for data in sse_data(lines()):
                    if data == '[DONE]':
                        done = True
                        break
                    event = json.loads(data)
                    events.write(json.dumps(dict(elapsed_s=time.monotonic() - start, data=event),
                                            ensure_ascii=False) + '\n')
                    events.flush()
                    if event.get('error'):
                        raise RuntimeError(event['error'])
                    usage = event.get('usage') or usage
                    for choice in event.get('choices', []):
                        delta = choice.get('delta') or {}
                        reasoning.write(delta.get('reasoning_content') or delta.get('reasoning') or '')
                        content.write(delta.get('content') or '')
                        calls.extend(delta.get('tool_calls') or [])
                        finish = choice.get('finish_reason') or finish
                    reasoning.flush()
                    content.flush()
            if not done or finish is None:
                raise RuntimeError('Incomplete SSE response: missing [DONE] or finish_reason')
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            error = str(exc)
    result = dict(done=done, finish=finish, error=error, usage=usage,
                  elapsed_s=time.monotonic() - start,
                  reasoning=analyze_text((out / 'reasoning.txt').read_text(encoding='utf-8'), tokenizer),
                  content=analyze_text((out / 'content.txt').read_text(encoding='utf-8'), tokenizer))
    save(out / 'tool_calls.json', calls)
    save(out / 'result.json', result)
    return result


def main():
    data = json.loads((HERE / 'cases.json').read_text(encoding='utf-8'))
    cases = {c['id']: c for c in data['cases']}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1'))
    parser.add_argument('--model', required=True, help='Model ID served by the API')
    parser.add_argument('--cases', choices=list(cases), nargs='+', default=list(cases))
    parser.add_argument('--temperature', type=float, nargs='+', default=[0.6])
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--reasoning-effort', choices=['low', 'medium', 'xhigh'])
    parser.add_argument('--tokenizer', type=Path, help='Optional local tokenizer.json or model directory')
    parser.add_argument('--out', type=Path, help='New output directory; existing directories are not overwritten')
    parser.add_argument('--timeout', type=float, default=120, help='Socket timeout in seconds')
    parser.add_argument('--deadline', type=float, default=600, help='Total request deadline in seconds')
    parser.add_argument('--fail-on-cycle', action='store_true', help='Exit 2 on a detected exact cycle')
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1 or args.top_k < 1:
        parser.error('repeats, max-tokens and top-k must be positive')
    if not 0 < args.top_p <= 1 or any(not math.isfinite(t) or t < 0 for t in args.temperature):
        parser.error('top-p must be in (0, 1]; temperature must be finite and nonnegative')
    if any(not math.isfinite(x) or x <= 0 for x in [args.timeout, args.deadline]):
        parser.error('timeout and deadline must be positive and finite')
    tokenizer = load_tokenizer(args.tokenizer)
    base = args.base_url.rstrip('/')
    url = base + ('/chat/completions' if base.endswith('/v1') else '/v1/chat/completions')
    out = args.out or HERE / 'results' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'metadata.json', dict(endpoint=url, model=args.model, tokenizer=str(args.tokenizer) if args.tokenizer else None,
                                     analysis_unit='tokens' if tokenizer else 'characters'))
    rows = []
    for case_id in dict.fromkeys(args.cases):
        for temperature in dict.fromkeys(args.temperature):
            for repeat in range(args.repeats):
                label = f'{case_id}_t{temperature:g}_r{repeat + 1}'
                payload = make_payload(cases[case_id], data['tools'], model=args.model, temperature=temperature,
                                       top_k=args.top_k, top_p=args.top_p, max_tokens=args.max_tokens,
                                       effort=args.reasoning_effort)
                print(f'BEGIN {label}', flush=True)
                row = probe(url, payload, out / label, tokenizer=tokenizer,
                            api_key=os.environ.get('OPENAI_API_KEY'), timeout=args.timeout, deadline=args.deadline)
                rows.append(dict(label=label, **row))
                save(out / 'results.json', rows)
                print(json.dumps(dict(label=label, finish=row['finish'], error=row['error'],
                                      cycle=bool(row['reasoning']['cycle'] or row['content']['cycle'])), ensure_ascii=False), flush=True)
    failed = sum(r['error'] is not None for r in rows)
    cycles = sum(bool(r['reasoning']['cycle'] or r['content']['cycle']) for r in rows)
    save(out / 'summary.json', dict(requests=len(rows), errors=failed, exact_cycles=cycles,
                                    length_limits=sum(r['finish'] == 'length' for r in rows)))
    print(f'Results: {out}', flush=True)
    return 1 if failed else 2 if args.fail_on_cycle and cycles else 0


if __name__ == '__main__':
    raise SystemExit(main())
