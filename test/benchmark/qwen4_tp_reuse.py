#!/usr/bin/env python3
"""Real-checkpoint TP cache isolation across reuse, growth and cancellation.

Run against a Qwen4 TP server with CUDA Graph enabled; no debug flags required.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18080')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--long-prompt', type=Path,
        help='Optional 128K prompt file for repeated-prefill memory regression')
    parser.add_argument('--long-repeats', type=int, default=6)
    args = parser.parse_args()
    if args.long_prompt and args.long_repeats < 3:
        parser.error('--long-repeats must be at least 3')
    with urllib.request.urlopen(args.url + '/v1/models', timeout=10) as response:
        model = json.load(response)['data'][0]['id']
    prompts = {
        'a': ('Ordinary notes about software testing and parallel computing. ' * 400)
            + '\nThe verification code is AZURE-7391. Explain what this code is used for.',
        'b': ('A travel diary describes rivers, mountains, villages and railway stations. ' * 1000)
            + '\nThe verification code is CORAL-2856. Explain what this code is used for.',
    }
    if args.long_prompt:
        prompts['long'] = args.long_prompt.read_text()

    def request(name, cancel=False):
        output_tokens = 256 if name == 'long' else 64
        payload = dict(model=model, messages=[dict(role='user', content=prompts[name])],
            temperature=0, max_tokens=output_tokens, min_tokens=output_tokens, ignore_eos=True,
            chat_template_kwargs={'enable_thinking': False}, stream=True,
            stream_options={'include_usage': True})
        req = urllib.request.Request(args.url + '/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        chunks, usage = [], None
        with urllib.request.urlopen(req, timeout=600) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                raw = line[6:].strip()
                if raw == b'[DONE]':
                    break
                event = json.loads(raw)
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    text = (delta.get('content') or '') + (delta.get('reasoning_content') or '')
                    if text:
                        chunks.append(text)
                if cancel and len(chunks) >= 3:
                    return dict(case=name, cancelled_after_chunks=len(chunks))
        assert not cancel, 'Stream ended before cancellation could be tested'
        assert usage and usage['completion_tokens'] == output_tokens, usage
        assert usage['prompt_tokens'] >= 2048, usage
        if name == 'long':
            assert usage['prompt_tokens'] >= 120000, 'Long prompt must exercise the 128K capacity class'
        return dict(case=name, output=''.join(chunks), usage=usage)

    records = []
    def save(record):
        records.append(record)
        args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2))

    reference_a = request('a')
    save(dict(phase='reference', **reference_a))
    save(dict(phase='cancel', **request('b', cancel=True)))
    time.sleep(1)  # Allow the disconnected stream's request cleanup to finish.
    reference_b = request('b')
    save(dict(phase='reference', **reference_b))
    references = {'a': reference_a, 'b': reference_b}

    def check(phase, row):
        reference = references[row['case']]
        row['matches_reference'] = (row['output'] == reference['output']
                                    and row['usage'] == reference['usage'])
        save(dict(phase=phase, **row))
        assert row['matches_reference'], row

    for name in ['a', 'b', 'a']:
        check('reuse', request(name))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(request, ['a', 'b']):
            check('concurrent', row)
    if args.long_prompt:
        references['long'] = request('long')
        save(dict(phase='long_reference', **references['long']))
        for _ in range(args.long_repeats - 1):
            check('long_repeat', request('long'))
        # Exercise eviction of the large idle slot, then admission of another
        # long request into a pool that has served different shapes and graphs.
        save(dict(phase='after_long_cancel', **request('b', cancel=True)))
        time.sleep(1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for row in pool.map(request, ['a', 'b']):
                check('after_long_concurrent', row)
        for _ in range(2):
            check('long_readmit', request('long'))
    print('PASS: request reuse, different contents/lengths, cancellation and concurrency')


if __name__ == '__main__':
    main()
