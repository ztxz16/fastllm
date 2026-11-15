"""Real-model V4.1 concurrency regression, with sampled DSpark and churn.

Pass the ordinary ftllm model arguments, --max_batch 4, --dspark 5 and
--test-tokenizer PATH/tokenizer.json. Requires an instruction-tuned checkpoint.
The caller may also use run_regression() on an already loaded model.
"""
import ctypes
import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def run_regression(model, tokenizer, output):
    from ftllm import llm
    assert model.get_max_batch() == 4
    lib = llm.fastllm_lib
    stop_count, stops = model.stop_token_ctypes(None)

    def encode(question):
        return tokenizer.encode('<｜begin▁of▁sentence｜><｜User｜>' + question +
                                '<｜Assistant｜></think>', add_special_tokens=False).ids

    cases = []
    for index, (question, answer) in enumerate([
            ('19*17', '323'), ('144/12', '12'), ('37+58', '95'),
            ('17 squared', '289'), ('123+456', '579'), ('81-37', '44')]):
        tag = 'R' + str(51 + index)
        cases.append(dict(name=tag, tokens=encode(
            'Compute ' + question + '. Reply exactly as ' + tag +
            '=NUMBER, replacing NUMBER with the result. No other text.'),
            limit=32, expected=tag + '=' + answer,
            # Mix supported DSpark with ordinary decoding in the same queue.
            repeat_penalty=1.05 if index % 3 == 2 else 1.))
    for start, end in [(1, 30), (41, 70)]:
        cases.insert(1, dict(name='sequence_' + str(start), tokens=encode(
            f'Reply exactly with integers from {start} to {end}, separated by commas. No other text.'),
            limit=128, expected=list(range(start, end + 1))))
    for index, secret in enumerate(['copper-river-7319', 'silver-lantern-8462']):
        filler = ''.join(f'Record {j}: This is an unrelated inventory entry about books and desks.\n'
                         for j in range(180 + index * 3))
        cases.append(dict(name='secret_' + str(index), tokens=encode(
            'Remember this exact secret: ' + secret + '.\n' + filler +
            '\nReply with only the exact secret from the beginning.'), limit=32, expected=secret))
    for limit in [1, 2, 3, 5, 6, 7, 11, 17]:
        cases.append(dict(name='limit_' + str(limit), limit=limit, exact_length=True,
            tokens=encode('List all integers from 1 to 1000, separated by commas. No other text.')))

    result = dict(status='running', waves=[], max_batch=4, do_sample=True,
                  temperature=.8, top_p=.95, top_k=50)

    def save():
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2))

    def wave(concurrency):
        pending = list(cases)
        live, futures, finished, events = {}, {}, [], []
        started = time.monotonic()
        result['current'] = dict(concurrency=concurrency, events=events, cases=finished)
        save()

        def fetch(handle):
            buffer, stats = (ctypes.c_int * 1)(), (ctypes.c_int * 3)()
            count = lib.fetch_response_tokens_batch_with_statistics_llm_model(
                model.model, handle, buffer, 1, stats)
            return count, buffer[0], list(stats)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while pending or live:
                assert time.monotonic() - started < 600, 'concurrent generation timed out'
                while pending and len(live) < concurrency:
                    case = pending.pop(0)
                    ids = case['tokens']
                    handle = lib.launch_response_llm_model(model.model, len(ids),
                        (ctypes.c_int * len(ids))(*ids), case['limit'], 0, True,
                        .95, 50, .8, case.get('repeat_penalty', 1.), False, stop_count, stops)
                    assert handle >= 0 and handle not in live
                    live[handle] = dict(case=case, tokens=[])
                    futures[handle] = pool.submit(fetch, handle)
                    events.append(dict(event='launch', name=case['name'], handle=handle,
                                       live=len(live), seconds=time.monotonic()-started))
                    save()
                ready, _ = wait(futures.values(), timeout=1, return_when=FIRST_COMPLETED)
                for handle in list(live):
                    if futures[handle] not in ready:
                        continue
                    count, token, stats = futures.pop(handle).result()
                    row = live[handle]
                    if count > 0:
                        row['tokens'].append(token)
                    elif count < 0:
                        assert count == -1, (handle, count)
                        case = row['case']
                        text = tokenizer.decode(row['tokens']).strip()
                        assert 0 < len(row['tokens']) <= case['limit'], case['name']
                        assert stats[2] == len(row['tokens']), (case['name'], stats)
                        if 'expected' in case:
                            got = [int(n) for n in re.findall(r'\d+', text)] if isinstance(case['expected'], list) else text
                            assert got == case['expected'], (case['name'], got, case['expected'])
                        if case.get('exact_length'):
                            assert len(row['tokens']) == case['limit'], (case['name'], text)
                        finished.append(dict(name=case['name'], handle=handle,
                            input_tokens=len(case['tokens']), output_tokens=len(row['tokens']),
                            text=text, repeat_penalty=case.get('repeat_penalty', 1.), passed=True))
                        del live[handle]
                        events.append(dict(event='finish', name=case['name'], handle=handle,
                                           live=len(live), seconds=time.monotonic()-started))
                        save()
                    if handle in live:
                        futures[handle] = pool.submit(fetch, handle)
        result['waves'].append(dict(concurrency=concurrency, cases=finished, events=events))
        save()

    try:
        # Preserve requests with unequal context/proposal/output lengths while
        # also exercising graph-size fallback and allocator/handle reuse.
        for concurrency in [1, 2, 4, 3, 1]:
            wave(concurrency)
        result.pop('current', None)
        result.update(status='passed', checks=sum(len(w['cases']) for w in result['waves']))
        save()
        return result
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        save()
        raise


def main():
    from ftllm.util import make_normal_parser, make_normal_llm_model
    from tokenizers import Tokenizer
    parser = make_normal_parser(__doc__)
    parser.add_argument('--test-tokenizer', required=True)
    parser.add_argument('--test-output', default='concurrency-results.json')
    args = parser.parse_args()
    assert args.max_batch == 4 and args.dspark > 0
    model = make_normal_llm_model(args)
    result = run_regression(model, Tokenizer.from_file(args.test_tokenizer), args.test_output)
    print('PASS:', result['checks'], 'sampled concurrent request checks', flush=True)
    # The native serving loop remains live after all response handles finish.
    os._exit(0)


if __name__ == '__main__':
    main()
