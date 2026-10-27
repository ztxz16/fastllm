"""Real-checkpoint regression for Qwen4 hybrid PP prefill with MTP.

Use two CUDA PP stages, CPU embedding and full device KV, for example:
  python test_qwen4_pp_mtp_prefill.py --report results.json MODEL \
    --device cudapp=2 --moe_device numa --mtp 3 --low_gpu_mem \
    --speculative_draft_model_path DRAFT --chunked_prefill_size 512

The history flag selects the existing serial fallback; prefix caching stays
off for those references. A deterministic code continuation compares the same
MTP mode across serial/PP execution, including ragged chunks and slot reuse.
KV streaming is tested separately because it cannot enable history caching.
Run both with all target experts on NUMA and with partial GPU residency. If a
resident expert belongs to another PP stage, prefill must use the serial
fallback; the same token, prefix-restore and request-reuse checks still apply.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--report', type=Path, required=True)
    test, argv = parser.parse_known_args()
    from ftllm import llm
    from ftllm.util import make_normal_parser, make_normal_llm_model
    args = make_normal_parser(__doc__).parse_args(argv)
    assert args.mtp >= 3 and args.low_gpu_mem, 'Requires MTP >= 3 and CPU embedding'
    assert not args.kvmem, 'Use full device KV for the serial history fallback'
    keys = ['FASTLLM_QWEN4_ENABLE_MTP', 'FASTLLM_QWEN4_MTP_MIN_P',
            'FASTLLM_PREFIX_CACHE', 'FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES']
    saved = {key: os.environ.get(key) for key in keys}
    os.environ['FASTLLM_PREFIX_CACHE'] = '0'
    os.environ['FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES'] = '1'
    model = make_normal_llm_model(args)
    model.enable_thinking = False
    model.set_verbose(True)
    lib = llm.fastllm_lib
    records = []
    terminal = model.encode(model.get_prompt(
        'Continue the Python list through 100. Output code only, one tuple per line.') +
        '```python\nitems = [\n' + ''.join(f'    ({i}, {i}),\n' for i in range(1, 17)))
    padding = model.encode('# Context: continue the numbered table.\n' * 1000)

    def request(name, length, serial, sampling=False, ids=None):
        model.set_save_history(serial)
        if ids is None:
            ids = padding[:length-len(terminal)] + terminal
        assert len(ids) == length and length > len(terminal)
        stops, stop_ids = model.stop_token_ctypes(None)
        handle = lib.launch_response_llm_model(
            model.model, length, (ctypes.c_int * length)(*ids), 64, 64,
            sampling, .9 if sampling else 1., 20 if sampling else 1,
            .95 if sampling else 1., 1., False, stops, stop_ids)
        assert handle >= 0
        output, times = [], []
        started = time.perf_counter()
        try:
            while True:
                if time.perf_counter() - started > 180:
                    raise TimeoutError(name)
                if not lib.can_fetch_response_llm_model(model.model, handle):
                    time.sleep(.0002)
                    continue
                token = lib.fetch_response_llm_model(model.model, handle)
                if token < 0:
                    assert token == -1, (name, token)
                    break
                output.append(int(token))
                times.append(time.perf_counter() - started)
                if len(output) == 64:
                    statistics = model.get_response_statistics(handle)
            assert len(output) == 64, (name, len(output))
        except BaseException:
            lib.abort_response_llm_model(model.model, handle)
            raise
        row = dict(name=name, length=length, serial=serial, output=output,
                   first_token_s=times[0], seconds=times[-1], statistics=statistics)
        records.append(row)
        test.report.parent.mkdir(parents=True, exist_ok=True)
        test.report.write_text(json.dumps(records, indent=2) + '\n')
        print('PREFILL_CHECK', name, row['first_token_s'], statistics, flush=True)
        return row

    try:
        model.set_chunked_prefill_size(512)
        for length, mtp, threshold in [(512, 1, .5), (513, 1, .5), (1025, 1, .5), (1024, 3, .5),
                                        (1025, 3, 0), (1031, 3, .5), (1025, 0, 0),
                                        (4097, 3, 0), (4103, 3, .5)]:
            os.environ['FASTLLM_QWEN4_ENABLE_MTP'] = str(mtp)
            os.environ['FASTLLM_QWEN4_MTP_MIN_P'] = str(threshold)
            name = f'n{length}-mtp{mtp}-p{threshold}'
            reference = request(name + '-serial', length, True)
            for repeat in range(2):
                actual = request(name + f'-pp-{repeat}', length, False)
                assert actual['statistics']['cached_input_tokens'] == 0
                assert actual['output'] == reference['output'], (name, 'serial/PP tokens differ')

        os.environ['FASTLLM_QWEN4_ENABLE_MTP'] = '3'
        os.environ['FASTLLM_QWEN4_MTP_MIN_P'] = '.5'
        prefix = padding[:4096-len(terminal)] + terminal
        extended = prefix + model.encode('    (17, 17),\n')
        reference = request('prefix-reference', len(extended), True, ids=extended)
        os.environ['FASTLLM_PREFIX_CACHE'] = '1'
        request('prefix-seed', len(prefix), False, ids=prefix)
        # A full-prompt snapshot is reusable only when at least one new token
        # follows it; pipelined prefill snapshots after the stages join.
        restored = request('prefix-restore', len(extended), False, ids=extended)
        assert restored['statistics']['cached_input_tokens'] > 0, 'MTP prefix was not restored'
        assert restored['output'] == reference['output'], 'MTP prefix restore changed tokens'
        os.environ['FASTLLM_PREFIX_CACHE'] = '0'
        request('sampling-fallback', 1025, False, sampling=True)
    finally:
        model.release_memory()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print('PASS Qwen4 PP MTP prefill, ragged chunks, slot reuse and prefix restore', flush=True)


if __name__ == '__main__':
    main()
