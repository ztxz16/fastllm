"""Prefill组合诊断的CPU协议与文件回归；不编译或启动CUDA。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import probe_qwen_moe_prefill_batches as probe
from run_qwen_moe_full import request


def write_trace(directory, rank, round_number, changed=False):
    """/** 模拟含非末Token差异的完整FP16轨迹，确保诊断不会仅比较最后一行。 */"""
    values = [1., -2., .5, .25, 2., 3., -1., 0., 4., 5., 6., 7., .125, -.5]
    if changed:
        values[8] += 1
    base = f'rank-{rank}-round-{round_number}'
    binary = directory / (base + '.f16')
    binary.write_bytes(struct.pack('<14e', *values))
    meta = dict(version=1, dtype='float16', byte_order='little', rank=rank, round=round_number,
        max_mib=256, tokens=2, hidden=2, layers=1, ids=[1, 2], binary=binary.name,
        snapshots=[dict(name=name, offset=offset, elements=size) for name, offset, size in (
            ('embedding', 0, 4), ('layer_0_attention', 4, 4), ('layer_0_moe', 8, 4), ('final_norm', 12, 2))])
    path = directory / (base + '.json')
    path.write_text(json.dumps(meta))
    return path, meta


class PrefillBatchTests(unittest.TestCase):
    def test_full_tensor_comparison_sees_difference_outside_last_token(self):
        result = probe.difference([1., 2., 3., 4.], [0., 2., 3., 4.])
        self.assertEqual(result['failed_elements'], 1)
        self.assertEqual(result['first_difference']['element_0based'], 0)
        self.assertEqual(result['max_abs_error'], 1.)
        self.assertEqual(result['mean_abs_error'], .25)
        for actual, expected in (([], []), ([1.], [1., 2.]), ([float('nan')], [1.]), ([1.], [float('inf')])):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                probe.difference(actual, expected)

    def test_trace_loader_rejects_incomplete_or_mismatched_files(self):
        for fault in ('missing', 'round', 'ids', 'offset', 'size', 'escape', 'duplicate', 'extra', 'limit'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                write_trace(directory, 0, 1)
                path, meta = write_trace(directory, 0, 2)
                if fault == 'missing':
                    path.unlink()
                elif fault == 'size':
                    (directory / meta['binary']).write_bytes(b'')
                elif fault == 'extra':
                    (directory / 'uncommitted.f16').write_bytes(b'')
                elif fault == 'duplicate':
                    (directory / 'duplicate.json').write_text(json.dumps(meta))
                else:
                    if fault == 'round':
                        meta['round'] = 3
                    elif fault == 'ids':
                        meta['ids'] = [2, 1]
                    elif fault == 'offset':
                        meta['snapshots'][1]['offset'] = 0
                    elif fault == 'escape':
                        meta['binary'] = '../trace.f16'
                    elif fault == 'limit':
                        meta.pop('max_mib')
                    path.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    probe.load_traces(directory, [0], [1, 2], 2, 1)

    def run_probe(self, fault=None):
        """/** 通过真实CLI验证精确重放门禁、完整阶段比较、关闭状态及四格式部分报告。 */"""
        sessions = []

        class Transport:
            def __init__(self, binary, worker, model, devices, output, **options):
                self.output, self.options = output, options
                self.round = 0
                sessions.append(self)

            def __enter__(self):
                self.output.mkdir(parents=True)
                return self

            def __exit__(self, exc_type, *unused):
                self.shutdown = dict(shutdown_complete=exc_type is None)

            def exchange(self, op, *, steps, trace):
                assert op == 'step'
                assert all(s['action'] in (0, 1, 3) for s in steps), 'no Decode in prefill diagnostic'
                begin = any(s['action'] == 1 for s in steps)
                active = [i for i, s in enumerate(steps) if s['action'] == 1]
                if begin:
                    self.round += 1
                    if fault == 'worker' and len(active) == 2:
                        raise RuntimeError('injected worker error')
                values, tokens, states = [], [], []
                for rank, step in enumerate(steps):
                    running = rank in active
                    state = dict(request_active=running, request_tokens=2 if running else 0,
                        request_capacity=5 if running else 0, request_kv_bytes=20 if running else 0,
                        captures=0, replays=0, prefix_cache=dict(last_reused_tokens=0))
                    states.append(state)
                    if running:
                        assert trace
                        assert step['token_ids'] == [1, 2] and step['capacity'] == 5
                        directory = Path(self.options['environment'][probe.TRACE_ENV])
                        _, manifest = write_trace(directory, rank, self.round, changed=len(active) == 2)
                        if fault == 'nonfinite' and len(active) == 2:
                            binary = directory / manifest['binary']
                            binary.write_bytes(struct.pack('<e', float('nan')) + binary.read_bytes()[2:])
                        logits = [0., 8. + (.25 if len(active) == 2 else 0.)] + [0.]*6
                        if fault == 'drift' and len(active) == 2:
                            logits[1] += .001
                        values.append(logits)
                        tokens.append(1)
                    else:
                        values.append([])
                        tokens.append(None)
                if fault == 'kv' and not begin:
                    states[0]['request_kv_bytes'] = 20
                return dict(status='PASS', execution='eager', states=states, token_ids=tokens, logits=values)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, build, baseline = root / 'model', root / 'build', root / 'baseline'
            for directory in (model, build, baseline):
                directory.mkdir()
            (model / 'config.json').write_text('{"vocab_size":8,"hidden_size":2,"num_hidden_layers":1}')
            (model / 'model.safetensors.index.json').write_text('{}')
            for name in ('CMakeCache.txt', 'fastllm_moe_generate', 'fastllm_tp_worker'):
                (build / name).write_text('mock only; never execute')
            job = request([1, 2], 4)
            old = dict(scope='controlled_admission_diagnostic', prefix_cache=False, first_waves=[4, 2],
                trace_tokens_1based=[1, 2, 79], request=job, git='old-commit', binary_sha256={},
                model_config_sha256=probe.file_hash(model / 'config.json'),
                model_index_sha256=probe.file_hash(model / 'model.safetensors.index.json'))
            if fault == 'provenance':
                old['model_config_sha256'] = 'different-model'
            (baseline / 'summary.json').write_text(json.dumps(dict(status='FAIL', complete=True, metadata=old)))
            for wave in (4, 2):
                directory = baseline / f'eager_first{wave}_then{4-wave}/repeat01'
                directory.mkdir(parents=True)
                for rank in range(wave):
                    report = dict(status='PASS', request=job, dp_rank=rank, steps=[dict(token_id=1,
                        execution='eager', logits=[0., 8. + (.25 if wave == 2 else 0.)] + [0.]*6)])
                    (directory / f'rank{rank}.json').write_text(json.dumps(report))
            argv = ['probe', '--baseline', str(baseline), '--model', str(model), '--build', str(build),
                    '--output', str(root / 'out')]
            with mock.patch('sys.argv', argv), mock.patch.object(probe, 'capture', return_value='mock hardware'), \
                    mock.patch.object(probe, 'RoundTransport', Transport), redirect_stdout(io.StringIO()):
                code = probe.main()
            output, = (root / 'out').iterdir()
            report = json.loads((output / 'summary.json').read_text())
            for extension in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((output / f'summary.{extension}').is_file())
        for session in sessions:
            self.assertFalse(session.options['cuda_graph'])
            self.assertFalse(session.options['prefix_cache'])
            self.assertEqual(session.options['environment'][probe.LIMIT_ENV], '256')
        return code, report, sessions

    def test_prefill_attribution_requires_baseline_replay_and_finds_first_changed_stage(self):
        code, report, sessions = self.run_probe()
        self.assertEqual(code, 1)
        self.assertTrue(report['complete'])
        self.assertEqual(report['expected_case_count'], 16)
        self.assertTrue(report['metadata']['trace_attribution_valid'])
        self.assertEqual(report['metadata']['first_different_stage'], {'0': 'layer_0_moe', '1': 'layer_0_moe'})
        row = next(r for r in report['cases'] if r['case'] == 'rank0_layer_0_moe')
        self.assertEqual(row['failed_elements'], 1)
        self.assertEqual(row['last_token']['failed_elements'], 0)
        self.assertEqual(row['first_difference']['prompt_token_1based'], 1)
        self.assertEqual(len(sessions), 2)
        self.assertTrue(all(s.shutdown['shutdown_complete'] for s in sessions))

    def test_output_drift_refuses_causal_stage_attribution(self):
        code, report, _ = self.run_probe('drift')
        self.assertEqual(code, 1)
        self.assertFalse(report['complete'])
        self.assertFalse(report['metadata']['trace_attribution_valid'])
        self.assertTrue(any(r['status'] == 'FAIL' and r['case'].endswith('baseline_replay') for r in report['cases']))
        self.assertNotIn('first_different_stage', report['metadata'])

    def test_provenance_worker_or_kv_failure_keeps_partial_reports(self):
        for fault in ('provenance', 'worker', 'kv', 'nonfinite'):
            with self.subTest(fault=fault):
                code, report, sessions = self.run_probe(fault)
                self.assertEqual(code, 1)
                self.assertFalse(report['complete'])
                self.assertFalse(report['metadata']['trace_attribution_valid'])
                if fault == 'provenance':
                    self.assertFalse(sessions)


if __name__ == '__main__':
    unittest.main()
