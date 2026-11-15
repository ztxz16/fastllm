"""完整前向诊断的CPU检查：真实边界连续性、二进制校验、公开入口和归因门禁。"""

import copy
import ctypes
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import probe_qwen_moe_forward_precision as probe


def trace_fixture(directory):
    """/** 两Token、两隐藏维、一层的独立二进制样本，包含可精确表示的正负FP16值。 */"""
    values = [1., -2., .5, .25, 2., 3., -1., 0., 4., 5., 6., 7., .125, -.5]
    binary = directory / 'trace.f16'
    binary.write_bytes(struct.pack('<14e', *values))
    meta = dict(version=1, dtype='float16', byte_order='little', rank=0, tokens=2,
                hidden=2, layers=1, ids=[1, 2], binary=binary.name, snapshots=[
                    dict(name='embedding', offset=0, elements=4),
                    dict(name='layer_0_attention', offset=4, elements=4),
                    dict(name='layer_0_moe', offset=8, elements=4),
                    dict(name='final_norm', offset=12, elements=2)])
    (directory / 'trace.json').write_text(json.dumps(meta))
    return meta


def reference_fixture(layers=2):
    """/** 两层HF完整前向连续边界，其他三个rank为空；用来拒绝独立输入被错当轨迹。 */"""
    tasks = []
    for layer in range(layers):
        before = [float(layer)]*4
        attended = [layer + .5]*4
        after = [float(layer+1)]*4
        for stage, values, output in (('full', before, after), ('attention', before, attended),
                                      ('moe', attended, after)):
            tasks.append(dict(name=f'real_layer_{layer}_{stage}',
                inputs=[dict(hidden=values), *[dict(hidden=[]) for _ in range(3)]],
                reference=[dict(output=output), *[dict(output=[]) for _ in range(3)]]))
    return dict(layers=tasks)


class ForwardPrecisionTests(unittest.TestCase):
    def test_binary_preserves_all_stages_and_fp16_values(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            trace_fixture(directory)
            meta, trace = probe.load_trace(directory, [1, 2], 2, 1)
            self.assertEqual(trace['embedding'], [1., -2., .5, .25])
            self.assertEqual(trace['final_norm'], [.125, -.5])
            self.assertEqual(list(trace), ['embedding', 'layer_0_attention', 'layer_0_moe', 'final_norm'])
            self.assertEqual(len(meta['binary_sha256']), 64)

    def test_binary_rejects_truncation_nonfinite_unexpected_files_and_path_escape(self):
        for mode in ('truncated', 'nan', 'extra', 'escape', 'absolute', 'duplicate_manifest', 'missing_manifest'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                meta = trace_fixture(directory)
                binary = directory / 'trace.f16'
                if mode == 'truncated':
                    binary.write_bytes(binary.read_bytes()[:-2])
                elif mode == 'nan':
                    binary.write_bytes(struct.pack('<e', float('nan')) + binary.read_bytes()[2:])
                elif mode == 'extra':
                    (directory / 'partial.f16').write_bytes(b'')
                elif mode in ('escape', 'absolute'):
                    meta['binary'] = '../trace.f16' if mode == 'escape' else str(binary.resolve())
                    (directory / 'trace.json').write_text(json.dumps(meta))
                elif mode == 'duplicate_manifest':
                    (directory / 'other.json').write_text(json.dumps(meta))
                else:
                    (directory / 'trace.json').unlink()
                with self.assertRaises(ValueError):
                    probe.load_trace(directory, [1, 2], 2, 1)

    def test_binary_rejects_wrong_history_order_and_offsets(self):
        for mode in ('history', 'rank', 'order', 'offset', 'overlap', 'elements', 'missing'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                meta = trace_fixture(directory)
                if mode == 'history':
                    meta['ids'] = [2, 1]
                elif mode == 'rank':
                    meta['rank'] = 1
                elif mode == 'order':
                    meta['snapshots'][1:3] = reversed(meta['snapshots'][1:3])
                elif mode in ('offset', 'overlap'):
                    meta['snapshots'][1]['offset'] = -1 if mode == 'offset' else 0
                elif mode == 'elements':
                    meta['snapshots'][0]['elements'] = 5
                else:
                    meta['snapshots'].pop()
                (directory / 'trace.json').write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    probe.load_trace(directory, [1, 2], 2, 1)

    def test_hf_trajectory_requires_continuous_saved_forward(self):
        reference = reference_fixture()
        result = probe.hf_trajectory(reference, 0, 2, 2, 2)
        self.assertEqual(result['embedding'], [0.]*4)
        self.assertEqual(result['layer_1_moe'], [2.]*4)
        for mode in ('input', 'attention', 'output', 'missing', 'duplicate', 'nonfinite', 'shape'):
            with self.subTest(mode=mode):
                broken = copy.deepcopy(reference)
                tasks = broken['layers']
                if mode == 'input':
                    tasks[3]['inputs'][0]['hidden'] = [3.]*4
                elif mode == 'attention':
                    tasks[2]['inputs'][0]['hidden'] = [4.]*4
                elif mode == 'output':
                    tasks[2]['reference'][0]['output'] = [8.]*4
                elif mode == 'missing':
                    tasks.pop()
                elif mode == 'duplicate':
                    tasks.append(tasks[0])
                elif mode == 'nonfinite':
                    tasks[0]['reference'][0]['output'] = [float('nan')]*4
                else:
                    tasks[0]['reference'][0]['output'] = [1.]
                with self.assertRaises(ValueError):
                    probe.hf_trajectory(broken, 0, 2, 2, 2)

    def test_exact_gate_is_distinct_from_original_tolerance_and_last_token(self):
        actual, expected = [0., .25, .01, .05], [0., 0., 0., 0.]
        row = probe.compare('tensor', actual, expected, 2)
        self.assertEqual(row['failed_elements'], 1)
        self.assertEqual(row['last_token']['failed_elements'], 0)
        self.assertEqual(row['exact_different_elements'], 3)
        gate = probe.compare('gate', actual, expected, 2, exact=True)
        self.assertEqual(gate['failed_elements'], 3)
        self.assertEqual(probe.compare('near', [.01], [0.], 1)['status'], 'PASS')
        self.assertEqual(probe.compare('near', [.01], [0.], 1, exact=True)['status'], 'FAIL')
        for values, target, width in (([], [], 1), ([1.], [1., 2.], 1), ([1.], [1.], 2),
                                      ([float('inf')], [1.], 1), ([1.], [1.], 0)):
            with self.subTest(values=values, width=width), self.assertRaises(ValueError):
                probe.compare('bad', values, target, width)

    def test_head_gate_uses_actual_native_decimal_wire_format_without_relaxing_tolerance(self):
        # 直接调用宿主libc的snprintf作为正式%.12f格式的oracle，不编译或加载CUDA。
        libc = ctypes.CDLL(None)
        actual = [struct.unpack('<f', struct.pack('<f', value))[0] for value in (.1, -.3, 2.1)]
        expected = []
        for value in actual:
            buffer = ctypes.create_string_buffer(32)
            libc.snprintf(buffer, ctypes.c_size_t(32), b'%.12f', ctypes.c_double(value))
            expected.append(float(buffer.value))
        raw = probe.compare('raw', actual, expected, 3, exact=True)
        self.assertEqual(raw['status'], 'FAIL')
        self.assertGreater(raw['max_abs_error'], 0.)
        row = probe.compare_native_head_replay(actual, expected, 3)
        self.assertEqual(row['status'], 'PASS')
        self.assertEqual(row['max_abs_error'], 0.)
        self.assertEqual(row['serialization_ambiguous_elements'], 0)
        self.assertEqual(row['raw_numeric_comparison']['max_abs_error'], raw['max_abs_error'])

        bits, = struct.unpack('<I', struct.pack('<f', actual[0]))
        adjacent = [struct.unpack('<f', struct.pack('<I', bits+1))[0], *actual[1:]]
        self.assertEqual(probe.compare('tolerance', adjacent, expected, 3)['status'], 'PASS')
        row = probe.compare_native_head_replay(adjacent, expected, 3)
        self.assertEqual(row['status'], 'FAIL')
        self.assertEqual(row['failed_elements'], 1)
        expected[0] += 1e-14
        self.assertEqual(probe.compare_native_head_replay(actual, expected, 3)['status'], 'FAIL')

    def test_head_gate_rejects_ambiguous_wire_identity_and_invalid_values(self):
        for value in (0., -0., 2.**-149, -(2.**-149), 2.**-30):
            with self.subTest(value=value):
                row = probe.compare_native_head_replay([value], [float(format(value, '.12f'))], 1)
                self.assertEqual(row['status'], 'FAIL')
                self.assertEqual(row['serialization_ambiguous_elements'], 1)
                self.assertIn('identity', row['reason'])
        for actual, expected in (([.1], [.1]), ([float('nan')], [0.]), ([1.], [float('inf')]),
                                  ([1e40], [1.]), ([2.**80], [1.]), ([], [])):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                probe.compare_native_head_replay(actual, expected, 1)

    def test_native_uses_public_session_and_child_only_trace_environment(self):
        args = SimpleNamespace(build=Path('/build'), model=Path('/model'), timeout=10)
        request = dict(token_ids=[1], max_new_tokens=9, eos_token_ids=[], temperature=0.)
        original = dict(request=request)
        with patch.dict(os.environ, {probe.TRACE_ENV: 'inherited', 'KEPT': 'yes'}), \
                patch.object(probe, 'QwenMoeProcessSession') as session_type:
            session = session_type.return_value.__enter__.return_value
            step = dict(execution='eager', logits=[1., 2.], joint_round=dict(
                actions=[1, 0, 0, 0], submitted_token_counts=[3, 0, 0, 0]))
            session.generate.return_value = dict(status='PASS', dp_rank=0, steps=[step])
            self.assertEqual(probe.native_forward(args, Path('/out'), original, [1, 2, 3]), [1., 2.])
            self.assertNotIn(probe.TRACE_ENV, session_type.call_args.kwargs['environment'])
            self.assertEqual(session_type.call_args.kwargs['environment']['KEPT'], 'yes')
            probe.native_forward(args, Path('/out'), original, [1, 2, 3], Path('/trace'))
            self.assertEqual(session_type.call_args.kwargs['environment'][probe.TRACE_ENV], '/trace')
            self.assertEqual(os.environ[probe.TRACE_ENV], 'inherited')
            self.assertFalse(session_type.call_args.kwargs['cuda_graph'])
            self.assertFalse(session_type.call_args.kwargs['prefix_cache'])
            self.assertEqual(session.generate.call_args.args[0]['max_new_tokens'], 1)
            self.assertEqual(request['max_new_tokens'], 9)
            step['execution'] = 'decode_cuda_graph'
            with self.assertRaisesRegex(RuntimeError, 'execution'):
                probe.native_forward(args, Path('/out'), original, [1, 2, 3])

    def test_head_replay_preserves_native_norm_and_both_gemm_shapes(self):
        import torch

        class Gemm:
            """/** CPU矩阵oracle：记录形状及输出类型，测试不会加载CUDA库。 */"""
            def __init__(self):
                self.calls = []

            def linear(self, torch, x, weight, fp32):
                self.calls.append((tuple(x.shape), fp32))
                return x.float() @ weight.float().T

        gemm = Gemm()
        weight = torch.tensor([[1., 2.], [3., 4.]])
        hf = torch.tensor([[1., 2.], [3., 4.]])
        native = hf + 1
        actual_norm = torch.tensor([[10., 20.]])
        result = probe.head_values(torch, lambda value: value*2, weight, hf, native, actual_norm, gemm)
        self.assertEqual(result['hf_norm'].tolist(), [[6., 8.]])
        self.assertEqual(result['hf_norm_on_native'].tolist(), [[8., 10.]])
        self.assertEqual(result['native_head_replay'].tolist(), [[50., 110.]])
        self.assertEqual(result['hf_head_full'].tolist(), [[22., 50.]])
        self.assertEqual(gemm.calls, [((1, 2), True), ((1, 2), True)])

    def test_early_failure_generates_all_report_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            argv = ['probe', '--diagnostic', '/missing', '--model', '/missing', '--output', temp]
            with patch('sys.argv', argv), patch.object(probe, 'capture', return_value='cpu'), \
                    patch.dict(os.environ, {probe.TRACE_ENV: 'bad'}), redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main(), 1)
            output, = Path(temp).iterdir()
            for suffix in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((output / f'summary.{suffix}').exists())
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(report['cases'][0]['case'], 'probe_error')
            self.assertIn('unset', report['cases'][0]['reason'])

    def test_full_probe_keeps_original_failure_and_marks_unstable_attribution(self):
        for head_error in (False, True):
            with self.subTest(head_error=head_error), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, diagnostic, build, model = [root / name for name in ('source', 'diagnostic', 'build', 'model')]
                for directory in (source, diagnostic, build, model):
                    directory.mkdir()
                (model / 'config.json').write_text(json.dumps(dict(model_type='qwen2_moe', hidden_size=2,
                    num_hidden_layers=1, vocab_size=2)))
                for name in ('CMakeCache.txt', 'fastllm_moe_generate', 'fastllm_tp_worker'):
                    (build / name).write_text('cpu fixture')
                (source / 'request.json').write_text(json.dumps(dict(dp_rank=0,
                    request=dict(token_ids=[1, 2]), steps=[dict(logits=[.1, 0.], execution='eager', token_id=0)])))
                (source / 'transformers-reference.json').write_text(json.dumps(
                    dict(rows=[dict(ids=[1, 2], logits=[0., 0.])])))
                (diagnostic / 'summary.json').write_text(json.dumps(dict(metadata=dict(source_dir=str(source),
                    request='request.json', target=dict(ids=[1, 2]), rank=0, git='source', completed_tasks=3),
                    cases=[dict(case='saved_full_model_logits', status='FAIL'),
                           dict(case='old_layer_failure', status='FAIL', reason='kept')])))
                (diagnostic / 'layer-inputs.json').write_text('[[1, 2], [], [], []]')
                reference = reference_fixture(1)
                reference['metadata'] = dict(dtype='float16')
                (diagnostic / 'reference.json').write_text(json.dumps(reference))

                def native(args, output, original, ids, trace=None):
                    """/** 模拟跨运行不稳定且采集存在，门禁应失败但原始观察仍落盘。 */"""
                    if trace is not None:
                        trace_fixture(trace)
                    return [.1 if output.name.endswith('1') else .11 if trace is None else .12, 0.]

                heads = {key: [0., 0.] for key in ('hf_norm', 'hf_norm_on_native', 'hf_head_full',
                    'hf_head_last', 'native_head_on_hf', 'native_head_replay', 'hf_head_on_native_norm')}
                argv = ['probe', '--diagnostic', str(diagnostic), '--model', str(model), '--build', str(build),
                        '--output', str(root / 'reports')]
                with patch('sys.argv', argv), patch.object(probe, 'capture', return_value='cpu'), \
                        patch.dict(os.environ, {probe.TRACE_ENV: ''}), \
                        patch.object(probe, 'native_forward', side_effect=native), \
                        patch.object(probe, 'head_probe', return_value=({}, heads),
                            side_effect=RuntimeError('head unavailable') if head_error else None), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(probe.main(), 1)
                output, = (root / 'reports').iterdir()
                report = json.loads((output / 'summary.json').read_text())
                self.assertEqual(report['cases'][0]['case'], 'saved_native_vs_saved_hf')
                self.assertEqual(report['cases'][0]['status'], 'FAIL')
                self.assertFalse(report['metadata']['trace_attribution_valid'])
                self.assertEqual(report['metadata']['previous_independent_failures'][0]['case'], 'old_layer_failure')
                self.assertIn('layer_0_moe', [row['case'] for row in report['cases']])
                if head_error:
                    self.assertEqual(report['cases'][-1]['reason'], 'head unavailable')
                else:
                    self.assertFalse(report['metadata']['head_attribution_valid'])
                    self.assertEqual(report['cases'][-1]['case'], 'head_native_vs_hf_same_native_norm')
                for suffix in ('json', 'md', 'csv', 'xlsx'):
                    self.assertTrue((output / f'summary.{suffix}').exists())


if __name__ == '__main__':
    unittest.main()
