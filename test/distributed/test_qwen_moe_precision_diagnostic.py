"""精度定位器的CPU检查；不启动GPU、原生worker或编译。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_qwen_moe_precision_diagnostic as diagnostic


class PrecisionDiagnosticTests(unittest.TestCase):
    def test_history_uses_teacher_forcing_and_original_tolerance(self):
        """/** 实际采样Token与教师Token不同仍须找到正确历史，不以贪心相同放行超差。 */"""
        report = dict(request=dict(token_ids=[1, 2]), teacher_forced_tokens=[4],
            steps=[dict(logits=[0, 1], token_id=9, execution='eager'),
                   dict(logits=[0.06, 1], token_id=1, execution='decode_cuda_graph')])
        reference = dict(rows=[dict(ids=[1, 2], logits=[0, 1]), dict(ids=[1, 2, 4], logits=[0, 1])])
        target = diagnostic.failing_history(report, reference)
        self.assertEqual(target['ids'], [1, 2, 4])
        self.assertEqual(target['step'], 1)
        self.assertEqual(target['metrics']['failed_elements'], 1)
        self.assertEqual(target['execution'], 'decode_cuda_graph')
        report['steps'][1]['logits'][0] = 0.05
        with self.assertRaisesRegex(ValueError, 'no logits exceeding'):
            diagnostic.failing_history(report, reference)

    def test_shapes_nonfinite_and_empty_ranks(self):
        """/** 形状和非有限值不能静默跳过，空闲rank可返回合法空张量。 */"""
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            diagnostic.difference([1], [])
        for value in (float('nan'), float('inf')):
            self.assertEqual(diagnostic.difference([value], [0])['failed_elements'], 1)
        self.assertEqual(diagnostic.difference([], [])['status'], 'PASS')

    def test_chunking_preserves_order_and_limits(self):
        """/** 任务超过字节或数量限制时分块，不裁剪张量，不将参考数据送给原生checker。 */"""
        layers = [dict(name=f'layer{i}', inputs=[dict(hidden=[0.25]*10)], reference=[0]*100) for i in range(7)]
        chunks = diagnostic.task_chunks(layers, byte_limit=400, count_limit=2)
        self.assertEqual([x for chunk in chunks for x in chunk], layers)
        for chunk in chunks:
            tasks = [{k: v for k, v in layer.items() if k != 'reference'} for layer in chunk]
            self.assertLessEqual(len(json.dumps(tasks).encode()), 400)
            self.assertLessEqual(len(chunk), 2)
        with self.assertRaisesRegex(ValueError, 'single task exceeds'):
            diagnostic.task_chunks(layers, byte_limit=10)

    def test_routing_failure_does_not_hide_numerical_output(self):
        """/** 专家集合不同仍继续记录输出误差，专家ID不能被当作浮点误差。 */"""
        reference = dict(selected_experts=[0], routing_weights=[0.5], router_logits=[1, 0], output=[0.0])
        actual = dict(reference, selected_experts=[1], output=[0.1])
        layer = dict(name='real_layer_9_moe', hidden=2, stage=2,
                     inputs=[dict(hidden=[1, 2])], reference=[reference])
        rows = diagnostic.compare_layer(layer, dict(task=layer['name'], ranks=[actual]),
                                        dict(num_experts=2, num_experts_per_tok=1))
        self.assertEqual(rows[0]['status'], 'FAIL')
        self.assertIsNone(rows[0]['max_abs_error'])
        output = next(row for row in rows if row['case'].endswith('/output'))
        self.assertEqual(output['failed_elements'], 1)
        self.assertAlmostEqual(output['max_abs_error'], 0.1)

    def test_reference_failure_keeps_original_failure_and_all_report_formats(self):
        """/** 参考进程失败时保存已有logits证据、四种报告并返回非零，禁止原生步骤继续。 */"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, build = root / 'source', root / 'build'
            source.mkdir()
            build.mkdir()
            (build / 'CMakeCache.txt').write_text('CUDA_ARCH:STRING=70')
            (source / 'summary.json').write_text(json.dumps(dict(metadata=dict(git='source-commit'))))
            (source / 'request.json').write_text(json.dumps(dict(dp_rank=2, request=dict(token_ids=[1, 2]),
                steps=[dict(logits=[0.06, 1], execution='eager', token_id=1)])))
            (source / 'transformers-reference.json').write_text(json.dumps(
                dict(rows=[dict(ids=[1, 2], logits=[0, 1])])))
            argv = ['diagnostic', '--result-dir', str(source), '--request', 'request.json',
                    '--model', str(root / 'model'), '--build', str(build), '--reference-python', 'reference-python',
                    '--output', str(root / 'reports')]
            with mock.patch('sys.argv', argv), mock.patch.object(diagnostic, 'capture', return_value='cpu-test'), \
                    mock.patch.object(diagnostic, 'run_logged', return_value=7) as run, redirect_stdout(io.StringIO()):
                self.assertEqual(diagnostic.main(), 1)
            self.assertEqual(run.call_count, 1)
            output, = (root / 'reports').iterdir()
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(report['metadata']['source_git'], 'source-commit')
            self.assertEqual(report['cases'][0]['failed_elements'], 1)
            self.assertEqual(json.loads((output / 'layer-inputs.json').read_text()), [[], [], [1, 2], []])
            for suffix in ('md', 'csv', 'json', 'xlsx'):
                self.assertTrue((output / f'summary.{suffix}').is_file())


if __name__ == '__main__':
    unittest.main()
