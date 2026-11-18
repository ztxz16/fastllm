"""性能入口隔离的纯Python回归；模拟原生会话，不编译或加载GPU权重。"""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_qwen_moe_full as runner


class PerformanceTests(unittest.TestCase):
    def run_performance(self, *, synthetic=False, fault=None):
        """/** 走真实CLI、测速循环和四格式报告，只替换GPU会话及外部环境探测。 */"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / 'CMakeCache.txt'
            cache.write_text('CUDA_ARCH:STRING=70\n')
            argv = ['runner', '--binary', '/unused', '--worker', '/unused', '--build-cache', str(cache),
                    '--output', str(root / 'out'), '--performance', '--requests', '/missing/requests.jsonl',
                    '--reference-python', '/missing/reference/python']
            if not synthetic:
                argv += ['--model', str(root / 'model')]
            sessions, snapshots = [], []
            original_save = runner.save_report

            def session(model, world, graph, prefix, trace, routing):
                """/** 提供正式输出结构并注入路径/输出错误，验证原有测速检查仍会拒绝。 */"""
                if fault == 'interrupt' and len(sessions) == 1:
                    raise KeyboardInterrupt()
                self.assertEqual(world, 4)
                self.assertFalse(trace)
                self.assertEqual(routing, 'prefix' if prefix else 'round_robin')
                sessions.append((model, graph, prefix))
                states = [dict(captures=int(graph), replays=int(graph and fault != 'graph'),
                               reserved_kv_bytes=128, workspace_bytes=64,
                               prefix_cache=dict(device_bytes=32)) for _ in range(4)]

                def generate(request, teacher_forced_tokens=None):
                    """/** 返回固定长度的生成结果，不提供HF参考或logits。 */"""
                    self.assertIsNone(teacher_forced_tokens)
                    self.assertEqual(request, runner.request([1, 2, 3, 4] * 8, 16))
                    reused = 31 if prefix else 0
                    if graph and fault == 'cache':
                        reused = 1
                    return dict(token_ids=[2 if graph and fault == 'output' else 1] * 16,
                                dp_rank=0, queue_ms=.01,
                                states=[dict(prefix_cache=dict(last_reused_tokens=reused))],
                                steps=[dict(elapsed_ms=index + .1) for index in range(16)])

                return nullcontext(SimpleNamespace(generate=generate, _states=states,
                                                    clear_prefix_cache=mock.Mock()))

            def save(output, rows, metadata):
                """/** 保留真实报告并记录中途状态，未完成矩阵不能显示总体通过。 */"""
                status = original_save(output, rows, metadata)
                snapshots.append(json.loads((output / 'summary.json').read_text()))
                return status

            console = io.StringIO()
            with mock.patch('sys.argv', argv), mock.patch.dict('sys.modules', transformers=None), \
                    mock.patch.object(runner, 'capture', return_value='mock hardware'), \
                    mock.patch.object(runner, 'fixture') as fixture, \
                    mock.patch.object(runner, 'case_process') as reference, \
                    mock.patch.object(runner.Suite, 'session', side_effect=session), \
                    mock.patch.object(runner.Suite, 'synthetic', side_effect=AssertionError('synthetic launched')) as synth, \
                    mock.patch.object(runner.Suite, 'generation_alignment', side_effect=AssertionError('alignment launched')) as alignment, \
                    mock.patch.object(runner, 'run_numeric_reference', side_effect=AssertionError('numeric launched')) as numeric, \
                    mock.patch.object(runner, 'save_report', side_effect=save), \
                    redirect_stdout(console), redirect_stderr(console):
                try:
                    code = runner.main()
                except KeyboardInterrupt:
                    code = 130
                for unused in (reference, synth, alignment, numeric):
                    unused.assert_not_called()

            output, = (root / 'out').iterdir()
            for suffix in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((output / f'summary.{suffix}').is_file())
            if synthetic:
                fixture.assert_called_once_with(output / 'f16_bias', **runner.SYNTHETIC_FIXTURES['f16_bias'])
                self.assertTrue(all(model == output / 'f16_bias' for model, _, _ in sessions))
            else:
                fixture.assert_not_called()
                self.assertTrue(all(model == root / 'model' for model, _, _ in sessions))
            self.assertFalse((output / 'requests.json').exists())
            self.assertFalse((output / 'transformers-generation.json').exists())
            return code, snapshots, console.getvalue()

    def test_real_performance_without_transformers_or_alignment_requests(self):
        """/** 真实模型仅运行18个性能case，五轮样本和独立完成状态必须保存。 */"""
        code, snapshots, console = self.run_performance()
        self.assertEqual(code, 0)
        result = snapshots[-1]
        self.assertEqual(result['expected_case_count'], 18)
        self.assertEqual(result['completed_case_count'], 18)
        self.assertTrue(result['complete'])
        self.assertEqual(result['metadata']['scope'], 'performance')
        self.assertEqual(result['metadata']['repeats'], 5)
        self.assertTrue(all(row['case'].startswith('perf_') and row['repeats'] == 5
                            for row in result['cases']))
        pending = [summary for summary in snapshots if summary['completed_case_count'] < 18]
        self.assertTrue(pending)
        self.assertTrue(all(not summary['complete'] and summary['status'] == 'FAIL' for summary in pending))
        self.assertNotIn('real_ep', console)
        self.assertNotIn('alignment/', console)

    def test_synthetic_performance_only_prepares_one_fixture(self):
        """/** 未指定真实模型时仅准备一个性能小模型，不执行完整合成验收。 */"""
        code, snapshots, _ = self.run_performance(synthetic=True)
        self.assertEqual(code, 0)
        self.assertEqual(snapshots[-1]['completed_case_count'], 18)

    def test_output_cache_and_graph_validation_still_fail(self):
        """/** 独立测速仍校验输出、缓存复用和实际Graph路径，失败返回非零。 */"""
        for fault, reason in (('output', 'different output histories'), ('cache', 'cache'),
                              ('graph', 'missed Graph')):
            with self.subTest(fault=fault):
                code, snapshots, _ = self.run_performance(fault=fault)
                self.assertEqual(code, 1)
                self.assertTrue(snapshots[-1]['complete'])
                self.assertTrue(any(row['status'] == 'FAIL' and reason in row['reason']
                                    for row in snapshots[-1]['cases']))

    def test_interrupt_preserves_incomplete_performance_report(self):
        """/** Ctrl+C保存已运行用例和预期总数，不启动后续对齐或标记为完整通过。 */"""
        code, snapshots, _ = self.run_performance(fault='interrupt')
        self.assertEqual(code, 130)
        self.assertEqual(snapshots[-1]['status'], 'FAIL')
        self.assertFalse(snapshots[-1]['complete'])
        self.assertEqual(snapshots[-1]['expected_case_count'], 18)
        self.assertEqual(snapshots[-1]['cases'][0]['status'], 'PASS')

    def test_alignment_options_are_rejected_with_performance(self):
        """/** 显式非默认对齐选项不能在性能模式中被忽略或触发隐藏的对齐阶段。 */"""
        base = ['runner', '--binary', '/unused', '--worker', '/unused', '--build-cache', '/unused', '--performance']
        for extra in (['--alignment-policy', 'exact'], ['--alignment-policy', 'numeric'],
                      ['--alignment-policy', 'all'], ['--alignment-topk', '2'], ['--near-tie-tol', '0']):
            with self.subTest(extra=extra), mock.patch('sys.argv', base + extra), \
                    mock.patch.object(runner, 'Suite') as suite, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
                suite.assert_not_called()


if __name__ == '__main__':
    unittest.main()
