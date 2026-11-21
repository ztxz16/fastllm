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
    def run_performance(self, *, synthetic=False, fault=None, prompt_lengths=None, output_tokens=None,
                        cache_states=None, concurrencies=None):
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
            if prompt_lengths is not None:
                argv += ['--performance-prompt-lengths', *map(str, prompt_lengths)]
            if output_tokens is not None:
                argv += ['--performance-output-tokens', str(output_tokens)]
            if cache_states is not None:
                argv += ['--performance-cache-states', *cache_states]
            if concurrencies is not None:
                argv += ['--performance-concurrencies', *map(str, concurrencies)]
            lengths, generated = prompt_lengths or [32], output_tokens or 16
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
                calls = []

                def generate(request, teacher_forced_tokens=None):
                    """/** 返回固定长度的生成结果，不提供HF参考或logits。 */"""
                    calls.append(request)
                    if fault == 'generation' and not graph and len(calls) == 4:
                        raise RuntimeError('injected generation failure')
                    self.assertIsNone(teacher_forced_tokens)
                    length = len(request['token_ids'])
                    self.assertIn(length, lengths)
                    expected_prompt = [index % 4 + 1 for index in range(length)]
                    self.assertEqual(request, runner.request(expected_prompt, generated))
                    reused = length - 1 if prefix else 0
                    if graph and fault == 'cache':
                        reused = 1
                    count = generated - 1 if fault == 'length' else generated
                    token = length % 5 + int(graph and fault == 'output')
                    return dict(token_ids=[token] * count,
                                dp_rank=0, queue_ms=.01,
                                states=[dict(prefix_cache=dict(last_reused_tokens=reused))],
                                steps=[dict(elapsed_ms=index + .1, execution='decode_cuda_graph' if graph else 'eager')
                                       for index in range(count)])

                return nullcontext(SimpleNamespace(generate=generate, _states=states,
                                                    output=root / f'mock-session-{len(sessions)}',
                                                    shutdown=dict(shutdown_complete=True),
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
            self.evidence = {}
            for row in snapshots[-1]['cases']:
                if row['case'].startswith('perf_'):
                    path = output / (row['case'] + '.json')
                    self.assertTrue(path.is_file(), row['case'])
                    self.evidence[row['case']] = json.loads(path.read_text())
            if fault is None:
                csv = (output / 'summary.csv').read_text()
                self.assertIn('prompt_length', csv)
                self.assertIn('output_tokens', csv)
                for row in snapshots[-1]['cases']:
                    evidence = json.loads((output / (row['case'] + '.json')).read_text())
                    self.assertEqual(len(evidence['samples_ms']), 5)
                    self.assertEqual(row['output_tokens'], generated)
                    self.assertAlmostEqual(row['tokens_per_second'] * row['median_ms'] / 1000,
                                           row['concurrency'] * generated)
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

    def test_long_context_matrix_counts_each_length_and_uses_matching_baselines(self):
        """/** 两个长输入各跑18项，总数不超过50，Eager/Graph只比较同长度基线。 */"""
        lengths = [1024, 1792]
        code, snapshots, _ = self.run_performance(prompt_lengths=lengths, output_tokens=128)
        self.assertEqual(code, 0)
        result = snapshots[-1]
        self.assertTrue(result['complete'])
        self.assertEqual(result['expected_case_count'], 36)
        self.assertEqual(result['completed_case_count'], 36)
        self.assertEqual(result['metadata']['prompt_lengths'], lengths)
        self.assertEqual(result['metadata']['max_new_tokens'], 128)
        for length in lengths:
            rows = [row for row in result['cases'] if row['prompt_length'] == length]
            self.assertEqual(len(rows), 18)
            self.assertTrue(all(row['case'].startswith(f'perf_prompt{length}_output128_') for row in rows))
            self.assertEqual(len(result['metadata']['prompt_token_ids_by_length'][str(length)]), length)

    def test_output_cache_and_graph_validation_still_fail(self):
        """/** 独立测速仍校验输出、缓存复用和实际Graph路径，失败返回非零。 */"""
        for fault, reason in (('output', 'different output histories'), ('cache', 'cache'),
                              ('length', 'output length differs'),
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
        interrupted = self.evidence[snapshots[-1]['cases'][1]['case']]
        self.assertIn('interrupted', interrupted['error'])

    def test_targeted_recheck_keeps_both_modes_and_saves_first_divergence(self):
        """/** 定向复测只运行匹配的两种模式，失败Token及完整五轮证据仍写盘且无有效加速比。 */"""
        code, snapshots, console = self.run_performance(
            prompt_lengths=[1024], output_tokens=128, cache_states=['off'], concurrencies=[4], fault='output')
        self.assertEqual(code, 1)
        result = snapshots[-1]
        self.assertTrue(result['complete'])
        self.assertEqual(result['expected_case_count'], 2)
        self.assertFalse(result['metadata']['performance_matrix_complete'])
        self.assertEqual([row['status'] for row in result['cases']], ['PASS', 'FAIL'])
        eager, graph = [self.evidence[row['case']] for row in result['cases']]
        self.assertEqual(len(graph['samples_ms']), 5)
        self.assertEqual([len(batch) for batch in graph['responses']], [4] * 5)
        self.assertEqual(Path(graph['eager_reference_file']).name, result['cases'][0]['case'] + '.json')
        comparison = graph['comparison']
        self.assertFalse(comparison['match'])
        self.assertEqual(comparison['mismatched_requests'], 20)
        self.assertEqual(comparison['token_mismatches'], 20 * 128)
        first = comparison['first_divergence']
        self.assertEqual((first['repeat_1based'], first['request_1based'], first['first_token_1based']), (1, 1, 1))
        self.assertEqual(first['expected_token'], eager['responses'][0][0]['token_ids'][0])
        self.assertEqual(first['actual_token'], graph['responses'][0][0]['token_ids'][0])
        self.assertEqual(first['actual_execution'], 'decode_cuda_graph')
        self.assertEqual(first['expected_execution'], 'eager')
        self.assertNotIn('summary', graph)
        self.assertIn('first_divergence=', console)

    def test_generation_failure_preserves_completed_batches(self):
        """/** 原生调用中途失败仍保存此前完成的轮次，Graph不能把失败Eager当有效基线。 */"""
        code, snapshots, _ = self.run_performance(cache_states=['off'], concurrencies=[1], fault='generation')
        self.assertEqual(code, 1)
        rows = snapshots[-1]['cases']
        eager = self.evidence[rows[0]['case']]
        self.assertEqual(len(eager['samples_ms']), 2)
        self.assertEqual(len(eager['responses']), 2)
        self.assertIn('injected generation failure', eager['error'])
        self.assertIn('Eager baseline failed or is missing', rows[1]['reason'])

    def test_differences_preserve_submission_order_and_length_mismatches(self):
        """/** 比较不按rank或Token排序掩盖差异；缺失轮次和截断输出均保持失败。 */"""
        reference = dict(token_ids=[10, 20, 30], dp_rank=2, steps=[dict(execution='eager')] * 3)
        actual = dict(token_ids=[10, 21], dp_rank=1, steps=[dict(execution='decode_cuda_graph')] * 2)
        comparison = runner.performance_token_differences([[reference], [reference, actual]],
                                                          [[reference], [reference, reference]])
        first = comparison['first_divergence']
        self.assertEqual((first['repeat_1based'], first['request_1based'], first['first_token_1based']), (2, 2, 2))
        self.assertEqual((first['actual_rank'], first['expected_rank']), (1, 2))
        self.assertEqual(comparison['token_mismatches'], 2)
        self.assertFalse(comparison['match'])
        self.assertFalse(runner.performance_token_differences([[]], [])['match'])

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

    def test_invalid_lengths_rejected_before_model_load(self):
        """/** 长度为零、重复、超上下文和误用于对齐时必须在启动GPU前拒绝。 */"""
        base = ['runner', '--binary', '/unused', '--worker', '/unused', '--build-cache', '/unused']
        for extra in (['--performance', '--performance-prompt-lengths', '0'],
                      ['--performance', '--performance-prompt-lengths', '32', '32'],
                      ['--performance', '--performance-output-tokens', '1'],
                      ['--performance', '--performance-output-tokens', '0'],
                      ['--performance', '--performance-prompt-lengths', '1922', '--performance-output-tokens', '128'],
                      ['--performance-prompt-lengths', '512'], ['--performance-output-tokens', '128'],
                      ['--performance-cache-states', 'off'], ['--performance-concurrencies', '4'],
                      ['--performance', '--performance-cache-states', 'off', 'off'],
                      ['--performance', '--performance-concurrencies', '4', '4'],
                      ['--performance', '--performance-concurrencies', '0'],
                      ['--performance', '--performance-cache-states', 'invalid']):
            with self.subTest(extra=extra), mock.patch('sys.argv', base + extra), \
                    mock.patch.object(runner, 'Suite') as suite, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
                suite.assert_not_called()

        argv = base + ['--performance', '--performance-prompt-lengths', '1921', '--performance-output-tokens', '128']
        with mock.patch('sys.argv', argv), mock.patch.object(runner, 'Suite', side_effect=RuntimeError('validated')):
            with self.assertRaisesRegex(RuntimeError, 'validated'):
                runner.main()


if __name__ == '__main__':
    unittest.main()
