"""四项对齐调度的纯Python检查；替换GPU/子进程，不编译或加载模型。"""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import run_qwen_moe_full as runner
from test_generation_alignment import report
from test_qwen_moe_generation_alignment import Session


class AllAlignmentTests(unittest.TestCase):
    def run_all(self, actual, expected, *, generation_error=None, numeric_error=None,
                numeric_fail=False, missing_numeric=False, prefix_reused=1):
        """/** 走正式CLI、生成矩阵和四格式报告，替换外部执行并记录阶段顺序及完成状态。 */"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, requests = root / 'cache', root / 'requests.jsonl'
            cache.write_text('CUDA_ARCH:STRING=70\n')
            requests.write_text(json.dumps(dict(messages=[], max_new_tokens=1)))
            argv = ['runner', '--binary', '/unused', '--worker', '/unused', '--model', str(root),
                    '--build-cache', str(cache), '--requests', str(requests), '--output', str(root / 'out'),
                    '--alignment-policy', 'all']
            calls, stages, snapshots = [], [], []
            tokenizer = SimpleNamespace(decode=lambda tokens, **kw: str(tokens))
            transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
            original_save = runner.save_report

            def reference_process(command, **kwargs):
                """/** 向真实参考路径写入固定报告，记录自由生成只启动一次参考子进程。 */"""
                stages.append('generation')
                output = Path(command[command.index('--output') + 1])
                output.write_text(json.dumps(dict(reports=[expected], metadata={})))
                return nullcontext()

            def numeric(suite, args, prompts):
                """/** 数值阶段模拟五项正常/失败用例，用于验证真实调度和汇总的隔离。 */"""
                stages.append('numeric')
                self.assertEqual(args.alignment_policy, 'numeric')
                self.assertEqual(prompts, [actual['request']])
                self.assertEqual(suite.output.name, 'numeric')
                if numeric_error:
                    raise numeric_error
                for index in range(4 if missing_numeric else 5):
                    def check(metrics):
                        """/** 保留可核对的数值误差，失败不得被生成结果覆盖。 */"""
                        metrics.update(max=0.123456, sum=0.123456, count=1,
                                       failed=int(numeric_fail and index == 0))
                        if metrics['failed']:
                            raise AssertionError('numeric logits exceed threshold')
                        return {}
                    suite.case(f'numeric_{index}', check)

            def save(output, rows, metadata):
                """/** 读取真实保存的汇总，检查阶段未完成时不能出现总体PASS。 */"""
                result = original_save(output, rows, metadata)
                data = json.loads((output / 'summary.json').read_text())
                snapshots.append((output.name, data.get('complete'), data['status'], len(rows)))
                return result

            console = io.StringIO()
            with mock.patch('sys.argv', argv), mock.patch.dict('sys.modules', transformers=transformers), \
                    mock.patch.object(runner, 'capture', return_value='mock hardware'), \
                    mock.patch('fastllm_pytools.qwen3_process_chat.prepare_request', return_value=actual['request']), \
                    mock.patch.object(runner, 'case_process', side_effect=reference_process), \
                    mock.patch.object(runner, 'wait_case', side_effect=generation_error, return_value=0), \
                    mock.patch.object(runner.Suite, 'session', side_effect=lambda *a: nullcontext(
                        Session(actual, calls, prefix_reused))), \
                    mock.patch.object(runner, 'run_numeric_reference', side_effect=numeric), \
                    mock.patch.object(runner, 'save_report', side_effect=save), \
                    redirect_stdout(console), redirect_stderr(console):
                try:
                    code = runner.main()
                except KeyboardInterrupt:
                    code = 130

            output, = (root / 'out').iterdir()
            reports = {}
            for name, directory in (('root', output), ('generation', output / 'generation'),
                                    ('numeric', output / 'numeric')):
                if not directory.exists():
                    continue
                reports[name] = json.loads((directory / 'summary.json').read_text())
                for suffix in ('json', 'md', 'csv', 'xlsx'):
                    self.assertTrue((directory / f'summary.{suffix}').is_file())
            csv = (output / 'summary.csv').read_text()
            self.assertIn('detail_report', csv)
            if not numeric_error and not generation_error:
                self.assertIn('0.123', csv)
                self.assertNotIn('0.123456', csv)
                markdown = (output / 'summary.md').read_text()
                with zipfile.ZipFile(output / 'summary.xlsx') as archive:
                    sheet = archive.read('xl/worksheets/sheet1.xml').decode()
                for display in (markdown, sheet):
                    self.assertIn('0.123', display)
                    self.assertNotIn('0.123456', display)
                self.assertEqual(reports['root']['cases'][-1]['max_abs_error'], 0.123456)
            return code, reports, stages, calls, snapshots, console.getvalue()

    def test_strict_failures_do_not_stop_numeric_or_hide_relaxed_pass(self):
        """/** 自由生成只跑一次，两个严格失败、一个宽松通过及numeric通过必须同时保留。 */"""
        actual = report([[10., 0., -1., -2., -3., -4.]])
        expected = report([[-4., 0., -1., -2., -3., 10.]])
        code, reports, stages, calls, _, console = self.run_all(actual, expected)
        self.assertEqual(code, 1)
        self.assertEqual(stages, ['generation', 'numeric'])
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(value is None for value in calls))
        summary = reports['root']
        self.assertTrue(summary['complete'])
        self.assertEqual({row['case']: row['status'] for row in summary['cases']},
                         dict(exact='FAIL', top5='FAIL', top5_near_tie='PASS', numeric='PASS'))
        self.assertIn('PASS: alignment/numeric', console)
        self.assertEqual(reports['generation']['expected_case_count'], 14)
        self.assertEqual(reports['numeric']['expected_case_count'], 5)

    def test_all_pass_requires_both_stages_complete(self):
        """/** 根报告只有四项全部完成且通过才能PASS，阶段中途快照必须保持FAIL。 */"""
        values = report([[5., 4., 3., 2., 1., 0.]])
        code, reports, _, _, snapshots, _ = self.run_all(values, values)
        self.assertEqual(code, 0)
        self.assertTrue(reports['root']['complete'])
        self.assertEqual(reports['root']['status'], 'PASS')
        for _, complete, status, _ in snapshots:
            if not complete:
                self.assertEqual(status, 'FAIL')

    def test_numeric_failure_is_not_downgraded_to_generation_warning(self):
        """/** 三项生成检查通过仍不能掩盖独立数值失败，最后返回非零。 */"""
        actual, expected = report([[9., 2., 1., 0., -1., -2.]]), report([[5., 1., 0., -1., -2., -3.]])
        code, reports, _, _, _, console = self.run_all(actual, expected, numeric_fail=True)
        self.assertEqual(code, 1)
        self.assertTrue(reports['root']['complete'])
        self.assertEqual([row['status'] for row in reports['root']['cases']], ['PASS'] * 3 + ['FAIL'])
        self.assertIn('numeric logits exceed threshold', reports['root']['cases'][-1]['reason'])
        self.assertIn('WARNING:', console)

    def test_stage_errors_or_missing_cases_never_claim_completion(self):
        """/** HF失败仍尝试numeric；numeric异常或少跑用例均保留已完成结果及不完整标记。 */"""
        values = report([[5., 4., 3., 2., 1., 0.]])
        for kwargs in (dict(generation_error=RuntimeError('HF failed')),
                       dict(numeric_error=RuntimeError('numeric failed')), dict(missing_numeric=True)):
            with self.subTest(kwargs=kwargs):
                code, reports, stages, _, _, _ = self.run_all(values, values, **kwargs)
                self.assertEqual(code, 1)
                self.assertEqual(stages, ['generation', 'numeric'])
                self.assertFalse(reports['root']['complete'])
                self.assertEqual(reports['root']['status'], 'FAIL')
                if 'generation_error' in kwargs:
                    self.assertEqual([row['status'] for row in reports['root']['cases']], ['FAIL'] * 3 + ['PASS'])
                else:
                    self.assertEqual([row['status'] for row in reports['root']['cases']], ['PASS'] * 3 + ['FAIL'])

    def test_execution_path_failure_applies_to_every_generation_policy(self):
        """/** 即使所有Token一致，真实Prefix路径证据缺失也必须使三种生成策略全部失败。 */"""
        values = report([[5., 4., 3., 2., 1., 0.]])
        code, reports, stages, _, _, _ = self.run_all(values, values, prefix_reused=0)
        self.assertEqual(code, 1)
        self.assertEqual(stages, ['generation', 'numeric'])
        self.assertEqual([row['status'] for row in reports['root']['cases']], ['FAIL'] * 3 + ['PASS'])

    def test_interrupt_preserves_partial_reports_and_does_not_launch_numeric(self):
        """/** 用户中断必须立即停止后续阶段，同时保留已知失败及四格式部分报告。 */"""
        values = report([[5., 4., 3., 2., 1., 0.]])
        code, reports, stages, _, _, _ = self.run_all(values, values, generation_error=KeyboardInterrupt())
        self.assertEqual(code, 130)
        self.assertEqual(stages, ['generation'])
        self.assertNotIn('numeric', reports)
        self.assertFalse(reports['root']['complete'])
        self.assertEqual(reports['root']['status'], 'FAIL')

    def test_numeric_interrupt_keeps_completed_generation_results(self):
        """/** numeric中断不能丢掉前三项已完成结论，也不能把仅计数齐全的根报告标记完成。 */"""
        values = report([[5., 4., 3., 2., 1., 0.]])
        code, reports, stages, _, _, _ = self.run_all(values, values, numeric_error=KeyboardInterrupt())
        self.assertEqual(code, 130)
        self.assertEqual(stages, ['generation', 'numeric'])
        self.assertEqual([row['status'] for row in reports['root']['cases'][:4]], ['PASS'] * 3 + ['FAIL'])
        self.assertFalse(reports['numeric']['complete'])
        self.assertFalse(reports['root']['complete'])

    def test_all_rejects_ambiguous_or_inapplicable_configuration(self):
        """/** 固定四项名称不能被自定义K/容差改变，模型缺失和性能混用在加载前拒绝。 */"""
        base = ['runner', '--binary', '/unused', '--worker', '/unused', '--build-cache', '/unused',
                '--alignment-policy', 'all']
        for extra in ([], ['--model', '/unused', '--performance'],
                      ['--model', '/unused', '--alignment-topk', '4'],
                      ['--model', '/unused', '--near-tie-tol', '0']):
            with self.subTest(extra=extra), mock.patch('sys.argv', base + extra), \
                    mock.patch.object(runner, 'Suite') as suite, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.main()
                self.assertEqual(raised.exception.code, 2)
                suite.assert_not_called()


if __name__ == '__main__':
    unittest.main()
