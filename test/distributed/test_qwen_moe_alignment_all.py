"""三级接受调度的纯Python检查；替换GPU/子进程，不编译或加载模型。"""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_qwen_moe_full as runner
from test_generation_alignment import report, all_failure
from test_qwen_moe_generation_alignment import Session


class AllAlignmentTests(unittest.TestCase):
    def run_all(self, actual, expected, *, generation_error=None, prefix_reused=1):
        """/** 运行真实CLI及四格式报告，所有模型和硬件调用均替换，numeric一旦调用即失败。 */"""
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
                """/** 为正式生成入口保存固定参考，记录只启动一次自由生成。 */"""
                stages.append('generation')
                output = Path(command[command.index('--output') + 1])
                output.write_text(json.dumps(dict(reports=[expected], metadata={})))
                return nullcontext()

            def save(output, rows, metadata):
                """/** 保留所有阶段中途状态，检查未完成报告不被提前标记PASS。 */"""
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
                    mock.patch.object(runner, 'run_numeric_reference', side_effect=AssertionError('numeric must not run')) as numeric, \
                    mock.patch.object(runner, 'save_report', side_effect=save), \
                    redirect_stdout(console), redirect_stderr(console):
                try:
                    code = runner.main()
                except KeyboardInterrupt:
                    code = 130
                numeric.assert_not_called()
            output, = (root / 'out').iterdir()
            self.assertFalse((output / 'numeric').exists())
            reports = {}
            for name, directory in (('root', output), ('generation', output / 'generation')):
                reports[name] = json.loads((directory / 'summary.json').read_text())
                for suffix in ('json', 'md', 'csv', 'xlsx'):
                    self.assertTrue((directory / f'summary.{suffix}').is_file())
            self.assertIn('accepted_by', (output / 'summary.csv').read_text())
            return code, reports, stages, calls, snapshots, console.getvalue()

    def test_top5_acceptance_changes_report_and_exit_code(self):
        a, b = report([[2.,1.,0.,-1.,-2.,-3.]]), report([[1.,2.,0.,-1.,-2.,-3.]])
        code, reports, stages, calls, _, console = self.run_all(a, b)
        self.assertEqual(code, 0)
        self.assertEqual(stages, ['generation'])
        self.assertEqual(len(calls), 10)
        self.assertEqual(reports['root']['status'], 'PASS')
        self.assertEqual(reports['root']['cases'][0]['accepted_by'], 'top5')
        self.assertIn('WARNING:', console)
        self.assertEqual(reports['generation']['expected_case_count'], 14)

    def test_near_tie_acceptance_preserves_two_warnings(self):
        a, b = report([[10.,0.,-1.,-2.,-3.,-4.]]), report([[-4.,0.,-1.,-2.,-3.,10.]])
        code, reports, _, _, _, _ = self.run_all(a, b)
        self.assertEqual(code, 0)
        row = reports['root']['cases'][0]
        self.assertEqual(row['accepted_by'], 'top5_near_tie')
        self.assertEqual(row['warning_count'], 20)

    def test_all_three_mismatch_returns_nonzero(self):
        code, reports, _, _, _, _ = self.run_all(*all_failure())
        self.assertEqual(code, 1)
        self.assertTrue(reports['root']['complete'])
        self.assertEqual(reports['root']['status'], 'FAIL')

    def test_exact_pass_never_computes_numeric_or_early_completion(self):
        a, b = report([[9.,2.,1.,0.,-1.,-2.]]), report([[5.,1.,0.,-1.,-2.,-3.]])
        code, reports, _, _, snapshots, _ = self.run_all(a, b)
        self.assertEqual(code, 0)
        self.assertEqual(reports['root']['cases'][0]['accepted_by'], 'exact')
        self.assertTrue(all(status == 'FAIL' for _, complete, status, _ in snapshots if not complete))

    def test_execution_failure_is_not_a_comparison_warning(self):
        values = report([[5.,4.,3.,2.,1.,0.]])
        code, reports, _, _, _, _ = self.run_all(values, values, prefix_reused=0)
        self.assertEqual(code, 1)
        self.assertIn('did not reuse cached tokens', reports['root']['cases'][0]['reason'])

    def test_missing_reference_is_functional_failure(self):
        values = report([[5.,4.,3.,2.,1.,0.]])
        code, reports, _, calls, _, _ = self.run_all(values, values, generation_error=RuntimeError('HF failed'))
        self.assertEqual(code, 1)
        self.assertFalse(reports['root']['complete'])
        self.assertEqual(calls, [])

    def test_interrupt_preserves_partial_reports(self):
        values = report([[5.,4.,3.,2.,1.,0.]])
        code, reports, stages, _, _, _ = self.run_all(values, values, generation_error=KeyboardInterrupt())
        self.assertEqual(code, 130)
        self.assertEqual(stages, ['generation'])
        self.assertFalse(reports['root']['complete'])

    def test_invalid_configuration_is_rejected_before_suite(self):
        base = ['runner', '--binary', '/unused', '--worker', '/unused', '--build-cache', '/unused']
        for extra in (['--alignment-policy','all'], ['--model','/unused','--alignment-policy','all','--performance'],
                      ['--model','/unused','--alignment-topk','4'], ['--model','/unused','--near-tie-tol','0'],
                      ['--model','/unused','--alignment-policy','numeric']):
            with self.subTest(extra=extra), mock.patch('sys.argv', base + extra), \
                    mock.patch.object(runner, 'Suite') as suite, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.main()
                self.assertEqual(raised.exception.code, 2)
                suite.assert_not_called()


if __name__ == '__main__':
    unittest.main()
