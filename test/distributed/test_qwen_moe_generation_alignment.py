"""生成验收入口及报告的纯Python检查；替换GPU执行，不编译或加载权重。"""

import copy
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_qwen_moe_full as runner
from test_generation_alignment import report, all_failure
from qwen_moe_transformers_reference import generation_reference


class Session:
    def __init__(self, output, calls, prefix_reused=1):
        """/** 只模拟正式会话的报告与路径证据，不修改验收实现。 */"""
        self.output, self.calls = output, calls
        self.prefix_reused = prefix_reused
        self.ready = dict(workers=[])
        self.shutdown = dict(workers_reaped=True)
        self._states = [dict(captures=1, replays=1) for _ in range(4)]

    def generate(self, request, teacher_forced_tokens=None):
        """/** 记录是否发生教师强制，并提供足够的前缀命中证据。 */"""
        self.calls.append(teacher_forced_tokens)
        result = copy.deepcopy(self.output)
        result.update(request=request, dp_rank=0,
                      states=[dict(prefix_cache=dict(hits=10, last_reused_tokens=self.prefix_reused))])
        return result

    def clear_prefix_cache(self):
        """/** 模拟完成正式缓存清理。 */"""
        pass


class RunnerTests(unittest.TestCase):
    def run_suite(self, actual, expected, policy='topk', tolerance=.1, reference_exit=0,
                  performance=False, prefix_reused=1):
        """/** 走正式suite和四种报告，只替换外部执行及tokenizer。 */"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite = runner.Suite.__new__(runner.Suite)
            suite.output, suite.rows, suite.metadata = root, [], {}
            suite.args = SimpleNamespace(reference_python='/unused/python', timeout=1,
                                         alignment_policy=policy, alignment_topk=5, near_tie_tol=tolerance,
                                         performance=performance)
            reference_path = root / 'transformers-generation.json'
            reference_path.write_text(json.dumps(dict(reports=[expected], metadata={})))
            calls = []
            session = Session(actual, calls, prefix_reused)
            tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens))
            console = io.StringIO()
            with mock.patch.object(runner, 'case_process', return_value=nullcontext()) as reference_process, \
                    mock.patch.object(runner, 'wait_case', return_value=reference_exit), \
                    mock.patch.object(suite, 'session', side_effect=lambda *a: nullcontext(session)), \
                    redirect_stdout(console):
                suite.generation_alignment(root, [actual['request']], tokenizer)
            summary = json.loads((root / 'summary.json').read_text())
            for suffix in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((root / f'summary.{suffix}').exists())
            csv = (root / 'summary.csv').read_text()
            self.assertIn('accepted_by', csv)
            self.assertNotIn('logit_failed_elements', csv)
            self.assertIn('exact_match', csv)
            self.assertTrue(calls)
            self.assertTrue(all(value is None for value in calls))
            if policy == 'all':
                reference_process.assert_called_once()
                self.assertEqual(len(calls), 10)
                evidence = json.loads((root / 'real_ep4_graph0_prefix0_cold_request000.json').read_text())
                self.assertEqual(evidence['actual']['token_ids'], actual['token_ids'])
                self.assertEqual(set(evidence['comparison']['comparisons']), {'exact', 'top5', 'top5_near_tie'})
                self.assertEqual(evidence['reference_request'], 0)
                for field in ('exact_status', 'top5_status', 'top5_near_tie_status'):
                    self.assertIn(field, csv)
            return summary, console.getvalue()

    def test_all_reuses_generation_and_keeps_independent_conclusions(self):
        """/** exact和严格候选失败后仍完成全部模式，只生成一次且三项结论分别保存。 */"""
        actual, expected = report([[10., 0., -1., -2., -3., -4.]]), report([[-4., 0., -1., -2., -3., 10.]])
        result, console = self.run_suite(actual, expected, policy='all')
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['complete'])
        cases = [row for row in result['cases'] if 'alignment_policy' in row]
        self.assertEqual(len(cases), 10)
        self.assertTrue(all(row['exact_status'] == row['top5_status'] == 'WARNING' and
                            row['top5_near_tie_status'] == 'PASS' for row in cases))
        self.assertEqual(console.count('ALIGNMENT:'), 10)

        result, _ = self.run_suite(actual, actual, policy='all')
        self.assertEqual(result['status'], 'PASS')

    def test_default_policy_does_not_run_numeric(self):
        """/** 同输出不同Logits通过exact，报告不再计算数值误差或输出numeric告警。 */"""
        actual, expected = report([[9., 2., 1., 0., -1., -2.]]), report([[5., 1., 0., -1., -2., -3.]])
        result, console = self.run_suite(actual, expected)
        self.assertEqual(result['status'], 'PASS')
        cases = [row for row in result['cases'] if 'alignment_policy' in row]
        self.assertEqual(len(cases), 10)
        self.assertTrue(all(row['accepted_by'] == 'exact' and row['failed_elements'] == 0 for row in cases))
        self.assertTrue(all('logit_diagnostic_status' not in row for row in cases))
        self.assertNotIn('WARNING:', console)
        self.assertEqual(cases[0]['configuration_scope'], 'shared_eager')
        self.assertEqual(cases[2]['configuration_scope'], 'ft_production_vs_hf_eager')

    def test_legacy_policy_names_use_the_same_cascade(self):
        """/** 旧exact/topk入口统一逐级接受，执行成功不能掩盖三级全部未通过。 */"""
        a, b = report([[10., 0., -1., -2., -3., -4.]]), report([[-4., 0., -1., -2., -3., 10.]])
        for policy in ('topk', 'exact', 'all'):
            with self.subTest(policy=policy):
                accepted, console = self.run_suite(a, b, policy=policy)
                self.assertEqual(accepted['status'], 'PASS')
                self.assertIn('WARNING:', console)
                failed, _ = self.run_suite(*all_failure(), policy=policy)
                self.assertEqual(failed['status'], 'FAIL')
                self.assertTrue(all(row['status'] == 'PASS' for row in failed['cases'] if row['case'].endswith('_execution')))

    def test_reference_failure_does_not_launch_ft(self):
        """/** HF失败不能降级成FT自比或使用缺失参考。 */"""
        values = report([[5., 4., 3., 2., 1., 0.]])
        with self.assertRaisesRegex(AssertionError, 'exited 7'):
            self.run_suite(values, values, reference_exit=7)

    def test_unfinished_summary_cannot_claim_pass(self):
        """/** 中途汇总即使已有用例全通过仍标记未完成；可选性能也须计入预期数量。 */"""
        values = report([[5., 4., 3., 2., 1., 0.]])
        snapshots = []
        original = runner.save_report

        def capture_summary(output, rows, metadata):
            """/** 使用正式四格式报告，只读取每次保存的完成状态。 */"""
            status = original(output, rows, metadata)
            summary = json.loads((output / 'summary.json').read_text())
            snapshots.append((summary['completed_case_count'], summary['complete'], status))
            return status

        with mock.patch.object(runner, 'save_report', side_effect=capture_summary):
            result, _ = self.run_suite(values, values)
        self.assertEqual(result['expected_case_count'], 14)
        self.assertEqual(snapshots[-1], (14, True, 'PASS'))
        self.assertTrue(all(count < 14 and not complete and status == 'FAIL'
                            for count, complete, status in snapshots[:-1]))

        result, _ = self.run_suite(values, values, performance=True)
        self.assertEqual(result['expected_case_count'], 14)
        self.assertTrue(result['complete'])
        self.assertEqual(result['status'], 'PASS')

    def test_old_prefix_hits_do_not_prove_current_request_reuse(self):
        """/** 累计命中不代表本次命中；当前请求没有复用Token时执行检查必须失败。 */"""
        values = report([[5., 4., 3., 2., 1., 0.]])
        result, _ = self.run_suite(values, values, prefix_reused=0)
        self.assertEqual(result['status'], 'FAIL')
        cases = [row for row in result['cases'] if row['case'].endswith('prefix1_execution')]
        self.assertEqual(len(cases), 2)
        self.assertTrue(all(row['status'] == 'FAIL' and 'did not reuse cached tokens' in row['reason']
                            for row in cases))

    def test_cli_propagates_failed_alignment_exit_code(self):
        """/** 主入口的候选失败返回非零并保存四格式报告，不被后续成功覆盖。 */"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, requests = root / 'cache', root / 'requests.jsonl'
            cache.write_text('CPU only')
            requests.write_text(json.dumps(dict(messages=[], max_new_tokens=1)))
            args = ['runner', '--binary', '/unused', '--worker', '/unused', '--model', str(root),
                    '--build-cache', str(cache), '--requests', str(requests), '--output', str(root / 'out')]

            def failed(suite, *unused):
                """/** 注入真实报告结构的请求失败，使用主入口现有汇总规则。 */"""
                suite.rows.extend([dict(case='generation', status='FAIL', failed_elements=1),
                                   dict(case='execution', status='PASS', failed_elements=0)])

            transformers = SimpleNamespace(AutoTokenizer=mock.Mock())
            with mock.patch('sys.argv', args), mock.patch.dict('sys.modules', transformers=transformers), \
                    mock.patch.object(runner, 'capture', return_value='CPU'), \
                    mock.patch('fastllm_pytools.qwen3_process_chat.prepare_request', return_value={}), \
                    mock.patch.object(runner.Suite, 'generation_alignment', failed), redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 1)


class ReferenceGenerationTests(unittest.TestCase):
    def test_public_generate_uses_shared_ids_and_unmodified_raw_logits(self):
        """/** 参考必须调用公开generate，动态KV且禁用EOS，并导出原始logits而非处理后scores。 */"""
        original = report([[5., 4., 3., 2., 1., 0.], [0., 1., 2., 3., 4., 5.]])
        model = mock.Mock()
        model.get_input_embeddings.return_value.weight.device = 'cuda:0'
        result = mock.MagicMock()
        result.sequences.__getitem__.return_value.tolist.return_value = original['token_ids']
        result.logits = []
        for step in original['steps']:
            tensor = mock.MagicMock()
            tensor.__getitem__.return_value.float.return_value.cpu.return_value.tolist.return_value = step['logits']
            result.logits.append(tensor)
        model.generate.return_value = result
        torch = SimpleNamespace(manual_seed=mock.Mock(), tensor=mock.Mock(return_value=SimpleNamespace(shape=(1, 1))),
                                ones_like=mock.Mock(return_value='mask'), inference_mode=lambda: nullcontext())
        transformers = SimpleNamespace(GenerationConfig=mock.Mock(side_effect=lambda **kw: SimpleNamespace(**kw)))
        tokenizer = SimpleNamespace(pad_token_id=None, decode=lambda tokens, **kw: str(tokens))
        before = copy.deepcopy(original)
        output, = generation_reference(model, tokenizer, torch, transformers, [original['request']])
        self.assertEqual(original, before)
        self.assertEqual(output['token_ids'], original['token_ids'])
        self.assertEqual([row['logits'] for row in output['steps']], [row['logits'] for row in original['steps']])
        self.assertIsNone(output['teacher_forced_tokens'])
        torch.tensor.assert_called_once_with([original['request']['token_ids']], device='cuda:0')
        model.generate.assert_called_once()
        kwargs = model.generate.call_args.kwargs
        config = kwargs['generation_config']
        self.assertIs(kwargs['use_model_defaults'], False)
        self.assertTrue(config.output_logits)
        self.assertTrue(config.use_cache)
        self.assertTrue(config.disable_compile)
        self.assertEqual(config.cache_implementation, 'dynamic')
        self.assertFalse(config.do_sample)
        self.assertEqual(config.num_beams, 1)
        self.assertEqual((config.eos_token_id, kwargs['eos_token_id']), ([], []))
        self.assertEqual(config.max_new_tokens, 2)
        self.assertEqual(runner.compare_generation(output, original)['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
