"""前缀分歧诊断的纯Python回归；不加载模型、不编译，不代表GPU数值结果。"""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import diagnose_qwen3_prefix as diag


REQUEST = dict(token_ids=[1, 2], max_new_tokens=3, temperature=0.0, top_k=0, top_p=1.0, seed=0, eos_token_ids=[])


def trace(warm=False, graph=False):
    """/** 构造首次分歧前数值接近、分歧后明显不同的样本，防止错误比较不同输入历史。 */"""
    values = [[2., 0., 0.], [0., .998, 1.001], [1000., 0., 0.]] if warm else [
        [2., 0., 0.], [0., 1., .999], [0., 2., 0.]]
    ids = [max(range(3), key=row.__getitem__) for row in values]
    prefix = dict(enabled=True, entries=1, device_bytes=32, lookups=2 if warm else 1,
                  hits=int(warm), reused_tokens=int(warm), last_reused_tokens=int(warm),
                  last_computed_tokens=1 if warm else 2, evictions=0, clears=0)
    state = dict(enabled=graph, active=False, captures=int(graph), replays=2 if graph else 0,
                 warmups=int(graph), eager_decodes=0 if graph else 2, workspace_bytes=0, prefix_cache=prefix)
    return dict(status='PASS', backend=diag.BACKEND, request=copy.deepcopy(REQUEST), token_ids=ids,
                steps=[dict(token_id=token, logits=values[i]) for i, token in enumerate(ids)], trace_logits=True,
                finish_reason='length', cache_after_generate=[diag.EMPTY.copy()], graph=[state],
                execution='decode_cuda_graph' if graph else 'eager')


class PrefixDiagnosticTests(unittest.TestCase):
    def test_compares_only_identical_histories_including_divergent_step(self):
        row = diag.compare_traces('cold_warm', trace(True), trace())
        self.assertEqual(row['status'], 'PASS')
        self.assertEqual((row['compared_steps'], row['compared_elements'], row['first_divergence_token_1based']), (2, 6, 2))
        self.assertEqual((row['token_mismatches'], row['failed_elements']), (2, 0))
        self.assertAlmostEqual(row['max_abs_error'], .002)
        self.assertAlmostEqual(row['mean_abs_error'], .004 / 6)
        self.assertEqual(row['first_divergence']['actual']['token_ids'], [2, 1])
        self.assertAlmostEqual(row['first_divergence']['actual']['margin'], .003)

    def test_same_state_token_mismatch_still_fails_with_small_logits_error(self):
        row = diag.compare_traces('repeat', trace(True), trace(), exact_tokens=True)
        self.assertEqual(row['status'], 'FAIL')
        self.assertEqual(row['failed_elements'], 0)
        self.assertIn('same-cache-state', row['reason'])

    def test_large_error_before_divergence_fails(self):
        actual = trace(True)
        actual['steps'][0]['logits'][0] += 1
        row = diag.compare_traces('bad', actual, trace())
        self.assertEqual((row['status'], row['failed_elements'], row['max_abs_error']), ('FAIL', 1, 1.))

    def test_repeated_trace_checks_entire_output(self):
        row = diag.compare_traces('repeat', trace(True), trace(True), exact_tokens=True)
        self.assertEqual((row['status'], row['compared_steps'], row['compared_elements']), ('PASS', 3, 9))
        self.assertEqual(row['max_abs_error'], 0.)

    def test_trace_rejects_bad_greedy_sampling_nonfinite_or_missing_evidence(self):
        for graph in (False, True):
            diag.validate_trace(trace(True, graph), REQUEST, 3, graph, 1)
        corruptions = [lambda r: r['steps'][0]['logits'].__setitem__(0, float('nan')),
                       lambda r: r['token_ids'].__setitem__(0, 2),
                       lambda r: r.__setitem__('trace_logits', False),
                       lambda r: r.__setitem__('execution', 'eager'),
                       lambda r: r['graph'][0]['prefix_cache'].__setitem__('last_reused_tokens', 0),
                       lambda r: r['steps'][0].__setitem__('logits', [])]
        for corrupt in corruptions:
            with self.subTest(corrupt=corrupt):
                report = trace(True, True)
                corrupt(report)
                with self.assertRaises(RuntimeError):
                    diag.validate_trace(report, REQUEST, 3, True, 1)

    def test_shape_and_request_changes_rejected(self):
        for change in ('request', 'shape'):
            report = trace()
            if change == 'request':
                report['request']['token_ids'] = [2, 1]
            else:
                report['steps'][0]['logits'] = []
            with self.assertRaises(ValueError):
                diag.compare_traces('bad', report, trace())

    def invoke(self, factory, request=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / 'config.json').write_text(json.dumps(dict(vocab_size=3)))
        (root / 'CMakeCache.txt').write_text('CUDA_ARCH:UNINITIALIZED=70\n')
        (root / 'request.json').write_text(json.dumps(request or REQUEST))
        flags = ['--model', str(root), '--generate', str(root / 'generate'), '--worker', str(root / 'worker'),
                 '--build-cache', str(root / 'CMakeCache.txt'), '--request-file', str(root / 'request.json'),
                 '--output', str(root / 'reports')]
        with mock.patch.object(diag, 'Qwen3ProcessSession', side_effect=factory) as create, \
                mock.patch.object(diag, 'capture', return_value='Python fixture; no GPU'), redirect_stdout(io.StringIO()):
            result = diag.main(flags)
        output = next((root / 'reports').iterdir())
        return result, output, json.loads((output / 'summary.json').read_text()), create

    def test_full_workflow_records_original_request_and_partial_scope(self):
        sessions = []

        def factory(*args, **kwargs):
            session = mock.MagicMock()
            session.__enter__.return_value = session
            session.generate.side_effect = [trace(warm, kwargs['cuda_graph']) for warm in (False, True, True, False)]
            sessions.append(session)
            return session

        status, output, report, create = self.invoke(factory)
        self.assertEqual((status, report['status'], report['complete'], len(report['cases'])), (0, 'PASS', True, 8))
        self.assertEqual(report['metadata']['request'], REQUEST)
        self.assertEqual(len(list((output / 'raw').glob('*.json'))), 8)
        for call, session in zip(create.call_args_list, sessions):
            self.assertTrue(call.kwargs['trace'])
            self.assertTrue(call.kwargs['prefix_cache'])
            self.assertFalse(call.kwargs['request_log'])
            self.assertEqual(session.generate.call_args_list, [mock.call(REQUEST)] * 4)
            session.clear_prefix_cache.assert_called_once()
            session.__exit__.assert_called_once()
        for extension in ('json', 'csv', 'md', 'xlsx'):
            self.assertTrue((output / f'summary.{extension}').exists())
        with zipfile.ZipFile(output / 'summary.xlsx') as workbook:
            self.assertIn('Cases', workbook.read('xl/workbook.xml').decode())

    def test_failure_preserves_raw_output_and_returns_nonzero(self):
        def factory(*args, **kwargs):
            session = mock.MagicMock()
            session.__enter__.return_value = session
            session.generate.side_effect = [trace(), trace()]
            return session

        status, output, report, _ = self.invoke(factory)
        self.assertEqual((status, report['status'], report['complete']), (1, 'FAIL', False))
        self.assertEqual(report['cases'][0]['case'], 'eager/warm')
        self.assertIn('reused_tokens=1', report['cases'][0]['reason'])
        self.assertTrue((output / 'raw/eager_warm.json').exists())

    def test_invalid_sampling_rejected_before_session(self):
        status, _, report, create = self.invoke(None, dict(REQUEST, temperature=.1))
        self.assertEqual(status, 1)
        self.assertIn('greedy/no-EOS', report['cases'][0]['reason'])
        create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
