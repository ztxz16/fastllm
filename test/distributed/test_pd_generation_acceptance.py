"""正式PD HTTP判定的纯Python检查；不发送HTTP、不加载模型或编译。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent / 'pd'))
import check_generation as checker


class PDGenerationAcceptanceTests(unittest.TestCase):
    def test_exact_acceptance_records_level(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            actual = dict(token_ids=[1,2])
            result = checker.compare_output(output, 'same', {}, actual, actual, 2)
            self.assertEqual(result['accepted_by'], 'exact')
            self.assertEqual(result['top5_status'], 'SKIP')
            self.assertEqual(json.loads((output/'same.evidence.json').read_text())['comparison']['status'], 'PASS')

    def test_missing_candidates_preserve_warning_and_fail_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            console = io.StringIO()
            with redirect_stdout(console), self.assertRaisesRegex(AssertionError, 'candidate evidence missing'):
                checker.compare_output(output, 'different', {}, dict(token_ids=[1,2]), dict(token_ids=[1,3]), 2)
            data = json.loads((output/'different.evidence.json').read_text())
            self.assertEqual(data['comparison']['exact_status'], 'WARNING')
            self.assertEqual(data['comparison']['top5_status'], 'NOT_RUN')
            self.assertEqual(data['first_divergence'], 1)
            self.assertIn('WARNING:', console.getvalue())
            self.assertNotIn('numeric', data['comparison'])

    def test_invalid_length_remains_functional_failure_with_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with self.assertRaisesRegex(ValueError, 'output length'):
                checker.compare_output(output, 'short', {}, dict(token_ids=[1]), dict(token_ids=[1,2]), 2)
            self.assertTrue((output/'short.evidence.json').exists())

    def test_formal_compare_reuses_actual_http_results(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(archive=None, record_reference=False, reference='reference', router='router', token='test')
            result = dict(token_ids=list(range(8)), elapsed_s=.12345)
            with mock.patch.object(checker, 'request', return_value=result) as request:
                row = checker.compare(args, Path(temp), 'request', [7])
            self.assertEqual(request.call_count, 2)
            self.assertEqual(row['accepted_by'], 'exact')
            self.assertEqual(row['latency_s'], .12345)


if __name__ == '__main__':
    unittest.main()
