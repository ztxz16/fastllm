"""真实MoE验收器的CPU检查；不启动原生进程、GPU或编译。"""

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


class RealComparisonTests(unittest.TestCase):
    def routing(self):
        """/** 构造中间两专家严格同分、集合相同但槽位相反的两侧原始证据。 */"""
        expected = dict(selected_experts=[0, 1, 2], routing_weights=[0.5, 0.25, 0.25],
                        router_logits=[1, 0, 0, -1, -2], output=[0.125],
                        routed_output=[0.075], shared_output=[0.05])
        actual = copy.deepcopy(expected)
        actual['selected_experts'] = [0.0, 2.0, 1.0]
        return actual, expected

    def test_tied_permutation_preserves_raw_evidence_and_pairs_weights(self):
        """/** 多Token对齐必须按专家ID配对权重，不能改写原始ID或直接排序权重。 */"""
        actual, expected = self.routing()
        actual['routing_weights'] = [0.5, 0.24, 0.26]
        for values in (actual, expected):
            for key in ('router_logits', 'selected_experts', 'routing_weights'):
                values[key] *= 2
        before = copy.deepcopy((actual, expected))
        weights, moved = runner.align_tied_routing(actual, expected, 2, 5, 3)
        self.assertEqual(weights, [0.5, 0.26, 0.24]*2)
        self.assertEqual(moved, 4)
        self.assertEqual((actual, expected), before)

    def test_changed_set_is_rejected_even_at_tied_cutoff(self):
        """/** 截断点同分但选择集合不同仍失败，不能用近似分数替代专家身份。 */"""
        actual, expected = self.routing()
        actual['selected_experts'] = [0, 3, 1]
        actual['router_logits'][3] = expected['router_logits'][3] = 0
        with self.assertRaisesRegex(AssertionError, 'different sets'):
            runner.align_tied_routing(actual, expected, 1, 5, 3)

    def test_non_tied_or_only_one_side_tied_is_rejected(self):
        """/** 两侧各自必须严格同分，微小分差也不能被容差当成同分。 */"""
        for side in (0, 1):
            actual, expected = self.routing()
            (actual, expected)[side]['router_logits'][1] = 1e-8
            with self.subTest(side=side), self.assertRaisesRegex(AssertionError, 'non-tied permutation'):
                runner.align_tied_routing(actual, expected, 1, 5, 3)

    def test_invalid_ids_duplicates_shapes_and_nonfinite_logits_fail(self):
        """/** 拒绝非法ID、重复专家、缺失槽位及非有限分数，不能静默对齐。 */"""
        for value in (-1, 5, 1.5, float('nan'), True):
            actual, expected = self.routing()
            actual['selected_experts'][1] = value
            with self.subTest(value=value), self.assertRaisesRegex(AssertionError, 'invalid ID'):
                runner.align_tied_routing(actual, expected, 1, 5, 3)
        actual, expected = self.routing()
        actual['selected_experts'] = [0, 1, 1]
        with self.assertRaisesRegex(AssertionError, 'duplicate expert'):
            runner.align_tied_routing(actual, expected, 1, 5, 3)
        actual, expected = self.routing()
        actual['routing_weights'].pop()
        with self.assertRaisesRegex(AssertionError, 'shape mismatch'):
            runner.align_tied_routing(actual, expected, 1, 5, 3)
        actual, expected = self.routing()
        expected['router_logits'][1] = float('inf')
        with self.assertRaisesRegex(AssertionError, 'nonfinite'):
            runner.align_tied_routing(actual, expected, 1, 5, 3)

    def test_top_one_and_idle_rank(self):
        """/** Top-1及空rank使用同样形状约束，不制造虚假的置换计数。 */"""
        for tokens in (0, 1):
            values = dict(selected_experts=[0]*tokens, routing_weights=[1.0]*tokens,
                          router_logits=[1.0]*tokens)
            self.assertEqual(runner.align_tied_routing(values, values, tokens, 1, 1),
                             ([1.0]*tokens, 0))

    def test_real_layer_verifier_excludes_ids_but_rejects_numeric_errors(self):
        """/** 走真实验收入口与四格式报告；同分置换可过，错误权重/层输出仍必须失败。 */"""
        for corrupt in (None, 'routing_weights', 'output'):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                model = root / 'model'
                model.mkdir()
                (model / 'config.json').write_text(json.dumps(dict(num_experts=5, num_experts_per_tok=3)))
                actual, expected = self.routing()
                if corrupt:
                    actual[corrupt][0] += 1.0
                layer = dict(name='real_layer_9', layer=9, stage=0, hidden=1,
                             inputs=[dict(hidden=[1.0]) for _ in range(4)],
                             reference=[copy.deepcopy(expected) for _ in range(4)])
                native = dict(status='PASS', workers_reaped=True, workers=[],
                              results=[dict(task=layer['name'], ranks=[copy.deepcopy(actual) for _ in range(4)])])
                suite = runner.Suite.__new__(runner.Suite)
                suite.output, suite.rows, suite.metadata = root, [], {}
                suite.args = SimpleNamespace(binary=Path('/unused/generate'), worker=Path('/unused/worker'), timeout=1)
                with mock.patch.object(runner, 'case_process', return_value=nullcontext(object())), \
                     mock.patch.object(runner, 'wait_case', return_value=0), \
                     mock.patch.object(runner, 'parse_report', return_value=native), redirect_stdout(io.StringIO()):
                    suite.real_layers(model, [layer])
                row = suite.rows[0]
                self.assertEqual(row['status'], 'FAIL' if corrupt else 'PASS')
                self.assertEqual(row['max_abs_error'], 1.0 if corrupt else 0.0)
                self.assertEqual(row['failed_elements'], 4 if corrupt else 0)
                for suffix in ('json', 'md', 'csv', 'xlsx'):
                    self.assertTrue((root / f'summary.{suffix}').is_file())
                if not corrupt:
                    evidence = json.loads((root / 'real_all_layers_router_experts_outputs.json').read_text())
                    self.assertEqual(evidence['tied_routing_slots'], 8)

    def test_same_greedy_token_does_not_hide_logit_failure(self):
        """/** 重现贪心相同但完整logits超差，拒绝并在错误中给出最大误差与词表位置。 */"""
        report = dict(request=dict(token_ids=[1]), steps=[dict(logits=[5.17468929, -0.236196697],
                      token_id=0, execution='eager')])
        metrics = dict(max=0.0, sum=0.0, count=0, failed=0)
        with self.assertRaisesRegex(AssertionError, r'2 failed logits; max_abs=.*worst_vocab_id=0'):
            runner.compare_trace(report, lambda _: [5.06640625, -0.33984375], metrics, .05, .01, 'real')
        self.assertEqual(metrics['failed'], 2)


if __name__ == '__main__':
    unittest.main()
