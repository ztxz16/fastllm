"""生成三级接受规则的纯Python检查；不加载GPU库、不编译。"""

import copy
import math
import unittest
from unittest import mock

from generation_alignment import (candidate_check, compare_generation, compare_generation_all,
                                  compare_token_ids, top_logprobs, validate_policy)


def report(vectors, tokens=None):
    """/** 构造完整自由生成证据；保持实际Token与对应贪心Logits一致。 */"""
    tokens = tokens or [max(range(len(row)), key=row.__getitem__) for row in vectors]
    return dict(status='PASS', finish_reason='length', teacher_forced_tokens=None,
                request=dict(token_ids=[0], max_new_tokens=len(tokens), temperature=0,
                             top_k=0, top_p=1, eos_token_ids=[], seed=1),
                token_ids=tokens, text=str(tokens),
                steps=[dict(token_id=token, logits=values) for token, values in zip(tokens, vectors)])


def all_failure():
    """/** 两侧选择互不在Top5，且一侧选中Token也未达到宽松门槛。 */"""
    return report([[.01] + [0.] * 19]), report([[-100.] * 15 + [10.] * 5])


class AlignmentTests(unittest.TestCase):
    def test_exact_short_circuits_candidates_and_has_no_numeric(self):
        actual, expected = report([[9., 2., 1., 0., -1., -2.]]), report([[5., 1., 0., -1., -2., -3.]])
        with mock.patch('generation_alignment.top_logprobs', side_effect=AssertionError('candidate executed')):
            result = compare_generation_all(actual, expected)
        self.assertEqual((result['status'], result['accepted_by']), ('PASS', 'exact'))
        self.assertEqual([result[k + '_status'] for k in ('exact','top5','top5_near_tie')], ['PASS','SKIP','SKIP'])
        self.assertEqual(result['warnings'], [])
        for key in ('logit_diagnostics','logit_diagnostic_status','max_abs_error','mean_abs_error','logit_failed_elements'):
            self.assertNotIn(key, result)

    def test_top5_accepts_and_preserves_exact_warning(self):
        a, b = report([[2., 1., 0., -1., -2., -3.]]), report([[1., 2., 0., -1., -2., -3.]])
        before = copy.deepcopy((a, b))
        with mock.patch('generation_alignment.top_logprobs', wraps=top_logprobs) as normalize:
            result = compare_generation_all(a, b)
        self.assertEqual(normalize.call_count, 2)
        self.assertEqual((result['status'], result['accepted_by']), ('PASS', 'top5'))
        self.assertEqual([result[k + '_status'] for k in ('exact','top5','top5_near_tie')], ['WARNING','PASS','SKIP'])
        self.assertFalse(result['exact_match'])
        self.assertEqual(result['warning_count'], 1)
        self.assertEqual(result['failed_elements'], 0)
        self.assertEqual((a, b), before)

    def test_near_tie_accepts_after_two_warnings_without_recomputing_top5(self):
        a, b = report([[10., 0., -1., -2., -3., -4.]]), report([[-4., 0., -1., -2., -3., 10.]])
        with mock.patch('generation_alignment.top_logprobs', wraps=top_logprobs) as normalize:
            result = compare_generation_all(a, b)
        self.assertEqual(normalize.call_count, 2)
        self.assertEqual((result['status'], result['accepted_by']), ('PASS', 'top5_near_tie'))
        self.assertEqual(result['warning_count'], 2)
        self.assertEqual(result['near_tie_accepted_directions'], 2)
        self.assertEqual(result['exact_status'], 'WARNING')
        self.assertEqual(result['top5_status'], 'WARNING')

    def test_three_warnings_mean_overall_fail(self):
        result = compare_generation_all(*all_failure())
        self.assertEqual(result['status'], 'FAIL')
        self.assertIsNone(result['accepted_by'])
        self.assertEqual(result['warning_count'], 3)
        self.assertEqual(result['failed_elements'], 1)
        self.assertEqual([v['status'] for v in result['comparisons'].values()], ['WARNING'] * 3)

    def test_single_level_mismatch_is_warning(self):
        a, b = all_failure()
        self.assertEqual(compare_generation(a, b, policy='exact')['status'], 'WARNING')
        self.assertEqual(compare_generation(a, b, near_tie_tol=0)['status'], 'WARNING')

    def test_first_divergence_limits_candidate_history(self):
        a = report([[2., 1., 0., -1., -2., -3.]] * 3)
        b = report([[2., 1., 0., -1., -2., -3.], [1., 2., 0., -1., -2., -3.], [-100., -90., -80., -70., -60., 100.]])
        result = compare_generation_all(a, b)
        self.assertEqual((result['compared_steps'], result['uncompared_steps']), (2, 1))
        self.assertEqual(result['first_divergence_token_1based'], 2)
        self.assertEqual(result['accepted_by'], 'top5')

    def test_invalid_tail_is_not_hidden_by_early_acceptance(self):
        a, b = all_failure()
        a['request']['max_new_tokens'] = b['request']['max_new_tokens'] = 2
        for r in (a, b):
            r['token_ids'] *= 2
            r['steps'] = copy.deepcopy(r['steps'] * 2)
        a['steps'][1]['logits'][0] = math.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            compare_generation_all(a, b)

    def test_exact_still_validates_output(self):
        good = report([[5., 4., 3., 2., 1., 0.]])
        mutations = [lambda r: r['request'].update(token_ids=[1]),
                     lambda r: r['request'].update(seed=2),
                     lambda r: r['request'].update(eos_token_ids=[1]),
                     lambda r: r['request'].update(temperature=.1),
                     lambda r: r.update(teacher_forced_tokens=[]),
                     lambda r: r.update(finish_reason='eos'), lambda r: r.update(status='FAIL'),
                     lambda r: r.update(steps=[]), lambda r: r.update(token_ids=[]),
                     lambda r: r.update(text=None), lambda r: r['steps'][0]['logits'].pop(),
                     lambda r: r['steps'][0].update(token_id=5)]
        for i, mutate in enumerate(mutations):
            bad = copy.deepcopy(good); mutate(bad)
            with self.subTest(index=i), self.assertRaises(ValueError):
                compare_generation_all(bad, good)

    def test_same_tokens_different_text_keeps_warning_and_acceptance_level(self):
        a = report([[5., 4., 3., 2., 1., 0.]])
        b = copy.deepcopy(a); b['text'] = 'different decoder'
        result = compare_generation_all(a, b)
        self.assertEqual(result['exact_status'], 'WARNING')
        self.assertEqual(result['accepted_by'], 'top5')
        self.assertFalse(result['text_match'])

    def test_near_tie_boundary(self):
        cutoff = -2.0 - .1
        for score, accepted in ((math.nextafter(cutoff, -math.inf), False), (cutoff, True),
                                (math.nextafter(cutoff, math.inf), True)):
            with self.subTest(score=score):
                self.assertEqual(candidate_check(1, {1: score}, {2: -2.}, .1)['accepted'], accepted)
        self.assertFalse(candidate_check(1, {1: -1.}, {2: -2.}, 0)['accepted'])

    def test_policy_validation_and_stable_normalization(self):
        for policy, k, tolerance in (('unknown',5,.1), ('topk',0,.1), ('topk',True,.1),
                                     ('topk',5,-.1), ('topk',5,math.nan)):
            with self.subTest(policy=policy, k=k, tolerance=tolerance), self.assertRaises(ValueError):
                validate_policy(policy, k, tolerance)
        self.assertEqual(top_logprobs([1000.,999.,998.],3), top_logprobs([0.,-1.,-2.],3))
        self.assertEqual(list(top_logprobs([1.]*6,5)), [0,1,2,3,4])

    def test_token_only_exact_can_accept_without_candidates(self):
        result = compare_token_ids([1,2],[1,2],2)
        self.assertEqual((result['status'], result['accepted_by']), ('PASS','exact'))
        self.assertEqual(result['top5_status'], 'SKIP')

    def test_token_only_mismatch_does_not_invent_candidate_pass(self):
        result = compare_token_ids([1,2],[1,3],2)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['exact_status'], 'WARNING')
        self.assertEqual(result['top5_status'], 'NOT_RUN')
        self.assertIn('evidence missing', result['reason'])

    def test_token_only_invalid_output_is_functional_failure(self):
        for value in ([1], [1,True], [1,-1], [1,'2']):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare_token_ids(value, [1,2], 2)


if __name__ == '__main__':
    unittest.main()
