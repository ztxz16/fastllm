"""生成验收规则的纯Python边界检查；不导入GPU库、不编译。"""

import copy
import math
import unittest

from generation_alignment import candidate_check, compare_generation, top_logprobs, validate_policy


def report(vectors, tokens=None):
    """/** 用完整原始logits构造固定长度自由生成证据，供输入与规则拒绝路径检查。 */"""
    tokens = tokens or [max(range(len(row)), key=row.__getitem__) for row in vectors]
    return dict(status='PASS', finish_reason='length', teacher_forced_tokens=None,
                request=dict(token_ids=[0], max_new_tokens=len(tokens), temperature=0,
                             top_k=0, top_p=1, eos_token_ids=[], seed=1),
                token_ids=tokens, text=str(tokens),
                steps=[dict(token_id=token, logits=values) for token, values in zip(tokens, vectors)])


class AlignmentTests(unittest.TestCase):
    def test_same_output_can_pass_generation_with_numeric_diagnostic_warning(self):
        """/** 相同Token/文本可通过两个生成等级，数值超差只记录WARNING并保留误差。 */"""
        actual, expected = report([[9., 2., 1., 0., -1., -2.]]), report([[5., 1., 0., -1., -2., -3.]])
        for policy in ('exact', 'topk'):
            result = compare_generation(actual, expected, policy=policy)
            self.assertEqual(result['status'], 'PASS')
            self.assertTrue(result['exact_match'])
            self.assertEqual(result['failed_elements'], 0)
            self.assertEqual(result['logit_failed_elements'], 6)
            self.assertEqual(result['logit_diagnostic_status'], 'WARNING')
            self.assertEqual(result['max_abs_error'], 4.)
            self.assertEqual(result['mean_abs_error'], 1.5)

    def test_mutual_candidates_pass_but_exact_output_fails(self):
        """/** 两边Top-1不同但互在Top-5时，候选级通过而严格级失败。 */"""
        a, b = report([[2., 1., 0., -1., -2., -3.]]), report([[1., 2., 0., -1., -2., -3.]])
        result = compare_generation(a, b, near_tie_tol=0)
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(result['exact_match'])
        self.assertEqual(result['first_divergence_token_1based'], 1)
        self.assertEqual(result['near_tie_accepted_directions'], 0)
        self.assertEqual(compare_generation(a, b, policy='exact')['status'], 'FAIL')

    def test_missing_mutual_membership_fails_when_tolerance_disabled(self):
        """/** 禁用边界容差后，不在对方Top-K的实际选择必须失败。 */"""
        a, b = report([[10., 0., -1., -2., -3., -4.]]), report([[-4., 0., -1., -2., -3., 10.]])
        result = compare_generation(a, b, near_tie_tol=0)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['failed_elements'], 1)
        self.assertTrue(all(not check['accepted'] for check in result['candidate_checks']))

    def test_boundary_below_equal_above_and_zero(self):
        """/** 门槛边界使用包含等号的单侧比较，零容差只允许集合成员。 */"""
        cutoff = -2.0 - .1
        for score, accepted in ((math.nextafter(cutoff, -math.inf), False),
                                (cutoff, True), (math.nextafter(cutoff, math.inf), True)):
            with self.subTest(score=score):
                check = candidate_check(1, {1: score}, {2: -2.}, .1)
                self.assertEqual(check['accepted'], accepted)
                self.assertEqual(check['near_tie_accepted'], accepted)
        self.assertFalse(candidate_check(1, {1: -1.}, {2: -2.}, 0)['accepted'])
        self.assertTrue(candidate_check(1, {1: -9.}, {1: -2.}, 0)['accepted'])
        self.assertFalse(candidate_check(1, {2: -1.}, {3: -2.}, .1)['accepted'])

    def test_cross_side_cutoff_is_not_same_token_numeric_closeness(self):
        """/** 如实复现规则的局限：本侧高分可越过对方门槛，并不证明同一Token两侧接近。 */"""
        a, b = report([[10., 0., -1., -2., -3., -4.]]), report([[-4., 0., -1., -2., -3., 10.]])
        result = compare_generation(a, b)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['near_tie_accepted_directions'], 2)
        self.assertEqual(result['max_abs_error'], 14.)
        self.assertEqual(result['logit_diagnostic_status'], 'WARNING')
        self.assertFalse(result['exact_match'])

    def test_one_direction_is_not_enough(self):
        """/** 平坦分布对尖锐分布时，一侧满足门槛仍不得代替双向条件。 */"""
        a, b = report([[.01, 0., 0., 0., 0., 0.]]), report([[-100., -10., -20., -30., -40., 10.]])
        result = compare_generation(a, b, topk=1)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(sum(check['accepted'] for check in result['candidate_checks']), 1)

    def test_first_divergence_stops_comparison_but_not_evidence_validation(self):
        """/** 首次分歧后不沿不同历史计算误差，仍拒绝缺失或非法的完整生成证据。 */"""
        a = report([[2., 1., 0., -1., -2., -3.]]*3)
        b = report([[2., 1., 0., -1., -2., -3.], [1., 2., 0., -1., -2., -3.],
                    [-100., -90., -80., -70., -60., 100.]])
        result = compare_generation(a, b, near_tie_tol=0)
        self.assertEqual((result['compared_steps'], result['uncompared_steps']), (2, 1))
        self.assertEqual(result['max_abs_error'], 1.)
        b['steps'][2]['logits'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            compare_generation(a, b)

    def test_prompt_settings_length_shape_and_forcing_are_rejected(self):
        """/** 输入条件或实际执行语义不一致不能转换成宽松候选检查。 */"""
        good = report([[5., 4., 3., 2., 1., 0.]])
        mutations = [lambda r: r['request'].update(token_ids=[1]),
                     lambda r: r['request'].update(seed=2),
                     lambda r: r['request'].update(eos_token_ids=[1]),
                     lambda r: r['request'].update(temperature=.1),
                     lambda r: r['request'].update(top_k=5),
                     lambda r: r['request'].update(top_p=.9),
                     lambda r: r.update(teacher_forced_tokens=[]),
                     lambda r: r.update(finish_reason='eos'),
                     lambda r: r.update(status='FAIL'),
                     lambda r: r.update(steps=[]),
                     lambda r: r.update(token_ids=[]),
                     lambda r: r.update(text=None),
                     lambda r: r['steps'][0]['logits'].pop(),
                     lambda r: r['steps'][0].update(token_id=5),
                     lambda r: r.update(token_ids=[5])]
        for index, mutate in enumerate(mutations):
            bad = copy.deepcopy(good)
            mutate(bad)
            with self.subTest(index=index), self.assertRaises(ValueError):
                compare_generation(bad, good)
        for value in (float('nan'), float('inf'), -float('inf')):
            bad = copy.deepcopy(good)
            bad['steps'][0]['logits'][2] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonfinite'):
                compare_generation(bad, good)

    def test_exact_text_requirement_is_separate(self):
        """/** Token一致但文本不同必须使严格等级失败，候选等级明确记录文本差异。 */"""
        a = report([[5., 4., 3., 2., 1., 0.]])
        b = copy.deepcopy(a)
        b['text'] = 'different decoder'
        self.assertEqual(compare_generation(a, b, policy='exact')['status'], 'FAIL')
        result = compare_generation(a, b)
        self.assertFalse(result['text_match'])
        self.assertFalse(result['exact_match'])

    def test_policy_validation_and_normalization(self):
        """/** 拒绝非法参数，稳定归一化不得因大logits溢出或整体平移而改变。 */"""
        for policy, k, tolerance in (('unknown', 5, .1), ('topk', 0, .1), ('topk', True, .1),
                                     ('topk', 5, -.1), ('topk', 5, math.nan), ('topk', 5, math.inf)):
            with self.subTest(policy=policy, k=k, tolerance=tolerance), self.assertRaises(ValueError):
                validate_policy(policy, k, tolerance)
        scores = top_logprobs([1000., 999., 998.], 3)
        self.assertAlmostEqual(sum(math.exp(value) for value in scores.values()), 1.)
        self.assertEqual(scores, top_logprobs([0., -1., -2.], 3))
        self.assertEqual(list(top_logprobs([1.]*6, 5)), [0, 1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, 'topk exceeds'):
            compare_generation(report([[1., 0.]]), report([[1., 0.]]))


if __name__ == '__main__':
    unittest.main()
