"""生成级HF对齐规则；纯Python检查，不加载模型、不修改推理或路由。"""

import heapq
import math


POLICY_SOURCE = {
    'url': 'https://github.com/vllm-project/vllm/blob/main/tests/models/utils.py',
    'utils_blob_sha': '0bec9ca9bec88f10d61770d821143f3528dbfdfe',
    'test_common_blob_sha': '07585525e9151e815106ea27bc2b4c55cfee0075',
    'scope': 'generated tokens: mutual top-k or cross-side near-tie cutoff; first divergence only',
    'normalization': 'shared Python float64 log_softmax of raw logits; top-k ties use lower token ID',
}


def validate_policy(policy, topk, near_tie_tol):
    """/** 拒绝未知策略、非法K或容差；参数不能随输出是否通过自动调整。 */"""
    if policy not in ('exact', 'topk'):
        raise ValueError('alignment policy must be exact or topk')
    if type(topk) is not int or topk < 1:
        raise ValueError('alignment topk must be a positive integer')
    if not math.isfinite(near_tie_tol) or near_tie_tol < 0:
        raise ValueError('near-tie tolerance must be finite and non-negative')


def validate_request(request):
    """/** 生成对照只接受固定长度、无EOS截断的贪心请求，拒绝混入采样配置。 */"""
    ids = request['token_ids']
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError('invalid prompt token IDs')
    if type(request['max_new_tokens']) is not int or request['max_new_tokens'] < 1:
        raise ValueError('generation length must be a positive integer')
    if (request['temperature'] != 0 or request['top_k'] != 0 or request['top_p'] != 1 or
            request['eos_token_ids'] != []):
        raise ValueError('alignment requires greedy, unfiltered, fixed-length generation with EOS disabled')


def validate_output(report, topk):
    """/**
     * 检查完整自由生成证据，不能靠zip截断、非有限数或教师强制轨迹取得通过。
     * @param report 包含请求、实际生成Token、文本和每步完整词表logits的报告。
     * @param topk 候选数量，必须不大于词表大小。
     * @return 词表大小；长度、形状、贪心选择或执行状态非法时抛出ValueError。
     */"""
    validate_request(report['request'])
    if report.get('status') != 'PASS' or report.get('finish_reason') != 'length':
        raise ValueError('generation did not complete successfully at the requested length')
    if report.get('teacher_forced_tokens') is not None:
        raise ValueError('teacher-forced traces are diagnostics, not independent generation')
    tokens, steps = report['token_ids'], report['steps']
    if len(tokens) != report['request']['max_new_tokens'] or len(steps) != len(tokens):
        raise ValueError('missing generated tokens or logits steps')
    if not isinstance(report['text'], str):
        raise ValueError('decoded text is required for exact comparison')
    vocab = len(steps[0]['logits'])
    if vocab < topk:
        raise ValueError('topk exceeds vocabulary size')
    for index, (token, step) in enumerate(zip(tokens, steps)):
        logits = step['logits']
        if len(logits) != vocab or not all(math.isfinite(value) for value in logits):
            raise ValueError(f'step={index}: invalid logits shape or nonfinite value')
        if type(token) is not int or not 0 <= token < vocab or step['token_id'] != token:
            raise ValueError(f'step={index}: invalid generated token ID')
        if logits[token] != max(logits):
            raise ValueError(f'step={index}: token is not a greedy maximum')
    if any(token >= vocab for token in report['request']['token_ids']):
        raise ValueError('prompt token outside vocabulary')
    return vocab


def top_logprobs(logits, topk):
    """/** 共享稳定log-softmax口径，仅导出Top-K；同分按Token ID升序稳定选择。 */"""
    maximum = max(logits)
    normalizer = math.log(math.fsum(math.exp(value - maximum) for value in logits))
    ids = heapq.nlargest(topk, range(len(logits)), key=lambda token: (logits[token], -token))
    return {token: (logits[token] - maximum) - normalizer for token in ids}


def candidate_check(token, own, other, tolerance):
    """/**
     * 对照vLLM生成候选规则：先查对方集合，再比较本侧分数与对方末位门槛。
     * 这是单侧下限判断，不是同一Token在两边的logprob误差，也不证明实际近同分。
     * @param token 本侧实际贪心选择。
     * @param own 本侧Top-K的Token到归一化logprob映射。
     * @param other 对方Top-K映射。
     * @param tolerance 大于零才启用额外门槛；零表示严格互包含。
     * @return 包含通过原因、实际分数及门槛的可审计记录。
     */"""
    cutoff = min(other.values())
    present = token in other
    near = not present and tolerance > 0 and token in own and own[token] >= cutoff - tolerance
    return dict(token_id=token, accepted=present or near, in_other_topk=present,
                near_tie_accepted=near, own_logprob=own.get(token),
                other_cutoff_logprob=cutoff, relaxed_cutoff_logprob=cutoff - tolerance)


def compare_generation(actual, expected, *, policy='topk', topk=5, near_tie_tol=0.1,
                       diagnostic_atol=0.05, diagnostic_rtol=0.01):
    """/**
     * 比较独立生成的完整输出，候选检查只到首次分歧；全量logits误差仅诊断相同历史。
     * 两侧共用原始Prompt ID、固定长度和贪心语义。即使候选容差通过，也保留严格
     * 不一致、首个分歧及未比较尾部长度；数值超差记录为非阻断WARNING并保留误差。
     * @param actual FT公开会话的自由生成报告，logits为每步完整词表的一维浮点列表。
     * @param expected HF公开generate的独立生成报告，字段含义相同。
     * @param policy exact要求Token与文本完全一致；topk要求双向候选规则通过。
     * @param topk 候选数，默认5；不会影响实际生成采样。
     * @param near_tie_tol 候选门槛的logprob余量，默认0.1，可显式设0。
     * @param diagnostic_atol 原始logits绝对误差诊断阈值，不作为生成验收门槛。
     * @param diagnostic_rtol 原始logits相对误差诊断阈值。
     * @return 选定策略的PASS/FAIL、严格一致性、候选证据和独立数值诊断。
     */"""
    validate_policy(policy, topk, near_tie_tol)
    for value in (diagnostic_atol, diagnostic_rtol):
        if not math.isfinite(value) or value < 0:
            raise ValueError('diagnostic tolerances must be finite and non-negative')
    actual_vocab = validate_output(actual, topk)
    expected_vocab = validate_output(expected, topk)
    if actual_vocab != expected_vocab or actual['request'] != expected['request']:
        raise ValueError('model vocabulary, prompt IDs or generation configuration differ')

    token_match = actual['token_ids'] == expected['token_ids']
    text_match = actual['text'] == expected['text']
    exact_match = token_match and text_match
    divergence, checks, diagnostics = None, [], []
    error_sum, error_count, error_max, bad_logits = 0.0, 0, 0.0, 0
    candidate_match = True
    for index, (left, right) in enumerate(zip(actual['steps'], expected['steps'])):
        errors = [abs(a - b) for a, b in zip(left['logits'], right['logits'])]
        if not all(math.isfinite(error) for error in errors):
            raise ValueError(f'step={index}: nonfinite logits difference')
        bad = sum(error > diagnostic_atol + diagnostic_rtol * abs(value)
                  for error, value in zip(errors, right['logits']))
        worst = max(range(len(errors)), key=errors.__getitem__)
        diagnostics.append(dict(step=index, max_abs_error=errors[worst],
                                mean_abs_error=math.fsum(errors) / len(errors),
                                failed_elements=bad, worst_vocab_id=worst))
        error_sum += math.fsum(errors)
        error_count += len(errors)
        error_max = max(error_max, errors[worst])
        bad_logits += bad
        if left['token_id'] != right['token_id']:
            divergence = index + 1
            a, b = top_logprobs(left['logits'], topk), top_logprobs(right['logits'], topk)
            checks = [dict(direction='ft_to_hf', **candidate_check(left['token_id'], a, b, near_tie_tol)),
                      dict(direction='hf_to_ft', **candidate_check(right['token_id'], b, a, near_tie_tol))]
            candidate_match = all(check['accepted'] for check in checks)
            break

    passed = exact_match if policy == 'exact' else candidate_match
    reason = '' if passed else ('generated token IDs or text differ' if policy == 'exact'
                               else f'first divergence at token {divergence}: mutual candidate check failed')
    return dict(status='PASS' if passed else 'FAIL', alignment_policy=policy, topk=topk,
                near_tie_tol=near_tie_tol, exact_match=exact_match, token_match=token_match,
                text_match=text_match, candidate_match=candidate_match,
                first_divergence_token_1based=divergence, compared_steps=len(diagnostics),
                uncompared_steps=len(actual['steps']) - len(diagnostics),
                near_tie_accepted_directions=sum(check['near_tie_accepted'] for check in checks),
                max_abs_error=error_max, mean_abs_error=error_sum / error_count,
                failed_elements=int(not passed), logit_failed_elements=bad_logits,
                logit_diagnostic_status='WARNING' if bad_logits else 'PASS', reason=reason,
                candidate_checks=checks, logit_diagnostics=diagnostics)


def compare_generation_all(actual, expected):
    """/**
     * 对同一份独立生成证据同时给出exact、严格Top-5及Top-5+0.1结论。
     * 完整输入验证、同历史logits诊断和Top-K计算仅执行一次。严格候选规则
     * 只读取真实集合成员关系，不能沿用宽松门槛的通过结果；数值诊断仍只警告。
     * @param actual FT公开会话保存的完整生成报告。
     * @param expected HF公开generate保存的独立生成报告。
     * @return 含三项独立结论及共同证据的报告；任一项失败则总体失败。
     */"""
    relaxed = compare_generation(actual, expected, policy='topk', topk=5, near_tie_tol=0.1)
    strict_checks = [dict(check, accepted=check['in_other_topk'], near_tie_accepted=False,
                          relaxed_cutoff_logprob=check['other_cutoff_logprob'])
                     for check in relaxed['candidate_checks']]
    strict_pass = all(check['accepted'] for check in strict_checks)
    strict = dict(relaxed, near_tie_tol=0.0, candidate_checks=strict_checks,
                  near_tie_accepted_directions=0, candidate_match=strict_pass,
                  status='PASS' if strict_pass else 'FAIL', failed_elements=int(not strict_pass),
                  reason='' if strict_pass else
                  f'first divergence at token {relaxed["first_divergence_token_1based"]}: mutual candidate check failed')
    exact = dict(relaxed, alignment_policy='exact',
                 status='PASS' if relaxed['exact_match'] else 'FAIL',
                 failed_elements=int(not relaxed['exact_match']),
                 reason='' if relaxed['exact_match'] else 'generated token IDs or text differ')
    comparisons = dict(exact=exact, top5=strict, top5_near_tie=relaxed)

    failures = [f'{name}: {result["reason"]}' for name, result in comparisons.items()
                if result['status'] != 'PASS']
    result = dict(relaxed, alignment_policy='all', comparisons=comparisons,
                  status='FAIL' if failures else 'PASS', failed_elements=int(bool(failures)),
                  reason='; '.join(failures))
    for name, comparison in comparisons.items():
        result[name + '_status'] = comparison['status']
        result[name + '_reason'] = comparison['reason']
    return result
