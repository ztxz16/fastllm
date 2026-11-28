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


def _pair(actual, expected, topk):
    """/** 校验完整生成证据并定位共同历史，绝不计算两侧Logits的数值误差。 */"""
    actual_vocab = validate_output(actual, topk)
    expected_vocab = validate_output(expected, topk)
    if actual_vocab != expected_vocab or actual['request'] != expected['request']:
        raise ValueError('model vocabulary, prompt IDs or generation configuration differ')
    token_match = actual['token_ids'] == expected['token_ids']
    text_match = actual['text'] == expected['text']
    divergence = next((i + 1 for i, (a, b) in enumerate(zip(actual['token_ids'], expected['token_ids']))
                       if a != b), None)
    compared = divergence or len(actual['token_ids'])
    return dict(exact_match=token_match and text_match, token_match=token_match, text_match=text_match,
                first_divergence_token_1based=divergence, compared_steps=compared,
                uncompared_steps=len(actual['token_ids']) - compared)


def _level(base, name, checks=None):
    """/** 保存已执行等级的PASS或WARNING；未通过单项不决定最终验收状态。 */"""
    passed = base['exact_match'] if name == 'exact' else all(c['accepted'] for c in checks)
    return dict(status='PASS' if passed else 'WARNING',
                reason='' if passed else ('generated token IDs or text differ' if name == 'exact'
                    else 'mutual candidate check did not pass'),
                candidate_checks=checks or [],
                near_tie_accepted_directions=sum(c['near_tie_accepted'] for c in (checks or [])))


def _candidates(actual, expected, base, topk):
    """/** 只在首个Token分歧处计算双侧Top-K，禁止沿已分歧的后续历史比较。 */"""
    position = base['first_divergence_token_1based']
    if position is None:
        return None
    i = position - 1
    return (actual['token_ids'][i], expected['token_ids'][i],
            top_logprobs(actual['steps'][i]['logits'], topk),
            top_logprobs(expected['steps'][i]['logits'], topk))


def _checks(candidates, tolerance):
    """/** 对固定的双侧候选执行指定容差检查，调用方决定是否需要本等级。 */"""
    if candidates is None:
        return []
    left, right, a, b = candidates
    return [dict(direction='ft_to_hf', **candidate_check(left, a, b, tolerance)),
            dict(direction='hf_to_ft', **candidate_check(right, b, a, tolerance))]


def compare_generation(actual, expected, *, policy='topk', topk=5, near_tie_tol=0.1):
    """/**
     * 检查单个生成等级，供诊断或三级判定的调用方使用；不运行numeric。
     * @param actual 实际完整生成及原始Logits证据。
     * @param expected 同输入与配置的完整参考生成。
     * @param policy exact或topk；该接口不代表三级整体接受结论。
     * @param topk 候选数，必须在词表范围内。
     * @param near_tie_tol 候选门槛容差；零仅检查集合互包含。
     * @return 已执行单项的PASS/WARNING与证据；输入或执行证据非法时抛出异常。
     */"""
    validate_policy(policy, topk, near_tie_tol)
    base = _pair(actual, expected, topk)
    checks = None if policy == 'exact' else _checks(_candidates(actual, expected, base, topk), near_tie_tol)
    level = _level(base, policy, checks)
    return dict(base, **level, alignment_policy=policy, topk=topk, near_tie_tol=near_tie_tol,
                candidate_match=None if checks is None else all(c['accepted'] for c in checks),
                failed_elements=0)


def compare_generation_all(actual, expected):
    """/**
     * 按exact、Top-5、Top-5+0.1逐级验收，任一通过即停止，全部未通过才FAIL。
     * 完整证据始终校验；未通过单项记WARNING，未执行项记SKIP，不运行numeric。
     * @param actual 实际完整自由生成报告，不接受教师强制轨迹。
     * @param expected 同输入与配置的完整参考生成报告。
     * @return 整体状态、接受等级、单项告警和首次分歧；证据非法时抛出异常。
     */"""
    base = _pair(actual, expected, 5)
    names = ('exact', 'top5', 'top5_near_tie')
    levels = {name: dict(status='SKIP', reason='earlier level accepted', candidate_checks=[],
                        near_tie_accepted_directions=0) for name in names}
    levels['exact'] = _level(base, 'exact')
    accepted = 'exact' if base['exact_match'] else None
    candidates = None
    if accepted is None:
        candidates = _candidates(actual, expected, base, 5)
        levels['top5'] = _level(base, 'top5', _checks(candidates, 0.0))
        if levels['top5']['status'] == 'PASS':
            accepted = 'top5'
        else:
            levels['top5_near_tie'] = _level(base, 'top5_near_tie', _checks(candidates, 0.1))
            if levels['top5_near_tie']['status'] == 'PASS':
                accepted = 'top5_near_tie'
    last = levels[accepted or 'top5_near_tie']
    warnings = [f'{name}: {levels[name]["reason"]}' for name in names if levels[name]['status'] == 'WARNING']
    result = dict(base, status='PASS' if accepted else 'FAIL', alignment_policy='all',
                  policy_version='generation_cascade_20261128', accepted_by=accepted,
                  topk=5, near_tie_tol=0.1, comparisons=levels, warnings=warnings,
                  warning_count=len(warnings), failed_elements=int(accepted is None),
                  reason='' if accepted else 'all three generation comparisons did not pass',
                  candidate_checks=last['candidate_checks'],
                  candidate_match=None if accepted == 'exact' else last['status'] == 'PASS',
                  near_tie_accepted_directions=last['near_tie_accepted_directions'])
    for name, level in levels.items():
        result[name + '_status'] = level['status']
        result[name + '_reason'] = level['reason']
    return result


def compare_token_ids(actual, expected, count):
    """/**
     * 检查仅包含Token ID的正式入口输出；exact未通过且缺少候选证据时不得猜测接受。
     * @param actual 实际输出Token ID列表。
     * @param expected 同输入及生成配置的参考Token ID列表。
     * @param count 约定输出数量；长度或ID非法属于功能故障。
     * @return 三级规则证据；exact通过即PASS，否则因候选证据缺失而FAIL并保留exact WARNING。
     */"""
    if type(count) is not int or count < 1:
        raise ValueError('invalid expected output length')
    for ids in (actual, expected):
        if not isinstance(ids, list) or len(ids) != count or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError('invalid token IDs or output length')
    same = actual == expected
    first = next((i + 1 for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
    return dict(status='PASS' if same else 'FAIL', policy_version='generation_cascade_20261128',
                accepted_by='exact' if same else None, exact_status='PASS' if same else 'WARNING',
                top5_status='SKIP' if same else 'NOT_RUN',
                top5_near_tie_status='SKIP' if same else 'NOT_RUN',
                first_divergence_token_1based=first,
                warnings=[] if same else ['exact: generated token IDs differ'],
                reason='' if same else 'candidate evidence missing; cannot complete generation cascade')
