#!/usr/bin/env python3
"""对同一失败Token历史比较缓存解码与完整重算；调用正式会话，不编译或修改容差。"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import traceback
import uuid

from run_process_tp import ROOT, capture, save_report
from run_qwen_moe_precision_diagnostic import ATOL, RTOL, difference, failing_history, run_logged
from fastllm_pytools.qwen_moe_process_session import QwenMoeProcessSession


def comparison(label, actual, expected):
    """/** 比较全词表并保存真正超差项及各自阈值；最大绝对误差项不一定超差。 */"""
    if not actual or not expected or any(not math.isfinite(x) for x in [*actual, *expected]):
        raise ValueError(f'{label}: empty or nonfinite logits')
    metrics = difference(actual, expected)
    failures = [dict(vocab_id=i, actual=a, expected=b, abs_error=abs(a-b),
                     threshold=ATOL + RTOL*abs(b), excess=abs(a-b)-ATOL-RTOL*abs(b))
                for i, (a, b) in enumerate(zip(actual, expected)) if abs(a-b) > ATOL + RTOL*abs(b)]
    failures.sort(key=lambda item: item['excess'], reverse=True)
    return dict(case=label, backend='independent_eager_cache_replay', **metrics,
                actual_greedy=max(range(len(actual)), key=actual.__getitem__),
                expected_greedy=max(range(len(expected)), key=expected.__getitem__),
                failed_vocab=failures, reason=f'full_vocab={len(actual)}; atol={ATOL}; rtol={RTOL}')


def replay_plan(report, reference):
    """/** 从原请求重建首次失败历史；仅使用实际输入，不将采样Token误当成教师强制输入。 */"""
    target = failing_history(report, reference)
    prompt = list(report['request']['token_ids'])
    ids = target['ids']
    if target['step'] < 1 or ids[:len(prompt)] != prompt or len(ids) != len(prompt) + target['step']:
        raise ValueError('probe requires a failing decode step after a nonempty prompt')
    if report['request']['eos_token_ids'] or not prompt:
        raise ValueError('probe requires a nonempty prompt and disabled EOS')
    return dict(prompt=prompt, ids=ids, suffix=ids[len(prompt):], step=target['step'],
                source_rank=report['dp_rank'], source_execution=target['execution'])


def hf_replay(model, torch, plan):
    """/** 同一HF模型先完整重算，再从空KV逐Token追加；校验每步缓存长度并保存末位置logits。 */"""
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        fresh = model(input_ids=torch.tensor([plan['ids']], device=device), use_cache=False)
        full = fresh.logits[0, -1].float().cpu().tolist()
        del fresh

        past, traces, seen = None, [], 0
        for ids in [plan['prompt'], *[[token] for token in plan['suffix']]]:
            result = model(input_ids=torch.tensor([ids], device=device), past_key_values=past, use_cache=True)
            past = result.past_key_values
            seen += len(ids)
            if past is None or past.get_seq_length() != seen:
                raise RuntimeError(f'HF cache length mismatch after {seen} tokens')
            traces.append(dict(input_length=seen, submitted_tokens=len(ids), cache_length=seen,
                               logits=result.logits[0, -1].float().cpu().tolist()))
    return dict(ids=plan['ids'], full=full, cached=traces[-1]['logits'], steps=traces)


def reference_process(args):
    """/** 独立参考进程沿用正式四卡分配与设备检查，进程退出后再启动原生会话释放显存。 */"""
    os.environ.pop('USE_TORCH', None)
    import torch
    import transformers
    from qwen_moe_transformers_reference import reference_device_map, check_reference_placement

    plan = json.loads(args.reference_plan.read_text())
    config = transformers.AutoConfig.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    if torch.cuda.device_count() != 4 or config.model_type != 'qwen2_moe':
        raise ValueError('probe requires Qwen2MoE and exactly four visible GPUs')
    placement = reference_device_map(config)
    model = transformers.AutoModelForCausalLM.from_pretrained(str(args.model), config=config,
        local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='eager', device_map=placement).eval()
    devices = check_reference_placement(model, placement)
    result = hf_replay(model, torch, plan)
    result['metadata'] = dict(torch=torch.__version__, transformers=transformers.__version__,
                              dtype='float16', attention='eager', device_map=placement, checked_devices=devices)
    args.reference_output.write_text(json.dumps(result, allow_nan=False))
    print('PASS: HF full-history and cached replay collected', flush=True)


def native_replay(args, output, original, plan, cached):
    """/** 每种模式新建正式四rank会话，仅rank0有请求；禁用Graph/Prefix，不模拟原并发调度。 */"""
    request = copy.deepcopy(original['request'])
    request.update(token_ids=plan['prompt'] if cached else plan['ids'],
                   max_new_tokens=plan['step'] + 1 if cached else 1)
    forced = plan['suffix'] if cached else None
    with QwenMoeProcessSession(args.build / 'fastllm_moe_generate', args.build / 'fastllm_tp_worker',
            args.model, '0,1,2,3', output, timeout=args.timeout, trace=True,
            cuda_graph=False, prefix_cache=False) as session:
        report = session.generate(request, teacher_forced_tokens=forced)
    if report['status'] != 'PASS' or report['dp_rank'] != 0 or len(report['steps']) != request['max_new_tokens']:
        raise RuntimeError('native replay returned incomplete request or unexpected DP rank')
    for index, trace in enumerate(report['steps']):
        count = 1 if cached and index else len(request['token_ids'])
        actions = [2 if cached and index else 1, 0, 0, 0]
        expected = dict(actions=actions, submitted_token_counts=[count, 0, 0, 0])
        if trace['execution'] != 'eager' or trace.get('joint_round') != expected:
            raise RuntimeError(f'native replay step {index}: unexpected joint round or execution path')
    return report


def main():
    """/** 原失败与八项对照独立落盘，数值失败不阻断其余对照，异常也保留四格式报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--request', default='session-0002/request-000001.json')
    parser.add_argument('--build', type=Path, default=Path('build-attention-ep-cu126'))
    parser.add_argument('--reference-python')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    parser.add_argument('--reference-plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--reference-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reference_plan:
        if not args.reference_output:
            parser.error('--reference-output is required with --reference-plan')
        reference_process(args)
        return 0
    if not args.result_dir or not args.reference_python:
        parser.error('--result-dir and --reference-python are required')
    args.build, args.model = args.build.resolve(), args.model.resolve()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_decode_probe_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    print(f'独立诊断目录：{output}', flush=True)
    rows = []
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), warmup=0, iterations=1, repeats=1, atol=ATOL, rtol=RTOL,
        expected_case_count=8, report_title='MoE缓存解码与完整历史重算定位',
        source_dir=str(args.result_dir.resolve()), request=args.request,
        report_description='保留原失败；新会话仅rank0有请求，其余rank参与EP。不重放原始混合准入调度，'
                           '不替代完整验收，不以HF缓存参考替换正式参考。')

    def record(label, actual, expected):
        """/** 每项完成立即保存；终端仅列最严重的五个真实超差项，JSON保留全部。 */"""
        row = comparison(label, actual, expected)
        rows.append(row)
        save_report(output, rows, metadata)
        print(f'{row["status"]}: {label}: max_abs={row["max_abs_error"]:.9g} '
              f'mean_abs={row["mean_abs_error"]:.9g} failed={row["failed_elements"]}', flush=True)
        for item in row['failed_vocab'][:5]:
            print(f'  vocab={item["vocab_id"]}: actual={item["actual"]:.9g} expected={item["expected"]:.9g} '
                  f'error={item["abs_error"]:.9g} threshold={item["threshold"]:.9g}', flush=True)

    try:
        metadata['build_cache'] = (args.build / 'CMakeCache.txt').read_text()
        metadata['binary_sha256'] = {}
        for name in ('fastllm_moe_generate', 'fastllm_tp_worker'):
            digest = hashlib.sha256()
            with (args.build / name).open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            metadata['binary_sha256'][name] = digest.hexdigest()
        source = json.loads((args.result_dir / 'summary.json').read_text())
        original = json.loads((args.result_dir / args.request).read_text())
        reference = json.loads((args.result_dir / 'transformers-reference.json').read_text())
        plan = replay_plan(original, reference)
        metadata.update(source_git=source.get('metadata', {}).get('git'), plan=plan)
        (output / 'plan.json').write_text(json.dumps(plan))
        saved_hf = next(row['logits'] for row in reference['rows'] if row['ids'] == plan['ids'])
        saved_native = original['steps'][plan['step']]['logits']
        record('saved_native_vs_saved_hf', saved_native, saved_hf)
        print(f'定位：step={plan["step"]} prompt={len(plan["prompt"])} history={len(plan["ids"])} '
              f'source_rank={plan["source_rank"]}; controlled_rank=0', flush=True)

        command = [args.reference_python, str(Path(__file__).resolve()), '--model', str(args.model),
                   '--reference-plan', str(output / 'plan.json'), '--reference-output', str(output / 'hf.json')]
        if run_logged(command, output / 'reference', args.timeout) != 0:
            raise RuntimeError(f'HF cache reference failed; inspect {output / "reference/stderr.log"}')
        hf = json.loads((output / 'hf.json').read_text())
        if hf['ids'] != plan['ids']:
            raise ValueError('HF reference used different Token IDs')
        metadata['reference'] = hf['metadata']
        record('hf_fresh_vs_saved_hf', hf['full'], saved_hf)
        record('hf_cached_vs_hf_fresh', hf['cached'], hf['full'])

        cached = native_replay(args, output / 'native-cached', original, plan, True)
        values = cached['steps'][-1]['logits']
        record('native_cached_vs_hf_cached', values, hf['cached'])
        record('native_cached_vs_hf_fresh', values, hf['full'])
        record('saved_native_vs_native_cached', saved_native, values)
        fresh = native_replay(args, output / 'native-fresh', original, plan, False)
        record('native_fresh_vs_hf_fresh', fresh['steps'][0]['logits'], hf['full'])
        record('native_cached_vs_native_fresh', values, fresh['steps'][0]['logits'])
    except Exception as error:
        (output / 'error.log').write_text(traceback.format_exc())
        rows.append(dict(case='probe_error', backend='independent_eager_cache_replay', status='FAIL',
                         max_abs_error=None, mean_abs_error=None, failed_elements=1, reason=str(error)))
        print(f'FAIL: probe_error: {error}', flush=True)
    status = save_report(output, rows, metadata)
    print(f'{status}: {output}（保留原失败的定向诊断，非完整模型验收）', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
