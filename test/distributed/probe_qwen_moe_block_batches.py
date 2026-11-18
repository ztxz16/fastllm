#!/usr/bin/env python3
"""正式第0层MoE的两/四请求内部诊断；先复现逐层基准，不执行HF或性能测试。"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import sys
import traceback
import uuid

import numpy as np

from probe_qwen_moe_prefill_batches import difference, file_hash, load_traces, prefill
from run_process_tp import ROOT, capture, save_report
from fastllm_pytools.qwen3_process_chat import validate_request

DTYPES = {'float16': '<f2', 'float32': '<f4', 'int32': '<i4'}
LIMIT = 256 * 1024 * 1024


def layout(meta):
    """/**
     * 检查维度并重建完整连续布局，拒绝溢出和不合法专家计数。
     * @param meta 原生manifest；包含rank、有效Token、padded容量、模型维度与专家行数。
     * @return 有序阶段描述及总字节数，张量分别保留FP16、FP32和int32类型。
     */"""
    keys = ('tokens', 'capacity', 'hidden', 'width', 'shared_width', 'experts', 'topk', 'rank', 'world')
    if any(type(meta.get(k)) is not int for k in keys):
        raise ValueError('invalid integer trace dimensions')
    t, c, h, m, s, e, k, rank, world = (meta[n] for n in keys)
    if (world != 4 or not 0 <= rank < world or not 0 <= t <= c or not 1 < c <= 2048 or
            not 0 < h <= 65536 or not 0 < m <= 65536 or not 0 < s <= 65536 or not 0 < k <= e <= 256):
        raise ValueError('invalid block trace dimensions')
    counts = meta['counts']
    if len(counts) != e or any(type(n) is not int or not 0 <= n <= c*world for n in counts):
        raise ValueError('invalid expert counts')
    entries, offset = [], 0

    def add(name, dtype, shape):
        """/** 生成连续字节布局，与正式C++观察器的类型及形状约定一致。 */"""
        nonlocal offset
        size = math.prod(shape) * np.dtype(DTYPES[dtype]).itemsize
        if not 0 < size <= LIMIT - offset:
            raise ValueError('block trace exceeds 256 MiB')
        entries.append(dict(name=name, dtype=dtype, offset_bytes=offset, bytes=size, shape=shape))
        offset += size

    if t:
        add('input', 'float16', [t, h])
    for name, dtype, shape in (
            ('norm', 'float16', [c, h]), ('router_logits', 'float16', [c, e]),
            ('selected_experts', 'int32', [c, k]), ('routing_weights', 'float16', [c, k]),
            ('gathered_norm', 'float16', [c*world, h]), ('gathered_experts', 'int32', [c*world, k]),
            ('gathered_weights', 'float16', [c*world, k]), ('expert_counts', 'int32', [e])):
        add(name, dtype, shape)
    for expert in range(rank, e, world):
        n = counts[expert]
        if n:
            for stage, dtype, shape in (('indices', 'int32', [n]), ('packed', 'float16', [n, h]),
                    ('gateup', 'float16', [n, 2*m]), ('activated', 'float16', [n, m]), ('down', 'float16', [n, h])):
                add(f'expert_{expert}_{stage}', dtype, shape)
    add('local_contributions', 'float16', [c, k, h])
    add('routed', 'float32', [c, h])
    if t:
        for stage, width in (('shared_gateup', 2*s), ('shared_activated', s), ('shared_down', h),
                             ('shared_gate', 1), ('shared_weighted', h), ('output', h)):
            add(stage, 'float16', [t, width])
    return entries, offset


def load_blocks(directory, wave, config, job):
    """/**
     * 校验四个rank的两轮快照，按原dtype映射第2轮；空闲rank的专家不能遗漏。
     * @param directory 本会话独占快照目录，拒绝缺失、重复或额外文件。
     * @param wave 从rank0开始的活动请求数，取2或4。
     * @param config 已核验来源的模型配置。
     * @param job 原始生成请求，保留完整Prompt长度。
     * @return rank到manifest/命名只读张量的映射；通信或Pack证据不一致则失败。
     */"""
    seen, allowed, result = set(), set(), {}
    for path in sorted(directory.glob('*.json')):
        meta = json.loads(path.read_text())
        key = (meta.get('rank'), meta.get('round'))
        if (any(type(v) is not int for v in key) or key in seen or key[0] not in range(4) or key[1] not in (1, 2)):
            raise ValueError('unexpected/duplicate block rank or round')
        expected = dict(version=1, byte_order='little', layer=0, max_mib=256, world=4,
            tokens=len(job['token_ids']) if key[0] < wave else 0, capacity=len(job['token_ids']),
            hidden=config['hidden_size'], width=config['moe_intermediate_size'],
            shared_width=config['shared_expert_intermediate_size'], experts=config['num_experts'],
            topk=config['num_experts_per_tok'])
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError('block trace model/input/layout metadata mismatch')
        entries, size = layout(meta)
        if entries != meta['snapshots'] or type(meta['bytes']) is not int or meta['bytes'] != size:
            raise ValueError('incomplete/reordered block stages or bad offsets')
        name = meta['binary']
        binary = directory / name
        if (Path(name).name != name or binary.resolve().parent != directory.resolve() or
                not binary.is_file() or binary.stat().st_size != size or {path, binary} & allowed):
            raise ValueError('invalid/reused/truncated block binary')
        seen.add(key)
        allowed.update((path, binary))
        storage = np.memmap(binary, mode='r', dtype='u1', shape=(size,))
        values = {entry['name']: np.ndarray(tuple(entry['shape']), dtype=DTYPES[entry['dtype']],
            buffer=storage, offset=entry['offset_bytes']) for entry in entries}
        for array in values.values():
            difference(array.reshape(-1), array.reshape(-1))
        if key[1] == 2:
            meta.update(manifest_sha256=file_hash(path), binary_sha256=file_hash(binary))
            result[key[0]] = (meta, values)
    if seen != {(r, n) for r in range(4) for n in (1, 2)} or set(directory.iterdir()) != allowed:
        raise ValueError('missing/incomplete or extra block trace files')
    validate_wave(result, wave)
    return result


def validate_wave(blocks, wave):
    """/**
     * 校验路由计数、padding、跨rank通信及专家Pack槽位，禁止按错误行序比较。
     * @param blocks 四个rank同一轮的manifest及原始命名张量。
     * @param wave 实际活动请求数；空闲请求的数据必须符合正式padding语义。
     * @return 无返回值；任何布局、索引或数据一致性错误都抛出包含阶段信息的异常。
     */"""
    meta, reference = blocks[0]
    c, k, e = meta['capacity'], meta['topk'], meta['experts']
    ids = reference['gathered_experts']
    active = ids[:wave*c]
    if ((active < 0).any() or (active >= e).any() or
            any(len(set(row)) != k for row in active) or (ids[wave*c:] != -1).any()):
        raise ValueError('invalid/duplicate active routes or nonempty padding')
    expected_counts = np.bincount(active.reshape(-1), minlength=e)
    for rank, (manifest, arrays) in blocks.items():
        if manifest['counts'] != expected_counts.tolist() or not np.array_equal(arrays['expert_counts'], expected_counts):
            raise ValueError(f'rank {rank}: expert counts disagree with actual routes')
        for field in ('gathered_norm', 'gathered_experts', 'gathered_weights'):
            if not np.array_equal(arrays[field], reference[field]):
                raise ValueError(f'rank {rank}: inconsistent {field} across owners')
        for local, gathered in (('norm', 'gathered_norm'), ('selected_experts', 'gathered_experts'),
                                ('routing_weights', 'gathered_weights')):
            if not np.array_equal(arrays[local], reference[gathered][rank*c:(rank+1)*c]):
                raise ValueError(f'rank {rank}: local/gathered {local} mismatch')
        if rank >= wave and any(np.any(arrays[name]) for name in ('norm', 'router_logits', 'routing_weights',
                                                                  'local_contributions', 'routed')):
            raise ValueError(f'rank {rank}: inactive request has nonzero values')
        for expert in range(rank, e, 4):
            indices = np.flatnonzero(ids.reshape(-1) == expert)
            if indices.size:
                prefix = f'expert_{expert}_'
                if not np.array_equal(arrays[prefix+'indices'], indices):
                    raise ValueError(f'expert {expert}: wrong/unstable packed slot map')
                if not np.array_equal(arrays[prefix+'packed'], reference['gathered_norm'][indices//k]):
                    raise ValueError(f'expert {expert}: Pack differs from gathered input')


def append_comparison(rows, case, actual, expected, **details):
    """/**
     * 比较完整张量并记录首个坐标、形状及零容差统计，保留原始精度。
     * @param rows 本次报告行列表，追加后由调用方落盘。
     * @param case 稳定唯一用例名。
     * @param actual 两请求张量或待重放输出，已按共同槽位对齐。
     * @param expected 同形状基准张量。
     * @param details 实际路径、rank、专家与双方原矩阵行数等附加证据。
     * @return 已追加的报告行；形状或有限值校验失败时抛出异常。
     */"""
    if actual.shape != expected.shape:
        raise ValueError(f'{case}: different compared shapes')
    metrics = difference(actual.reshape(-1), expected.reshape(-1))
    first = metrics['first_difference']
    if first:
        first['coordinates_0based'] = [int(i) for i in np.unravel_index(first['element_0based'], actual.shape)]
    row = dict(case=case, backend='formal_eager_moe_layer0', shape=list(actual.shape), **metrics, **details)
    rows.append(row)
    print(f'{row["status"]}: {case}; max_abs={row["max_abs_error"]:.6g}; '
          f'mean_abs={row["mean_abs_error"]:.6g}; different={row["failed_elements"]}', flush=True)
    return row


def compare_blocks(blocks, rows, metadata):
    """/**
     * 按数据依赖阶段比较共同请求，专家行按全局路由槽位匹配。
     * @param blocks 以活动请求数2/4及rank为索引的原始快照。
     * @param rows 累积报告行；不把不同原矩阵行数直接逐行对比。
     * @param metadata 写入最早差异及逐专家差异，路由ID变化时停止专家配对。
     * @return 无返回值；槽位不匹配立即失败，不借助宽松数值阈值继续归因。
     */"""
    phases = ('input', 'norm', 'router_logits', 'selected_experts', 'routing_weights',
              'packed', 'gateup', 'activated', 'down', 'local_contributions', 'routed',
              'shared_gateup', 'shared_activated', 'shared_down', 'shared_gate', 'shared_weighted', 'output')
    changed, expert_first = [], {}
    for phase in phases:
        if phase in ('packed', 'gateup', 'activated', 'down'):
            for expert in range(blocks[2][0][0]['experts']):
                owner = expert % 4
                am, a = blocks[2][owner]
                bm, b = blocks[4][owner]
                count = am['counts'][expert]
                if not count:
                    continue
                prefix = f'expert_{expert}_'
                indices, expected_indices = a[prefix+'indices'], b[prefix+'indices']
                positions = np.searchsorted(expected_indices, indices)
                if (positions >= expected_indices.size).any() or not np.array_equal(expected_indices[positions], indices):
                    raise ValueError(f'expert {expert}: cannot pair common request slots')
                row = append_comparison(rows, prefix+phase, a[prefix+phase], b[prefix+phase][positions],
                    phase=phase, owner_rank=owner, expert=expert, actual_rows=count, expected_rows=bm['counts'][expert])
                if row['failed_elements']:
                    changed.append(phase)
                    expert_first.setdefault(str(expert), phase)
                    first = row['first_difference']
                    slot = int(indices[first['coordinates_0based'][0]])
                    first.update(request_rank=slot//(am['capacity']*am['topk']),
                        prompt_token_1based=(slot//am['topk']) % am['capacity']+1,
                        routing_slot_0based=slot % am['topk'])
        else:
            for rank in (0, 1):
                a, b = blocks[2][rank][1][phase], blocks[4][rank][1][phase]
                row = append_comparison(rows, f'rank{rank}_{phase}', a, b, phase=phase, rank=rank)
                if row['failed_elements']:
                    changed.append(phase)
        if phase == 'selected_experts' and phase in changed:
            metadata['expert_pairing_valid'] = False
            metadata['comparison_stopped_reason'] = '路由ID已不同，不能把不同专家当成同一专家继续比较'
            break
    metadata.update(first_different_stage=next((p for p in phases if p in changed), None),
                    different_stages=[p for p in phases if p in changed], expert_first_difference=expert_first)


def main():
    """/** 先复现已保存logits及第0层输入/输出，再做内部比较；异常和部分失败均保留四格式报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='已完成的moe_prefill_batches结果目录')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, default=Path('build-attention-ep-cu126'))
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be positive and finite')
    args.model, args.build, args.baseline = args.model.resolve(), args.build.resolve(), args.baseline.resolve()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_moe_block_batches_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f'第0层MoE内部诊断：{output}', flush=True)
    rows, metadata = [], dict(scope='formal_moe_block_batch_trace_diagnostic', execution_complete=False,
        trace_attribution_valid=False, expert_pairing_valid=True, expected_case_count=18,
        baseline=str(args.baseline), command=shlex.join([sys.executable, *sys.argv]),
        warmup=1, iterations=1, repeats=1, layer=0, cuda_graph=False, prefix_cache=False,
        git=capture(['git', 'rev-parse', 'HEAD']), git_status=capture(['git', 'status', '--short']),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cpu=capture(['lscpu']), cuda=capture(['nvcc', '--version']),
        report_title='第0层MoE并发组合内部诊断',
        report_description='正式Eager首步Prefill，固定两请求/四请求。先精确重放旧logits和层输入/输出；'
            '通过后按原始槽位比较专家矩阵。零容差FAIL只用于定位，不更改验收阈值，不提供性能结论。')
    try:
        baseline = json.loads((args.baseline / 'summary.json').read_text())
        old = baseline['metadata']
        if (not baseline['complete'] or old['scope'] != 'formal_prefill_batch_trace_diagnostic' or
                not old['trace_attribution_valid'] or old['cuda_graph'] or old['prefix_cache'] or
                old['first_different_stage'] != {'0': 'layer_0_moe', '1': 'layer_0_moe'}):
            raise ValueError('baseline must be completed Prefill attribution with first difference at layer_0_moe')
        config = json.loads((args.model / 'config.json').read_text())
        job = old['request']
        validate_request(job, config['vocab_size'])
        if job['temperature'] != 0 or job['eos_token_ids'] or len(job['token_ids']) <= 1:
            raise ValueError('expected multi-token greedy Prefill with no EOS')
        metadata.update(request=job, model=str(args.model), baseline_git=old['git'],
            build_cache=(args.build / 'CMakeCache.txt').read_text(),
            binary_sha256={n: file_hash(args.build/n) for n in ('fastllm_moe_generate', 'fastllm_tp_worker')},
            baseline_binary_sha256=old['binary_sha256'])
        for filename, key in (('config.json', 'model_config_sha256'), ('model.safetensors.index.json', 'model_index_sha256')):
            metadata[key] = file_hash(args.model/filename)
            if metadata[key] != old[key]:
                raise ValueError(f'model provenance mismatch: {filename}')
        saved, old_traces = {}, {}
        for wave in (4, 2):
            saved[wave] = json.loads((args.baseline / f'first{wave}/prefill2.json').read_text())
            if saved[wave]['status'] != 'PASS' or saved[wave]['execution'] != 'eager':
                raise ValueError('baseline prefill protocol failed')
            old_traces[wave] = load_traces(args.baseline / f'first{wave}/trace', list(range(wave)),
                job['token_ids'], config['hidden_size'], config['num_hidden_layers'])
            for rank, (manifest, _) in old_traces[wave].items():
                for key in ('manifest_sha256', 'binary_sha256'):
                    if manifest[key] != old['traces'][str(wave)][str(rank)][key]:
                        raise ValueError('baseline snapshots changed since attribution')
        save_report(output, rows, metadata)
        print('旧报告及快照校验通过；开始两组正式Prefill采集。', flush=True)

        blocks = {}
        for wave in (4, 2):
            print(f'采集：{wave}请求；4个专家owner；warmup 1次、正式1次。', flush=True)
            result, directory, shutdown = prefill(args, output / f'first{wave}', job,
                list(range(wave)), config['vocab_size'], block_trace=True)
            metadata[f'first{wave}_shutdown'] = shutdown
            blocks[wave] = load_blocks(directory, wave, config, job)
            for rank in range(wave):
                row = append_comparison(rows, f'first{wave}_rank{rank}_logits_replay',
                    np.asarray(result['logits'][rank]), np.asarray(saved[wave]['logits'][rank]))
                row['token_match'] = result['token_ids'][rank] == saved[wave]['token_ids'][rank]
                if not row['token_match']:
                    row.update(status='FAIL', reason='采样Token未复现')
                manifest, values = old_traces[wave][rank]
                for stage, field in (('layer_0_attention', 'input'), ('layer_0_moe', 'output')):
                    entry, = [s for s in manifest['snapshots'] if s['name'] == stage]
                    expected = values[entry['offset']:entry['offset']+entry['elements']]
                    actual = blocks[wave][rank][1][field]
                    append_comparison(rows, f'first{wave}_rank{rank}_{field}_replay', actual, expected.reshape(actual.shape))
            save_report(output, rows, metadata)
        if len(rows) != 18 or any(row['status'] != 'PASS' for row in rows):
            raise ValueError('block trace did not exactly replay baseline; internal attribution refused')
        metadata['traces'] = {str(w): {str(r): meta for r, (meta, _) in blocks[w].items()} for w in blocks}
        compare_blocks(blocks, rows, metadata)
        metadata.update(trace_attribution_valid=True, execution_complete=True, expected_case_count=len(rows))
    except BaseException as error:
        metadata['trace_attribution_valid'] = False
        (output / 'diagnostic.traceback.log').write_text(traceback.format_exc())
        rows.append(dict(case='diagnostic_error_or_interrupt', status='FAIL', failed_elements=1,
                         reason=f'{type(error).__name__}: {error}'))
        print(f'FAIL: {type(error).__name__}: {error}', flush=True)
    finally:
        status = save_report(output, rows, metadata)
    if metadata['trace_attribution_valid']:
        print('首个不同阶段：' + str(metadata['first_different_stage']), flush=True)
        print('各专家首个差异：' + json.dumps(metadata['expert_first_difference'], ensure_ascii=False), flush=True)
        if not metadata['expert_pairing_valid']:
            print(metadata['comparison_stopped_reason'], flush=True)
    else:
        print('内部归因未完成；先检查采集、布局和精确重放失败。', flush=True)
    print(f'{status}: {output}', flush=True)
    return int(status != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
