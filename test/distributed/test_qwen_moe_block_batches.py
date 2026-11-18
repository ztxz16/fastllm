"""第0层MoE内部诊断的CPU回归；模拟原生快照，不启动GPU或编译。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import probe_qwen_moe_block_batches as probe
from probe_qwen_moe_prefill_batches import BLOCK_ENV, LIMIT_ENV, TRACE_ENV, prefill
from run_qwen_moe_full import request
from test_qwen_moe_prefill_batches import write_trace

CONFIG = dict(vocab_size=8, hidden_size=2, num_hidden_layers=1, moe_intermediate_size=2,
              shared_expert_intermediate_size=3, num_experts=4, num_experts_per_tok=2)


def write_blocks(directory, wave, fault=None):
    """/** 生成包含四个专家owner的异构快照；两请求时空闲rank2/3仍执行专家。 */"""
    directory.mkdir(parents=True)
    meta = dict(version=1, byte_order='little', layer=0, max_mib=256, world=4,
        capacity=2, hidden=2, width=2, shared_width=3, experts=4, topk=2, counts=[wave]*4)
    ids = np.tile([[0, 1], [2, 3]], (4, 1)).astype('<i4')
    ids[wave*2:] = -1
    norm = np.tile([[1., 2.], [3., 4.]], (4, 1)).astype('<f2')
    norm[wave*2:] = 0
    probabilities = np.full((8, 2), .5, dtype='<f2')
    probabilities[wave*2:] = 0
    saved = {}
    for repeat in (1, 2):
        for rank in range(4):
            m = dict(meta, tokens=2 if rank < wave else 0, rank=rank, round=repeat)
            entries, size = probe.layout(m)
            arrays = {entry['name']: np.zeros(entry['shape'], dtype=probe.DTYPES[entry['dtype']]) for entry in entries}
            arrays.update(norm=norm[rank*2:rank*2+2], selected_experts=ids[rank*2:rank*2+2],
                routing_weights=probabilities[rank*2:rank*2+2], gathered_norm=norm.copy(),
                gathered_experts=ids.copy(), gathered_weights=probabilities.copy(), expert_counts=np.array([wave]*4, dtype='<i4'))
            prefix = f'expert_{rank}_'
            slots = np.flatnonzero(ids.reshape(-1) == rank).astype('<i4')
            arrays[prefix+'indices'] = slots
            arrays[prefix+'packed'] = norm[slots//2].copy()
            for stage in ('gateup', 'activated', 'down'):
                arrays[prefix+stage].fill(1)
                if wave == 2 and rank == 2:
                    arrays[prefix+stage][0, 0] += .125
            if rank < wave:
                arrays['input'][:] = [[2., 3.], [-1., 0.]]
                arrays['output'][:] = [[5. if wave == 2 else 4., 5.], [6., 7.]]
                arrays['router_logits'][:] = [[4., 3., 2., 1.], [1., 2., 4., 3.]]
                arrays['local_contributions'].fill(1)
                arrays['routed'].fill(2)
            if fault == 'packed' and rank == 2 and repeat == 2:
                arrays[prefix+'packed'][0, 0] += 1
            if fault == 'slot' and rank == 2 and repeat == 2:
                arrays[prefix+'indices'][0] += 1
            if fault == 'nan' and rank == 1 and repeat == 2:
                arrays['router_logits'][0, 0] = np.nan
            base = f'rank-{rank}-round-{repeat}'
            binary = directory / (base+'.bin')
            binary.write_bytes(b''.join(arrays[entry['name']].tobytes() for entry in entries))
            assert binary.stat().st_size == size
            m.update(bytes=size, snapshots=entries, binary=binary.name)
            if fault == 'offset' and rank == 0 and repeat == 2:
                m['snapshots'][1]['offset_bytes'] += 1
            if fault == 'escape' and rank == 0 and repeat == 2:
                m['binary'] = '../escape.bin'
            if fault == 'shape' and rank == 0 and repeat == 2:
                m['width'] += 1
            if fault == 'truncated' and rank == 0 and repeat == 2:
                binary.write_bytes(binary.read_bytes()[:-1])
            manifest = directory / (base+'.json')
            manifest.write_text(json.dumps(m))
            if fault == 'missing_idle' and rank == 3:
                manifest.unlink()
            if repeat == 2:
                saved[rank] = (m, arrays)
    return saved


class BlockBatchTests(unittest.TestCase):
    def test_layout_includes_idle_experts_and_rejects_invalid_counts_or_budget(self):
        meta = dict(tokens=0, capacity=1024, hidden=2048, width=1408, shared_width=5632,
                    experts=60, topk=4, rank=2, world=4, counts=[64]*60)
        entries, size = probe.layout(meta)
        self.assertLess(size, probe.LIMIT)
        self.assertIn('expert_2_gateup', [e['name'] for e in entries])
        self.assertNotIn('input', [e['name'] for e in entries])
        for fault in ({'counts': [-1]*60}, {'counts': [4097]*60}, {'counts': [4096]*60},
                      {'capacity': True}, {'world': 3}, {'rank': 4}, {'topk': 61}):
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                probe.layout(dict(meta, **fault))

    def test_loader_validates_all_ranks_layout_finite_values_and_packing(self):
        for fault in (None, 'packed', 'slot', 'nan', 'offset', 'escape', 'shape', 'truncated', 'missing_idle'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)/'trace'
                write_blocks(directory, 2, fault)
                if fault:
                    with self.assertRaises(ValueError):
                        probe.load_blocks(directory, 2, CONFIG, request([1, 2], 4))
                else:
                    blocks = probe.load_blocks(directory, 2, CONFIG, request([1, 2], 4))
                    self.assertEqual(set(blocks), {0, 1, 2, 3})
                    self.assertEqual(blocks[2][0]['tokens'], 0)
                    self.assertEqual(blocks[2][1]['expert_2_gateup'].shape, (2, 4))

    def test_experts_paired_by_slot_and_first_difference_is_gateup(self):
        with tempfile.TemporaryDirectory() as temp:
            blocks, rows, metadata = {}, [], {}
            for wave in (4, 2):
                blocks[wave] = write_blocks(Path(temp)/str(wave), wave)
            with redirect_stdout(io.StringIO()):
                probe.compare_blocks(blocks, rows, metadata)
            self.assertEqual(metadata['first_different_stage'], 'gateup')
            self.assertEqual(metadata['expert_first_difference'], {'2': 'gateup'})
            row = next(r for r in rows if r['case'] == 'expert_2_gateup')
            self.assertEqual((row['actual_rows'], row['expected_rows'], row['owner_rank']), (2, 4, 2))
            self.assertEqual(row['first_difference']['prompt_token_1based'], 2)
            self.assertEqual(row['first_difference']['request_rank'], 0)
            self.assertTrue(all(r['status'] == 'PASS' for r in rows if r['case'].endswith('_packed')))

    def test_changed_routes_stop_expert_attribution(self):
        with tempfile.TemporaryDirectory() as temp:
            blocks = {w: write_blocks(Path(temp)/str(w), w) for w in (4, 2)}
            blocks[2][0][1]['selected_experts'] = blocks[2][0][1]['selected_experts'].copy()
            blocks[2][0][1]['selected_experts'][0, 0] = 3
            rows, metadata = [], {}
            with redirect_stdout(io.StringIO()):
                probe.compare_blocks(blocks, rows, metadata)
            self.assertEqual(metadata['first_different_stage'], 'selected_experts')
            self.assertFalse(metadata['expert_pairing_valid'])
            self.assertFalse(any(r['case'].startswith('expert_') for r in rows))

    def run_main(self, fault=None):
        """/** 用已有逐层报告和模拟正式采集执行完整CLI，验证重放门禁和四格式部分报告。 */"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, build, baseline = (root/n for n in ('model', 'build', 'baseline'))
            for path in (model, build, baseline):
                path.mkdir()
            (model/'config.json').write_text(json.dumps(CONFIG))
            (model/'model.safetensors.index.json').write_text('{}')
            for name in ('CMakeCache.txt', 'fastllm_moe_generate', 'fastllm_tp_worker'):
                (build/name).write_text('mock; never execute')
            job = request([1, 2], 4)
            old = dict(scope='formal_prefill_batch_trace_diagnostic', trace_attribution_valid=True,
                first_different_stage={'0': 'layer_0_moe', '1': 'layer_0_moe'}, cuda_graph=False,
                prefix_cache=False, request=job, git='old', binary_sha256={}, traces={},
                model_config_sha256=probe.file_hash(model/'config.json'),
                model_index_sha256=probe.file_hash(model/'model.safetensors.index.json'))
            for wave in (4, 2):
                directory = baseline/f'first{wave}/trace'
                directory.mkdir(parents=True)
                for rank in range(wave):
                    for repeat in (1, 2):
                        write_trace(directory, rank, repeat, changed=wave == 2)
                traces = probe.load_traces(directory, list(range(wave)), [1, 2], 2, 1)
                old['traces'][str(wave)] = {str(r): m for r, (m, _) in traces.items()}
                result = dict(status='PASS', execution='eager', token_ids=[1]*wave+[None]*(4-wave),
                    logits=[[1., 8., 0., 0., 0., 0., 0., 0.]]*wave+[[]]*(4-wave))
                (baseline/f'first{wave}/prefill2.json').write_text(json.dumps(result))
            if fault == 'provenance':
                old['model_config_sha256'] = 'wrong'
            (baseline/'summary.json').write_text(json.dumps(dict(status='FAIL', complete=True, metadata=old)))

            def collect(args, output, request_job, ranks, vocab, *, block_trace):
                """/** 模拟已确认路径的正式Prefill输出；按故障项破坏logits、层边界或快照。 */"""
                assert block_trace and request_job == job and vocab == 8
                if fault == 'worker':
                    raise RuntimeError('injected worker failure')
                wave = len(ranks)
                result = json.loads((baseline/f'first{wave}/prefill2.json').read_text())
                directory = output/'trace'
                write_blocks(directory, wave, 'truncated' if fault == 'trace' else None)
                if fault == 'logits':
                    result['logits'][0][1] += .001
                if fault == 'token':
                    result['token_ids'][0] = 2
                if fault == 'output':
                    meta = json.loads((directory/'rank-0-round-2.json').read_text())
                    entry, = [e for e in meta['snapshots'] if e['name'] == 'output']
                    values = np.memmap(directory/meta['binary'], mode='r+', dtype='<f2',
                                       offset=entry['offset_bytes'], shape=(2, 2))
                    values[0, 0] += 1
                    values.flush()
                return result, directory, dict(shutdown_complete=True)

            argv = ['probe', '--baseline', str(baseline), '--model', str(model), '--build', str(build),
                    '--output', str(root/'out')]
            with mock.patch('sys.argv', argv), mock.patch.object(probe, 'capture', return_value='mock'), \
                    mock.patch.object(probe, 'prefill', side_effect=collect), redirect_stdout(io.StringIO()):
                code = probe.main()
            output, = (root/'out').iterdir()
            report = json.loads((output/'summary.json').read_text())
            for ext in ('json', 'csv', 'md', 'xlsx'):
                self.assertTrue((output/f'summary.{ext}').is_file())
            return code, report

    def test_main_replays_all_boundaries_and_localizes_idle_owner_expert(self):
        code, report = self.run_main()
        self.assertEqual(code, 1)
        self.assertTrue(report['complete'])
        self.assertTrue(report['metadata']['trace_attribution_valid'])
        self.assertEqual(report['metadata']['first_different_stage'], 'gateup')
        self.assertTrue(all(row['status'] == 'PASS' for row in report['cases'][:18]))

    def test_main_refuses_output_drift_and_preserves_partial_reports(self):
        for fault in ('provenance', 'worker', 'logits', 'token', 'output', 'trace'):
            with self.subTest(fault=fault):
                code, report = self.run_main(fault)
                self.assertEqual(code, 1)
                self.assertFalse(report['complete'])
                self.assertFalse(report['metadata']['trace_attribution_valid'])
                self.assertNotIn('first_different_stage', report['metadata'])

    def test_block_prefill_clears_inherited_forward_trace_environment(self):
        # 在构造transport时截获参数并中止，绝不启动原生产物。
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict('os.environ',
                {TRACE_ENV: 'old', LIMIT_ENV: '32', BLOCK_ENV: 'stale'}):
            args = type('Args', (), dict(build=Path(temp), model=Path(temp), timeout=1))()
            with mock.patch('probe_qwen_moe_prefill_batches.RoundTransport', side_effect=RuntimeError('stop')) as transport:
                with self.assertRaisesRegex(RuntimeError, 'stop'):
                    prefill(args, Path(temp)/'out', request([1, 2], 4), [0, 1], 8, block_trace=True)
            environment = transport.call_args.kwargs['environment']
            self.assertNotIn(TRACE_ENV, environment)
            self.assertNotIn(LIMIT_ENV, environment)
            self.assertEqual(environment[BLOCK_ENV], str(Path(temp)/'out/trace'))


if __name__ == '__main__':
    unittest.main()
