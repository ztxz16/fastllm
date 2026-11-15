"""双输入归因的CPU验证：完整形状、重放门禁、旧失败保留及证据失配拒绝。"""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import probe_qwen_moe_stage_inputs as probe
from test_qwen_moe_forward_recheck import saved_probe


def fields(output):
    """/** 两Token、两隐藏维、两专家的确定性层报告；空闲rank字段由调用方构造。 */"""
    return dict(output=output, router_logits=[1., 0., 1., 0.], selected_experts=[0, 0],
                routing_weights=[1., 1.], routed_output=[1.]*4, shared_output=[0.]*4)


class StageInputTests(unittest.TestCase):
    def test_half_wire_recovers_subnormal_without_tolerating_corruption(self):
        values = [0., 2.**-24, -2.**-24, .125, 65504.]
        wire = [float(format(value, '.12f')) for value in values]
        self.assertNotEqual(values[1], wire[1])
        self.assertEqual(probe.restore_half_wire(wire), values)
        for value in (.1, float('inf'), float('nan'), 1e20, wire[1]+1e-12):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.restore_half_wire([value])

    def test_distinguishes_input_propagation_and_same_input_error(self):
        refs = dict(hf=fields([1.]*4), native=fields([1.125]*4))
        native = dict(hf=fields([1.]*4), native=fields([1.25]*4))
        rows, tokens, valid = probe.attribution_rows(13, [5, 9], 2, [1.125]*4, [1.]*4,
            [1.25]*4, refs['hf'], refs, native)
        lookup = {row['case']: row for row in rows}
        self.assertTrue(valid)
        self.assertEqual(lookup['layer13/propagated_by_hf']['max_abs_error'], .125)
        self.assertEqual(lookup['layer13/same_native_input']['max_abs_error'], .125)
        self.assertEqual(lookup['layer13/total_output_difference']['max_abs_error'], .25)
        self.assertEqual(len(tokens), 8)
        self.assertTrue(all(row['attribution_valid'] for row in tokens))
        self.assertEqual({row['token_id'] for row in tokens}, {5, 9})

    def test_any_replay_mismatch_denies_attribution_even_below_tolerance(self):
        for side in ('native', 'hf_router', 'hf_output'):
            with self.subTest(side=side):
                refs = dict(hf=fields([1.]*4), native=fields([1.]*4))
                actual = copy.deepcopy(refs)
                if side == 'native':
                    actual['native']['output'][0] = 1.0009765625
                elif side == 'hf_router':
                    refs['hf']['selected_experts'][0] = 1
                else:
                    refs['hf']['output'][0] = 1.0009765625
                rows, tokens, valid = probe.attribution_rows(13, [5, 9], 2, [1.]*4, [1.]*4,
                    [1.]*4, fields([1.]*4), refs, actual)
                self.assertFalse(valid)
                self.assertTrue(any(row['status'] == 'FAIL' for row in rows))
                self.assertTrue(all(not row['attribution_valid'] for row in tokens))

    def test_observation_preserves_public_forward_and_cleans_hooks_on_failure(self):
        try:
            import torch
        except ImportError:
            self.skipTest('requires an existing PyTorch installation for CPU operations')
        import test_qwen_moe_routing_observer as routing
        from torch.overrides import _get_current_function_mode
        fixture = routing.RoutingObserverTests()
        fixture.torch = torch
        previous = _get_current_function_mode()
        for fail in (False, True):
            with self.subTest(fail=fail):
                model = fixture.model(torch.float16, True, fail=fail)
                layer = model.model.layers[0]
                x = torch.tensor([[[.125, -.5, 2.], [1., .25, -.125]]], dtype=torch.float16)
                original = x.clone()
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'injected expert failure'):
                        probe.observe_moe(torch, layer.mlp, layer.post_attention_layernorm, x)
                else:
                    with torch.inference_mode():
                        expected = x + layer.mlp(layer.post_attention_layernorm(x))[0]
                    result = probe.observe_moe(torch, layer.mlp, layer.post_attention_layernorm, x)
                    self.assertEqual(result['output'], expected.reshape(-1).tolist())
                    for field, value in layer.mlp.evidence[-1].items():
                        self.assertEqual(result[field], value.reshape(-1).tolist())
                fixture.assert_no_hooks(model)
                self.assertTrue(torch.equal(x, original))
                self.assertIs(_get_current_function_mode(), previous)

    def run_fixture(self, root, mode='normal'):
        """/** 使用真实报告/证据复核逻辑；仅替换GPU进程与HF计算，不执行编译或外部二进制。 */"""
        source = saved_probe(root)
        model, build = root / 'model', root / 'build'
        config_path = model / 'config.json'
        config = json.loads(config_path.read_text())
        config.update(num_experts=2, num_experts_per_tok=1)
        config_path.write_text(json.dumps(config))
        (build / 'fastllm_moe_layer_check').write_text('not executable')
        if mode == 'worker_changed':
            (build / 'fastllm_tp_worker').write_text('changed')
        if mode == 'trace_changed':
            path = source / 'trace/trace.f16'
            path.write_bytes(struct.pack('<e', 2.) + path.read_bytes()[2:])
        before = {str(p): probe.digest(p) for p in root.rglob('*') if p.is_file()}
        refs = dict(hf=fields([1.]*4), native=fields([4., 5., 6., 7.]))

        def native(command, directory, timeout):
            """/** 读真实任务文件，确保两个输入都保留全部Token且只激活rank0。 */"""
            self.assertEqual(Path(command[0]).name, 'fastllm_moe_layer_check')
            tasks = json.loads(Path(command[command.index('--tasks')+1]).read_text())
            self.assertEqual([t['inputs'][0]['hidden'] for t in tasks], [[.5]*4, [2., 3., -1., 0.]])
            self.assertTrue(all(t['stage'] == 2 and len(t['inputs']) == 4 for t in tasks))
            self.assertTrue(all(not r['hidden'] for t in tasks for r in t['inputs'][1:]))
            directory.mkdir()
            results = []
            for task, name in zip(tasks, refs):
                empty = {key: [] for key in refs[name]}
                results.append(dict(task=task['name'], ranks=[refs[name], empty, empty, empty]))
            if mode == 'incomplete':
                results.pop()
            (directory / 'stdout.log').write_text(json.dumps(dict(
                status='PASS', workers_reaped=True, results=results)))
            return 0

        argv = ['probe', '--probe', str(source), '--model', str(model), '--build', str(build),
                '--layers', '0', '--output', str(root / 'new-reports')]
        with patch('sys.argv', argv), patch.object(probe, 'capture', return_value='CPU fixture'), \
                patch.object(probe, 'run_logged', side_effect=native) as runner, \
                patch.object(probe, 'hf_pair', return_value=(refs, dict(device='CPU fixture'))) as hf, \
                patch.dict('os.environ', {probe.TRACE_ENV: ''}), redirect_stdout(io.StringIO()):
            code = probe.main()
        output, = (root / 'new-reports').iterdir()
        self.assertEqual(code, 1)  # 历史真实模型失败必须保留。
        for name, sha in before.items():
            self.assertEqual(probe.digest(Path(name)), sha)
        for suffix in ('json', 'md', 'csv', 'xlsx'):
            self.assertTrue((output / f'summary.{suffix}').is_file())
        return json.loads((output / 'summary.json').read_text()), output, runner, hf

    def test_complete_pipeline_preserves_failure_files_and_both_input_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            report, output, runner, hf = self.run_fixture(Path(temp))
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(report['metadata']['layer_attribution_valid'], {'0': True})
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(hf.call_count, 1)
            self.assertFalse(any(row['case'] == 'stage_inputs_error' for row in report['cases']))
            for suffix in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((output / 'tokens' / f'summary.{suffix}').is_file())

    def test_rejects_changed_evidence_or_incomplete_native_run(self):
        for mode in ('worker_changed', 'trace_changed', 'incomplete'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                report, output, runner, hf = self.run_fixture(Path(temp), mode)
                self.assertEqual(report['cases'][-1]['case'], 'stage_inputs_error')
                self.assertEqual(hf.call_count, 0)
                if mode != 'incomplete':
                    self.assertEqual(runner.call_count, 0)


if __name__ == '__main__':
    unittest.main()
