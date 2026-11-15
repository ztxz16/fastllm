"""专家贡献诊断的CPU验证：真实公开算子观察、舍入边界、证据拒绝与报告保留。"""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import probe_qwen_moe_reduction_precision as probe
import test_qwen_moe_stage_inputs as stage_tests


def rounding_case():
    """/** 四专家构造FP16半ULP累加差异，两个Token保留非自然顺序的专家批次。 */"""
    observed = dict(selected_experts=[0, 1, 2, 3] * 2, routing_weights=[1.] * 8,
                    shared_output=[0., 0.], routed_output=[1., 1.], output=[1., 1.])
    records = [dict(expert=expert, tokens=[1, 0], weighted=[value, value], expert_output=[value, value])
               for expert, value in enumerate([1., 2.**-11, 2.**-11, 0.])]
    return records, observed


class ReductionTests(unittest.TestCase):
    def test_half_ulp_accumulation_distinguishes_same_contributions(self):
        records, observed = rounding_case()
        original = copy.deepcopy(records)
        result = probe.replay_contributions(records, observed, [0., 0.], 1, 4, 4)
        self.assertEqual(result['fp16']['output'], [1., 1.])
        self.assertEqual(result['fp32']['output'], [1.0009765625] * 2)
        self.assertEqual(records, original)

    def test_shared_and_residual_have_separate_half_rounding(self):
        records, observed = rounding_case()
        observed['shared_output'] = [2.**-11] * 2
        result = probe.replay_contributions(records, observed, [2.**-11] * 2, 1, 4, 4)
        self.assertEqual(result['fp16']['output'], [1., 1.])
        self.assertEqual(result['fp32']['output'], [1.001953125] * 2)

    def test_rejects_incomplete_duplicate_or_corrupted_contributions(self):
        for mode in ('missing', 'duplicate', 'token', 'product', 'nan', 'shape', 'id', 'route'):
            with self.subTest(mode=mode):
                records, observed = rounding_case()
                if mode == 'missing':
                    records.pop()
                elif mode == 'duplicate':
                    records.append(copy.deepcopy(records[0]))
                elif mode == 'token':
                    records[0]['tokens'] = [0, 0]
                elif mode == 'product':
                    records[0]['weighted'][0] = .5
                elif mode == 'nan':
                    records[0]['expert_output'][0] = float('nan')
                elif mode == 'shape':
                    records[0]['weighted'].pop()
                elif mode == 'id':
                    observed['selected_experts'][0] = -1
                else:
                    observed['selected_experts'][0] = 1
                with self.assertRaises((ValueError, OverflowError)):
                    probe.replay_contributions(records, observed, [0., 0.], 1, 4, 4)

    def test_public_hf_style_forward_capture_and_exception_cleanup(self):
        try:
            import torch
        except ImportError:
            self.skipTest('requires existing PyTorch for CPU operations only')
        from torch.overrides import _get_current_function_mode
        from test_qwen_moe_routing_observer import RoutingObserverTests
        fixture = RoutingObserverTests()
        fixture.torch = torch
        previous = _get_current_function_mode()
        for normalize in (False, True):
            for fail in (False, True):
                with self.subTest(normalize=normalize, fail=fail):
                    model = fixture.model(torch.float16, normalize, fail=fail)
                    layer = model.model.layers[0]
                    x = torch.tensor([[[.125, -.5, 2.], [1., .25, -.125]]], dtype=torch.float16)
                    before = x.clone()
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'injected expert failure'):
                            probe.capture_contributions(torch, layer.mlp, layer.post_attention_layernorm, x)
                    else:
                        expected = probe.observe_moe(torch, layer.mlp, layer.post_attention_layernorm, x)
                        actual, records = probe.capture_contributions(torch, layer.mlp, layer.post_attention_layernorm, x)
                        self.assertEqual(actual, expected)
                        result = probe.replay_contributions(records, actual, x.reshape(-1).tolist(), 3, 2, 4)
                        self.assertEqual(result['fp16']['routed_output'], actual['routed_output'])
                        self.assertEqual(result['fp16']['output'], actual['output'])
                    self.assertTrue(torch.equal(x, before))
                    fixture.assert_no_hooks(model)
                    self.assertIs(_get_current_function_mode(), previous)

    def test_saved_pair_rechecks_complete_inputs_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = stage_tests.StageInputTests()
            _, pair, _, _ = fixture.run_fixture(root)
            model = root / 'model'
            result = probe.load_pair(pair, model.resolve(), 0)
            self.assertEqual(list(result[3]), ['hf', 'native'])
            self.assertEqual(result[3]['hf'], [.5] * 4)
            tasks_path = pair / 'layer0-tasks-0.json'
            original = tasks_path.read_text()
            tasks = json.loads(original)
            tasks[0]['inputs'][0]['hidden'][0] += .125
            tasks_path.write_text(json.dumps(tasks))
            with self.assertRaisesRegex(ValueError, 'paired task changed'):
                probe.load_pair(pair, model.resolve(), 0)
            tasks_path.write_text(original)
            refs_path = pair / 'layer0-hf.json'
            refs = json.loads(refs_path.read_text())
            refs['hf']['output'][0] += .0009765625
            refs_path.write_text(json.dumps(refs))
            with self.assertRaisesRegex(ValueError, 'gate no longer passes'):
                probe.load_pair(pair, model.resolve(), 0)

    def test_pipeline_preserves_old_failure_and_reports_counterfactual(self):
        records, observed = rounding_case()
        observed['router_logits'] = [0.] * 8
        config = dict(hidden_size=1, num_experts_per_tok=4, num_experts=4)
        parent = dict(cases=[dict(case='historical_failure', status='FAIL', failed_elements=1, reason='original')],
                      metadata=dict(git='historical', build_cache='sm_70', reference_environments={'0': {}}))
        native = copy.deepcopy(observed)
        native['output'] = [1.0009765625] * 2
        native['routed_output'] = [1.0009765625] * 2
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                expected = copy.deepcopy(observed)
                if corrupt:
                    expected['output'][0] += .0009765625
                data = (parent, config, [7, 9], dict(hf=[0., 0.], native=[0., 0.]),
                        dict(hf=expected, native=expected), dict(hf=native, native=native), {})

                def hf(model, config, layer, inputs, versions, observer):
                    """/** CPU流水线夹具仅替换GPU权重加载，调用真实捕获回调及重放/报告流程。 */"""
                    return ({name: observer(None, None, None, values) for name, values in inputs.items()}, {})

                argv = ['probe', '--pair', str(root / 'pair'), '--model', str(root / 'model'),
                        '--tokens', '0', '1', '--output', str(root / 'results')]
                with patch('sys.argv', argv), patch.object(probe, 'capture', return_value='CPU fixture'), \
                        patch.object(probe, 'load_pair', return_value=data), \
                        patch.object(probe, 'hf_pair', side_effect=hf), \
                        patch.object(probe, 'capture_contributions', return_value=(observed, records)), \
                        redirect_stdout(io.StringIO()):
                    code = probe.main()
                self.assertEqual(code, 1)
                output, = (root / 'results').iterdir()
                for suffix in ('json', 'md', 'csv', 'xlsx'):
                    self.assertTrue((output / f'summary.{suffix}').is_file())
                report = json.loads((output / 'summary.json').read_text())
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(report['metadata']['attribution_valid'], not corrupt)
                self.assertEqual(report['cases'][0]['case'], 'historical_failure')
                if not corrupt:
                    counterfactual = [row for row in report['cases'] if row.get('counterfactual')]
                    self.assertTrue(counterfactual)
                    self.assertTrue(all(row['exact_different_elements'] == 0 for row in counterfactual))
                else:
                    self.assertFalse(any(row.get('counterfactual') for row in report['cases']))
                    self.assertTrue(any(row['case'] == 'hf/hf_capture/output' and row['status'] == 'FAIL'
                                        for row in report['cases']))


if __name__ == '__main__':
    unittest.main()
