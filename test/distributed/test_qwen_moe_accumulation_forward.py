"""完整累加实验的CPU检查：真实公开前向、舍入变化、隔离恢复及历史失败保留。"""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import probe_qwen_moe_accumulation_forward as probe
from test_qwen_moe_forward_recheck import saved_probe
import test_qwen_moe_routing_observer as routing_tests


class AccumulationForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """/** 仅使用已安装PyTorch的CPU算子；不下载、不访问GPU、不编译。 */"""
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('requires existing PyTorch for CPU operations only')
        cls.torch = torch

    def model(self, *, rounding=False, fail=False, corrupt=False):
        """/** 两层公开模型夹具，四专家贡献产生可手算的FP16半ULP累加差异。 */"""
        torch = self.torch
        fixture = routing_tests.RoutingObserverTests()
        fixture.torch = torch
        model = fixture.model(torch.float16, False, fail=fail, corrupt=corrupt)
        model.model.norm = torch.nn.Identity()
        original = model.forward

        def forward(input_ids, use_cache=False):
            """/** 使用真实层和Tensor运算，提供标准logits返回约定。 */"""
            return SimpleNamespace(logits=model.model.norm(original(input_ids, use_cache=use_cache)))

        model.forward = forward
        if rounding:
            model.config.num_experts_per_tok = 4
            torch.nn.init.zeros_(model.model.embed_tokens.weight)
            for layer in model.model.layers:
                block = layer.mlp
                block.experts = torch.nn.ModuleList(torch.nn.Linear(3, 3, dtype=torch.float16) for _ in range(4))
                for expert, value in zip(block.experts, (4., 2.**-9, 2.**-9, 0.)):
                    torch.nn.init.zeros_(expert.weight)
                    torch.nn.init.constant_(expert.bias, value)
                torch.nn.init.zeros_(block.shared_expert.weight)

                def moe_forward(hidden, block=block):
                    """/** 原始HF式四专家逐次FP16累加，未接入实验重放函数。 */"""
                    shape = hidden.shape
                    hidden = hidden.reshape(-1, 3)
                    logits = block.gate(hidden)
                    weights, selected = torch.topk(torch.nn.functional.softmax(logits, dim=1, dtype=torch.float), 4)
                    weights = weights.to(hidden.dtype)
                    routed = torch.zeros_like(hidden)
                    for expert in range(4):
                        token, slot = torch.where(selected == expert)
                        routed.index_add_(0, token, block.experts[expert](hidden[token]) * weights[token, slot, None])
                    shared = block.shared_expert(hidden) * torch.sigmoid(block.shared_expert_gate(hidden))
                    return (routed + shared).reshape(shape), logits

                block.forward = moe_forward
        return model

    def test_plain_capture_exact_and_fp32_changes_downstream_input(self):
        torch = self.torch
        from torch.overrides import _get_current_function_mode
        model = self.model(rounding=True)
        previous = _get_current_function_mode()
        with torch.inference_mode():
            before = model(torch.tensor([[1, 2]])).logits.clone()
        baseline = probe.observed_forward(model, torch, [1, 2])
        candidate = probe.observed_forward(model, torch, [1, 2], fp32=True)
        self.assertEqual(baseline['logits'], [2.5] * 3)
        self.assertEqual(candidate['logits'], [2.501953125] * 3)
        self.assertEqual(baseline['layers'][0]['output'], [1.25] * 6)
        self.assertEqual(candidate['layers'][0]['output'], [1.2509765625] * 6)
        self.assertEqual(candidate['layers'][1]['input'], candidate['layers'][0]['output'])
        for field in ('router_logits', 'routing_weights', 'shared_output', 'attention_input'):
            self.assertEqual(baseline['layers'][0][field], candidate['layers'][0][field])
        routing_tests.RoutingObserverTests().assert_no_hooks(model)
        self.assertIs(_get_current_function_mode(), previous)
        with torch.inference_mode():
            self.assertTrue(torch.equal(before, model(torch.tensor([[1, 2]])).logits))

    def test_original_forward_preserved_for_real_tensor_observation(self):
        model, torch = self.model(), self.torch
        with torch.inference_mode():
            expected = model(torch.tensor([[1, 2, 3]])).logits[0, -1].float().tolist()
        actual = probe.observed_forward(model, torch, [1, 2, 3])
        self.assertEqual(actual['logits'], expected)
        self.assertEqual(actual['layers'][1]['input'], actual['layers'][0]['output'])
        routing_tests.RoutingObserverTests().assert_no_hooks(model)

    def test_failure_and_composition_mismatch_remove_all_hooks(self):
        from torch.overrides import _get_current_function_mode
        previous = _get_current_function_mode()
        for kwargs, message in ((dict(fail=True), 'injected expert failure'), (dict(corrupt=True), 'do not reconstruct')):
            for fp32 in (False, True):
                with self.subTest(kwargs=kwargs, fp32=fp32):
                    model = self.model(**kwargs)
                    with self.assertRaisesRegex(RuntimeError, message):
                        probe.observed_forward(model, self.torch, [1, 2], fp32=fp32)
                    routing_tests.RoutingObserverTests().assert_no_hooks(model)
                    self.assertIs(_get_current_function_mode(), previous)

    def test_reduction_rejects_bad_coverage_product_and_replay(self):
        torch = self.torch
        record = dict(selected_experts=torch.tensor([[0, 1], [0, 1]]),
                      routing_weights=torch.ones((2, 2), dtype=torch.float16),
                      routed_output=torch.ones((2, 1), dtype=torch.float16),
                      shared_output=torch.zeros((2, 1), dtype=torch.float16))
        records = [(expert, torch.tensor([1, 0]), torch.full((2, 1), .5, dtype=torch.float16),
                    torch.full((2, 1), .5, dtype=torch.float16)) for expert in (0, 1)]
        single, candidate = probe.reduce_captured(torch, record, records)
        self.assertEqual(single.flatten().tolist(), [1., 1.])
        self.assertEqual(candidate.flatten().tolist(), [1., 1.])
        for mode in ('missing', 'duplicate', 'token', 'product', 'nan', 'replay', 'dtype'):
            with self.subTest(mode=mode):
                obs, values = copy.deepcopy(record), copy.deepcopy(records)
                if mode == 'missing':
                    values.pop()
                elif mode == 'duplicate':
                    values.append(values[0])
                elif mode == 'token':
                    values[0][1][0] = 0
                elif mode == 'product':
                    values[0][3][0] = .25
                elif mode == 'nan':
                    values[0][2][0] = float('nan')
                elif mode == 'replay':
                    obs['routed_output'][0] = 0
                else:
                    obs['shared_output'] = obs['shared_output'].float()
                with self.assertRaises(ValueError):
                    probe.reduce_captured(torch, obs, values)

    def test_saved_evidence_validates_trace_and_model_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = saved_probe(root)
            evidence = probe.load_evidence(directory, root / 'model')
            self.assertEqual(evidence['ids'], [1, 2])
            self.assertIn(str(directory / 'summary.json'), evidence['hashes'])
            config_path = root / 'model/config.json'
            config = json.loads(config_path.read_text())
            config['hidden_size'] = 3
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'model config differs'):
                probe.load_evidence(directory, root / 'model')

    def test_pipeline_keeps_old_failure_and_stops_candidate_on_bad_gate(self):
        model, torch = self.model(rounding=True), self.torch
        baseline = probe.observed_forward(model, torch, [1, 2])
        reference = dict(metadata={}, layers=[dict(name=f'real_layer_{i}_moe', reference=[{
            key: row[key] for key in ('output', 'router_logits', 'selected_experts', 'routing_weights', 'routed_output', 'shared_output')
        }]) for i, row in enumerate(baseline['layers'])])
        data = dict(report=dict(cases=[dict(case='historical_failure', status='FAIL', failed_elements=1, reason='original')],
            metadata=dict(git='saved', build_cache='sm_70', source_rank=0, head_attribution_valid=False)),
            config=dict(hidden_size=3, vocab_size=3, num_experts_per_tok=4), ids=[1, 2], reference=reference,
            hf=baseline['trajectory'], native=baseline['trajectory'], logits=baseline['logits'],
            native_logits=[3.5] * 3, hashes={})
        for bad in (False, True):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                current = copy.deepcopy(data)
                if bad:
                    current['logits'][0] += .001
                argv = ['probe', '--probe', str(root / 'source'), '--model', str(root / 'model'), '--output', str(root)]
                with patch('sys.argv', argv), patch.object(probe, 'capture', return_value='CPU fixture'), \
                        patch.object(probe, 'load_evidence', return_value=current), \
                        patch.object(probe, 'load_model', return_value=(torch, model, {})), \
                        patch.object(probe, 'observed_forward', wraps=probe.observed_forward) as forward, \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(probe.main(), 1)
                output, = root.iterdir()
                report = json.loads((output / 'summary.json').read_text())
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(report['cases'][0]['case'], 'historical_failure')
                self.assertEqual(report['metadata']['attribution_valid'], not bad)
                self.assertEqual(forward.call_count, 1 if bad else 2)
                self.assertEqual((output / 'candidate.json').exists(), not bad)
                for suffix in ('json', 'csv', 'md', 'xlsx'):
                    self.assertTrue((output / ('summary.' + suffix)).is_file())
                if not bad:
                    rows = {row['case']: row for row in report['cases']}
                    self.assertTrue(rows['native_vs_fp32_candidate/logits']['counterfactual'])
                    self.assertEqual(rows['gate/plain_hf_after_candidate']['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
