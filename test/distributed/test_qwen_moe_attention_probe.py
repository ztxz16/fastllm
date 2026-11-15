"""Attention独立诊断的CPU与模拟C ABI检查；不编译或执行CUDA。"""

import ctypes
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import probe_qwen_moe_attention_precision as probe
import test_qwen_moe_router_probe as fixtures


class AttentionProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """/** 仅导入已有CPU PyTorch用于张量与hook验证。 */"""
        import torch
        cls.torch = torch

    def attention(self, width=4, bias=True, fail=False):
        """/** 可观测的简单模块，检查实验注入与hook清理；不模拟真实Attention内核。 */"""
        torch = self.torch
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(width, width, bias=bias).half()
                self.k_proj = torch.nn.Linear(width, width, bias=bias).half()
                self.v_proj = torch.nn.Linear(width, width, bias=bias).half()
                self.o_proj = torch.nn.Linear(width, width, bias=False).half()

            def forward(self, hidden_states, **kwargs):
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
                if fail:
                    raise RuntimeError('injected attention failure')
                return self.o_proj(q+k+v), None
        return Attention()

    def invoke(self, attention, mode, gemm=None, x=None):
        """/** 固定CPU输入并经同一实验函数调用，不触发参考环境或原生二进制。 */"""
        torch = self.torch
        if x is None:
            x = torch.ones(1, 2, attention.q_proj.weight.shape[1]).half()
        return probe.attention_variant(torch, attention, x, x*.5, x,
                                       torch.arange(x.shape[1]).reshape(1, -1), None, mode, gemm)

    def test_separate_bias_exposes_double_rounding(self):
        """/** 精确构造一次与两次FP16舍入的差别，确保bias实验确实改变所测因素。 */"""
        torch = self.torch
        attention = self.attention(width=1)
        with torch.no_grad():
            attention.q_proj.weight.fill_(1.0009765625)
            attention.q_proj.bias.fill_(-1.001953125)
        x = torch.tensor([[[1.0009765625]]], dtype=torch.float16)
        _, hf = self.invoke(attention, 'hf_replay', x=x)
        _, split = self.invoke(attention, 'split_bias', x=x)
        self.assertEqual(hf['q'].item(), 2**-20)
        self.assertEqual(split['q'].item(), 0)
        for module in attention.modules():
            self.assertFalse(module._forward_hooks)

    def test_modes_inject_only_selected_projection_and_leave_weights(self):
        """/** 检查合并/FP32投影选择、norm分支与权重只读，覆盖有无bias。 */"""
        torch = self.torch
        torch.manual_seed(17)
        def linear(torch, x, weight, fp32=False, bias=None):
            value = torch.nn.functional.linear(x.float(), weight.float(), None if bias is None else bias.float())
            return value if fp32 else value.half()
        for bias in (False, True):
            attention = self.attention(bias=bias)
            saved = {k: v.clone() for k, v in attention.state_dict().items()}
            for mode in probe.MODES:
                with self.subTest(bias=bias, mode=mode):
                    gemm = SimpleNamespace(linear=mock.Mock(side_effect=linear))
                    output, raw = self.invoke(attention, mode, gemm)
                    self.assertTrue(torch.equal(output, raw['output']))
                    self.assertEqual(set(raw), {'norm', 'q', 'k', 'v', 'context', 'o', 'output'})
                    self.assertTrue(torch.equal(raw['norm'], torch.full_like(raw['norm'],
                        .5 if mode in ('native_norm', 'combined_projections', 'current_projections') else 1)))
                    calls = gemm.linear.call_args_list
                    expected_count = {'merged_qkv': 1, 'fp32_o': 1, 'combined_projections': 2,
                                      'fused_qkv': 1, 'current_projections': 2}.get(mode, 0)
                    self.assertEqual(len(calls), expected_count)
                    if mode in ('merged_qkv', 'combined_projections', 'fused_qkv', 'current_projections'):
                        self.assertEqual(calls[0].args[2].shape, (12, 4))
                        self.assertFalse(calls[0].kwargs.get('fp32', False))
                        self.assertEqual(calls[0].kwargs['bias'] is not None,
                                         bias and mode in ('fused_qkv', 'current_projections'))
                    if mode in ('fp32_o', 'combined_projections', 'current_projections'):
                        self.assertTrue(calls[-1].kwargs['fp32'])
                        self.assertEqual(calls[-1].args[2].shape, (4, 4))
                    for module in attention.modules():
                        self.assertFalse(module._forward_hooks)
                    for key, value in attention.state_dict().items():
                        self.assertTrue(torch.equal(value, saved[key]))

    def test_fused_projection_adds_bias_before_the_only_half_rounding(self):
        """/** 消去用例必须区分当前融合bias与旧单独bias，并且不会重复加bias。 */"""
        torch = self.torch
        attention = self.attention(width=1)
        with torch.no_grad():
            attention.q_proj.weight.fill_(1.0009765625)
            attention.q_proj.bias.fill_(-1.001953125)
        def linear(torch, x, weight, fp32=False, bias=None):
            value = torch.nn.functional.linear(x.float(), weight.float(), None if bias is None else bias.float())
            return value if fp32 else value.half()
        gemm = SimpleNamespace(linear=linear)
        x = torch.tensor([[[1.0009765625]]], dtype=torch.float16)
        _, fused = self.invoke(attention, 'fused_qkv', gemm, x)
        _, legacy = self.invoke(attention, 'merged_qkv', gemm, x)
        _, hf = self.invoke(attention, 'hf_replay', x=x)
        self.assertEqual(fused['q'].item(), 2**-20)
        self.assertEqual(fused['q'].item(), hf['q'].item())
        self.assertEqual(legacy['q'].item(), 0)

    def test_router_replay_preserves_half_weights_and_normalization(self):
        """/** 通过独立的常量logits检查softmax、TopK和可选归一化，再验证路由边界记录。 */"""
        torch = self.torch
        config = SimpleNamespace(hidden_size=1, num_experts_per_tok=2, norm_topk_prob=False)
        hidden = torch.ones(1, 1, 1, dtype=torch.float16)
        gate = torch.tensor([[0.], [2.], [1.], [-1.]], dtype=torch.float16)
        output = probe.replay_router(torch, hidden, torch.nn.Identity(), gate, config)
        self.assertEqual(output['selected_experts'], [1, 2])
        expected = torch.softmax(gate[:, 0].float(), dim=0)[[1, 2]]
        self.assertEqual(output['routing_weights'], expected.half().tolist())
        config.norm_topk_prob = True
        normalized = probe.replay_router(torch, hidden, torch.nn.Identity(), gate, config)
        self.assertEqual(normalized['routing_weights'], (expected/expected.sum()).half().tolist())

        tied = dict(router_logits=[0., 2., 2., -1.], selected_experts=[1, 2], routing_weights=[.4, .4])
        swapped = dict(tied, selected_experts=[2, 1])
        self.assertEqual(probe.routing_comparison('tied', tied, swapped, 4, 2)['status'], 'PASS')
        non_tied = dict(tied, router_logits=[0., 2.001953125, 2., -1.])
        self.assertEqual(probe.routing_comparison('order', non_tied, swapped, 4, 2)['status'], 'FAIL')
        changed = dict(output, selected_experts=[1, 0])
        row = probe.routing_comparison('set', changed, output, 4, 2)
        self.assertEqual(row['failed_elements'], 1)
        self.assertEqual(row['details'][0]['token'], 0)
        self.assertEqual([item['expert'] for item in row['details'][0]['scores']], [0, 1, 2])

    def test_router_comparison_rejects_malformed_shapes_and_ids(self):
        """/** 拒绝不完整或超长字段及非法专家ID，避免错误切片掩盖坏输入。 */"""
        values = dict(router_logits=[0., 2., 1., -1.], selected_experts=[1, 2], routing_weights=[.6, .2])
        for field in values:
            for delta in (-1, 1):
                bad = dict(values, **{field: values[field][:-1] if delta < 0 else values[field] + [0]})
                with self.subTest(field=field, delta=delta), self.assertRaisesRegex(ValueError, 'shape mismatch'):
                    probe.routing_comparison('invalid', values, bad, 4, 2)
        for expert in (-1, 4, .5, float('nan'), True):
            bad = dict(values, selected_experts=[1, expert])
            with self.subTest(expert=expert), self.assertRaisesRegex(ValueError, 'invalid expert ID'):
                probe.routing_comparison('invalid', bad, values, 4, 2)

    def test_exception_removes_hooks_and_invalid_modes_reject(self):
        """/** 子模块异常或非法实验名称不遗留hook，不以其他后端掩盖失败。 */"""
        attention = self.attention(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            self.invoke(attention, 'split_bias')
        for module in attention.modules():
            self.assertFalse(module._forward_hooks)
        with self.assertRaises(ValueError):
            self.invoke(attention, 'unknown')

    def test_baseline_mismatch_stops_attribution(self):
        """/** 参考重放不一致时，即使实验更接近原生日志也不得继续作因果判断。 */"""
        probe.require_hf_replay({'status': 'PASS'})
        with self.assertRaisesRegex(RuntimeError, 'stop projection attribution'):
            probe.require_hf_replay({'status': 'FAIL'})

    def test_missing_cuda_records_incomplete_four_format_report(self):
        """/** 无GPU提前失败也生成全部报告格式，禁止进入cuBLAS或加载完整模型。 */"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            argv = ['probe', '--diagnostic', str(root/'source'), '--model', str(root/'model'),
                    '--worker', str(root/'worker'), '--output', str(root/'reports')]
            with mock.patch('sys.argv', argv), mock.patch.object(probe, 'capture', return_value='cpu-test'), \
                    mock.patch.object(self.torch.cuda, 'is_available', return_value=False), \
                    mock.patch.object(probe, 'MoeGemm') as gemm, redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main(), 1)
            gemm.assert_not_called()
            output, = (root/'reports').iterdir()
            report = json.loads((output/'summary.json').read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertFalse(report['complete'])
            self.assertEqual(report['expected_case_count'], 96)
            for suffix in ('md', 'csv', 'json', 'xlsx'):
                self.assertTrue((output/f'summary.{suffix}').is_file())


class AttentionProbeBlasTests(unittest.TestCase):
    def test_current_moe_abi_for_half_and_float_output(self):
        """/** 验证32F/TENSOR_OP、矩阵方向、FP16/FP32输出类型及资源释放。 */"""
        fixture = fixtures.RouterProbeBlasTests()
        library = fixture.library()
        with mock.patch('ctypes.CDLL', return_value=library):
            gemm = probe.MoeGemm(Path('/unused'), 9876)
        for fp32 in (False, True):
            output = fixture.tensor((25, 60), 3000)
            torch = SimpleNamespace(float16='fp16', float32='fp32', empty=mock.Mock(return_value=output))
            self.assertIs(gemm.linear(torch, fixture.tensor((25, 2048), 1000),
                                      fixture.tensor((60, 2048), 2000), fp32=fp32), output)
            args = library.cublasGemmEx.call_args.args
            self.assertEqual(args[1:6], (1, 0, 60, 25, 2048))
            self.assertEqual((args[7].value, args[8], args[9]), (2000, 2, 2048))
            self.assertEqual((args[10].value, args[11], args[12]), (1000, 2, 2048))
            self.assertEqual((args[14].value, *args[15:]), (3000, 0 if fp32 else 2, 60, 68, 99))
            self.assertEqual(ctypes.cast(args[6], ctypes.POINTER(ctypes.c_float))[0], 1)
            self.assertEqual(ctypes.cast(args[13], ctypes.POINTER(ctypes.c_float))[0], 0)
            self.assertEqual(torch.empty.call_args.kwargs['dtype'], 'fp32' if fp32 else 'fp16')
        gemm.close()
        gemm.close()
        library.cublasDestroy_v2.assert_called_once()

    def test_bias_broadcast_precedes_beta_one_gemm_and_invalid_bias_rejects(self):
        """/** 检查有bias的C初始化与beta=1，覆盖FP16/FP32输出和偏置维度拒绝。 */"""
        fixture = fixtures.RouterProbeBlasTests()
        library = fixture.library()
        with mock.patch('ctypes.CDLL', return_value=library):
            gemm = probe.MoeGemm(Path('/unused'), 9876)
        x, weight, bias = fixture.tensor((26, 2048), 1), fixture.tensor((60, 2048), 2), fixture.tensor((60,), 3)
        for fp32 in (False, True):
            events = []
            output = fixture.tensor((26, 60), 4)
            output.copy_ = mock.Mock(side_effect=lambda _: events.append('bias'))
            library.cublasGemmEx.side_effect = lambda *args: events.append('gemm') or 0
            torch = SimpleNamespace(float16='fp16', float32='fp32', empty=mock.Mock(return_value=output))
            gemm.linear(torch, x, weight, fp32=fp32, bias=bias)
            output.copy_.assert_called_once_with(bias)
            self.assertEqual(events, ['bias', 'gemm'])
            self.assertEqual(ctypes.cast(library.cublasGemmEx.call_args.args[13], ctypes.POINTER(ctypes.c_float))[0], 1)
        library.cublasGemmEx.reset_mock()
        with self.assertRaisesRegex(ValueError, 'bias'):
            gemm.linear(torch, x, weight, bias=fixture.tensor((59,), 3))
        library.cublasGemmEx.assert_not_called()
        gemm.close()

    def test_invalid_layout_and_blas_error_do_not_fallback(self):
        """/** 无效尺寸不进入C接口，cuBLAS运行失败直接上报。 */"""
        fixture = fixtures.RouterProbeBlasTests()
        library = fixture.library(fail_gemm=True)
        with mock.patch('ctypes.CDLL', return_value=library):
            gemm = probe.MoeGemm(Path('/unused'), 0)
        torch = SimpleNamespace(float16='fp16', empty=mock.Mock(return_value=fixture.tensor((1, 60), 3)))
        with self.assertRaises(ValueError):
            gemm.linear(torch, fixture.tensor((1, 2048), 1), fixture.tensor((60, 1024), 2))
        library.cublasGemmEx.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'status=7'):
            gemm.linear(torch, fixture.tensor((1, 2048), 1), fixture.tensor((60, 2048), 2))
        gemm.close()


if __name__ == '__main__':
    unittest.main()
