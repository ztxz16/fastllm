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
        def linear(torch, x, weight, fp32=False):
            value = torch.nn.functional.linear(x.float(), weight.float())
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
                        .5 if mode in ('native_norm', 'combined_projections') else 1)))
                    calls = gemm.linear.call_args_list
                    expected_count = {'merged_qkv': 1, 'fp32_o': 1, 'combined_projections': 2}.get(mode, 0)
                    self.assertEqual(len(calls), expected_count)
                    if mode in ('merged_qkv', 'combined_projections'):
                        self.assertEqual(calls[0].args[2].shape, (12, 4))
                        self.assertFalse(calls[0].kwargs.get('fp32', False))
                    if mode in ('fp32_o', 'combined_projections'):
                        self.assertTrue(calls[-1].kwargs['fp32'])
                        self.assertEqual(calls[-1].args[2].shape, (4, 4))
                    for module in attention.modules():
                        self.assertFalse(module._forward_hooks)
                    for key, value in attention.state_dict().items():
                        self.assertTrue(torch.equal(value, saved[key]))

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
            self.assertEqual(report['expected_case_count'], 72)
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
