"""Router独立诊断的CPU/FFI参数检查；不编译、不调用CUDA或加载cuBLAS。"""

import ctypes
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from probe_qwen_moe_router_precision import PedanticGemm, exact_metrics, norm_variants
import probe_qwen_moe_router_precision as probe


class RouterProbeNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """/** 使用已有PyTorch CPU算子，不安装依赖或加载GPU。 */"""
        import torch
        cls.torch = torch

    def test_native_order_matches_independent_float32_tree(self):
        """/** 用独立标量FP32舍入重建线程累加及树归约，检查FP16边界与scale乘法。 */"""
        torch = self.torch
        generator = torch.Generator().manual_seed(20261115)
        x = torch.randn(3, 512, generator=generator).half()
        weight = torch.randn(512, generator=generator).half()
        hf, simulated = norm_variants(torch, x, weight, 1e-6)

        def f32(value):
            """/** 独立按IEEE float32舍入，不使用PyTorch归约实现。 */"""
            return struct.unpack('<f', struct.pack('<f', value))[0]

        expected = []
        for row in x.tolist():
            lanes = [f32(f32(row[i]*row[i]) + f32(row[i+256]*row[i+256])) for i in range(256)]
            for step in (128, 64, 32, 16, 8, 4, 2, 1):
                lanes = [f32(lanes[i]+lanes[i+step]) for i in range(step)]
            scale = torch.rsqrt(torch.tensor(f32(f32(lanes[0]/512) + f32(1e-6)))).item()
            expected.append(torch.tensor([f32(value*scale) for value in row]).half() * weight)
        self.assertTrue(torch.equal(simulated, torch.stack(expected)))
        expected_hf = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)+1e-6)).half()*weight
        self.assertTrue(torch.equal(hf, expected_hf))

    def test_invalid_shapes_types_and_nonfinite_rejected(self):
        """/** 不能把不支持的维度或无效值当作原生Norm模拟成功。 */"""
        torch = self.torch
        x, w = torch.ones(2, 256).half(), torch.ones(256).half()
        for values, weight, eps in ((x[:, :-1], w[:-1], 1e-6), (x.float(), w, 1e-6),
                                  (x, w[:-1], 1e-6), (x, w, 0), (x, w, float('nan')),
                                  (x*float('inf'), w, 1e-6)):
            with self.assertRaises(ValueError):
                norm_variants(torch, values, weight, eps)

    def test_exact_diagnostics_detect_one_half_step(self):
        """/** 一个FP16间隔也须计入定位差异；该标准不修改正式模型验收阈值。 */"""
        torch = self.torch
        a = torch.tensor([-4.0390625, -3.0], dtype=torch.float16)
        b = torch.tensor([-4.04296875, -3.0], dtype=torch.float16)
        result = exact_metrics(torch, a, b)
        self.assertEqual(result['failed_elements'], 1)
        self.assertEqual(result['max_abs_error'], 0.00390625)
        self.assertEqual(result['mean_abs_error'], 0.001953125)
        self.assertEqual(exact_metrics(torch, a, a)['status'], 'PASS')
        self.assertEqual(exact_metrics(torch, a*float('nan'), b)['failed_elements'], 2)
        with self.assertRaises(ValueError):
            exact_metrics(torch, a, b[:1])

    def test_setup_failure_still_writes_incomplete_reports(self):
        """/** CUDA不可用时不继续运行，四种报告保留失败且标记未完成。 */"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            argv = ['probe', '--diagnostic', str(root / 'source'), '--model', str(root / 'model'),
                    '--worker', str(root / 'worker'), '--output', str(root / 'reports')]
            with mock.patch('sys.argv', argv), mock.patch.object(probe, 'capture', return_value='cpu-test'), \
                    mock.patch.object(self.torch.cuda, 'is_available', return_value=False), \
                    mock.patch.dict('sys.modules', safetensors=SimpleNamespace(safe_open=mock.Mock())), \
                    mock.patch.object(probe, 'PedanticGemm') as gemm, redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main(), 1)
            gemm.assert_not_called()
            output, = (root / 'reports').iterdir()
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertFalse(report['complete'])
            self.assertIn('CUDA reference environment', report['cases'][0]['reason'])
            for suffix in ('md', 'csv', 'json', 'xlsx'):
                self.assertTrue((output / f'summary.{suffix}').is_file())


class RouterProbeBlasTests(unittest.TestCase):
    def library(self, fail_mode=False, fail_gemm=False):
        """/** 模拟C ABI返回值和handle写入，仅记录参数，不访问设备指针。 */"""
        library = SimpleNamespace()
        def create(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 1234
            return 0
        def version(handle, pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 120604
            return 0
        for name in ('cublasCreate_v2', 'cublasDestroy_v2', 'cublasSetStream_v2', 'cublasSetMathMode',
                     'cublasGetVersion_v2', 'cublasGemmEx'):
            setattr(library, name, mock.Mock(return_value=0))
        library.cublasCreate_v2.side_effect = create
        library.cublasGetVersion_v2.side_effect = version
        library.cublasSetMathMode.return_value = 7 if fail_mode else 0
        library.cublasGemmEx.return_value = 7 if fail_gemm else 0
        return library

    def tensor(self, shape, pointer):
        """/** 仅用于ABI约束的假张量；没有实际存储，错误调用会在测试中暴露。 */"""
        return SimpleNamespace(shape=shape, ndim=2, is_cuda=True, device='cuda:0', dtype='fp16',
                               is_contiguous=lambda: True, data_ptr=lambda: pointer)

    def test_layout_precision_and_private_handle_lifecycle(self):
        """/** 验证row-major转置、leading dimension、FP16输出和PEDANTIC枚举与正式调用一致。 */"""
        library = self.library()
        with mock.patch('ctypes.CDLL', return_value=library):
            gemm = PedanticGemm(Path('/unused/libcublas.so.12'), 9876)
        output = self.tensor((25, 60), 3000)
        torch = SimpleNamespace(float16='fp16', empty=mock.Mock(return_value=output))
        self.assertIs(gemm.linear(torch, self.tensor((25, 2048), 1000), self.tensor((60, 2048), 2000)), output)
        args = library.cublasGemmEx.call_args.args
        self.assertEqual(len(args), 19)
        self.assertEqual(args[1:6], (1, 0, 60, 25, 2048))
        self.assertEqual((args[7].value, args[8], args[9]), (2000, 2, 2048))
        self.assertEqual((args[10].value, args[11], args[12]), (1000, 2, 2048))
        self.assertEqual((args[14].value, *args[15:]), (3000, 2, 60, 69, -1))
        self.assertEqual(ctypes.cast(args[6], ctypes.POINTER(ctypes.c_float))[0], 1)
        self.assertEqual(ctypes.cast(args[13], ctypes.POINTER(ctypes.c_float))[0], 0)
        self.assertEqual(gemm.version, 120604)
        gemm.close()
        gemm.close()
        library.cublasDestroy_v2.assert_called_once()

    def test_configuration_failure_releases_handle(self):
        """/** stream/math mode配置失败也须回收私有handle。 */"""
        library = self.library(fail_mode=True)
        with mock.patch('ctypes.CDLL', return_value=library), self.assertRaisesRegex(RuntimeError, 'status=7'):
            PedanticGemm(Path('/unused'), 0)
        library.cublasDestroy_v2.assert_called_once()

    def test_gemm_failure_has_no_fallback_and_bad_inputs_are_rejected(self):
        """/** cuBLAS失败原样报错；布局不合要求时禁止进入C接口。 */"""
        library = self.library(fail_gemm=True)
        with mock.patch('ctypes.CDLL', return_value=library):
            gemm = PedanticGemm(Path('/unused'), 0)
        torch = SimpleNamespace(float16='fp16', empty=mock.Mock(return_value=self.tensor((1, 60), 3)))
        with self.assertRaisesRegex(RuntimeError, 'status=7'):
            gemm.linear(torch, self.tensor((1, 2048), 1), self.tensor((60, 2048), 2))
        with self.assertRaises(ValueError):
            gemm.linear(torch, self.tensor((1, 2048), 1), self.tensor((60, 1024), 2))
        self.assertEqual(library.cublasGemmEx.call_count, 1)
        gemm.close()


if __name__ == '__main__':
    unittest.main()
