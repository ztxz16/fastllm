"""独立Lt探针的CPU ABI及资源生命周期检查；不加载真实CUDA库、不编译。"""

import ctypes as ct
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from probe_qwen_moe_cublaslt import Heuristic, LtGemm, linked_cublaslt
import test_qwen_moe_router_probe as fixtures


class LtProbeTests(unittest.TestCase):
    def fixture(self, count=1, state=0, workspace=128, matmul_status=0, fail_layout=0):
        """/** 模拟CUDA公开ABI的返回结构及描述创建，所有设备地址均为未解引用的整数。 */"""
        library, handles, layouts, attrs = SimpleNamespace(), [], [], []
        def create(pointer, *args):
            handle = len(handles)+100
            handles.append(handle)
            ct.cast(pointer, ct.POINTER(ct.c_void_p))[0] = handle
            return 0
        def layout(pointer, *args):
            layouts.append(args)
            return 7 if len(layouts) == fail_layout else create(pointer, *args)
        def attribute(descriptor, key, value, size):
            attrs.append((descriptor.value, key, ct.string_at(value, size)))
            return 0
        def heuristic(*args):
            result = ct.cast(args[8], ct.POINTER(Heuristic)).contents
            result.workspaceSize, result.state, result.algo[0] = workspace, state, 123
            ct.cast(args[9], ct.POINTER(ct.c_int))[0] = count
            return 0
        for kind in ('cublasLt', 'cublasLtMatmulDesc', 'cublasLtMatrixLayout', 'cublasLtMatmulPreference'):
            setattr(library, kind+'Create', mock.Mock(side_effect=layout if kind == 'cublasLtMatrixLayout' else create))
            setattr(library, kind+'Destroy', mock.Mock(return_value=0))
        for kind in ('cublasLtMatmulDesc', 'cublasLtMatmulPreference'):
            setattr(library, kind+'SetAttribute', mock.Mock(side_effect=attribute))
        library.cublasLtGetVersion = mock.Mock(return_value=120604)
        library.cublasLtMatmulAlgoGetHeuristic = mock.Mock(side_effect=heuristic)
        library.cublasLtMatmul = mock.Mock(return_value=matmul_status)
        tensor = fixtures.RouterProbeBlasTests().tensor
        x, w, bias = tensor((26, 2048), 256), tensor((3072, 2048), 512), tensor((3072,), 768)
        output, scratch = tensor((26, 3072), 1024), tensor((1024,), 1280)
        torch = SimpleNamespace(float16='fp16', uint8='byte',
            empty=mock.Mock(side_effect=[output, scratch]),
            cuda=SimpleNamespace(current_stream=mock.Mock(return_value=SimpleNamespace(cuda_stream=9876))))
        with mock.patch('ctypes.CDLL', return_value=library):
            gemm = LtGemm(Path('/unused'), 1024)
        return gemm, library, torch, x, w, bias, output, layouts, attrs

    def test_bias_epilogue_abi_layout_stream_and_cleanup(self):
        """/** 核对FP16布局、FP32计算、beta=0、bias指针和当前stream，以及描述释放。 */"""
        gemm, lib, torch, x, w, bias, output, layouts, attrs = self.fixture()
        self.assertIs(gemm.linear(torch, x, w, bias), output)
        self.assertEqual(ct.sizeof(Heuristic), 96)
        self.assertEqual(layouts, [(2, 2048, 3072, 2048), (2, 2048, 26, 2048), (2, 3072, 26, 3072)])
        self.assertEqual(lib.cublasLtMatmulDescCreate.call_args.args[1:], (68, 0))
        desc = lib.cublasLtMatmulDescSetAttribute.call_args_list[0].args[0].value
        actual = {key: int.from_bytes(data, 'little') for handle, key, data in attrs if handle == desc}
        self.assertEqual(actual, {3: 1, 4: 0, 7: 4, 8: 768})
        args = lib.cublasLtMatmul.call_args.args
        self.assertEqual(ct.cast(args[2], ct.POINTER(ct.c_float))[0], 1)
        self.assertEqual(ct.cast(args[7], ct.POINTER(ct.c_float))[0], 0)
        self.assertEqual([args[i].value for i in (3, 5, 8, 10, 13, 15)], [512, 256, 1024, 1024, 1280, 9876])
        self.assertEqual(args[14], 1024)
        self.assertEqual(gemm.calls[0]['algorithm_words'][0], 123)
        self.assertEqual(gemm.calls[0]['workspace_required'], 128)
        lib.cublasLtMatmulDescDestroy.assert_called_once()
        self.assertEqual(lib.cublasLtMatrixLayoutDestroy.call_count, 3)
        lib.cublasLtMatmulPreferenceDestroy.assert_called_once()
        gemm.close()
        gemm.close()
        lib.cublasLtDestroy.assert_called_once()

    def test_no_bias_keeps_default_epilogue(self):
        """/** 无bias时不设置bias属性，仍以beta=0执行并记录真实配置。 */"""
        gemm, lib, torch, x, w, _, _, _, _ = self.fixture()
        gemm.linear(torch, x, w)
        self.assertEqual([call.args[1] for call in lib.cublasLtMatmulDescSetAttribute.call_args_list], [3, 4])
        self.assertEqual(gemm.calls[0]['epilogue'], 'DEFAULT')
        gemm.close()

    def test_unsupported_heuristic_does_not_execute_and_releases_descriptors(self):
        """/** 无算法、算法错误或超出工作区都直接失败，不尝试其他后端。 */"""
        for options in ({'count': 0}, {'state': 7}, {'workspace': 2048}):
            gemm, lib, torch, x, w, bias, *_ = self.fixture(**options)
            with self.subTest(options=options), self.assertRaisesRegex(RuntimeError, 'no valid algorithm'):
                gemm.linear(torch, x, w, bias)
            lib.cublasLtMatmul.assert_not_called()
            self.assertEqual(lib.cublasLtMatrixLayoutDestroy.call_count, 3)
            lib.cublasLtMatmulPreferenceDestroy.assert_called_once()
            gemm.close()

    def test_partial_setup_and_execution_failures_release_resources(self):
        """/** 中途创建或执行失败时保留失败状态，且只释放已成功创建的描述。 */"""
        for options, layout_count in (({'fail_layout': 2}, 1), ({'matmul_status': 7}, 3)):
            gemm, lib, torch, x, w, bias, *_ = self.fixture(**options)
            with self.subTest(options=options), self.assertRaisesRegex(RuntimeError, 'status=7'):
                gemm.linear(torch, x, w, bias)
            self.assertEqual(lib.cublasLtMatrixLayoutDestroy.call_count, layout_count)
            lib.cublasLtMatmulDescDestroy.assert_called_once()
            self.assertFalse(gemm.calls)
            gemm.close()

    def test_invalid_input_rejected_before_allocation(self):
        """/** 错误bias与形状不创建描述或输出缓冲，已关闭handle不能再执行。 */"""
        gemm, lib, torch, x, w, *_ = self.fixture()
        with self.assertRaisesRegex(ValueError, 'bias'):
            gemm.linear(torch, x, w, fixtures.RouterProbeBlasTests().tensor((3071,), 768))
        torch.empty.assert_not_called()
        lib.cublasLtMatmulDescCreate.assert_not_called()
        gemm.close()
        with self.assertRaisesRegex(ValueError, 'inputs/handle'):
            gemm.linear(torch, x, w)

    def test_library_resolution_requires_existing_native_dependency(self):
        """/** 仅接受原生cuBLAS依赖列出的现有Lt路径，拒绝缺库后的系统搜索fallback。 */"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'libcublasLt.so.12'
            path.touch()
            with mock.patch('subprocess.run', return_value=SimpleNamespace(stdout=f' libcublasLt.so.12 => {path} (0x0)')):
                self.assertEqual(linked_cublaslt(Path('/native/libcublas.so.12')), path)
            with mock.patch('subprocess.run', return_value=SimpleNamespace(stdout='libcublasLt.so.12 => not found')):
                with self.assertRaisesRegex(RuntimeError, 'cannot resolve'):
                    linked_cublaslt(Path('/native/libcublas.so.12'))


if __name__ == '__main__':
    unittest.main()
