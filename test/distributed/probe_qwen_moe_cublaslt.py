"""独立cuBLASLt bias epilogue诊断；仅加载已有动态库，不编译或替换正式算子。"""

from contextlib import ExitStack
import ctypes as ct
from pathlib import Path
import re
import subprocess


class Heuristic(ct.Structure):
    """/** CUDA公开ABI：64字节算法描述、工作区大小、状态、waves和保留字段。 */"""
    _fields_ = [('algo', ct.c_uint64 * 8), ('workspaceSize', ct.c_size_t),
                ('state', ct.c_int), ('wavesCount', ct.c_float), ('reserved', ct.c_int * 4)]


def linked_cublaslt(cublas):
    """/** 从已确认的原生cuBLAS依赖中解析Lt，缺失即失败，不换用另一套CUDA库。 */"""
    text = subprocess.run(['ldd', str(cublas)], check=True, capture_output=True, text=True, timeout=15).stdout
    match = re.search(r'^\s*libcublasLt\.so(?:\.\d+)*\s+=>\s+(\S+)', text, re.M)
    if not match or not Path(match[1]).is_file():
        raise RuntimeError('cannot resolve native cuBLASLt dependency: ' + text)
    return Path(match[1]).resolve()


class LtGemm:
    """/** 独立FP16 Lt GEMM，固定32F计算和显式bias epilogue，失败不fallback。 */"""

    def __init__(self, library, workspace_bytes=1024*1024):
        """/** 加载指定Lt库并创建私有handle；记录每次实际选中的算法及矩阵形状。 */"""
        if not 0 <= workspace_bytes <= 1024**3:
            raise ValueError('Lt workspace must be between 0 and 1 GiB')
        self.lib, self.workspace_bytes = ct.CDLL(str(library)), workspace_bytes
        p, i, size = ct.c_void_p, ct.c_int, ct.c_size_t
        signatures = {
            'cublasLtCreate': [ct.POINTER(p)], 'cublasLtDestroy': [p],
            'cublasLtMatmulDescCreate': [ct.POINTER(p), i, i],
            'cublasLtMatmulDescDestroy': [p],
            'cublasLtMatmulDescSetAttribute': [p, i, p, size],
            'cublasLtMatrixLayoutCreate': [ct.POINTER(p), i, ct.c_uint64, ct.c_uint64, ct.c_int64],
            'cublasLtMatrixLayoutDestroy': [p],
            'cublasLtMatmulPreferenceCreate': [ct.POINTER(p)],
            'cublasLtMatmulPreferenceDestroy': [p],
            'cublasLtMatmulPreferenceSetAttribute': [p, i, p, size],
            'cublasLtMatmulAlgoGetHeuristic': [p, p, p, p, p, p, p, i, ct.POINTER(Heuristic), ct.POINTER(i)],
            'cublasLtMatmul': [p]*14 + [size, p],
        }
        for name, signature in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = signature, i
        self.lib.cublasLtGetVersion.argtypes, self.lib.cublasLtGetVersion.restype = [], size
        self.version = self.lib.cublasLtGetVersion()
        self.handle, self.calls = p(), []
        self.check(self.lib.cublasLtCreate(ct.byref(self.handle)), 'create')

    @staticmethod
    def check(status, phase):
        """/** 带阶段报告Lt失败，保留实际后端，不改用GemmEx或PyTorch。 */"""
        if status:
            raise RuntimeError(f'cuBLASLt {phase} failed: status={status}')

    def close(self):
        """/** 释放私有handle，可重复调用；调用方先完成输出读取以暴露异步错误。 */"""
        if self.handle:
            self.check(self.lib.cublasLtDestroy(self.handle), 'destroy')
            self.handle = ct.c_void_p()

    def linear(self, torch, x, weight, bias=None):
        """/**
         * 在当前CUDA stream计算Y[T,N]=X[T,K]*W[N,K]^T+bias[N]。
         * 独立实验使用FP32计算、FP16输入输出、beta=0和可选BIAS epilogue。
         * 每次创建描述并查询一个算法，记录其原始配置；不缓存、不计时、不fallback。
         * @param torch 已安装的PyTorch模块，仅提供设备张量及当前stream。
         * @param x 连续非空FP16 CUDA [T,K]。
         * @param weight 同设备连续FP16 [N,K]。
         * @param bias 可空的同设备连续FP16 [N]，只由epilogue相加一次。
         * @return FP16 [T,N]；参数、算法或执行失败抛异常，描述在异常路径同样释放。
         */"""
        if (not self.handle or x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
                or not x.is_cuda or x.device != weight.device or x.dtype != torch.float16
                or weight.dtype != x.dtype or not x.is_contiguous() or not weight.is_contiguous()
                or min(*x.shape, *weight.shape) <= 0):
            raise ValueError('invalid FP16 CUDA Lt inputs/handle')
        t, k = x.shape
        n = weight.shape[0]
        if bias is not None and (bias.shape != (n,) or bias.device != x.device
                                 or bias.dtype != x.dtype or not bias.is_contiguous()):
            raise ValueError('invalid FP16 CUDA Lt bias')
        output = torch.empty((t, n), dtype=x.dtype, device=x.device)
        workspace = torch.empty((self.workspace_bytes,), dtype=torch.uint8, device=x.device)
        p, i = ct.c_void_p, ct.c_int

        with ExitStack() as cleanup:
            def create(kind, *args):
                """/** 成功创建后立即登记对应析构，后续任何阶段失败仍释放已有描述。 */"""
                descriptor = p()
                self.check(getattr(self.lib, kind+'Create')(ct.byref(descriptor), *args), kind+'Create')
                cleanup.callback(lambda: self.check(getattr(self.lib, kind+'Destroy')(descriptor), kind+'Destroy'))
                return descriptor

            def attribute(kind, descriptor, key, value):
                """/** 按CUDA公开ABI的实际类型大小传递属性，不硬编码指针宽度。 */"""
                self.check(getattr(self.lib, kind+'SetAttribute')(descriptor, key, ct.byref(value), ct.sizeof(value)),
                           f'{kind} attribute {key}')

            desc = create('cublasLtMatmulDesc', 68, 0)  # COMPUTE_32F, CUDA_R_32F scale
            attribute('cublasLtMatmulDesc', desc, 3, i(1))  # TRANSA=T
            attribute('cublasLtMatmulDesc', desc, 4, i(0))  # TRANSB=N
            if bias is not None:
                attribute('cublasLtMatmulDesc', desc, 7, i(4))  # EPILOGUE_BIAS
                attribute('cublasLtMatmulDesc', desc, 8, p(bias.data_ptr()))
            # 使用列主序视图：(W[K,N])^T * X[K,T] = Y[N,T]。
            a = create('cublasLtMatrixLayout', 2, k, n, k)
            b = create('cublasLtMatrixLayout', 2, k, t, k)
            c = create('cublasLtMatrixLayout', 2, n, t, n)
            preference = create('cublasLtMatmulPreference')
            attribute('cublasLtMatmulPreference', preference, 1, ct.c_size_t(self.workspace_bytes))
            # 与PyTorch 2.7.1 gemm_and_bias一致：D偏好取bias地址对齐，最大256字节。
            for key, tensor in zip((5, 6, 7, 8), (weight, x, output, bias)):
                address = tensor.data_ptr() if tensor is not None else 0
                alignment = min(256, address & -address) if address else 256
                attribute('cublasLtMatmulPreference', preference, key, ct.c_uint32(alignment))

            chosen, count = Heuristic(), i()
            self.check(self.lib.cublasLtMatmulAlgoGetHeuristic(self.handle, desc, a, b, c, c, preference,
                       1, ct.byref(chosen), ct.byref(count)), 'heuristic')
            if count.value != 1 or chosen.state or chosen.workspaceSize > self.workspace_bytes:
                raise RuntimeError(f'cuBLASLt no valid algorithm: count={count.value} state={chosen.state}')
            alpha, beta = ct.c_float(1), ct.c_float(0)
            stream = torch.cuda.current_stream(x.device).cuda_stream
            self.check(self.lib.cublasLtMatmul(self.handle, desc, ct.byref(alpha), p(weight.data_ptr()), a,
                       p(x.data_ptr()), b, ct.byref(beta), p(output.data_ptr()), c, p(output.data_ptr()), c,
                       ct.byref(chosen.algo), p(workspace.data_ptr()), self.workspace_bytes, p(stream)), 'matmul')
            self.calls.append(dict(t=t, n=n, k=k, bias=bias is not None, backend='cublasLtMatmul',
                compute='32F', epilogue='BIAS' if bias is not None else 'DEFAULT', stream=stream,
                workspace_limit=self.workspace_bytes, workspace_required=chosen.workspaceSize,
                algorithm_words=list(chosen.algo)))
        return output
