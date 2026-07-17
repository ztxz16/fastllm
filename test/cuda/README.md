# CUDA Kernel Tests

这个目录用于放独立 CUDA 算子正确性测试。

测试目标是不依赖完整模型加载，直接比较：

```text
CPU reference vs GPU kernel output
```

每个测试建议输出：

```text
max_abs_error
mean_abs_error
max_rel_error
PASS / FAIL
```

启用方式：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cmake -S . -B build-fastllm \
  -DUSE_CUDA=ON \
  -DENABLE_VLLM_KERNEL=ON \
  -DENABLE_CUDA_TESTS=ON \
  -DCUDA_ARCH=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

cmake --build build-fastllm --target test_awq_gemm_compare -j$(nproc)

./build-fastllm/test/cuda/test_awq_gemm_compare
```

5090 环境下 AWQ GEMM 测试期望输出类似：

```text
[AWQ GEMM GPU compare no bias] max_abs=0.000894308/0.02 mean_abs=0.000172735/0.002 max_rel=0.000456023/0.05 max_abs_index=1 PASS
[AWQ GEMM GPU compare with bias] max_abs=0.000922203/0.02 mean_abs=0.00017659/0.002 max_rel=0.000439798/0.05 max_abs_index=85 PASS
[AWQ GEMM] GPU compare PASS
```

如果找不到测试可执行文件：

```bash
find build-fastllm -name test_awq_gemm_compare -type f
```

如果 CMake 找不到 `nvcc`，先确认：

```bash
which nvcc
nvcc --version
ls /usr/local/cuda/bin/nvcc
```

## W4A8 Cutlass Validation

W4A8 迁移分阶段接入时，先用独立入口做合法性验证，再接数值和性能测试。

阶段 6 可用下面的环境变量打开入口验证日志：

```bash
export FASTLLM_CUDA_W4A8_VALIDATE=1
export FASTLLM_CUDA_W4A8_TRACE=1
```

只验证权重缓存、activation quant、output buffer 准备时，可分别打开：

```bash
export FASTLLM_CUDA_W4A8_PREPARE_CACHE=1
export FASTLLM_CUDA_W4A8_PREPARE_ACTIVATION=1
export FASTLLM_CUDA_W4A8_PREPARE_OUTPUT=1
```

真正允许 W4A8 GEMM 接管 `DoCudaLinear` 时，再打开：

```bash
export FASTLLM_CUDA_W4A8_ENABLE_GEMM=1
```

步骤 5 新增了最小独立测试：

```bash
cmake --build build-fastllm --target test_w4a8_cutlass_linear -j$(nproc)
./build-fastllm/test/cuda/test_w4a8_cutlass_linear
```

这个 target 只会在 CMake 判定 `FASTLLM_CUTLASS_W4A8_ENABLED=ON` 时生成。
也就是需要同时满足：

```text
USE_CUDA=ON
ENABLE_CUDA_TESTS=ON
ENABLE_VLLM_KERNEL=ON
CUTLASS include 可用
CUDA_ARCH 包含 90 / 90a / 9.0a
CUDA 编译器版本 >= 12.0
```

不满足时 CMake 会打印：

```text
CUDA W4A8 CUTLASS test: OFF
```

此时不会生成 `test_w4a8_cutlass_linear`，普通构建和其它 CUDA 测试不受影响。

第一版测试先覆盖入口行为：

```text
编译未开启 FASTLLM_ENABLE_VLLM_CUTLASS_W4A8 时返回 false
非 SM90 时返回 false
m/k 非 128 对齐时返回 false
input/weight dtype 不满足时返回 false
bias 不满足时返回 false
groupCnt 不是 128 时返回 false
```

当编译开启 W4A8 且运行在 SM90 上时，测试会继续打开
`FASTLLM_CUDA_W4A8_ENABLE_GEMM=1`，跑最小数值比较：

```text
W4A8 output vs CPU baseline
n = 1
n = 4 / 8 / 16
n = 64 / 128
with bias / no bias
FP16 input / BF16 input
```

合法性用例应覆盖：

```text
n = 1
small batch: n = 4 / 8 / 16
large batch: n = 64 / 128
m/k: 128 对齐和非 128 对齐
input dtype: FP16 / BF16 / 非 FP16/BF16
weight dtype: INT4_GROUP / 非 INT4_GROUP
bias: empty / FLOAT32[k] / 非 FLOAT32 / 长度不等于 k
runtime arch: SM90 和非 SM90
```

期望行为：

```text
SM90 + dtype/shape/bias/weight 全满足时打印 validation ready
非 SM90 打印 skip: runtime arch is not SM90
m/k 非 128 对齐打印 skip: m/k is not 128-aligned
dtype 不满足打印对应 dtype skip reason
weight 不满足打印对应 INT4_GROUP / group / scale-min skip reason
```

完整数值正确性和性能测试要等 W4A8 GEMM dispatch 接通后再跑，届时和 FP16 cuBLAS、Marlin INT4_GROUP、INT4_GROUP128 baseline 分别比较。
