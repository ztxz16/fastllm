# Windows wheel 构建

需要同时生成 wheel、Electron 绿色包和完整验证报告时，使用
[`make_release.ps1`](../make_release.ps1)，参见 [Windows 发布流程](windows-release.md)。

仓库根目录的 `make_whl.ps1` 是 Windows 对应的 wheel 构建脚本。默认行为与
Linux 的 `make_whl.sh` 对齐：先构建 CPU 动态库，再构建 CUDA 动态库，最后把
Python 包、Triton 编译服务、构建信息、CUDA 主库和 CPU 降级库一起打入 wheel。

## 环境要求

- 64 位 Windows 10/11 或 Windows Server；
- Visual Studio 2022，并安装“使用 C++ 的桌面开发”和 Windows SDK；
- CMake 3.23+。脚本也会查找 Visual Studio 自带的 CMake；
- 64 位 Python，以及 `setuptools` 和 `wheel`；
- 构建 CUDA wheel 时，需要 CUDA Toolkit 12.x 和与 MSVC 兼容的 `nvcc`。

先准备 Python 打包工具：

```powershell
python -m pip install -U setuptools wheel
```

## 常用命令

在仓库根目录的 PowerShell 中运行：

```powershell
# CPU + CUDA wheel；Windows 默认不要求 NCCL
.\make_whl.ps1

# 只编译当前需要的架构，可明显缩短编译时间（例如 Ada / SM 89）
.\make_whl.ps1 -CudaArch "89"

# CPU-only wheel，不需要 CUDA Toolkit
.\make_whl.ps1 -CpuOnly

# nightly 包
.\make_whl.ps1 -Nightly -CudaArch "89"

# 中断后续编：保留已生成的目标文件，参数应与上次构建一致
.\make_whl.ps1 -Incremental -BuildDirectory "build-fastllm-windows"
```

多架构构建中的 GGML、GGUF 和注意力内核可能耗时较长。重新运行时使用
`-Incremental`，并保持原来的 `-BuildDirectory`、`-CudaArch` 和 `-CMakeArgs`；
不加 `-Incremental` 会先清理指定的构建目录。

CUDA 12.9 的部分 CCCL 头文件在 Windows 上使用了宽度不匹配的 `long2`。
CMake 检测到该代码后，会在构建目录生成修正副本，供相关的 Thrust、CUB 和
FlashInfer 源文件使用，无需修改系统安装的 CUDA Toolkit。

如果当前 PowerShell 会话禁止执行本地脚本，可以只对当前进程放开：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

默认输出位于：

```text
build-fastllm-windows\cuda\tools\dist\*.whl   # CUDA wheel
build-fastllm-windows\cpu\tools\dist\*.whl    # CPU-only wheel
```

脚本构建结束时会打开 wheel 并验证关键文件。CUDA wheel 至少包含：

```text
ftllm/fastllm_tools.dll       # CUDA 主库
ftllm/fastllm_tools-cpu.dll   # CUDA 依赖加载失败时的 CPU 降级库
ftllm/build_info.json
ftllm/fastllm_triton_server.py
ftllm/*.py
```

CUDA wheel 会声明 Windows 已有构建的 `nvidia-cuda-runtime-cu12` 和
`nvidia-cublas-cu12`；CPU-only wheel 不声明 NVIDIA 依赖，NCCL 包也只会用于实际
启用 NCCL 的 Linux 构建。加载器会从 wheel 目录、`CUDA_PATH\bin`、Python 环境、
这些 NVIDIA 包、`ftllmdepend` 以及 `PATH` 中的有效目录中查找依赖。如果 CUDA DLL 不完整，
会自动尝试随包附带的 CPU 动态库并输出警告。

## NCCL 与无 NCCL 降级

普通 Windows CUDA Toolkit 通常不包含可直接链接的 NCCL，因此 CMake 在 Windows
上默认 `USE_NCCL=OFF`，`make_whl.ps1 -UseNccl Auto` 找不到完整 NCCL SDK 时也会
自动关闭它。这个配置仍会编译 CUDA 后端：

- 单卡 CUDA 不需要 NCCL；
- 支持的多卡拓扑和张量会优先使用 FastLLM 的 CUDA P2P 自定义 all-reduce；
- Windows 双卡通过启动自检后，CUDA Graph 中每卡不超过 64 KiB 的 broadcast、
  reduce 和 all-reduce 可以使用 GPU 协调的映射主机内存通信，不要求 P2P；
- eager 路径使用主机内存同步汇聚；双卡大块 FP16/BF16/FP32/INT8/INT32 SUM 可复用锁页缓冲区并在
  GPU 上计算。未通过自检的拓扑、更多 GPU 和超限消息
  不支持这一 Graph 后端，需要由模型退出图模式；不支持自动退出的调用方应关闭
  Graph。主机中转的性能不能按 NCCL 预期使用。

通信后端自动选择，无需新增环境变量。图模式沿用 `FASTLLM_CUDA_GRAPH` 总开关；
需要关闭时，在启动前设置 `$env:FASTLLM_CUDA_GRAPH="0"`。支持范围、测试和示例见
[Windows 双卡 CUDA Graph](windows-tp-graph.md)。

如果已有兼容 Windows 的 NCCL SDK，其中包含 `include\nccl.h`、`nccl.lib` 和运行时
DLL，可以显式启用：

```powershell
.\make_whl.ps1 -UseNccl ON -NcclRoot "D:\sdk\nccl" -CudaArch "89"
```

运行时应保留 `NCCL_ROOT`，或把 NCCL DLL 所在目录加入 `PATH`。`-UseNccl ON` 下
找不到头文件或导入库会在 CMake 配置阶段直接报出明确错误。

## 其他参数

```powershell
# 指定 Python、CMake、并行数和额外 CMake 选项
.\make_whl.ps1 `
  -Python "C:\Python311\python.exe" `
  -CMakePath "C:\Program Files\CMake\bin\cmake.exe" `
  -Jobs 12 `
  -CMakeArgs "-DCUDA_NO_TENSOR_CORE=OFF"
```

默认架构列表与 `make_whl.sh` 一致。发布包适合使用默认列表；本地验证建议通过
`-CudaArch` 只编译目标显卡架构。

Windows 脚本直接读取 `make_whl.sh` 中的默认架构；公共工具位于
`tools/windows/common.ps1`。增量打包时会清理构建目录中上次生成的包元数据，
确保新 wheel 使用当前源码的版本号。
