# Windows 支持与维护范围

Windows 开发分支为 `windows-nvidia`，`windows-1.9.1` 用于准备 1.9.1 发布。
构建 wheel 和 Electron 绿色包使用仓库根目录的 `make_release.ps1`，详见
[Windows 发布流程](windows-release.md)。

## 平台与运行时

支持 Windows x64 的 NVIDIA CUDA 和 CPU 后端，CPU 需要 AVX2/FMA/F16C。
CUDA 包包含 CPU 降级库；绿色包内置 Python、Electron、用户态 CUDA 库和
VC++ runtime，目标机器仍需安装 NVIDIA 显卡驱动。

Windows 构建保持 `USE_NUMAS=OFF`、`USE_NCCL=OFF`。普通内存专家分流和
GPU 专家缓存不依赖 NUMA，CUDA Graph 沿用 `FASTLLM_CUDA_GRAPH` 总开关。
支持条件和关闭方法见 [Windows wheel 构建](windows-wheel.md)。

Windows 不提供 AMD/ROCm 后端和运行时 Triton 编译服务；对应计算使用已有
原生 CUDA 内核及回退路径。MiniMax-H3 等可选原生媒体后端有独立构建选项，
不包含在默认 wheel 构建目标中。

## 维护边界

| 范围 | 主要位置 | 要点 |
| --- | --- | --- |
| MSVC/CUDA 编译 | `CMakeLists.txt`、CPU/CUDA 源文件 | C++17/20 配置、UTF-8、整数宽度和编译器扩展兼容 |
| 第三方内核 | `third_party/` | 平台兼容修改保持局部；CUDA Toolkit 头文件修正生成在构建目录 |
| CPU/磁盘运行时 | `src/devices/disk/`、`src/models/basellm.cpp` | Windows 文件读取、对齐分配、内存预算和后台执行 |
| 多卡通信 | `src/devices/multicuda/` | 可选 NCCL、P2P 和主机中转回退 |
| Python 运行时 | `tools/fastllm_pytools/`、`tools/scripts/setup.py` | DLL 搜索、CPU 降级、依赖声明和进程管理 |
| 发布打包 | `make_*.ps1`、`portable/windows/`、`desktop/windows/`、`tools/windows/` | 运行时锁定、缓存、原生入口、依赖验证和校验和 |

保持共用模型和算子实现，平台差异通过编译条件或独立平台工具处理。
升级 Python、Electron、Pi 等依赖时同步维护 `portable/windows/runtime-lock.json`
中的版本、下载地址和 SHA256。

## 多卡与发布验证

无 NCCL 时，支持的拓扑优先使用 CUDA P2P；其余路径使用主机中转。
Windows 双卡小消息 Graph 通信要求启动自检通过，具体限制见
[Windows 双卡 CUDA Graph](windows-tp-graph.md)。单卡验证不能替代多卡验证。

发布入口验证 wheel 元数据、CUDA 架构、CPU 降级、DLL 依赖、路径迁移、
Electron 启停和归档哈希。使用 `-RequireCuda -SmokeModel <模型目录>` 可增加
本机 GPU 实测，结果记录在输出目录的 `release-verification.json` 中。

分支名称和编译成功不代表所有 GPU 架构均已实测。发布时应以产物的构建信息、
架构清单和验证报告说明支持范围；多卡及不同 GPU 架构需要相应硬件验证。
