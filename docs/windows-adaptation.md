# Windows 适配整理与分支建议

本文记录 Windows 适配的维护边界和验证范围，验证基线为 `origin/master`
`0f1b4daf`（2026-09-08）及 Windows 适配改动。构建和使用方法见
[Windows 发布流程](windows-release.md)。

## 已有工作及维护边界

| 工作 | 主要位置 | 作用 |
| --- | --- | --- |
| MSVC / CUDA 编译 | `CMakeLists.txt`、`include/`、CPU/CUDA 源文件 | C++17/20 分文件配置、AVX512 架构识别、UTF-8、标准预处理器、整数宽度与编译器扩展兼容 |
| 第三方 CUDA 头文件兼容 | `third_party/flashinfer`、`deep_gemm`、`turbomind` | 标准对齐和共享内存声明、模板解析及 MSVC 不支持的语法；CUDA 12.9 CCCL 修正副本仅生成在构建目录 |
| CPU / 磁盘运行时 | `src/devices/disk/diskdevice.cpp`、`src/models/basellm.cpp` | Windows 文件定位读取、对齐分配、分块读取，以及替代 `__int128` 的预算计算 |
| 可选 NCCL | `src/devices/multicuda/`、`CMakeLists.txt` | Windows 默认关闭 NCCL，保留 P2P 自定义 all-reduce 和主机内存中转的集合通信 |
| Python 运行时 | `tools/fastllm_pytools/`、`tools/scripts/setup.py` | DLL 搜索、CPU 降级、按构建特性声明依赖、Windows 内存检测、后台子进程和 SQLite 连接释放 |
| Windows Agent | `tools/ftllm_agent_runtime/` | Pi Windows 可执行文件、平台 wheel 标签、PowerShell 工具及其本地依赖 |
| wheel 与绿色包 | `make_whl.ps1`、`make_portable.ps1`、`portable/windows/` | 原生构建、内嵌 Python、CLI 启动器、运行时版本锁定与离线缓存 |
| Electron 与发布 | `desktop/package.ps1`、`desktop/windows/`、`make_release.ps1`、`tools/windows/` | 桌面组装、退出时清理子进程、资源和 DLL 验证、归档及校验和 |

这些改动覆盖 Windows x64 的 NVIDIA CUDA 路径和 CPU 降级路径；没有实现 Windows
AMD/ROCm 后端。Windows 下运行时 Triton 编译服务仍不可用，已有原生 CUDA 内核
和 fallback 负责相应计算。无 NCCL 的主机中转通信包含同步和设备/主机数据复制，
不能按 NCCL 的性能预期使用，也不支持在 CUDA Graph 中执行。

## 本次冗余清理

- 删除 `include/devices/multicuda/fastllm-nccl.h` 的整套假 NCCL 类型和函数。
  NCCL 专用头文件、通信器和真实调用仅在 `FASTLLM_USE_NCCL` 下编译；删除无调用方
  的 `GetNcclComm`。无 NCCL 分支直接使用现有主机中转实现。
- 合并 broadcast / reduce / all-reduce 的单卡复制处理，以及自定义 all-reduce
  环境变量的默认值和 `auto` 判断。
- 将 wheel 的元数据、平台标签、必要文件和 CLI 入口检查统一到
  `tools/windows/verify_wheel.py`，wheel 和绿色包入口共用，发布阶段继续执行
  额外的二进制架构、源码和原生库校验。
- CMake 统一维护 Python 包的资源复制命令，增加 `fastllm_python_package` 目标。
  PowerShell 打包直接构建该目标，增量构建即使不重新链接 DLL，也会刷新 Python、
  Triton 服务和 `build_info.json`。验证器补充服务文件与源码的一致性检查。
- DLL 搜索去除重复的 NVIDIA 目录 glob、重复搜索根和重复异常类型；构建目录
  解析复用 PowerShell 公共路径函数，保留递归清理前的目录范围和链接检查。
- 增量构建结束时按生成时间选择源码归档，修复输出误指向旧版本源码包的问题。
- 将 `PATH` 中的有效目录注册到 Windows DLL 搜索路径，修复没有 `CUDA_PATH`、
  仅通过 `PATH` 提供 CUDA 或 NCCL DLL 时的加载问题；继续保留 DLL 搜索句柄和
  CPU 降级逻辑。

保留四个 PowerShell 入口：它们分别服务 wheel、CLI/浏览器包、Electron 和完整
发布，有独立用途。第三方头文件兼容、磁盘读取降级和无 NCCL 通信也有实际用途，
不作为冗余删除。已有 `build-*` 产物和历史日志不属于源码，本次没有清理它们。

## 本次验证

- Windows DLL 搜索、打包与归档回归：9 项通过，包含 `PATH` 目录注册、无效目录
  过滤、去重、搜索句柄保留，以及旧 Triton 服务混入增量 wheel 的回归。
- 启动器回归：43 项通过；Launcher WebUI：16 项通过。
- Electron 运行时：6 项通过，覆盖 Linux/Windows 环境构造和分段 UTF-8 解码。
- 6 个 PowerShell、17 个 Python 文件语法检查通过，`git diff --check` 通过。
- MSVC Release CPU 动态库编译、加载通过，确认 CPU 后端可用且不注册 CUDA。
- 故意提供无法加载的主 DLL 后，加载器成功使用此次编译的 CPU 降级库。
- 清除 `CUDA_PATH`，仅通过 `PATH` 提供 CUDA DLL 后，加载成功并确认 CUDA
  后端可用；同一进程在注册搜索目录前加载失败，确认没有借用预加载的依赖。
- 故意污染构建暂存区的 Python、Triton 服务和构建信息后，打包目标恢复了
  与源码一致的内容；原生 DLL 修改时间未变，确认不重新链接也能刷新资源。
- CUDA SM86 Release 构建和 wheel 打包通过；源码/原生库一致性、SM86 机器码
  和 PTX 校验通过，构建信息确认 `USE_CUDA=true`、`USE_NCCL=false`。
- 此次编译的 CUDA DLL 加载和显卡枚举通过；RTX 3090 加载本地
  `Qwen3-4B-FP8`，完成 16 token 上限的确定性生成，返回 `hello`。

验证使用本机 Windows x64、Visual Studio 2022、CUDA Toolkit 12.9 和 RTX 3090
（SM86）。旧的 `build-release-0.1.8.2` 成品基于 `546d76c6`，不能代表这次更新后
代码的完整发布验证。本次未重新生成默认多架构 Electron 发布包；多 GPU、
SM90/SM120 实机及 Linux/ROCm 回归需要对应环境。

## 是否建立 windows-nvidia

建议建立，定位为适配、集成验证和 Windows 发布分支，最终逐步合回 `master`。
当前改动跨越构建、原生运行时、第三方内核和桌面发布，主线又在持续修改 Qwen4
和多卡执行；单独承载这些工作，便于明确验证基线和管理 Windows 发布节奏。
仓库现有 GitHub Actions 仅运行 Linux CPU 构建，尚未形成 Windows 自动验证。

建议按以下逻辑组织后续提交；相互依赖的变更应一起提交：

1. MSVC / CUDA 构建及原生代码兼容，包括所需第三方补丁。
2. 可选 NCCL 与无 NCCL 通信、相应构建开关和限制说明。
3. Python 平台支持、wheel 构建与依赖元数据。
4. Windows Pi Agent 运行时。
5. 绿色包与 Electron 支持。
6. 发布验证和配套文档。

拆分时按依赖关系确定顺序，`CMakeLists.txt` 按修改块暂存，共用工具先于调用方
进入提交；测试随对应功能提交。

其中标准 C++ 修复、SQLite 连接关闭、UTF-8 流解码和通用打包修复适合较早合回
`master`；Windows 专用入口留在独立目录。保持共用模型和算子实现，避免形成另一套
Windows 模型代码。分支发布前应补 Windows CPU 构建/脚本 CI，并单独安排 CUDA
实机、多卡和默认多架构构建验证。

当前已满足提交到 `windows-nvidia` 开发分支的基线要求：已复现的 `PATH` DLL
加载问题已修复，针对性回归和真实 DLL 加载验证通过，增量 wheel 已重新校验。
正式发布或合入主线前，还需完成上述多卡、跨平台和完整发布验证。
