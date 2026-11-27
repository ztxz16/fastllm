# 第三方组件版本记录

## GGUF

本仓库以源码快照（vendoring）集成 **FastLLM 上游的完整 GGUF 兼容组件**，不是 Git submodule。
升级单位是上游的整个 `third_party/gguf/` 子树；不能只替换个别报错文件后宣称组件升级。
该组件自身整合了 llama.cpp 与 ik_llama.cpp 的代码；本记录不代表这两个原始项目的完整版本升级。

- 上游仓库：<https://github.com/ztxz16/fastllm>
- 固定提交：`e33b3cb795119e9d0b00d65b9c9991c4ca054483`（修复无 AVX2 的 x86 编译兼容性）。
- 完整组件目录：`third_party/gguf/`。
- 上游 Git tree：`15451516e39d050823f8791f79c83b39ac85fade`。
- 版本清单：[gguf.lock.json](gguf.lock.json)，记录全部文件的 Git blob、SHA-256 与文件模式。
- 全部 7 个文件：`README.md`、`ggml-dequantize.cpp`、`ggml-iqk.cpp`、`ggml-quant.cpp`、
  `gguf-adapter.cpp`、`gguf.cpp`、`gguf.h`。文件内容和原有版权说明均保持上游原样。

2026-11-27 完整同步后，7 个文件中有 4 个相对升级前发生变化，另 3 个原本就与该版本一致。
公共头文件未变。除 AVX 辅助函数与 IQK 指令集条件外，完整版本还带入了元数据标量/数组读取、
GGUF 权重标记、部分反量化入口与 Qwen3.5 名称映射变化，因此需覆盖相应回归。
之前仅替换两个文件的尝试已撤销，本次按完整子树重新导入并校验。

## 实现与适配边界

| 模块 | 实现语言/技术 | 职责与边界 |
| --- | --- | --- |
| 上游 GGUF 完整组件 | C++、CPU SIMD | 格式读取、量化/反量化、矩阵计算与上游模型映射；保持固定版本原样 |
| FT CPU 算子条件修正 | C++ 条件编译 | 位于 `src/devices/cpu/cpudevice.cpp`；两处 FP16 转换要求 F16C，BF16 SIMD 转换要求 AVX2，保留原函数体和标量路径 |
| 版本清单与校验工具 | JSON、Python 标准库 | 离线检查文件清单、文件内容、模式与 tree；不参与推理、不触发编译、不修改文件 |

需要本地兼容适配时放在 `third_party/` 外，通过 FT 调用侧或构建接入；不在锁定快照中打补丁。
上游版本里已有的适配代码随完整组件保留，不属于新增本地修改。
GPU 专用构建解耦留到后续，本次继续使用现有构建入口。

## 完整性检查

在仓库根目录运行，或从任意目录使用脚本的绝对路径：

```bash
python3 tools/third_party/verify_gguf.py
```

此命令只做静态文件校验，文件缺失、多余、改动、符号链接或模式不一致时返回非零。
不依赖本地是否已抓取上游 Git 对象；检查清单本身是否与记录的 tree 一致。
清单作为受版本控制的信任基线，离线校验不能替代对上游来源的审核。

本次还直接对照了上游提交中的全部 7 个 Git blob 和 tree，确认没有混用版本。
源码 Review 与完整性检查已完成；没有本地编译、模型验证或数值回归，尚不能标记验收成功。

## 后续升级流程

1. 从上述上游仓库获取并固定完整提交号，审阅整个组件与现有 FT API 的差异。
2. 在组件没有未提交修改时，通过该提交的 Git 归档导入完整 `third_party/gguf/` 子树，
   核对新增、删除与文件模式；不能保留上个版本的额外文件，也不能按报错挑选文件。
3. 从同一上游提交重新生成完整版本清单，记录 revision、目录 tree、全部文件 blob 和 SHA-256。
   原样保留上游内容，所有本地适配留在目录外。
4. 运行完整性校验并直接对照上游 Git tree，Review 适配；按项目规范在服务器构建、回归。
   版本清单、完整组件、必要适配及升级说明应一同纳入版本管理。

## 待服务器验收

- Xeon E5-2680 v2：使用本机 CPU 架构，关闭 `MAKE_WHL_X86`，不要强制启用 AVX2/FMA；
  V100 的 `CUDA_ARCH=70` 只控制 GPU 架构。
- 构建 PD 原生模块与三个模型执行程序，继续正式模型 PD/非 PD 对照。
- 回归 GGUF 元数据读取、模型映射、量化与普通点积；升级完整性不等于数值正确性。
- 上游在能力不足时关闭 IQK 重排内核，不意味着所有 IQ 格式都有普通点积实现。
  无 AVX2 路径下 IQ2_XS、IQ2_S、IQ3_XXS 的点积仍未提供，不能宣称支持所有格式。
- CPU FP16/BF16 条件修正仍需实际输出验证；按项目规则分别记录 exact/Top-K 的 FAIL
  与仅 numeric 的 WARNING，保留所有差异证据。

## FlashInfer 的 CCCL 构建依赖（2026-11-27）

CUDA 12.6 配套 CCCL 2.5，缺少当前 FlashInfer 使用的 `cuda::maximum`、`cuda::minimum`。
现有快速除法封装还使用 `cuda::fast_mod_div`，仅补齐前两个接口不够。
FT 构建层按当前 FlashInfer 上游锁定的 CCCL **v3.3.2** 完整提交接入：
`876867684f7fac130e0f5911236e0a92a970d4fd`，仓库为 <https://github.com/NVIDIA/cccl>。
不改写 FlashInfer/CCCL 源码，也不向 CUDA 命名空间补造同名类型。

来源依据：

- [当前 FlashInfer 的 CCCL 子模块](https://github.com/flashinfer-ai/flashinfer/tree/5f3d1b3fc6e1ed8a79429986b3637802f1bd2b57/3rdparty/cccl)。
- [CCCL v3.3.2 的版本与兼容说明](https://github.com/NVIDIA/cccl/blob/876867684f7fac130e0f5911236e0a92a970d4fd/README.md)。

| 模块 | 实现语言/技术 | 职责 |
| --- | --- | --- |
| FT 依赖选择与固定版本校验 | CMake、Git | `cmake/fastllm_cccl.cmake` 检查 Toolkit 头文件，必要时获取完整固定版本，校验提交和干净工作区 |
| CCCL | CUDA C++ 头文件库 | 同一版本的 CUB、Thrust、libcudacxx 同时接入现有 CUDA 编译目标 |
| FlashInfer 与 FT 推理调用 | CUDA C++ | 保持现有采样、注意力和调用入口；本次不修改算子或采样规则 |

默认先检查当前 NVCC 对应 Toolkit 的归约与快速除法头文件。满足要求时继续使用 Toolkit 自带版本；
缺少接口时，CMake 通过 Git 获取完整固定版本到构建目录的 `_deps/fastllm_cccl-src`。
只使用其头文件，不配置或编译 CCCL 自身的项目；不覆盖系统 CUDA 安装。
三个组件以普通 `-I` 统一置于 Toolkit 隐式头文件之前，避免只替换 libcudacxx 导致混用版本。
配置时打印所选版本和来源；显式固定版本低于 Toolkit 自带版本时拒绝降级。

首次自动获取需要 Git、网络和 CMake 3.18 以上。可在有网络的环境准备完整离线 checkout：

```bash
git clone --branch v3.3.2 --depth 1 https://github.com/NVIDIA/cccl.git /home/deps/cccl-3.3.2
git -C /home/deps/cccl-3.3.2 rev-parse HEAD
```

提交号应为上述固定值；保留 `.git`，整个目录不要改写或添加本地文件。
服务器配置命令可增加 `-DFASTLLM_CCCL_ROOT=/home/deps/cccl-3.3.2`；
`-DFASTLLM_FETCH_CCCL=OFF` 关闭自动获取，缺少接口且未指定目录时明确报错。
默认配置不必增加参数，原有 `CUDA_ARCH=70` 与 CPU 指令集配置保持不变。

本地仅核对了上游固定提交、三个组件版本、所需头文件及 FT 构建接入，并完成源码 Review 和静态检查。
未执行本地 CMake、NVCC 或 GPU 测试。CCCL 3.x 支持 CUDA 12.x，但官方不将 SM70 列入常规 CI；
本次是否通过 V100 构建和实际推理仍需服务器验证，不能仅凭头文件存在宣称验收成功。
