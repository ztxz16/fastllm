# ROCm 编译

## 0. 支持平台

ROCm 编译目前仅支持Linux平台。

目前支持的GPU型号如下：

- AMD Radeon Instinct MI系列，如MI50, MI100，MI210等
- AMD Radeon RDNA RX 7000 游戏卡和工作站卡系列，W7800，W7900等
- 海光系列GPU，如K100等（未验证，理论可行）

## 1. 安装 ROCm，获取 ROCm Arch

请参考 [ROCm 官方文档](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) 安装 ROCm。

可以在 [架构列表](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)的LLVM target列中找到GPU的 ROCm Arch。

常见GPU对应的架构：
| 架构代号 | 架构系列 | 代表产品示例                                | 推荐 ROCm 版本 |
|----------|-----------|---------------------------------------------|----------------|
| gfx900   | GCN5.0    | Radeon Instinct MI25                        | ❌不支持         |
| gfx906   | GCN5.1    | Radeon VII, Instinct MI50                   | [6.3.3](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.3/install/quick-start.html) |
| gfx908   | CDNA      | Radeon Instinct MI100                       | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |
| gfx90a   | CDNA2     | Radeon Instinct MI210/MI250/MI250X         | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |
| gfx942   | CDNA3     | Instinct MI300A/MI300X/MI325X              | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |
| gfx1030  | RDNA2     | Radeon PRO W6800/V620                       | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |
| gfx1100  | RDNA3     | Radeon PRO W7800/W7900, RX 7900 XT/XTX/GRE  | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |
| gfx1101  | RDNA3     | Radeon PRO V710                         | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |



把需要编译的GPU架构用`;`分隔，填入`-DROCM_ARCH`参数中。如果不填这个参数，会自动检测。

注意，部分GPU（比如RX6000系列、MI50不支持矩阵乘法加速`rocwmma`，只要列表中有一个GPU不支持`rocwmma`，则编译时不会使用`rocwmma`。

## 2. 编译

如果使用自动检测，直接运行以下命令。适用于只在本机运行，不拷贝到有其他GPU的机器上运行的情况。

``` sh
bash install.sh -DUSE_ROCM=ON
```

如果需要编译成支持多个GPU的版本，或者在其他机器上运行，需要手动指定`ROCM_ARCH`参数。

``` sh
bash install.sh -DUSE_ROCM=ON -DROCM_ARCH="gfx908;gfx90a;gfx1100"
```

### 独立编译共享库（Python ROCm SDK 环境）

使用 Python 安装的 ROCm SDK 时，应在对应虚拟环境中设置 `ROCM_PATH`。CPU 源码使用主机 C++ 编译器，GPU 源码通过 CMake 的 HIP 语言编译。以 SDK 10.0.0 的目录布局及 `gfx1100` 为例：

```sh
export ROCM_PATH="$(rocm-sdk path --root)"
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S . -B build-rocm-10 -G Ninja \
  -DUSE_ROCM=ON -DUSE_CUDA=OFF -DROCM_ARCH=gfx1100 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++
cmake --build build-rocm-10 --target fastllm_tools --parallel 24
```

产物是 `build-rocm-10/tools/ftllm/libfastllm_tools.so`。默认启用 NUMA，Ubuntu 需要安装 `libnuma-dev`；SDK 需要提供 hipBLAS、hipBLASLt、rocPRIM、RCCL 及 HIP 开发包。HIP 兼容头文件不参与二次 hipify，转换后的头文件和源码分别保存在构建目录的 `hip-headers/`、`hip-src/`，原始头文件仍用于主机代码。

本次适配的验证环境为 Ubuntu 22.04 x86_64、RX 7900 GRE（gfx1100）、SDK 包版本 10.0.0（内置 HIP 7.15、Clang 23）、GCC 11 和 CMake 4。其他架构和上表中的旧 SDK 版本未在本次适配中回归验证。

### 构建隔离与环境设置

ROCm 的 SDK 查找、hipify、编译选项、源文件和依赖集中在 `cmake/ROCm.cmake`，只有 `USE_ROCM=ON` 才加载。NVIDIA 构建不会引入 HIP 文件、头文件搜索路径、宏或链接库。GPU 架构仍由 `ROCM_ARCH` 指定；SDK 编译器与 CMake 包路径从 `ROCM_PATH` 推导，无需同时设置 `ROCM_HOME`、`HIP_PATH`、`CMAKE_PREFIX_PATH` 或重复传递编译器路径。

Python SDK 环境保留虚拟环境激活、`ROCM_PATH` 和运行时库搜索路径；需要直接使用 `hipcc`、`rocminfo` 时将 SDK 的 `bin` 加入 `PATH`。源码构建的 Python 包还需要将 `build-rocm-10/tools` 加入 `PYTHONPATH`。使用本地模型不需要额外设置 Hugging Face/Transformers 的离线环境变量。

### CUDA 专用优化的回退

ROCm 不编译 CUTLASS、Marlin、TurboMind SM70 和 NVIDIA PTX 自定义 all-reduce 内核。相关能力检查和可选加速入口返回 `false`，保留原生 HIP、hipBLAS 或 RCCL 路径；GGUF MMQ/MMVQ 专用加速不可用时，继续使用已有反量化或通用计算路径。

FlashInfer 注意力被禁用，使用已有原生注意力。普通生成的温度、top-k/top-p、重复惩罚和 typical acceptance 由 `src/devices/rocm/fastllm-rocm-sampling.hip` 实现。结果可返回主机或留在显存中；GPU 随机序列在每次执行时推进，图重放不会反复使用捕获时的同一随机数。top-k/top-p 对原始 softmax 概率联合筛选，与原 CUDA FlashInfer 路径一致；同概率项按相同阈值处理，greedy 平局选择较小 token id。typical acceptance 的主机返回接口仅在需要验收候选时拷贝对应概率行。

DFlash 专用 top-k 和拒绝采样仍未移植，不能据此认为 DFlash 推测解码已可用。

HIP 编译必须保留 `-fgpu-default-stream=per-thread`：隐式 kernel launch、hipBLAS 和图捕获需要使用同一个每线程流，否则图可能遗漏模型计算，产生重复 token。兼容层将主机标量模式下的 FP16 GemmEx 累加提升至 FP32，保留 FP16 输入/输出，避免长向量点积的精度损失。

### Qwen3.5 完整模型验证

RX 7900 GRE 上已验证 Qwen3.5-4B 的 FP16 文本生成，包含图执行、单请求和双请求并发、greedy、top-k/top-p 随机采样与重复惩罚。算术、翻译、中文生成和随机文本检查通过。测试输入为 33 个 token，每请求连续生成 128 个 token，prefix cache 关闭：单请求解码约 56 tokens/s，随机采样约 53 tokens/s；双请求合计解码约 112 / 106 tokens/s。短输入测量不代表所有上下文长度下的速度。未验证其他模型、视觉输入或 DFlash。

激活 SDK 环境并编译后，可以运行本地已下载的模型：

```sh
python -m pip install numpy pillow transformers jinja2
export PYTHONPATH="$PWD/build-rocm-10/tools${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m ftllm.chat /path/to/Qwen3.5-4B \
  --device cuda:0 --dtype float16 --atype float16 \
  --max_batch 1 --chunked_prefill_size 256 --enable_thinking false
```

原生采样回归测试为 `test/ops/rocmSampling.cpp`，验证统计分布、temperature/top-k/top-p、重复惩罚、typical acceptance、无效输入、随机图重放，以及修改输入后隐式算子的图重算。它使用主机 C++ 编译器链接已构建的共享库和 HIP runtime：

```sh
g++ -O2 -std=c++17 -pthread -DUSE_ROCM -DUSE_CUDA -D__HIP_PLATFORM_AMD__ \
  -Iinclude -Iinclude/utils -Ithird_party/gguf -Ithird_party/json11 \
  -I"$ROCM_PATH/include" test/ops/rocmSampling.cpp \
  -Lbuild-rocm-10/tools/ftllm -L"$ROCM_PATH/lib" \
  -lfastllm_tools -lamdhip64 \
  -Wl,-rpath,"$PWD/build-rocm-10/tools/ftllm" -o build-rocm-10/rocm-sampling-test
build-rocm-10/rocm-sampling-test
```

ROCm 共享库启用未定义符号检查，缺少入口会在链接阶段报错。

## TODO

- [ ] 海光系列GPU的验证
- [ ] 支持`rocwmma`，能使用矩阵乘法加速

## 鸣谢

[leavelet](https://github.com/leavelet) 提供ROCM支持
