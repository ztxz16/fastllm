# ROCm 编译与 wheel 打包

本文维护在 `master` 分支，以下构建命令在 **`rocm` 分支**执行。该分支提供独立的 HIP 构建、采样实现和 `ftllm-rocm` 打包入口；本文中的 Python SDK 版本为 **10.0.0**。

```sh
git clone --branch rocm https://github.com/ztxz16/fastllm.git
cd fastllm
```

已有仓库也可以在保存当前工作后切换到 `rocm`。新版脚本需要包含本文末尾列出的打包提交；旧的系统 ROCm 6.x 安装说明不适用于这组 Python SDK wheel。

## 构建与运行条件

| 用途 | 条件 |
|---|---|
| Docker 编译 | Linux x86_64、可用的 Docker 及访问权限、下载基础镜像和依赖的网络；无需 GPU、AMD 驱动或宿主 ROCm SDK |
| 宿主直接编译 | Linux x86_64、Python >= 3.10、ROCm Python 开发 SDK、GCC/G++、CMake、Ninja、libnuma 开发包及 Python 打包工具 |
| 安装 wheel 后运行 | Linux x86_64、支持 AVX2/F16C/FMA 的 CPU、Python >= 3.10、兼容的 AMD 内核驱动和固件、GPU 设备访问权限及匹配的运行库设备包 |

当前分发基线是 Ubuntu 22.04 / Python 3.10，实机为 RX 7900 GRE（gfx1100）。Ubuntu 24.04 / Python 3.12 也能构建，但该环境生成的原生库引用了 `GLIBC_2.38`，不能作为 Ubuntu 22.04 兼容包分发。`linux_x86_64` 标签不代表支持所有 Linux 发行版，也不承诺 manylinux2014/glibc 2.17。

当前脚本不生成 ARM64 或 Windows wheel。ARM 主机上通过 Docker 模拟 x86_64 的构建方式尚未验证。

## 默认覆盖的 GPU 架构

`--architectures` 默认值为 **`all`**。脚本读取当前 `rocm` 包的依赖元数据，保留适用于当前平台的设备包，再与 SDK 编译器支持的目标取交集。无需维护固定的短架构列表；升级 SDK 后，新增且满足条件的目标会自动进入默认编译范围。

当前 ROCm Python SDK 10.0.0 提供以下 **25 个目标**：

| 目标组 | 编译目标 / pip extra |
|---|---|
| gfx9 | `gfx908`、`gfx90a`、`gfx942`、`gfx950` |
| gfx10.1 | `gfx1010`、`gfx1011`、`gfx1012` |
| gfx10.3 | `gfx1030`、`gfx1031`、`gfx1032`、`gfx1033`、`gfx1034`、`gfx1035`、`gfx1036` |
| gfx11.0 | `gfx1100`、`gfx1101`、`gfx1102`、`gfx1103` |
| gfx11.5 | `gfx1150`、`gfx1151`、`gfx1152`、`gfx1153` |
| gfx12.0 | `gfx1200`、`gfx1201` |
| gfx12.5 | `gfx1250` |

编译目标覆盖不等于对应硬件已完成运行验收。请根据目标显卡选择架构，并在对应硬件上验证模型、精度和性能。MI50/Radeon VII 的 `gfx906` 虽然能被编译器识别，但不在这组 SDK 的运行库设备包中，因此本 wheel 不包含它。

脚本会检查最终共享库中的每个 HIP bundle，确认其包含全部请求的目标。缺少运行库设备包、编译不支持或代码对象不完整都会报错，不会静默移除显卡目标后继续生成包。

## 使用 Docker 编译 wheel

推荐从仓库根目录运行：

```sh
# 查看当前镜像 SDK 能覆盖的目标；不编译 FastLLM
bash make_whl_rocm_docker.sh --list-architectures

# 默认编译全部可用架构
bash make_whl_rocm_docker.sh --jobs 12
```

首次执行会构建 Ubuntu 22.04 + ROCm SDK 10.0.0 开发镜像。镜像只安装用户态构建依赖，不安装 DKMS 或内核驱动，不挂载 GPU，也不需要 `--privileged`。默认的 25 架构构建会比单架构明显耗时；`--jobs` 控制构建任务并行数，单个 HIP 源文件内部仍可能逐架构编译。

默认产物位于：

```text
build-rocm-docker-22.04/dist/
  ftllm_rocm-0.1.8.1.post1-py3-none-linux_x86_64.whl
  ftllm_rocm-0.1.8.1.post1-py3-none-linux_x86_64.whl.sha256
  README.txt
```

文件归调用者的 UID/GID 所有。宿主用户没有 Docker socket 权限时，可以使用 `sudo bash make_whl_rocm_docker.sh ...`，脚本仍会按原调用者的身份写入产物。

常用选项：

```sh
# 只编译指定子集，支持逗号或分号分隔
bash make_whl_rocm_docker.sh --architectures 'gfx1100;gfx942' --jobs 12

# 复用已构建镜像
bash make_whl_rocm_docker.sh --skip-image-build --jobs 12

# 只重新打包已有构建；架构列表必须与该构建一致
bash make_whl_rocm_docker.sh --skip-image-build --skip-build \
  --architectures 'gfx1100;gfx942'

# 指定相对于挂载仓库的输出目录
bash make_whl_rocm_docker.sh --dist-dir dist/rocm --jobs 12

# Ubuntu 24.04，输出到 build-rocm-docker-24.04/dist/
bash make_whl_rocm_docker.sh --ubuntu 24.04 --jobs 12
```

`--build-dir` 和 `--dist-dir` 为容器内路径，相对路径从 `/workspace/fastllm`（挂载的源码仓库）解析。容器构建目录与宿主构建目录分开，以免混用 CMake 缓存。更换 SDK 版本或自定义基础镜像时，应选择新的构建目录；SDK 升级导致 `all` 的架构列表变化时，旧构建需要重新编译。

Docker Hub 无法访问时，可以使用 Ubuntu 官方 ECR 镜像：

```sh
bash make_whl_rocm_docker.sh \
  --base-image public.ecr.aws/ubuntu/ubuntu:22.04 --jobs 12

# 24.04 需选择对应的基础镜像
bash make_whl_rocm_docker.sh --ubuntu 24.04 \
  --base-image public.ecr.aws/ubuntu/ubuntu:24.04 --jobs 12
```

Ubuntu 镜像来源见 [Canonical 文档](https://ubuntu.com/docs/oci-registries/oci-how-to/getting-started/)。`--base-image` 也可以指定镜像 digest；它应与 `--ubuntu` 选择的发行版一致。`--image` 可指定构建镜像名称，`--rocm-version` 可指定 Python SDK 版本。当前验证版本为 10.0.0，其他 SDK 版本需要重新验证。

如需只构建镜像，两个 Dockerfile 的 context 都使用 `whl_docker_rocm/`：

```sh
docker build -t ftllm-rocm-builder:ubuntu22.04-rocm10.0.0 whl_docker_rocm
docker build -f whl_docker_rocm/24.04/Dockerfile \
  -t ftllm-rocm-builder:ubuntu24.04-rocm10.0.0 whl_docker_rocm
```

镜像通过共用的 `install-sdk.sh` 安装固定版本的开发工具。构建 context 不包含源码、模型或已有构建产物；运行镜像时再挂载源码仓库。Docker 与宿主打包入口最终调用相同的 `tools/scripts/build_rocm_wheel.py`。

## 使用宿主 SDK 编译 wheel

以下命令适用于 Ubuntu 22.04；SDK 的 10.0.0 是 Python 发行包版本：

```sh
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential ca-certificates git libnuma-dev python3-venv

python3 -m venv "$HOME/venvs/fastllm-rocm-build"
. "$HOME/venvs/fastllm-rocm-build/bin/activate"
python -m pip install 'pip==26.2.1'
python -m pip install \
  'rocm[libraries,devel]==10.0.0' \
  'cmake==4.4.3' 'ninja==1.13.2' 'setuptools==84.0.0' 'wheel==0.48.0' \
  --extra-index-url https://stable.repo.amd.com/rocm/whl-next/

bash make_whl_rocm.sh --list-architectures
bash make_whl_rocm.sh --jobs 12
```

脚本自动定位当前 Python SDK，不需要固定的虚拟环境路径，也不需要提前设置 `ROCM_PATH`、`HIP_PATH`、`CMAKE_PREFIX_PATH` 或 `LD_LIBRARY_PATH`。默认构建目录为 `build-rocm-wheel`，默认输出目录为 `<build-dir>/dist`。

```sh
# 选择子集和独立构建目录
bash make_whl_rocm.sh --architectures 'gfx1100;gfx942' \
  --build-dir build-rocm-subset --jobs 12

# 复用该构建，只重新生成 wheel
bash make_whl_rocm.sh --architectures 'gfx1100;gfx942' \
  --build-dir build-rocm-subset --skip-build
```

脚本不删除原有构建目录；`--skip-build` 会检查 ROCm 后端、CPU 构建基线和架构配置，不接受不匹配的 CUDA 或其他架构构建。

### 只编译原生共享库

开发和调试时也可以直接调用 CMake：

```sh
# 在上面的 SDK 虚拟环境中执行
export ROCM_PATH="$(python -m rocm_sdk path --root)"
cmake -S . -B build-rocm-dev -G Ninja \
  -DUSE_ROCM=ON -DUSE_CUDA=OFF -DUSE_NUMAS=ON \
  -DROCM_ARCH='gfx1100;gfx942' \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++
cmake --build build-rocm-dev --target fastllm_tools --parallel 12
```

这里生成的是开发用共享库。分发 Python 包时使用前面的 wheel 脚本，它会设置统一 CPU 基线、生成运行库加载代码、附带 libnuma 和许可证，并声明对应 GPU 设备包的 pip extras。不要直接把开发目录中的 `.so` 当作完整 pip 包分发。

## 用户安装 wheel

在已有 Python/pip 和兼容 AMD 驱动的机器上，用户不需要安装 ROCm 开发 SDK、hipcc、GCC、CMake、PyTorch 或 Triton。ROCm 核心运行库、数学库、指定 GPU 的设备数据和应用 Python 依赖会由 pip 安装；模型文件单独准备。

以 RX 7900 GRE 的 `gfx1100` 为例，在 wheel 所在目录运行：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install \
  './ftllm_rocm-0.1.8.1.post1-py3-none-linux_x86_64.whl[gfx1100]' \
  --extra-index-url https://stable.repo.amd.com/rocm/whl-next/
python -m pip check

ftllm chat /path/to/Qwen3.5-4B \
  --device cuda:0 --dtype float16 --atype float16 \
  --max_batch 1 --kv_cache_limit 1G --chunked_prefill_size 256 --enable_thinking false

ftllm server /path/to/Qwen3.5-4B \
  --device cuda:0 --dtype float16 --atype float16 \
  --max_batch 1 --kv_cache_limit 1G --chunked_prefill_size 256 --enable_thinking false
```

AMD 后端沿用 `cuda:0` 设备参数。需要访问 `/dev/kfd` 和 `/dev/dri/renderD*`；驱动、固件及设备访问权限由宿主提供。已安装 wheel 的程序不需要手动设置 ROCm 库路径环境变量。

至少选择一个与显卡对应、且包含在该 wheel 中的架构 extra。多种 GPU 可以组合为 `[gfx1100,gfx942]`；`[all-gpus]` 安装该 wheel 所覆盖全部架构的设备数据，体积会显著增加。普通 `[all]` 保持应用功能依赖的含义，不代表全部 GPU。

`ftllm-rocm` 与 NVIDIA 版使用相同的 `ftllm` Python 模块和 CLI，请使用独立虚拟环境。本文描述的是本地生成 wheel 的安装方式，不能假定 PyPI 上已有相同版本或相同架构覆盖的发布包。

## 构建标识、验证与限制

wheel 内的 `ftllm/rocm_build_info.json` 记录 SDK 版本、实际架构、HIP bundle 检查数、源码提交、工作区是否有未提交修改、原生库 SHA256、构建系统、Python 和 glibc 版本。构建时的 glibc 版本不是自动计算出的最低运行版本。

当前已验证的能力包括 Qwen3.5-4B FP16 推理、OpenAI API、ROCm 原生采样和图重放。gfx942 的 FP8 兼容层使用软件转换保留 OCP E4M3 格式，转换回归覆盖所有双字节组合、舍入、次正规数、正负零、NaN 和饱和边界。DFlash 专用 top-k 和拒绝采样尚未完成移植；其他模型、量化路径及视觉功能需要分别验证。

构建端的架构选择测试无需 GPU：

```sh
python test/test_rocm_wheel_architectures.py -v
```

2026-09-05 验证记录：

| 构建环境 | 验证结果 |
|---|---|
| Ubuntu 22.04 / Python 3.10 宿主 SDK | 25 目标完整编译和打包通过；18 个 HIP bundle 均含全部目标，wheel 为 85.37 MiB |
| Ubuntu 22.04 Docker | gfx1100 + gfx942 编译和打包、镜像缓存复用、仅重新打包通过；产物安装和模型 API 验证通过 |
| Ubuntu 24.04 Docker / Python 3.12 | gfx1100 编译和打包通过；未进行该产物的 GPU 运行验收 |

25 目标 wheel 已在独立运行环境中安装并通过 `pip check`、CLI 导入、Qwen3.5-4B FP16 API、原生采样及图重放回归。运行环境不含开发 SDK、NVIDIA 依赖、PyTorch 或 Triton；FastLLM/ROCm 库从该环境加载，未借用源码构建目录。实机验证仅使用 RX 7900 GRE，其他目标目前完成了编译检查。gfx942 的 FP8 软件转换逻辑还在 gfx1100 上通过强制软件分支验证，这不能替代 MI300 实机验收。

85.37 MiB 仅为 FastLLM wheel 的体积，不包括 pip 安装的 ROCm 运行库、设备数据、其他 Python 依赖和模型文件。

对应 `rocm` 分支提交：`e9e5e4a0`（多架构 FP8 兼容）、`dfe41a6b`（动态全架构 wheel 打包）、`be211b1b`（Docker 构建入口）。本次 25 目标 wheel 记录的源码提交为 `be211b1b`，构建时工作区干净。

## 鸣谢

[leavelet](https://github.com/leavelet) 提供了早期 ROCm 支持。
