# fastllm

| [快速开始](#快速开始) | [部署DeepSeek](docs/deepseek.md) | [部署Qwen3](docs/qwen3.md) | [版本日志](docs/version.md) | [English Document](README_EN.md)

# 引用说明

本项目参考了许多开源项目的代码和相关文章，具体请参考 [参考代码和文章](#参考代码和文章)

## 介绍

fastllm是c++实现自有算子替代Pytorch的高性能全功能大模型推理库，可以推理Qwen, Llama, Phi等稠密模型，以及DeepSeek, Qwen-moe等moe模型
- 具有优良的兼容性，支持M40, K80到5090全系列N卡，支持MI50，7900等A卡，支持天数，沐曦等国产卡，支持ThinkForce NPU推理
- 支持任意显卡的FP8推理
- 任意显卡只需要显存 > 10G就可以支持单卡推理满血DeepSeek R1 671B模型
- 双路9004/9005服务器 + 单显卡部署DeepSeek R1 671B - FP8原版模型，单并发速度可达20左右，部署INT4模型单并发速度可达30左右，最高并发速度可达60+

部署交流QQ群： 831641348

微信群：目前群聊超过200人，请添加小助手微信号`fastllmxzs`加群: 

## 新功能速览

- Fastllm目前支持Qwen3-Next模型的混合推理了！
- Fastllm目前支持导出通用动态量化模型了！参考[动态量化说明](docs/dtype_config.md)
- Fastllm目前可以支持部分GGUF模型的读取了！需要注意，目前需要使用`--ori`参数指定源模型配置文件夹，请阅读 [使用指南](#使用指南)

## 亮点功能

- 🚀 安装使用简单方便，一条命令就能成功安装，一条命令就能成功运行。
- 🚀 支持CPU + GPU混合推理MOE大参数模型（单显卡即可推理DEEPSEEK 671B）。
- 🚀 使用C++实现自有底层算子，不依赖PyTorch。
- 🚀 兼容性好，PIP安装支持可以支持到P100、MI50等老卡，源码安装支持更多设备。
- 🚀 支持多卡张量并行推理，支持3、5、7等奇数张卡。
- 🚀 支持GPU + CPU混合张量并行推理
- 🚀 支持CPU和显卡实现FP8运算，老设备也可以运行
- 🚀 支持多CPU加速，且只占用1份内存
- 🚀 支持ROCM，AMD GPU；支持天数，沐曦，燧原；支持华为昇腾。
- 🚀 支持动态Batch，流式输出；前后端分离设计，可跨平台移植，可在安卓上直接编译。
- 🚀 支持Python[自定义模型结构](docs/custom.md)

## 快速开始

### 安装


- `pip`安装支持`Nvidia GPU`和`AMD GPU`，其余`GPU`请使用[源码安装](#源码安装)
- `pip`安装速度慢时，可使用镜像加速

```
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

#### Linux系统 + Nvidia GPU:

由于目前PyPI限制库大小，安装包中不含CUDA依赖，安装ftllm之前建议先手动安装CUDA 12以上版本 (已安装cuda可跳过)
```
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run
```

然后用pip安装，命令如下：

```
pip install ftllm -U
```

#### Linux系统 + AMD GPU:

由于目前PyPI限制库大小，安装包中不含ROCM依赖，安装ftllm之前建议先手动安装ROCM 6.3.3 (若已安装ROCM可跳过)
```
wget wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/jammy/amdgpu-install_6.3.60303-1_all.deb
apt install ./amdgpu-install_6.3.60303-1_all.deb -y
amdgpu-install --usecase=hiplibsdk,rocm,dkms -y
```

然后用pip安装，命令如下：

```
pip install ftllm-rocm -U
```

#### Windows系统 （仅支持Nvidia GPU）:

第一次安装前需要安装依赖库:

```
pip install https://www.modelscope.cn/models/huangyuyang/fastllmdepend-windows/resolve/master/ftllmdepend-0.0.0.1-py3-none-win_amd64.whl
```

然后用pip安装，命令如下：

```
pip install ftllm -U
```

- Hint

Conda下安装有时候会出现环境错误，如果出现可以尝试在Conda外或使用venv等虚拟环境尝试

（若使用时报错，可参考[ftllm报错](docs/faq.md#ftllm加载报错) )

### 运行demo程序

可以运行一个较小模型测试安装是否成功, 以Qwen/Qwen3-0.6B模型为例

#### 命令行聊天：

```
ftllm run Qwen/Qwen3-0.6B
```

#### WebUI:

```
ftllm webui Qwen/Qwen3-0.6B
```

#### API Server (OpenAI 风格):

```
ftllm server Qwen/Qwen3-0.6B
```

## 使用指南

### 1. 如何启动模型

基本的启动命令格式如下：

```
ftllm run Qwen/Qwen3-0.6B # 启动本地对话
ftllm webui Qwen/Qwen3-0.6B # 启动WebUI
ftllm server Qwen/Qwen3-0.6B # 启动API Server
```

根据你需要开启的服务，选择相应的命令。以 `server` 命令为例，格式如下：

```
ftllm server model
```

这里的`model`可以是:

- Huggingface上的模型，例如 `Qwen/Qwen3-0.6B` 代表 [千问3-0.6B模型](https://hf-mirror.com/Qwen/Qwen3-0.6B)
- 本地模型路径。例如`/mnt/Qwen3-0.6B`，高速下载模型可以参考 [模型下载](#模型下载)

无论是在线模型还是本地模型，目前支持以下几种格式 （均以在线模型举例，可以在Huggingface上搜到对应模型）:

- `FP16`, `BF16`格式的原始模型，例如`Qwen/Qwen3-0.6B`
- `FP8`格式的模型，例如`Qwen/Qwen3-0.6B-FP8`
- `AWQ`格式的模型，例如`Qwen/Qwen3-14B-AWQ`
- `Fastllm`格式的模型，例如`fastllm/DeepSeek-V3-0324-INT4`。也可以下载原始模型后通过 [模型导出](#模型导出) 中的命令导出
- `GGUF` 格式的模型，需要用`--ori`参数指定原始模型路径，例如

``` sh
ftllm server DeepSeek-V3-0324-Q4_K_M-00001-of-00009.gguf --ori DeepSeek-V3
```

这里的`DeepSeek-V3`是原始模型文件夹，无需下载权重文件，可以参考如下命令下载：

``` sh
ftllm download deepseek-ai/DeepSeek-V3 --exclude "*safetensors*"
```


如果您是第一次使用ftllm，建议直接使用基本的启动命令启动，所有的参数都会自动设置。如果您希望继续调参，请参照下面的参数设置说明

### 2. 如何设定推理精度

当启动的模型为浮点精度时（`BF16`, `FP16`, `FP8`）时，可以通过以下参数来设置模型的推理精度：

- `--dtype`:
  - **描述**: 指定模型的数据类型。
  - **可选值**: `int4g` `int4` `int8` `fp8` `float16` 或其他支持的数据类型。
  - **示例**: `--dtype int4`

- `--moe_dtype`:
  - **描述**: 指定模型MOE层的数据类型。
  - **可选值**: `int4g` `int4` `int8` `fp8` `float16` 或其他支持的数据类型。
  - **示例**: `--moe_dtype int4`
  - **说明**: 如果指定的模型不是`moe`结构的模型，这个参数不会生效

命令示例：

```
ftllm server Qwen/Qwen3-0.6B --dtype int8 
# 上面的命令会读取原始模型（这个模型是BF16精度），并在线量化为INT8精度推理

ftllm server deepseek-ai/DeepSeek-V3-0324 --dtype fp8 --moe_dtype int4
# 上面的命令会读取原始模型（这个模型是FP8精度），并使用FP8 + INT4的混合精度推理
```

- `--dtype_config`:
  - **描述**: 指定动态量化配置文件。
  - **说明**: 参考[动态量化说明](docs/dtype_config.md)

若不设定这些参数，ftllm会使用模型中设定的精度来进行推理

若使用的模型已经是量化好的模型（例如AWQ模型，Fastllm导出的量化模型等），建议不指定这些参数

### 3. 如何设定运行设备

可以通过以下参数来设定执行推理的设备

- `--device`:
  - **描述**: 指定模型运行的计算设备。
  - **示例**: `--device cpu`, `--device cuda`
  - **常用值说明**: 
    - `cpu` 使用`cpu`推理
    - `cuda` 使用`gpu`推理 
    - `numa` 使用多路`numa`节点加速推理，在多CPU的机器才会有提升。使用numa加速时，强烈建议关闭系统numa平衡。（ubuntu中可使用命令`sudo sysctl -w kernel.numa_balancing=0`)
    - `multicuda` 使用多设备张量并行推理
      - **使用多显卡**: `--device multicuda:0,1`
      - **使用多显卡+CPU**: `--device multicuda:0,cpu`
      - **按比例使用多显卡+CPU**: `--device multicuda:0:4,1:5,cpu:1` 
      (`cuda:0`计算4/10, `cuda:1`计算5/10, `cpu`计算1/10)
  - **串行计算**: 一些场景下可以指定不同的device串行执行。例如
    - `--device "{'cuda:0':3,'cuda:1':2}"`: 这样`3/5`的层会运行在`cuda:0`上，`2/5`的层会运行在`cuda:1`上
    - `--device "{'multicuda:0,1':3,'cuda:1':2}"`: 这样`3/5`的层会使用`cuda:0`,`cuda:1`张量并行，`2/5`的层仅仅运行在`cuda:1`上

- `--moe_device`:
  - **描述**: 指定 MOE（Mixture of Experts）层的计算设备。
  - **示例**: `--moe_device cpu`, `--moe_device numa`
  - **常用值说明**: 
    - `cpu` 使用`cpu`推理
    - `numa` 使用多路`numa`节点加速推理，在多CPU的机器才会有提升
    - `cuda` 使用`gpu`推理 （MOE层需要大量显存，一般不建议指定为`cuda`）
  - **说明**: 一般和device指定为不同的设备实现混合推理，例如
  `--device cuda --moe_device cpu`来实现MOE模型的单卡+CPU混合推理。
   `--device cuda --moe_device numa` 来实现MOE模型的单卡+多NUMA节点加速推理
   如果指定的模型不是`moe`结构的模型，这个参数不会生效

若不设定这些参数，会使用默认配置来推理，默认配置如下：

| 模型类型 | device | moe_device |
|-------:|--------|------------:|
| 稠密模型 | cuda   |   不生效    |
| MOE模型  | cuda   |   cpu    |

对于发烧友而言，如果想更进一步榨干硬件，可以参考 [混合推理指南](docs/mixforward.md)

### 4. 如何设定运行参数

可以通过下列参数设置运行参数。

需要注意的是，速度和参数设置并不一定正相关，如果对性能要求高，可以多方向尝试一下

- `-t` 或 `--threads`:
  - **描述**: 设置使用的CPU线程数。
    - 当`device`为`cpu`时，这个参数决定了推理使用的线程数
    - 当`device`为`numa`时，推理线程数主要由环境变量`FASTLLM_NUMA_THREADS`决定，`threads`参数请设得小一点（推荐设为1）
  - **示例**: `-t 27`

例如我们在多CPU设备上用GPU + 多CPU混合部署一个`MOE`模型`fastllm/DeepSeek-V3-0324-INT4`，可以尝试这些命令：

``` bash
export FASTLLM_NUMA_THREADS=27 && ftllm server fastllm/DeepSeek-V3-0324-INT4 --device cuda --moe_device numa -t 1 
# 使用多numa推理，每个numa节点使用27个线程

export FASTLLM_NUMA_THREADS=16 && ftllm server fastllm/DeepSeek-V3-0324-INT4 --device cuda --moe_device numa -t 1 
# 使用多numa推理，每个numa节点使用16个线程

numactl -C 0-31 -m 0 ftllm server fastllm/DeepSeek-V3-0324-INT4 --device cuda --moe_device cpu -t 27 
# 绑定单numa节点，使用CPU推理，使用27线程
```

不同硬件上，不同参数发挥出的性能有很大不同。一般而言，CPU上使用的线程数不建议超过物理核数


### 5. 其它参数

- `--moe_experts`:
  - **描述**: 指定 MOE（Mixture of Experts）层使用的专家数。不设定则根据模型配置设定。减少专家数可以提高推理速度，但可能降低推理准确度
  - **示例**: `--moe_experts 6`

- `--cuda_se`:
  - **描述**: 指定 MOE中的共享专家 是否在cuda上执行，默认为true
  - **示例**: `--cuda_se false`

- `--port`:
  - **描述**: 指定服务运行的端口号。
  - **示例**: `--port 8080`

- `--help`:
  - **描述**: 查看模块参数详细信息。
  - **示例**: `ftllm server --help`

- `--version` 或 `-v`:
  - **描述**: 查看ftllm版本号。
  - **示例**: `ftllm -v`

- `--hide_input`:
  - **描述**: server模式隐藏日志中的请求信息。
  - **示例**: `ftllm server --hide_input`
 
- `--api_key`:
  - **描述**: server模式设定api_key。
  - **示例**: `ftllm server --api_key xxxxxxxx` 
 
- `--max_token`:
  - **描述**: webui模式指定最大输出。
  - **示例**: `ftllm webui --max_token`
 
- `--think`:
  - **描述**: 强制思考。
  - **示例**: `ftllm webui --think`

- `--cache_dir`:
  - **描述**: 指定在线Huggingface模型的缓存目录
  - **示例**: `ftllm --cache_dir /mnt`

- `--chat_template`:
  - **描述**: 指定chat_template文件
  - **示例**: `ftllm --chat_template deepseekv31.jinja`

## 工具调用

目前以下模型支持工具调用：

- GLM4.5, GLM4.5-AIR
- Qwen3-Instruct系列
- Qwen3-Coder系列
- Kimi-K2
- DeepSeekV3.1, 需要指定chat_template, 文件位于本项目`example/chat_template/deepseekv31.jinja`

## 模型获取

### 模型下载

可以使用如下命令将模型下载到本地（使用高速镜像，无需科学上网）

```
ftllm download deepseek-ai/DeepSeek-R1
```


### 模型导出

如果使用量化加载模型（如`--dtype int4`），那么每次读取模型时会在线量化，读取速度较慢。

ftllm export 是一个用于导出和转换模型权重的工具。它支持将模型权重转换为不同的数据类型。以下是如何使用 ftllm export 的详细说明。

#### 命令格式

``` sh
ftllm export <模型路径> -o <输出路径> --dtype <数据类型> -t <线程数>
```

#### 示例命令

``` sh
ftllm export /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-INT4 --dtype int4 -t 16
```

#### 混合精度

可以通过指定`--moe_dtype`来实现混合精度，例如

``` sh
ftllm export /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-FP16INT4 --dtype float16 --moe_dtype int4 -t 16
```

#### 加载导出后的模型

导出后的模型使用方法和原始模型类似，使用导出模型时`--dtype`参数将被忽略

例如

``` sh
ftllm run /mnt/DeepSeek-V3-INT4/
```

### 支持的模型

如果需要运行更多早期的模型，请参考[支持模型列表](docs/models.md)

### 源码安装

若pip安装失败或有其它特殊需求，可以用源码编译安装
源码安装后如果需要卸载，方法和PIP安装一样
```
pip uninstall ftllm
```

建议使用cmake编译，需要提前安装gcc，g++ (建议9.4以上), make, cmake (建议3.23以上)

GPU编译需要提前安装好CUDA编译环境，建议使用尽可能新的CUDA版本

使用如下命令编译

``` sh
bash install.sh -DUSE_CUDA=ON -D CMAKE_CUDA_COMPILER=$(which nvcc) # 编译GPU版本
# bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=89 -D CMAKE_CUDA_COMPILER=$(which nvcc) # 可以指定CUDA架构，如4090使用89架构
# bash install.sh # 仅编译CPU版本
```

##### 其他平台编译

其他不同平台的编译可参考文档

[TFACC平台](docs/tfacc.md)  
[ROCm平台](docs/rocm.md)

编译中遇到问题可参考 [FAQ文档](docs/faq.md)

## 参考代码和文章

### 大量NN底层算子的实现思路

[pytorch](https://github.com/pytorch/pytorch)

### 大量LLM具体的模型实现

[transfomers](https://github.com/huggingface/transformers)

### GGML中的一些量化方法、以及计算Kernel

[llama.cpp](https://github.com/ggml-org/llama.cpp)

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)

### MOE算子线程不平衡时动态调度的思路

[KTransformers 0.3 思路介绍](https://zhuanlan.zhihu.com/p/1900318746402329329)

[KT中关于线程调度的相关代码](https://github.com/kvcache-ai/ktransformers/blob/main/csrc/ktransformers_ext/cpu_backend/backend.cpp)

### 基于numa改进的MOE动态调度算子

[lvllm中的实现](https://github.com/guqiong96/Lvllm/blob/main/csrc/lk/moe.cpp)

### Function call解析相关的代码

[vllm中的实现](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers)

### json的构造和解析

[json11](https://github.com/dropbox/json11)

感谢大佬对开源社区的贡献！如发现未标明的引用代码可在issue中提出