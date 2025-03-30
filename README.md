# fastllm

[English Document](README_EN.md)

## 介绍

fastllm是c++实现，后端无依赖（仅依赖CUDA，无需依赖PyTorch）的高性能大模型推理库。

可实现MOE模型混合推理，eypc 9374f*2 + 24路DDR5 4800 + 4090 24G，推理DeepSeek R1 671B INT4模型单路可达20+tps。

部署交流QQ群： 831641348

部署交流微信群: ![二维码](docs/wechat_group0.jpg)

| [快速开始](#快速开始) | [DeepSeek部署指南](docs/deepseek.md) | [版本日志](docs/version.md) |

## 亮点功能

- 🚀 DeepSeek混合推理，消费级单卡即可多并发部署，后续将支持多卡提速
- 🚀 双CPU仅占用1份内存，部署DeepSeek R1 671b int4 共占用内存340G
- 🚀 支持多NUMA节点加速
- 🚀 支持动态Batch，流式输出
- 🚀 支持多卡部署，支持GPU + CPU混合部署
- 🚀 前后端分离设计，便于支持新的计算设备
- 🚀 后端纯c++实现，便于跨平台移植，可在安卓上直接编译
- 🚀 支持Python自定义模型结构

## 快速开始

### 安装

- PIP安装

Linux系统可尝试直接pip安装，命令如下：

```
pip install ftllm
```

（由于目前pypi限制库大小，安装包中不含cuda依赖，建议先手动安装cuda12以上版本）

（若使用时报错，可参考[ftllm报错](docs/faq.md#ftllm报错) )

- 源码安装

若pip安装失败或有其它特殊需求，可以用源码编译安装

建议使用cmake编译，需要提前安装gcc，g++ (建议9.4以上), make, cmake (建议3.23以上)

GPU编译需要提前安装好CUDA编译环境，建议使用尽可能新的CUDA版本

使用如下命令编译

``` sh
bash install.sh -DUSE_CUDA=ON -D CMAKE_CUDA_COMPILER=$(which nvcc) # 编译GPU版本
# bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=89 -D CMAKE_CUDA_COMPILER=$(which nvcc) # 可以指定CUDA架构，如4090使用89架构
# bash install.sh # 仅编译CPU版本
```

其他不同平台的编译可参考文档
[TFACC平台](docs/tfacc.md)

### 运行demo程序

以Qwen/Qwen2-0.5B-Instruct模型为例，可以运行一个较小模型测试安装是否成功

#### 命令行聊天：

```
ftllm run Qwen/Qwen2-0.5B-Instruct
```

#### webui:

```
ftllm webui Qwen/Qwen2-0.5B-Instruct
```

#### api server (openai风格):

```
ftllm server Qwen/Qwen2-0.5B-Instruct
```

#### 本地模型

可以启动本地下载好的Hugging Face模型，假设本地模型路径为 `/mnt/Qwen/Qwen2-0.5B-Instruct/`
则可以用如下命令启动（webui, server类似）

```
ftllm run /mnt/Qwen/Qwen2-0.5B-Instruct/
```

#### 模糊启动

如果记不住模型名，可以输入大概的模型名（不保证能匹配成功）
例如：
```
ftllm run qwen2-7b-awq
```

```
ftllm run deepseek-v3-0324-int4
```

#### 设置缓存目录

模型会下载到缓存目录（默认~/.cache），可以通过环境变量 `FASTLLM_CACHEDIR` 来设置缓存目录，例如在Linux下:

```
export FASTLLM_CACHEDIR=/mnt/
```

#### 参数说明

以下是运行 `ftllm` 模块时常用的参数说明：

##### 通用参数

- `-t` 或 `--threads`:
  - **描述**: 设置使用的CPU线程数。
  - **示例**: `-t 27`

- `--dtype`:
  - **描述**: 指定模型的数据类型。
  - **可选值**: `int4` 或其他支持的数据类型。
  - **示例**: `--dtype int4`
  
- `--device`:
  - **描述**: 指定模型运行的计算设备。
  - **常用值**: `cpu` 或 `cuda`或`numa`
  - **示例**: `--device cpu` 或 `--device cuda`

- `--moe_device`:
  - **描述**: 指定 MOE（Mixture of Experts）层的计算设备。
  - **常用值**: `cpu` 或 `cuda`或`numa`
  - **示例**: `--moe_device cpu`

- `--moe_experts`:
  - **描述**: 指定 MOE（Mixture of Experts）层使用的专家数。不设定则根据模型配置设定。减少专家数可以提高推理速度，但可能降低推理准确度
  - **示例**: `--moe_experts 6`

- `--port`:
  - **描述**: 指定服务运行的端口号。
  - **示例**: `--port 8080`

### 模型下载

可以使用如下命令将模型下载到本地

```
ftllm download deepseek-ai/DeepSeek-R1
```

### 模型导出

如果使用量化加载模型（如`--dtype int4`），那么每次读取模型时会在线量化，读取速度较慢。

ftllm.export 是一个用于导出和转换模型权重的工具。它支持将模型权重转换为不同的数据类型。以下是如何使用 ftllm.export 的详细说明。

#### 命令格式

``` sh
python3 -m ftllm.export -p <模型路径> -o <输出路径> --dtype <数据类型> -t <线程数>
```

#### 示例命令

``` sh
python3 -m ftllm.export -p /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-INT4 --dtype int4 -t 16
```

#### 加载导出后的模型

导出后的模型使用方法和原始模型类似，使用导出模型时`--dtype`参数将被忽略

例如

``` sh
ftllm run /mnt/DeepSeek-V3-INT4/
```