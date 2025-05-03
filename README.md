# fastllm

[English Document](README_EN.md)

| [快速开始](#快速开始) | [DeepSeek部署指南](docs/deepseek.md) | [Qwen3部署指南](docs/qwen3.md) | [版本日志](docs/version.md) |

## 介绍

fastllm是c++实现，后端无依赖（仅依赖CUDA，无需依赖PyTorch）的高性能大模型推理库，兼容Qwen、QWQ等稠密模型和DeepSeek MOE模型。

eypc 9655*2  + 24路DDR5 6400 + 4090 24G 推理DeepSeek R1 671B模型，INT4量化单路可达27+tps，并发40+tps。
eypc 9374f*2 + 24路DDR5 4800 + 4090 24G 推理DeepSeek R1 671B模型，INT4量化单路可达22+tps，并发35+tps；INT8量化单路可达17+tps,并发30+tps。


部署交流QQ群： 831641348

部署交流微信群: ![二维码](docs/wechat_group0.jpg)

## 亮点功能

- 🚀 DeepSeek混合推理，消费级单卡即可多并发部署，后续将支持多卡提速
- 🚀 双CPU仅占用1份内存，部署DeepSeek R1 671b int4 共占用内存340G
- 🚀 支持ROCM，可使用AMD GPU推理
- 🚀 支持国产GPU 天数，沐曦，燧原，均支持单卡混合推理671B模型
- 🚀 支持多NUMA节点加速
- 🚀 支持动态Batch，流式输出
- 🚀 支持多卡部署，支持GPU + CPU混合部署
- 🚀 前后端分离设计，便于支持新的计算设备
- 🚀 后端纯c++实现，便于跨平台移植，可在安卓上直接编译
- 🚀 支持Python[自定义模型结构](docs/custom.md)

## 快速开始

### 安装


- PIP安装（目前仅支持Nvidia GPU，其余GPU请使用源码安装）

#### Windows系统:

第一次安装前需要安装依赖库:

```
pip install https://hf-mirror.com/fastllm/fastllmdepend-windows/resolve/main/ftllmdepend-0.0.0.1-py3-none-win_amd64.whl
```

然后用pip安装，命令如下：

```
pip install ftllm
```

#### Linux系统:

由于目前PyPI限制库大小，安装包中不含CUDA依赖，安装ftllm之前建议先手动安装CUDA 12以上版本 (已安装cuda可跳过)
```
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run
```

然后用pip安装，命令如下：

```
pip install ftllm
```

- Hint

Conda下安装有时候会出现环境错误，如果出现可以尝试在Conda外或使用venv等虚拟环境尝试

（若使用时报错，可参考[ftllm报错](docs/faq.md#ftllm加载报错) )

#### 源码安装

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

#### NUMA加速

若想使用单NUMA节点，建议用numactl绑定numa节点

可以设定环境变量来激活多NUMA节点加速（PIP版本可直接激活，源码安装时需要在编译时加入-DUSE_NUMA=ON选项）

```
export FASTLLM_USE_NUMA=ON
# export FASTLLM_NUMA_THREADS=27 # 选用，这个变量用于设定每个numa节点开启的线程数
```

#### 启动本地模型

可以启动本地下载好的Hugging Face模型（支持原始模型，AWQ模型，FASTLLM模型，暂不支持GGUF模型），假设本地模型路径为 `/mnt/Qwen/Qwen2-0.5B-Instruct/`
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

模型会下载到缓存目录（默认~/.cache），可以通过参数`--cache_dir`来设置，例如

```
ftllm run deepseek-v3-0324-int4 --cache_dir /mnt/
```

也可以通过环境变量 `FASTLLM_CACHEDIR` 来设置，例如在Linux下:

```
export FASTLLM_CACHEDIR=/mnt/
```

## 参数设置

首次体验ftllm时建议不加任何参数，程序会自动选择参数。

当需要对性能进行调优时，再增加参数测试

以下是运行 `ftllm` 模块时常用的参数说明：

### 通用参数

- `--device`:
  - **描述**: 指定模型运行的计算设备。
  - **常用值**: `cpu` 或 `cuda`或`numa`或`multicuda`
  - **示例**: `--device cpu` 或 `--device cuda`
  - **使用多显卡**: `--device multicuda:0,1`
  - **使用显卡+CPU**: `--device multicuda:0,cpu`
  - **按比例使用多显卡+CPU**: `--device multicuda:0:4,1:5,cpu:1`
  (cuda:0计算4/10, cuda:1计算5/10, cpu计算1/10)

- `--moe_device`:
  - **描述**: 指定 MOE（Mixture of Experts）层的计算设备。
  - **常用值**: `cpu` 或 `cuda`或`numa`
  - **示例**: `--moe_device cpu`
  - **说明**: 一般和device指定为不同的设备实现混合推理，例如
  `--device cuda --moe_device cpu`来实现MOE模型的单卡+CPU混合推理。
   `--device cuda --moe_device numa` 来实现MOE模型的单卡+多NUMA节点加速推理

- `-t` 或 `--threads`:
  - **描述**: 设置使用的CPU线程数。但当device为numa时，会开启计算服务器，计算线程数由环境变量`FASTLLM_NUMA_THREADS`决定，threads参数请设得小一点（可以设为1）
  - **示例**: `-t 27`

- `--dtype`:
  - **描述**: 指定模型的数据类型。
  - **可选值**: `int4` `int8` `float16` 或其他支持的数据类型。
  - **示例**: `--dtype int4`
  - **说明**: 使用原始模型时，指定此参数可以在线量化模型。例如下述命令会将DeepSeek-R1在线量化为int4后运行。
  ```
  ftllm run deepseek-ai/DeepSeek-R1 --dtype int4
  ```
  若使用的模型已经是量化好的模型（例如AWQ模型，Fastllm导出的量化模型等），建议不指定该参数

- `--moe_dtype`:
  - **描述**: 指定模型MOE层的数据类型。
  - **可选值**: `int4` `int8` `float16` 或其他支持的数据类型。
  - **示例**: `--moe_dtype int4`
  若使用的模型已经是量化好的模型（例如AWQ模型，Fastllm导出的量化模型等），建议不指定该参数

- `--moe_experts`:
  - **描述**: 指定 MOE（Mixture of Experts）层使用的专家数。不设定则根据模型配置设定。减少专家数可以提高推理速度，但可能降低推理准确度
  - **示例**: `--moe_experts 6`

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

### 模块参数
  
各个模块的参数说明请参考[参数说明](docs/demo_arguments.md)

## 模型获取

### 模型下载

可以使用如下命令将模型下载到本地

```
ftllm download deepseek-ai/DeepSeek-R1
```


### 模型导出

如果使用量化加载模型（如`--dtype int4`），那么每次读取模型时会在线量化，读取速度较慢。

ftllm export 是一个用于导出和转换模型权重的工具。它支持将模型权重转换为不同的数据类型。以下是如何使用 ftllm export 的详细说明。

#### 命令格式

``` sh
ftllm export -p <模型路径> -o <输出路径> --dtype <数据类型> -t <线程数>
```

#### 示例命令

``` sh
ftllm export -p /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-INT4 --dtype int4 -t 16
```

#### 混合精度

可以通过指定`--moe_dtype`来实现混合精度，例如

``` sh
ftllm export -p /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-FP16INT4 --dtype float16 --moe_dtype int4 -t 16
```

#### 加载导出后的模型

导出后的模型使用方法和原始模型类似，使用导出模型时`--dtype`参数将被忽略

例如

``` sh
ftllm run /mnt/DeepSeek-V3-INT4/
```

### 支持的模型

如果需要运行更多早期的模型，请参考[支持模型列表](docs/models.md)
