## DeepSeek模型介绍

DeepSeek是深度求索公司出品的模型，目前主要产品为DeepSeek-V3和DeepSeek-R1。

### 编译Fastllm

建议使用cmake编译，需要提前安装gcc，g++ (建议9.4以上), make, cmake (建议3.23以上)

在Ubuntu下可以使用如下命令安装

``` sh
apt-get install gcc g++ make cmake
```

编译GPU版本需要提前安装好CUDA编译环境，建议使用尽可能新的CUDA版本

使用如下命令编译

``` sh
git clone https://www.github.com/ztxz16/fastllm
cd fastllm
bash install.sh -DUSE_CUDA=ON -D CMAKE_CUDA_COMPILER=$(which nvcc) # 编译GPU版本
# bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=89 -D CMAKE_CUDA_COMPILER=$(which nvcc) # 可以指定CUDA架构，如4090使用89架构
# bash install.sh # 仅编译CPU版本
```

### 模型下载

国内建议使用镜像站下载

[镜像站使用说明](https://hf-mirror.com/)

[DeepSeek-R1模型](https://hf-mirror.com/deepseek-ai/DeepSeek-R1)

[DeepSeek-V3模型](https://hf-mirror.com/deepseek-ai/DeepSeek-V3)

[DeepSeek-V2模型](https://hf-mirror.com/deepseek-ai/DeepSeek-V2.5-1210)

### 运行demo程序

假设模型存储在`/mnt/DeepSeek-V3/`目录下

编译完成之后可以使用下列服务:

（如果使用多numa机器，建议用numactl绑定在一个numa节点上运行）

``` sh
# webui，运行后可以在浏览器访问，使用纯CPU运行
python3 -m ftllm.webui -p /mnt/DeepSeek-V3/ -t 27 --device cpu --dtype int4 --port 8080

# webui，运行后可以在浏览器访问，MOE层使用CPU运行，其余层使用CUDA运行
python3 -m ftllm.webui -p /mnt/DeepSeek-V3/ -t 27 --moe_device cpu --device cuda --dtype int4 --port 8080

# 命令行对话的Demo
python3 -m ftllm.chat -p /mnt/DeepSeek-V3/ -t 27 --moe_device cpu --device cuda --dtype int4

# openai api server, 这里在8080端口打开了一个模型名为deepseek的server
python3 -m ftllm.server -p /mnt/DeepSeek-V3/ -t 27 --moe_device cpu --device cuda --dtype int4 --port 8080 --model_name deepseek
```

#### 参数说明

以下是运行 `ftllm` 模块时常用的参数说明：

##### 通用参数

- `-p` 或 `--path`:
  - **描述**: 指定模型路径。
  - **示例**: `-p /mnt/DeepSeek-V3/`

- `-t` 或 `--threads`:
  - **描述**: 设置使用的线程数。
  - **示例**: `-t 27`

- `--device`:
  - **描述**: 指定模型运行的计算设备。
  - **可选值**: `cpu` 或 `cuda`。
  - **示例**: `--device cpu` 或 `--device cuda`

- `--moe_device`:
  - **描述**: 指定 MOE（Mixture of Experts）层的计算设备。
  - **可选值**: `cpu` 或 `cuda`。
  - **示例**: `--moe_device cpu`

- `--dtype`:
  - **描述**: 指定模型的数据类型。
  - **可选值**: `int4` 或其他支持的数据类型。
  - **示例**: `--dtype int4`

- `--port`:
  - **描述**: 指定服务运行的端口号。
  - **示例**: `--port 8080`


以上demo均可使用参数 --help 查看详细参数，详细参数说明可参考 [参数说明](docs/demo_arguments.md)

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
# webui，运行后可以在浏览器访问，MOE层使用CPU运行，其余层使用CUDA运行
python3 -m ftllm.webui -p /mnt/DeepSeek-V3-INT4/ -t 27 --moe_device cpu --device cuda --port 8080
```