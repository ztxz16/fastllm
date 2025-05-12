## DeepSeek模型介绍

DeepSeek是深度求索公司出品的模型，目前主要产品为DeepSeek-V3和DeepSeek-R1。

### 安装Fastllm

- PIP安装

Linux系统可尝试直接pip安装，命令如下：
```
pip install ftllm -U
```
若安装失败则参考[源码安装](../README.md#安装)

### 运行示例

#### 命令行聊天：

```
ftllm run fastllm/DeepSeek-V3-0324-INT4
```

#### webui:

```
ftllm webui fastllm/DeepSeek-V3-0324-INT4
```

#### api server (openai风格):

```
ftllm server fastllm/DeepSeek-V3-0324-INT4
```

#### NUMA加速

若想使用单NUMA节点，建议用numactl绑定numa节点

可以设定环境变量来激活多NUMA节点加速（PIP版本可直接激活，源码安装时需要在编译时加入-DUSE_NUMA=ON选项）

```
export FASTLLM_USE_NUMA=ON
# export FASTLLM_NUMA_THREADS=27 # 选用，这个变量用于设定每个numa节点开启的线程数
```

#### 本地模型

可以启动本地下载好的模型（支持原始模型，AWQ模型，FASTLLM模型，暂不支持GGUF模型），假设本地模型路径为 `/mnt/DeepSeek-R1`
则可以用如下命令启动（webui, server类似）

```
ftllm run /mnt/DeepSeek-R1
```

#### 模糊启动

如果记不住模型名，可以输入大概的模型名（不保证能匹配成功）
例如：
```
ftllm run deepseek-v3-0324-int4 # 这条命令会直接运行deepseek-v3-0324版本的int4量化模型
```

#### 设置缓存目录

如果不想使用默认的缓存目录，可以通过环境变量 `FASTLLM_CACHEDIR` 来设置缓存目录，例如在Linux下:

```
export FASTLLM_CACHEDIR=/mnt/
```

#### 一些推荐模型

目前推荐使用的一些模型：

- fastllm/DeepSeek-V3-0324-INT4
- fastllm/DeepSeek-R1-INT4
- deepseek-ai/DeepSeek-R1
- deepseek-ai/DeepSeek-V3
- deepseek-ai/DeepSeek-V3-0324
- deepseek-ai/DeepSeek-V2.5-1210

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

以上demo均可使用参数 --help 查看详细参数，详细参数说明可参考 [参数说明](docs/demo_arguments.md)

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
ftllm export <模型路径> -o <输出路径> --dtype <数据类型> -t <线程数>
```

#### 示例命令

``` sh
ftllm export /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-INT4 --dtype int4 -t 16
```

#### 混合精度

可以通过指定moe_dtype来实现混合精度，例如

``` sh
ftllm export /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-FP16INT4 --dtype float16 --moe_dtype int4 -t 16
```

#### 加载导出后的模型

导出后的模型使用方法和原始模型类似，使用导出模型时`--dtype`参数将被忽略

例如

``` sh
ftllm run /mnt/DeepSeek-V3-INT4/
```