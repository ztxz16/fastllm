## Qwen3模型介绍

Qwen3是阿里巴巴出品的系列模型

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
ftllm run fastllm/Qwen3-235B-A22B-INT4MIX
ftllm run Qwen/Qwen3-30B-A3B
```

#### webui:

```
ftllm webui fastllm/Qwen3-235B-A22B-INT4MIX
ftllm webui Qwen/Qwen3-30B-A3B
```

#### api server (openai风格):

```
ftllm server fastllm/Qwen3-235B-A22B-INT4MIX
ftllm server Qwen/Qwen3-30B-A3B
```

#### 参数建议

如有需要，可以将以下参数可以加在运行命令中

- 硬思考模式: 千问3的独有模式，该模式默认打开，可以通过enable_thinking参数来关闭，关闭后模型将不生成思考。例如

```bash
ftllm server Qwen/Qwen3-30B-A3B --enable_thinking false
```

- 推理设备: 非MOE模型默认使用显卡推理，若显存容量不足希望使用纯CPU推理，可以设置`--device cpu`, 或`--device numa`使用多路numa加速
- 量化: Qwen3系列模型目前建议使用参数`--dtype int4g256`指定4bit量化，`--dtype int8`指定8bit量化


- MOE模型（Qwen3-30B-A3B, Qwen3-235B-A22B）默认使用cpu+gpu混合推理，若希望使用cuda推理需要指定device参数，例如
``` bash
ftllm server Qwen/Qwen3-30B-A3B --device cuda --dtype int4g256
ftllm server Qwen/Qwen3-30B-A3B --device cuda --dtype int8
```

- 更多参数信息可参考 [常用参数](../README.md#常用参数)

#### NUMA加速

若想使用单NUMA节点，建议用numactl绑定numa节点

可以设定环境变量来激活多NUMA节点加速（PIP版本可直接激活，源码安装时需要在编译时加入-DUSE_NUMA=ON选项）

```
export FASTLLM_USE_NUMA=ON
# export FASTLLM_NUMA_THREADS=27 # 选用，这个变量用于设定每个numa节点开启的线程数
```

#### 本地模型

可以启动本地下载好的模型（支持原始模型，AWQ模型，FASTLLM模型，暂不支持GGUF模型），假设本地模型路径为 `/mnt/Qwen/Qwen3-30B-A3B`
则可以用如下命令启动（webui, server类似）

```
ftllm run /mnt/Qwen/Qwen3-30B-A3B
```

### 模型下载

可以使用如下命令将模型下载到本地

```
ftllm download Qwen/Qwen3-30B-A3B
```
