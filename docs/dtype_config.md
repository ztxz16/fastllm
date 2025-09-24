## 基本用法

通过 原始模型 + 量化配置文件，可以使用、导出不同格式的量化模型

例如我们使用模型

`Qwen3-Next-80B-A3B-Instruct` （https://hf-mirror.com/Qwen/Qwen3-Next-80B-A3B-Instruct）

和量化配置文件：

`UD-Q2_K_S.json` : （[这个文件](../example/quant/qwen3_next/UD-Q2_K_S.json)）

下面这条命令会读会读取`UD-Q2_K_S.json`中的的规则来进行动态量化，并部署`api server`
```
ftllm server Qwen3-Next-80B-A3B-Instruct --dtype_config UD-Q2_K_S.json
```

下面这条命令会读会读取`/root/dtype_config.json`中的的规则来进行动态量化，并导出模型
```
ftllm export Qwen3-Next-80B-A3B-Instruct --dtype_config UD-Q2_K_S.json -o Qwen3-Next-80B-A3B-Instruct-UD-Q2_K_S
```

# 如何编写配置文件

这个文档以`DeepSeek-V3-0324`模型为例，介绍如何使用动态量化功能

（注意，目前的量化功能为初版，后续会支持动态设定量化方法，以及设置重要性矩阵来提高量化精度）

编写动态量化配置时，最好先了解如下知识：
- 需要对`正则表达式`有一定了解
- 需要对`json`格式有一定了解
- 需要对模型的结构有一定了解。可以通过查看模型文件夹下的`model.safetensors.index.json`来查看模型的权重名称。

我们以如下示例来说明如何编写配置文件：

```
[
    {
        "key" : "(.)*",
        "dtype" : "int4g",
        "comment": "default use dtype int4g"
    },
    {
        "key" : "model\\.layers\\.([0-9]|[1][0-9]|2[0-5])\\.(.)*",
        "dtype" : "fp8",
        "comment": "layer 0~25 use dtype fp8"
    },
    {
        "key" : "(.)*mlp\\.gate\\.weight",
        "dtype" : "float16",
        "comment": "gate use float16"
    },
    {
        "key" : "(.)*experts(.)*[gate|up|down]_proj(.)*",
        "dtype" : "int4",
        "comment": "moe use int4"
    }
]
```
将以上json代码保存为`/root/dtype_config.json`（这里的路径只是举例，你可以保存到任意位置）
然后就可以使用`--dtype_config /root/dtype_config.json`来读取其中的配置（如果你的文件不是保存在`/root/dtype_config.json`，那么请使用你保存文件的路径）

**说明**：
- 配置文件中是一个json的数组，数组中每个元素代表一条量化规则。
- 每条规则中，目前有两个关键参数
    - `key`: 指定匹配权重的正则表达式
    - `value`: 当权重和`key`指定的正则表达式匹配时，使用`value`指定的类型
- 越靠后的规则优先级越高，一个权重如果匹配上了多个规则，那么会使用最后一条规则
- 规则中的其余参数不会有影响。例如上例中的`comment`参数仅为注释
- 上例中:
    - 0~25层的MLA部分使用`fp8`类型
    - 剩余层的MLA部分使用`int4g`类型
    - gate权重均使用`float16`类型
    - moe部分均使用`int4`类型
