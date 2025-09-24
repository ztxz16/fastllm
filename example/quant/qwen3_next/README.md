本文件夹下为qwen3_next结构模型的通用动态量化配置

基本模板如下：

``` json
[
    {
        "key" : "(.)*",
        "dtype" : "float16",
        "comment": "default"
    },
    {
        "key" : "(.)*expert(.)*[gate|up]_proj(.)*",
        "dtype" : "ggml_q2_k",
        "comment": "mlp gate_up"
    },
    {
        "key" : "(.)*expert(.)*[down]_proj(.)*",
        "dtype" : "ggml_q4_k",
        "comment": "mlp down"
    },
    {
        "key" : "(.)*shared_expert(.)*[gate|up|down]_proj(.)*",
        "dtype" : "float16",
        "comment": "shared expert"
    }
]
```

可根据需要修改每部分的量化类型

如果对模型结构了解，也可以新增量化规则