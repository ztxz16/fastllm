### 自定义模型

对于Fastllm框架中没有支持的模型，可以通过自定义模型结构来支持

Pyhton 自定义模型只需要一个python文件来描述模型结构，可参考 [QWEN](../example/python/qwen2.py) 中的实现

### Python自定义模型的使用

使用ftllm.chat, ftllm.webui, ftllm.server时，可以加入参数--custom来指定自定义模型文件

假设我们的模型位于 `~/Qwen2-7B-Instruct/` 目录，自定义模型位于 `~/qwen2.py`

那么可以使用命令

``` sh
python3 -m ftllm.chat -t 16 -p ~/Qwen2-7B-Instruct/ --custom ~/qwen2.py 
```

来通过自定义模型文件加在Qwen2模型，server和webui用法类似

### Python自定义模型的写法

自定义模型时，需要实现一个模型的描述类，继承自ftllm.llm.ComputeGraph

对应 [QWEN](../example/python/qwen2.py) 中的代码

``` python
from ftllm.llm import ComputeGraph
class Qwen2Model(ComputeGraph):
```

文件最后需要定义 `__model__` 变量来指定自定义模型结构对应的class, 对应代码

``` python
__model__ = Qwen2Model
```

模型描述类中需要实现build方法，来获取模型参数、描述计算流程

这里以示例代码为例介绍

``` python
class Qwen2Model(ComputeGraph):
    def build(self):
        # 1. 获取weight, data, config
        weight, data, config = self.weight, self.data, self.config

        # 2. 设置一些config
        config["max_positions"] = 128000

        # 3. 描述计算流程
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        self.Embedding(data["inputIds"], weight["model.embed_tokens.weight"], data["hiddenStates"]);
        # 以下是计算流程，具体参见示例代码
```

#### `self.config`

模型配置，默认会从模型文件夹下的 `config.json` 文件中读取

build方法中可以修改config中的参数，例如改动 `max_positions` 可以修改上下文长度

有一些模型的 `config.json` 中使用的变量名不一致，需要在build过程中手动为config赋值。

例如在TeleChat7B模型的配置中没有 `max_positions` 变量，而是用 `seq_length` 变量代表长度，那么在build方法中需要用如下代码赋值：

``` python 
self.config["max_positions"] = self.config["seq_length"]
```

config中，有以下变量必须要赋值（如果config.json中变量名一致，可以不处理）：

``` python
self.config["max_positions"] #代表最长上下文长度
```

#### `self.weight`

代表权重数据

`self.weight[weightName]` 代表模型文件中名为weightName的参数（对应HF模型文件夹中.safetensors文件中的参数名）

#### ```self.data```

代表计算流程的中间变量和输入变量

`self.data[dataName]` 代表名为dataName的中间变量，`dataName` 可以使用除以下输入变量名之外的任意字符串

输入变量：

``` python
data["inputIds"] # 输入token
data["positionIds"] # 位置信息
data["attentionMask"] # mask信息
data["sin"] # 用于旋转编码的sin
data["cos"] # 用于旋转编码的cos
data["atype"] # 推理中的数据类型
data["pastKey."][i] # 第i个block的key cache
data["pastValue."][i] # 第i个block的value cache
```

#### 计算流程及算子

使用基类ComputeGraph添加算子的函数来描述计算流程

目前支持的算子见文档 [自定义模型算子](./custom_op.md)

### cpp版本的自定义模型

（cpp版本的自定义模型接口还在修改中...）
