## 自定义模型算子文档

### `AddTo`
```python
def AddTo(self, input0, input1, alpha = 1.0):
    """
    将两个输入节点相加，并乘以一个可选的缩放因子 alpha。

    参数:
    input0 (GraphNode): 第一个输入节点。
    input1 (GraphNode): 第二个输入节点。
    alpha (float, optional): 缩放因子，默认为 1.0。

    返回:
    无返回值，结果存储在内部图结构中。
    """
    self.graph.append({"type": "AddTo", 
                       "nodes": {"input0": input0, "input1": input1, "alpha": FloatGraphNode(alpha)}})
```

### `DataTypeAs`
```python
def DataTypeAs(self, input, input1):
    """
    将输入节点的数据类型转换为另一个输入节点的数据类型。

    参数:
    input (GraphNode): 需要转换数据类型的输入节点。
    input1 (GraphNode): 目标数据类型的输入节点。

    返回:
    无返回值，结果存储在内部图结构中。
    """
    self.graph.append({"type": "DataTypeAs", 
                       "nodes": {"input": input, "input1": input1}})
```

### `Embedding`
```python
def Embedding(self, input, weight, output):
    """
    执行嵌入操作，将输入索引映射到嵌入权重。

    参数:
    input (GraphNode): 输入索引节点。
    weight (GraphNode): 嵌入权重节点。
    output (GraphNode): 输出节点。

    返回:
    无返回值，结果存储在内部图结构中。
    """
    self.graph.append({"type": "Embedding", 
                       "nodes": {"input": input, "weight": weight, "output": output}})
```

### `ExpandHead`
```python
def ExpandHead(self, input, headDim):
    """
    把input最后一维展开成[-1, headDim]。

    参数:
    input (GraphNode): 输入节点。
    headDim (int): 头部维度大小。

    返回:
    无返回值，结果存储在内部图结构中。
    """
    self.graph.append({"type": "ExpandHeads", 
                       "nodes": {"input": input, "headDim": IntGraphNode(headDim)}})
```

### `FusedAttention`
```python
def FusedAttention(self, q, k, v, curk, curv, original, mask, output, seqLens, 
                   scale, maskType=0, unitLen=128):
    """
    执行Attention操作。

    参数:
    q (GraphNode): 查询节点。
    k (GraphNode): key cache 
    v (GraphNode): value cache 
    curk (GraphNode): 当前key
    curv (GraphNode): 当前value
    original (GraphNode): 原始节点，用于恢复计算后的shape
    mask (GraphNode): 掩码
    output (GraphNode): 输出
    seqLens (GraphNode): 序列长度
    scale (float): 缩放因子
    maskType (int, optional): 掩码类型，默认为 0。
    unitLen (int, optional): 单元长度，默认为 128。

    返回:
    无返回值，结果存储在内部图结构中。
    """
    self.graph.append({"type": "FusedAttention", 
                       "nodes": {"q": q, "k": k, "v": v, "curk": curk, "curv": curv, 
                                "original": original, "mask": mask, "output": output, "seqLens": seqLens, 
                                 "scale": FloatGraphNode(scale), 
                                 "maskType": IntGraphNode(maskType), "unitLen": IntGraphNode(unitLen)}})
```

### `Linear`
```python
def Linear(self, input, weight, bias, output):
    """
    执行线性变换操作。

    参数:
    input (GraphNode): 输入节点。
    weight (GraphNode): 权重节点。
    bias (GraphNode): 偏置节点。
    output (GraphNode): 输出节点。

    返回:
    无返回值，结果存储在内部图结构中。
    """
    self.graph.append({"type": "Linear", 
                       "nodes": {"input": input, "weight": weight, "bias": bias, "output": output}})
```

### `LlamaRotatePosition2D`
```python
def LlamaRotatePosition2D(self, input, positionIds, sin, cos, rotaryDim):
    """
    执行 Llama 模型的二维位置旋转操作。

    参数:
    input (GraphNode): 输入节点。
    positionIds (GraphNode): 位置 ID 节点。
    sin (GraphNode): 正弦节点。
    cos (GraphNode): 余弦节点。
    rotaryDim (int): 旋转维度大小。

    返回:
    无返回值，结果存储在内部图结构中。
    """
    self.graph.append({"type": "LlamaRotatePosition2D", 
                       "nodes": {"input": input, "positionIds": positionIds, "sin": sin, "cos": cos, "rotaryDim": IntGraphNode(rotaryDim)}})
```

### `MulTo`
```python
def MulTo(self, input0, input1):
    """
    将两个输入节点相乘。

    参数:
    input0 (GraphNode): 第一个输入节点。
    input1 (GraphNode): 第二个输入节点。

    返回:
    无返回值，结果存储在内部图结构中。
    """
    self.graph.append({"type": "MulTo", 
                       "nodes": {"input0": input0, "input1": input1}})
```

### `RMSNorm`
```python
def RMSNorm(self, input, weight, eps, output):
    """
    执行 RMS 归一化操作。

    参数:
    input (GraphNode): 输入节点。
    weight (GraphNode): 权重节点。
    eps (float): 小常数，用于防止除零错误。
    output (GraphNode): 输出节点。

    返回:
    无返回值，结果存储在内部图结构中。
    """
    self.graph.append({"type": "RMSNorm", 
                       "nodes": {"input": input, "weight": weight, "eps": FloatGraphNode(eps), "output": output}})
```

### `Silu`
```python
def Silu(self, input, output):
    """
    执行 SiLU（Sigmoid Linear Unit）激活函数操作。

    参数:
    input (GraphNode): 输入节点。
    output (GraphNode): 输出节点。

    返回:
    无返回值，结果存储在内部图结构中。
    """
    self.graph.append({"type": "Silu", 
                       "nodes": {"input": input, "output": output}})
```

### `SplitLastTokenStates`
```python
def SplitLastTokenStates(self, input, seqLens, output):
    """
    分割batch输入中每个batch的最后一个 token 状态。

    参数:
    input (GraphNode): 输入节点。
    seqLens (GraphNode): 序列长度节点。
    output (GraphNode): 输出节点。

    返回:
    无返回值，结果存储在内部图结构中。
    """
    self.graph.append({"type": "SplitLastTokenStates", 
                       "nodes": {"input": input, "output": output, "seqLens": seqLens}})
```