# 混合推理使用说明

这个文档以`DeepSeek-V3-0324-INT4`模型为例，介绍如何使用混合推理来榨干硬件

## 基本用法

假设我们在一台有两张48G的显卡上部署`DeepSeek-V3-0324-INT4`模型，一般用法是这样的


```
ftllm server fastllm/DeepSeek-V3-0324-INT4
```

这时候会默认将模型的moe部分运行在cpu上，非moe部分运行在cuda上，等价于如下命令：

```
ftllm server fastllm/DeepSeek-V3-0324-INT4 --device cuda --moe_device cpu
```

（注意：之后的优化目前仅对`cuda`和`cpu`的混合推理有效，`numa`无法使用这些功能）

## 将部分moe层运行在单张显卡上

用上述命令运行时，显存会有大量剩余，我们可以通过设置`moe_device`，将一部分moe层
指定在cuda上运行

```
ftllm server fastllm/DeepSeek-V3-0324-INT4 --device cuda --moe_device "{'cuda':1,'cpu':19}"
```

上述命令中将`moe_device`设置为`"{'cuda':1,'cpu':19}"`，代表`1/20`的moe层运行在cuda上，`19/20`的moe层运行在cpu上

这样能轻微提升decode速度，但是可能会降低上下文长度

## 将部分moe层运行在多张显卡上

使用下面的命令可以使用多张显卡来加速部分moe层

```
ftllm server fastllm/DeepSeek-V3-0324-INT4 --device cuda --moe_device "{'multicuda:0,1':15,'cpu':85}"
```

上述命令中将`moe_device`设置为`"{'multicuda:0,1':15,'cpu':85}"`，代表`15/100`的moe层使用0,1两张gpu张量并行推理，`85/100`的moe层运行在cpu上

这样能进一步提升decode速度

（建议看到这里就结束，但如果想了解更多的花活也可以继续往下看）

## 将部分moe层使用混合张量并行推理

使用下面的命令可以使用混合张量并来加速部分moe层

```
ftllm server fastllm/DeepSeek-V3-0324-INT4 --device cuda --moe_device "{'multicuda:0:3,1:3,cpu:2':15,'cpu':85}"
```

上述命令中将`moe_device`设置为`"{'multicuda:0:3,1:3,cpu:2':15,'cpu':85}"`，代表：
- `15/100`的moe层使用混合张量并行，这时候两张显卡和cpu会同时工作，`3/8`的计算量在显卡0上，`3/8`的计算量在显卡1上，`2/8`的计算量在cpu上
- `85/100`的moe层运行在cpu上

这样理论上能更进一步提升decode速度，但目前实现效率不高，速度还不如上一步，后续会继续优化

