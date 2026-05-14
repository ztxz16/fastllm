# 混合推理使用说明

这个文档以`DeepSeek-V4-Flash`模型为例，介绍如何使用混合推理来榨干硬件

## 基本用法

假设我们在一台有两张48G的显卡上部署`DeepSeek-V4-Flash`模型，一般用法是这样的


```
ftllm server fastllm/DeepSeek-V4-Flash
```

这时候会默认将模型的moe部分运行在numa上，非moe部分运行在cuda上，等价于如下命令：

```
ftllm server fastllm/DeepSeek-V4-Flash --device cuda --moe_device numa
```

## 将部分moe层运行在单张显卡上

用上述命令运行时，显存会有大量剩余，我们可以通过设置`moe_device`，将一部分moe层
指定在cuda上运行

```
ftllm server fastllm/DeepSeek-V4-Flash --device cuda --moe_device "{'cuda':1,'numa':19}"
```

上述命令中将`moe_device`设置为`"{'cuda':1,'numa':19}"`，代表`1/20`的moe层运行在cuda上，`19/20`的moe层运行在numa上

这样能轻微提升decode速度，但是可能会降低上下文长度

## moe部分使用显卡、numa、硬盘一起推理

当机器的内存不足以放下全部moe权重，或者希望把显卡、numa和SSD都利用起来时，可以把`moe_device`同时设置为`cuda`、`numa`和`disk`

```
ftllm server fastllm/DeepSeek-V4-Flash --device cuda --moe_device "{'cuda':1,'numa':8,'disk':1}"
```

上述命令中将`moe_device`设置为`"{'cuda':1,'numa':8,'disk':1}"`，代表`1/10`的moe层运行在cuda上，`8/10`的moe层运行在numa上，`1/10`的moe层从硬盘读取并推理

硬盘推理速度主要受SSD读取速度影响，一般会比numa慢很多，建议只把少量moe层放到`disk`上作为容量补充。如果上下文较长，可以配合`--chunked_prefill_size`降低prefill阶段的显存峰值

## 将部分moe层运行在多张显卡上

使用下面的命令可以使用多张显卡来加速部分moe层

```
ftllm server fastllm/DeepSeek-V4-Flash --device cuda --moe_device "{'multicuda:0,1':15,'numa':85}"
```

上述命令中将`moe_device`设置为`"{'multicuda:0,1':15,'numa':85}"`，代表`15/100`的moe层使用0,1两张gpu张量并行推理，`85/100`的moe层运行在numa上

这样能进一步提升decode速度


