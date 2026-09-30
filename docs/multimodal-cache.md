# 多模态缓存

## 当前实现

Qwen3.5 native 图片路径支持图片 SHA-256 → 最终视觉 embedding 缓存。同一模型实例再次收到相同图片时，可跳过 native resize、patch 构造、视觉编码器和 merger 计算。特征保留原有 FP32 精度，以 CPU LRU 管理。

缓存默认上限为 **512 MiB**，最多保留 **128 个条目**。首次处理带有效缓存键的图片请求时才创建缓存池；模型加载、纯文本、纯视频和关闭缓存的请求均不触发创建。内存随条目写入逐步分配，不预分配整个容量。

## 配置

在 `ftllm server` 模型启动命令中添加参数：

```bash
--image-embedding-cache 1g  # 上限为 1 GiB
--image-embedding-cache 0   # 关闭图片 embedding 缓存
```

也支持别名 `--image_embedding_cache`。显式命令行参数优先于环境变量；未指定参数时保留 `FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES` 的值，环境变量未设置时使用 512 MiB。

直接使用 Python 模型入口时，可在加载模型前设置环境变量，值必须是非负十进制字节数，例如：

```bash
export FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES=536870912
```

环境变量为空、格式错误或超出 native `size_t` 范围时回退到默认容量。容量在首次创建缓存池时读取，修改启动配置后需重新加载模型。

## 缓存键与生命周期

- Python 对实际发送的连续 FP32 图片数据、shape、grid、处理配置及身份版本计算 SHA-256。摘要以现有 payload 的 `[图片数量, 8]` int32 tensor 传入 native，保留完整 256 位，不改模型 token 或 C API。
- native 在键中加入模型计算类型，并以模型实例隔离缓存。每张图片单独缓存；`[A, B, A]` 中两次 A 可复用同一个结果，合并后的特征仍按请求顺序排列。
- 命中时复制到本次请求自己的输出，淘汰不会使在途请求持有失效的指针。缓存只保留 CPU 特征，不额外常驻 GPU 显存。
- 容量不足时按最近最少使用顺序淘汰。单张图片特征超过容量时正常计算，但不写入缓存，也不为它清空已有条目。
- 重新初始化模型参数、加载权重时清空缓存，卸载模型时释放。缓存容量限制保留的特征数组字节数；条目数上限另行限制索引元数据，本次请求的工作内存不计入缓存容量。

## 作用范围

当前仅缓存 Qwen3.5 native 图片路径的最终视觉特征；其他模型、视频和未携带缓存键的旧 payload 沿用原有计算路径。

图片读取、base64 传输、解码、Python payload 准备、哈希和语言模型预填充仍可能执行。图片缓存命中不等于 KV 前缀命中，不会增加 `cached_input_tokens`。

带有多模态输入的请求不写入或恢复仅按 token ID 索引的通用历史缓存和分页前缀缓存。图片占位 token 不能标识像素内容或多模态位置状态；Qwen3.5/3.8 恢复纯文本前缀后还可能误入解码分支，跳过视觉编码与分块预填充。该限制保留单次请求内的 KV 复用、图片 embedding 缓存和纯文本请求的前缀缓存；长多模态历史仍需重新完成语言模型预填充。

Qwen3.8-Flash-Next（Qwen4Exp）另有模型内部的前缀快照记录路径。其请求状态保留多模态标记，首次视觉 prefill 及后续 decode 均不记录这种快照；直接调用 C++ `ForwardMultimodal` 也会设置该标记。纯文本独立快照仍可记录和恢复。

开启模型 verbose 日志后，可用 `Image embedding cache hit` / `Image embedding cache miss` 确认命中情况。工作室中的等待提示不代表缓存状态。

## 验证

无需模型权重的回归测试：

```bash
python3 test/api/test_image_embedding_cache.py
g++ -std=c++17 -pthread -O1 -g -fsanitize=address,undefined \
    -fno-omit-frame-pointer -Iinclude \
    test/basic/test_image_embedding_cache.cpp -o /tmp/test_image_embedding_cache
/tmp/test_image_embedding_cache
```

Python 测试覆盖图片内容、尺寸、grid 和处理参数失效、输入布局等价、重复图片顺序、关闭缓存及文本/视频路径。C++ 测试覆盖容量解析、LRU、字节与条目上限、超大条目、请求输出独立性、实例隔离和并发访问。

启用 CMake 测试后，`multimodal_history_cache` 检查多模态状态不会污染 token 历史缓存、图片请求不会恢复文本 KV，并验证纯文本缓存仍可记录与恢复。模型集成回归还应在开启前缀缓存时先预热相同文本前缀，再分别提交不同图片，确认图片内容正确且多模态请求没有 token 前缀命中。

`qwen4_multimodal_cache` 使用小型 CPU KV/QSA 状态执行真实快照存储和恢复，检查图片、视频请求在后续 decode 阶段仍不记录快照，同时覆盖直接 C++ 多模态入口与请求状态复用。它不运行视觉编码器或完整模型推理。

模型集成验证应关闭 KV 前缀缓存以隔离视觉缓存效果，比较冷请求、命中请求和关闭图片缓存时的 logits，并检查命中日志。还应验证先运行纯文本，再提交图片时才创建缓存，以及多图片 `[A, B, A]` 的顺序。已有验证覆盖 Qwen3.5-2B 单卡、Qwen3.5 架构的 27B FP8 模型 TP2 和 CPU 延迟初始化路径；单卡/TP2 logits 使用 `rtol=atol=1e-5`，CPU 冷/热请求 logits 完全一致。

预处理缓存、多模态 KV 身份与完整状态恢复、工作室媒体引用等后续方案见[设计草案](multimodal-cache-design.md)，尚未纳入当前实现。

双 2080 Ti 的 [32 张真实收据完整 Pi 任务](benchmarks/qwen38_27b_multimodal_agent_2080ti.md)记录了逐图覆盖、三次上下文压缩、显存和自然结束状态。流程完成，独立验收 350/351；仍有一项日期识别差异，不能将缓存隔离验证等同于全部内容正确。
