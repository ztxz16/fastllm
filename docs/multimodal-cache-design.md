# 多模态缓存设计草案

状态：后续设计草案，以下分层缓存、KV 身份/检查点、媒体引用等尚未实现。调研日期：2026-09-05。审查基线：`fe0fa800`。

当前已实现 Qwen3.5 图片 embedding 缓存，配置、作用范围与验证方式见[多模态缓存](multimodal-cache.md)。本草案保留对完整方案的分析，不作为当前功能说明。

建议采用按媒体项组织的预处理缓存、视觉特征缓存，以及带媒体身份和模型附加状态的 KV 前缀缓存。复用现有分页分配器和调度器；首先接入 Qwen3.5 图片路径，再逐个适配其他模型和视频。目标是精确复用计算结果，不改变图片分辨率、输入内容或推理精度。

最重要的执行顺序是：**准备身份和布局 → 查找可恢复的 KV 前缀 → 只为剩余区间取得视觉特征 → 分块预填充**。不能先把全部历史图片编码一遍，再检查 KV。

## 基线现状和问题

- `webui_server.py::build_api_messages` 每轮重新读取历史附件并发送 data URL。保留历史图片的语义是正确的，重复传输可以另外优化。
- `fastllm_completion.py::_load_image_from_url` 每次解析图片；`llm.py` 的 token 计数和生成入口分别调用多模态准备函数，缺少共同持有的准备结果。
- `Qwen3_5Model::EncodeVisualItems` 每次执行视觉编码，没有模型实例级的结果缓存。`PrepareVision` 只复用模型准备状态。
- `PagedCacheManager::Record/Query` 当前按 token 页建立 Trie。`Qwen35LinearPrefixSnapshot` 也按 token 序列匹配，没有媒体身份。相同尺寸、不同内容的图片可以产生相同的占位 token，因而不能直接用这些键证明多模态缓存相同。
- `Qwen35ForwardMultimodal` 用已有 past KV 判断后续 forward，并从请求中的 `mrope_position_delta` 调整位置；新请求恢复前缀时，不能假定它已经生成了这些位置数据，也不能假定剩余输入中没有新图片。
- Qwen3.5 多模态预填充有独立入口。线性注意力快照要求页边界，不能仅把最终状态的长度向下取整就当作较早位置的状态。

以上来自本地代码审查；不同图片串缓存属于待回归验证的正确性缺口，尚未据此宣称用户会话已出现错误回答。

以 1920×2560 图片、4800 个视觉 token 和 5120 维输出为例，视觉特征本体在 FP32 下约 93.75 MiB，在 FP16/BF16 下约 46.875 MiB。当前 C++ 输出容器使用 FP32；缓存必须保留实际结果精度，不能为达到后一数字额外降精度。这是容量计算，不是端到端性能测试。

## 从 vLLM 和 SGLang 采用什么

调研固定到 vLLM `7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719`、SGLang `bd16c22a04b0eb9bc2e775795bda6b11727a5d38`，避免把不同版本的设计混在一起。

| 机制 | 已核对的实现 | FastLLM 的选择 |
| --- | --- | --- |
| 预处理缓存 | vLLM 缓存 processor 的媒体输出；SGLang 将内容摘要、processor 指纹和处理参数组成 artifact key | 按媒体项缓存与文本无关的结果，单独重建本轮 prompt 和位置 |
| 视觉特征缓存 | vLLM `EncoderCacheManager` 按媒体身份管理请求引用和可回收条目；SGLang 有按字节限制的 embedding LRU | 采用每项缓存、引用保护、按字节回收；跨轮默认放 CPU |
| KV 身份 | vLLM 把 image hash 纳入相关 block 的 extra hash；SGLang 的媒体项带 hash、offset，并构建媒体相关的 padded input IDs | 在现有分页 Trie 增加独立的多模态页键，不改实际模型 token |
| 计算调度 | vLLM 只调度与本次待计算 token 区间相交、尚未缓存的 encoder 输入 | KV 优先查询；特征按需取得；不让历史图片无条件触发 ViT |
| 混合注意力 | SGLang 的 Mamba Radix 缓存同时管理普通 KV 和 recurrent state | Qwen3.5 把 KV、线性状态、位置边界及必要的 draft 状态作为一个恢复计划 |

参考：[vLLM 预处理](https://github.com/vllm-project/vllm/blob/7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719/docs/design/mm_processing.md)、[vLLM encoder 生命周期](https://github.com/vllm-project/vllm/blob/7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719/vllm/v1/core/encoder_cache_manager.py)、[vLLM 前缀键](https://github.com/vllm-project/vllm/blob/7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719/docs/design/prefix_caching.md)、[vLLM 调度器](https://github.com/vllm-project/vllm/blob/7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719/vllm/v1/core/sched/scheduler.py)。

参考：[SGLang 媒体身份](https://github.com/sgl-project/sglang/blob/bd16c22a04b0eb9bc2e775795bda6b11727a5d38/python/sglang/srt/multimodal/cache/identity.py)、[SGLang 预处理缓存](https://github.com/sgl-project/sglang/blob/bd16c22a04b0eb9bc2e775795bda6b11727a5d38/python/sglang/srt/multimodal/cache/preprocess_cache.py)、[SGLang embedding LRU](https://github.com/sgl-project/sglang/blob/bd16c22a04b0eb9bc2e775795bda6b11727a5d38/python/sglang/srt/mem_cache/multimodal_cache.py)、[SGLang 媒体项与位置](https://github.com/sgl-project/sglang/blob/bd16c22a04b0eb9bc2e775795bda6b11727a5d38/python/sglang/srt/managers/schedule_batch.py)、[SGLang 混合状态缓存](https://github.com/sgl-project/sglang/blob/bd16c22a04b0eb9bc2e775795bda6b11727a5d38/python/sglang/srt/mem_cache/mamba_radix_cache.py)。

这里采用其分层、身份、生命周期思想，不引入这些框架作为运行依赖。SGLang 的哈希占位值适合其输入处理体系；FastLLM 存在 int token 到 float tensor 的路径，直接塞入大整数会有精度和语义问题，故选择旁路元数据。

## 数据模型与缓存键

以下名称为拟议内部接口。

```text
MediaIdentity
  content_digest, modality, identity_schema

PreparedMedia
  processor_key, immutable_payload, grid/crop/frame metadata
  feature_count, size_bytes

MediaOccurrence
  encoder_key, ordered feature_spans, position metadata

PreparedMultimodalRequest
  actual_token_ids, occurrences, full_prompt_position_plan
  namespace, prefix_page_keys, source/feature leases

EncodedMedia
  encoder_key, immutable feature tensors, dtype/layout, byte_size

PrefixCheckpoint
  namespace, prefix_digest, committed_length
  KV page references, recurrent states, position boundary state
  optional compatible draft state, readiness and ownership
```

一个媒体项可以在同一请求出现多次，也可以对应多个不连续的特征区间。例如视频帧之间有时间戳文本。`MediaOccurrence` 必须显式记录 token 区间到特征行的映射，不能假定所有多模态 token 连成一段。

身份逐层构建：

```text
content_digest = SHA256(实际接收并用于解码的媒体字节)
processor_key  = H(版本, modality, content_digest,
                   processor 指纹, 完整有效处理参数)
encoder_key    = H(processor_key, 模型加载代次,
                   视觉权重/适配器代次, 编码配置, 输出格式)
```

- 使用带类型、长度和固定字节序的规范序列化，保存完整摘要。禁止 Python `hash()`、指针地址、文件名或 URL 作为内容身份。
- PIL/NumPy/C++ tensor 入口对不可变快照的像素、shape、dtype 和影响解释的元数据构造身份。不同输入表示暂时未能去重只导致安全的 miss。
- 处理指纹包括实际 resize/crop/patch/merge/归一化策略与实现版本；视频还包括取帧区间、顺序、采样参数和实际选帧。未建模的新参数保守纳入键或关闭该项复用。
- 模型加载代次由服务端生成。换权重、适配器或改变影响计算的配置即换代，不能只根据模型目录名判断相同。
- 同图换问题应命中 encoder；温度、输出长度等生成设置不进入图片 encoder key。文本相关 processor/encoder 必须由适配器声明依赖，并把相关输入加入键，不能强行跨问题复用。
- URL/本地文件先读取实际内容再定身份；默认不假定同一 URL 永久不变。外部传入的 digest/UUID 只作提示，服务器验证后才能成为可信身份。
- 公共部署若有租户隔离，命名空间由服务端鉴权结果决定，并贯穿媒体引用、processor、encoder、KV 三层。单用户本地模型可在本实例内共享。

## 第一层：预处理缓存

位于模型服务的 Python 准备层，供 OpenAI/Anthropic API 和 Python 模型入口共同使用。缓存不可变的媒体结果和小型布局元数据；不保存整段对话的 prompt/position IDs 到图片条目中。

`get_input_token_len` 和生成共用一个 `PreparedMultimodalRequest`。API 在请求生命周期内显式持有它；旧的独立计数/生成接口通过同一媒体缓存取得结果。这样无需依赖线程恰好相同，也不会因 tools、thinking 或模板参数变化误用整段 prompt。

冷请求只进行一次内容解码和必要处理；热请求先识别媒体，再查询布局及 payload。对于 Qwen3.5，token 数和 grid 可由图片元数据与处理配置得到，不需要运行 ViT。重型 tensor 被淘汰后仍可利用轻量元数据构建前缀键；若连元数据也丢失，则从原始附件重建。

同一项的并发 miss 合并为一次计算，其他请求等待同一个结果。锁只保护索引和状态，不跨越图片处理或 GPU forward。共享 tensor 只读，调用方需要原地修改时另建工作副本。

CPU LRU 按真实存储字节计费，并设置条目数上限。不要同时常驻 PIL、RGB、float payload 和拼接后的 payload 多份副本；优先保持紧凑媒体表示，按需生成实际上传格式。

## 第二层：视觉特征缓存

归属 C++ 模型实例，缓存视觉编码器及 projector/merger 的最终输出，位于与语言模型文本 embedding 合并之前。这样独立 C++/Python/API 调用共享能力，也不把大型 tensor 放在工作室 UI 缓存中。

以单张图片为基本单位，按原始顺序组装本轮所需特征。请求 `[A] → [A,B] → [A,B,A]` 只应首次编码 A、B 各一次，重复 A 共享同一只读存储，但保留不同的 occurrence 和位置。

默认将跨请求条目放在 **CPU 普通内存**；本次 forward 所需特征才搬到 GPU。预留有上限的 pinned staging buffer，NUMA0 部署在节点 0 分配。只有实测传输成为瓶颈时再开启单独 GPU 热缓存预算；所有副本都要计费，不按 TP 数量无条件复制完整 CPU 条目。

这适合当前显存需要同时容纳权重、KV 和推测解码状态的环境。KV 全覆盖图片时连 CPU→GPU 特征传输都跳过；KV 不命中而 encoder 命中时，仍需做语言模型预填充。

缓存管理与执行工作区分开。只持有完成结果，不保留 ViT 层中间激活、整段请求或指向临时工作区的裸指针。共享所有权必须能接入现有 `ResponseContext` 析构，避免其删除共享的 `Data*`。

若后续模型还有 DeepStack 等媒体独立的旁路输出，适配器需将全部必要 tensor 组成一个缓存结果；不能只缓存最终 embedding 后漏掉额外输入。

## 第三层：多模态 KV 前缀缓存

### 页键必须表达实际计算输入

为多模态增加版本化的 `PrefixKeyView`；实际 `inputIds/currentTokens/allTokens` 保持模型原 token。页键覆盖：

```text
page_key[i] = H(namespace, page_key[i-1], actual_tokens[i],
                本页涉及的媒体身份及特征区间,
                本页 position IDs 与必要的 attention/type 元数据)
```

媒体描述包含 encoder key、特征行区间及其 token 内偏移。每个与图片相关的页都包含相应身份；后续文本页通过父键继承依赖。新追加图片不能进入它之前各页的键，否则 `[A] → [A,B]` 会无谓丢失 A 的前缀命中。

同图改变位置、图片换序、改变系统提示词，可能仍命中 encoder，但 KV 只能复用实际依赖一致的连续前缀。由适配器声明媒体的最早依赖位置；有双向图像注意力等机制的模型不能把任意图像内部边界当作可独立重算的边界。

`PagedCacheManager` 保留现有页分配、引用和淘汰机制；给多模态页增加完整摘要键及独立版本命名空间。底层哈希表可用摘要的一部分选桶，但相等判断使用完整摘要，不能只比较一个截断整数。页物理所有权和回收反向索引同时覆盖新索引，不能让同一个可变物理页留下多个陈旧映射。

首版保留现有纯文本索引，不与旧 token-only 多模态条目混用；可接受暂不共享“纯文本请求”与“多模态请求”的共同文本开头。以后在明确位置/掩码等价后再统一索引。通用 `PastKVCacheManager`、专用 scheduler、线性状态快照都必须使用同一套多模态身份；未适配的入口同时禁止多模态查找和写入旧缓存。

### 先恢复前缀，再按区间取得特征

```text
prepared = prepare_identity_layout_and_prompt(request)
plan = prefix_cache.lookup_and_pin(prepared.prefix_keys)
L = validate_all_required_states(plan)
restore_atomically(plan, L)

for chunk in remaining_prompt[L:]:
    features = get_or_encode(items_overlapping(chunk))
    forward_chunk(actual_tokens, position_slice, feature_slices)
    publish_checkpoint_if_selected_boundary()
```

必须处理三种情况：

| 恢复点 | 处理 |
| --- | --- |
| 已覆盖历史图片 | 跳过对应图片的编码和特征搬运 |
| 位于图片特征区间内部 | 取得整项 encoder 结果，只合并剩余特征行；必要时重算整项视觉编码，不能拿图片裁片代替 |
| 位于新图片之前 | 先继续文本，再为新图片取得特征；不能因 past KV 非空就跳过新图片 |

若整个 prompt 命中，但没有可直接使用的下一 token logits，选严格小于 prompt 长度的最近完整检查点，重算末尾得到 logits。不能用空 forward 开始生成；对页对齐的完整命中，这可能需要重算最后一整页。

位置计划应在视觉编码前可用，由完整本轮布局生成，再按真实区间切片。图片本地 grid 可以跨请求复用；整段 prompt 的 MRoPE position/delta 不能按图片 ID 独立缓存。恢复点状态与当前后缀布局一起确定 decode 位置；不能把上一轮最终 delta 无条件用于含新图片的新一轮。

### Qwen3.5 必须恢复完整状态

一个可用检查点包含同一已提交 token 边界上的：普通注意力 KV 页、每层线性注意力卷积缓存和 recurrent state、位置边界信息，以及当前执行方式需要的 MTP/DFlash 状态。

状态查询以 `namespace + prefix_digest + committed_length` 匹配。先找到 KV 页和全部必要附加状态共同支持的最长边界，再统一取得引用、恢复并提交；失败释放临时引用，退回较短完整检查点或冷计算。不能只恢复普通 KV 后继续运行线性层。

空闲检查点保存可校验的页身份和代次，不为保留一份线性状态而永久锁住整段普通 KV。恢复时才原子取得有效页的 lease；页已淘汰或被覆盖则该检查点不可恢复。附加状态自身也要按字节淘汰，过期的孤立快照及时清理。

推测生成未验证的 token 不得发布。首版沿用现有“必需 draft 状态必须存在”的约束；缺少 draft 状态时降到较短检查点或冷路径。后续只有在验证了 draft 重建或本请求回退普通解码之后，才可解除这个限制。

### 主动生成有用的检查点

视觉编码按媒体项执行一次，语言模型预填充使用特征切片按页对齐分块。优先在图片结束附近的有效页边界、prompt 最后一个完整页边界保存状态，长输入再按较大间隔补点。没有真实 token 时不为了对齐添加 padding。

尤其应在越过 `floor(prompt_len / page_len) * page_len` 前切块并取得真实状态。最终状态无法回滚成该边界的线性状态。末尾不足一页的部分下一轮重新计算即可；不要求首版新增非整页 KV 快照。

快照位置优先级先采用“prompt 尾部、媒体边界、间隔点”，按字节限制总量和在途复制。不要每页保存一份大 recurrent state。CPU 保存附加状态是默认建议；GPU 复制/回读成本、每个模型的状态大小需基准验证。不同 TP 分片恢复必须一致。

## 生命周期与资源上限

每项采用 `FILLING → READY → EVICTABLE` 生命周期，计算失败有独立失败完成路径。管理器发放 lease；同一请求重复引用媒体时，在最后一次使用及异步传输完成前保持保护。

- miss 的生产者取消时仍要唤醒等待者；一个等待者取消不能取消其他请求需要的共享结果。所有生产任务须受服务生命周期管理。
- `clear`、模型卸载、配置换代推进 generation。旧的在途任务不得重新把过期结果写回缓存。
- 只淘汰没有活跃 lease 的条目。索引删除但仍有外部引用的存储继续计入存活字节，直至真正释放；不能把从字典移除当作显存已经释放。
- 除常驻缓存预算，还限制在途媒体数、估算峰值字节及 encoder 工作并发。所有 CUDA 事件完成后才能释放或复用相关内存，不能通过无限堆积待复制条目绕过预算。
- 单项大于缓存容量时正常执行但不长期保留；工作区仍受现有请求资源限制。KV/状态被淘汰不应级联删除仍可复用的视觉特征，反向也一样。
- 默认不持久化 encoder/KV 到磁盘，不做跨机器共享。工作室原始附件仍承担重启后的可靠来源。

建议以 CPU 预处理 256 MiB、CPU encoder 1 GiB、CPU 多模态附加状态 512 MiB、额外 GPU 驻留 0 作为本机基准的起始配置，按需分配。这些是待实测的候选值，不是已确定的产品默认值。预算是每模型实例/服务的总额，不能被每个请求或 TP rank 再乘一遍。

拟议参数可以是 `--mm-processor-cache 256m`、`--mm-encoder-cache 1g`、`--mm-prefix-state-cache 512m`；GPU 热缓存后续单独配置。各层设为 0 不再跨请求保留该层结果，必要的本次请求工作内存仍存在；附加状态容量为 0 时，无法组成完整状态的混合模型多模态 KV 前缀也不复用。关闭缓存不能重新启用旧的 token-only 多模态匹配。纯文本行为保留现有配置。

## 工作室传输优化

计算缓存首先在 API/native 层完成，标准 OpenAI `image_url` 请求无需修改就能受益。

第二阶段可让 API server 注册不可变媒体对象，返回绑定服务实例和访问权限的 `media_ref`。工作室第一次上传后保存引用，之后带引用；API server 直接解析到可信身份和媒体元数据。先通过能力查询启用扩展，其他 API server 继续使用现有 data URL。

服务器重启、引用过期或对象被删除时，工作室从已保存附件重传并重建引用，再启动生成。恢复应发生在生成前，避免重试重复生成。客户端任意填写的相同 UUID 不得覆盖已登记的另一张图片。普通 URL 相同仍不代表图片相同。

不把这个传输扩展作为修复首轮后的视觉编码重复计算的前置条件，也不引入单机没有必要的共享内存 IPC 或分离 encoder 服务。

## 实施顺序与边界

| 阶段 | 交付内容 | 收益与发布条件 |
| --- | --- | --- |
| 0 | 复现与分段计时；堵住多模态读写 token-only 缓存的入口 | 先确立正确基线；该保守修复可能降低既有错误复用产生的命中数 |
| 1 | 统一媒体身份和布局；共享 Python 准备结果；C++ 每图片 encoder 缓存；所有权/预算/清理 | Qwen3.5 相同图片只首次运行 ViT；KV 尚未适配时继续保守禁用 |
| 2 | 多模态页键、全布局位置生成、按剩余区间 forward、主动页边界检查点、完整状态恢复 | 后续轮次跳过历史图片及大部分前缀预填充，覆盖 TP 和推测解码组合 |
| 3 | 工作室媒体引用；其他架构适配；视频媒体组；按数据决定是否增加 GPU 热缓存 | 减少传输，并把已验证缓存能力逐个扩展 |

第一阶段可以先在 native 按实际收到的 tensor 内容及 grid 计算 encoder key，兼容旧 Python payload。减少重复 payload 传送再通过版本化入口/注册句柄完成。新版接口必须协商能力，不能让旧 native 默默忽略新字段后执行不安全的恢复。

预期改动主要落点：

- Python：新增共享多模态准备/缓存模块，接入 `llm.py`、`fastllm_completion.py` 和各模型准备器。
- native 接口：`tools/src/pytools.cpp` 与相应 ctypes 声明；保留旧入口，增加版本化描述及所有权交接。
- 通用 C++：`ResponseContext` 的媒体描述和 lease，`PagedCacheManager` 的多模态页键，通用 scheduler 的恢复契约。
- Qwen3.5：拆分 `EncodeVisualItems`、位置规划、特征合并和分块 forward；扩展 `Qwen35LinearPrefixSnapshot` 身份及状态发布/恢复。
- 工作室：后续接入引用协议，不承担核心计算缓存。

Gemma、Step 等模型先声明能力：encoder 是否只依赖媒体、特征区间布局、允许的分块边界、附加恢复状态。未验证的模型可只启用 processor/encoder 层；KV 层保持安全 miss。视频先按完整采样媒体组缓存，不能把时序 patch 相互依赖的帧擅自拆成独立图片编码。

## 验证和可观测性

新增阶段指标：媒体读取/哈希、processor、payload 传输、encoder、特征搬运、prefix lookup/restore、LLM prefill、TTFT。分别报告 processor/encoder 命中项数、encoder 实际调用数、KV 命中 token、附加状态 miss 原因、常驻/在途/被引用字节和淘汰数。`cached_input_tokens` 仍只统计真正恢复的语言模型 KV，不能把 encoder 命中计作 KV 命中。

| 场景 | 应验证的行为 |
| --- | --- |
| 一图后连续追问 | 预算足够且未淘汰时，首次后 encoder 调用为 0；完整前缀命中覆盖图片时特征 H2D 为 0 |
| 同图换问题或换系统提示 | encoder 命中；KV 仅命中实际相同的连续前缀 |
| 相同尺寸两张不同图、相同 token 占位 | 从最早受图片影响处开始不能误命中；输出与关闭缓存基线一致 |
| `[A] → [A,B] → [A,B,A]`、换序、删除或替换历史图 | 逐项命中与 occurrence 映射准确；新增 B 不破坏更早未受影响的前缀键 |
| 相同 URL/文件内容改变；处理参数/权重改变 | 对应层正确失效，不复用旧结果 |
| 页边界落在图片前/中/后，尾部非整页 | token、特征切片、MRoPE 和线性状态均与冷路径一致 |
| 整个 prompt 完全命中 | 无现成 logits 时安全重算末尾，首个生成 token 正确 |
| 新图片出现在已有 KV 之后 | 新图片确实编码并影响输出，decode 位置不沿用错误的旧 delta |
| 并发相同图片、重复项、取消、clear、卸载 | 只产生所需的一次填充，无悬挂等待、重复释放、过期回填或存活字节漏计 |
| encoder 或 KV/状态单独淘汰，超大图片，小预算 | 正常重算，无依赖悬空；常驻与在途内存都在各自边界内 |
| 单卡、TP2、MTP/DFlash 开关、纯文本 | 完整恢复或安全回退；纯文本性能与输出无可归因退化 |

数值验证比较逐 token logits、position IDs 及相关状态，与同精度关闭缓存路径使用预先规定的数值容差比较；不能仅凭回答看起来相同判定正确。补充 greedy 多轮输出对照；必要时分析分块导致的正常浮点差异。

性能基准沿用 `numactl -C 0-31 -m 0`，固定模型、模板、图片、生成长度和思考设置，分开测试冷缓存、只命中 encoder、命中完整前缀、淘汰后重算，记录多次请求的 TTFT 分布和阶段计数。大图长会话与并发测试记录峰值及结束后的 CPU/GPU 内存。设计阶段不承诺固定加速倍数。

缓存不能消除新 token 对历史 KV 的注意力读取成本，也不能抵消历史模板改变或缓存被淘汰。验收目标是避免不必要的重复视觉编码和历史预填充，并保持推理结果和资源生命周期正确。
