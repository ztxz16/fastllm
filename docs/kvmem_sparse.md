# KVMem 稀疏 KV 存储基础设施

`KvMemSparseCache` 保存完整的 pinned DDR KV 和有容量上限的 GPU 热缓存，接收已有 Indexer 的 GPU TopK，返回对应的 GPU 物理行。它不重新选 token，不改变 TopK 顺序、重复项或注意力算术。同时提供 MiMo/Naive N0.5 的 CUDA TP 适配器；完整模型吞吐与容量以实测记录为准。

这与 `kvmem.md` 中普通注意力的近似检索策略是两个接口：本接口只改变存储位置，不缩减已有 DSA 选出的集合。Indexer 本身和其历史索引仍由调用方管理。

## 数据流

1. 每层、每张 GPU 建立一个缓存对象，预先分配所有请求的 DDR KV、GPU 槽位、地址表和工作区。构造时检查主机预算。
2. 预填充可以在同步后初始化 `HostKey/HostValue`；GPU 生成的新 KV 调用 `Write`，写入 DDR，同时更新该 token 已存在的 GPU 副本。传入的是已经量化或转换好的存储字节。
3. `Resolve` 接收设备上的请求槽号、有效长度、TopK 和有效项数。先保护全部命中，再为缺失 token 选择空槽或按旋转顺序淘汰未选中的槽。重复 TopK 只搬运一次，但输出仍保留其重复项与顺序。
4. 缺失行由 CUDA kernel 从映射的 pinned DDR 读入 GPU。后续注意力使用 `Indices()`、`CachePointers()`、`Lengths()`，只读取 GPU KV。

设置 `pageLen > 0` 可启用共享分页 DDR。`totalTokens` 指定独立的全局物理容量，零值保持 `R × L`；容量和 `L` 必须页对齐，物理池至少容纳一个最长请求。全局基址是 `HostKey/Value(0)`，分页模式不接受其他 request 参数。调用方填写固定地址的 `KeyPages/ValuePages(request)` 设备表，负责有效页号、引用计数与写时复制。K/V 页号可以不同，共享页必须保持只读。`Gather` 按页表把单层历史精确搬入预分配的 prefill 工作区。`pageLen=0` 保持原有连续存储接口。

缓存以 token 行为搬运单位。K/V 可以有不同的行字节数，FP16、BF16、FP8 和内嵌 scale/padding 均按字节保留。缓存不负责量化，也不提供量化精度保证。

## CUDA Graph 与生命周期

对象必须在捕获前创建，存活到所有 graph executable 销毁之后。`Write`、`Resolve` 没有动态分配、主机回调或设备到主机的元数据读取；设备数组内容可在每次回放前更新。长度、TopK、请求顺序和命中率变化不需要重新捕获。`maxBatch` 包括补齐行，可以大于 `maxRequests`，例如 220 个请求使用 224 行 Graph，补齐行传 `request=-1`。

所有调用由指定 CUDA stream 排序。不能同时在不同 stream/Graph 上访问同一个对象：输出和工作区共享，每个 owner 的槽位状态也共享。请求回收时在同一 stream 调用 `Reset(owner, stream)`，完成后再填入新请求的 DDR 数据。Reset 保持地址不变，旧 Graph 可以继续使用。DDR 长度以内的 KV 必须已经由调用方初始化，不能在 GPU 访问期间直接改写主机数据。

一次 `Resolve` 中每个非负 owner 只能出现一次。多个 query 共享请求时，应由调用方先分批或构建并集；本接口没有隐含的 prefill 合并策略。`maxSelected <= residentTokens`，保证最坏情况下全部选中行都能同时驻留。当前没有容量不足时直接读 DDR 的注意力回退。

负索引和超出当前有效长度的 TopK 项被映射为 `-1`。错误的 owner、长度、有效项数或写入位置会设置 sticky `DeviceErrors()`，无效的 resolve 行输出长度为零。调用方应在批次完成后的错误检查中读取该值，不能将错误批次作为有效推理结果。统计读取同样由调用方显式完成，不在回放关键路径同步。

## 容量

设请求槽数为 `R`，历史上限为 `L`，GPU 常驻 token 数为 `C`，捕获 batch 为 `B`，TopK 上限为 `K`，每 token 的 K/V 合计字节数为 `W`：

- pinned DDR KV：`totalTokens × W`，未指定独立总池时为 `R × L × W`，启动时一次性分配。
- GPU KV：`R × C × W`。
- GPU 历史位置到槽位表：`R × L × 4` 字节。
- GPU 槽位表：`R × C × 4` 字节。
- 分页模式额外需要 K/V 页表：`R × round_up(L / pageLen, 4) × 8` 字节，保证各请求页表的 16 字节对齐。
- 批次工作区主要为 `B × (2C + 2K) × 4` 字节，另有地址、计数器和统计。

`HostBytes()` 和 `DeviceBytes()` 返回对象申请的实际字节数，不含 CUDA context、Graph 对象、Indexer 或注意力自身的工作区。逻辑位置表占用随上下文增长，但每步命中和淘汰工作只扫描 TopK 与常驻槽，不扫描完整历史。

主机换入的带宽与 GPU HBM 带宽不同。一次缺失的 K/V 都需要经过主机到 GPU 的链路；高命中率有助于减少传输，不能由节省显存推导出整模型吞吐提升。

## 验证与测速

启用 `USE_CUDA=ON`、`UNIT_TEST=ON` 后：

```sh
cmake --build BUILD --target cuda_kvmem_sparse_test -j 4
ctest --test-dir BUILD -R '^cuda_kvmem_sparse$' --output-on-failure
BUILD/cuda_kvmem_sparse_test --benchmark
```

该目标直接构建缓存模块和测试，不加载模型权重，不依赖完整模型库的构建。CUDA 12 及以上的测试还使用仓库现有 DSA 的 score、softmax、PV CUDA 实现，将 DDR 缓存的输出与全 GPU KV 的输出作逐位比较。旧工具链只构建存储与 Graph 测试，不引入 FP8 头文件依赖。

覆盖内容包括 BF16/FP8 DSA、非对齐字节布局、重复和无效索引、冷热切换、全命中零换入、请求换位、220→224 补齐、20 万 token 容量、Graph 内追加/覆盖 KV、请求回收和固定地址回放。测速使用 10 万上下文、TopK 2048、常驻 4096 token，记录不同 batch 和命中率下两段缓存 kernel 的耗时、实际搬运字节及有效主机传输带宽；不包含 Indexer、DSA 或完整模型的运行时间。

## MiMo/Naive N0.5 适配

在模型预热前配置 `--max_batch`，再通过已有 `--kvmem` JSON 开启，例如：

```sh
--tp 4 --ep 4 --max_batch 64 --kv_cache_dtype fp8_e4m3 \
--indexer_tp_storage layer_shard --cache_history true \
--chunked_prefill_size 8192 \
--kvmem '{"max_tokens":66048,"resident_tokens":16384,"prefill_tokens":8192,"host_mib":65536}'
```

本适配器保留 DSA 的完整 Indexer 历史与原始 TopK；`sink_tokens`、`recent_tokens`、`retrieval_tokens` 和 `retrieval_interval` 不参与 DSA 选择。主 KV 使用 pinned DDR，Indexer 单独放在 GPU，继续遵循 `--indexer_dtype` 与 `--indexer_tp_storage`。SWA 保留 GPU 环形窗口。支持 BF16/FP8 KV、多请求解码、攒 prefill 和已有 CUDA Graph 档位。支持前缀复用，暂不支持草稿模型。

`max_tokens` 是每个请求的输入加输出长度上限。已有 `--tokens` 参数可以独立限制总物理池，向下取整到完整页；不指定时，仍按最大 Graph 档位乘以页对齐的 `max_tokens` 预分配。DDR 主 KV 和 GPU Indexer 按总池分配，热缓存按最大 Graph 档位分配；逻辑位置表、页表和单层 staging 仍需覆盖单请求最大长度。`resident_tokens` 是每层、每请求的主 KV 热缓存，必须至少容纳全部 TopK。`host_mib` 是所有 TP rank、全部 DSA 层合计的主 KV 预算。GPU 容量检查另外计入热 KV、完整 Indexer、映射表、页表、SWA 检查点和 staging；运行时工作区由 `--gpu_mem_ratio` 的保留空间覆盖。

例如 `--max_batch 400 --tokens 52428800 --max_context_length 200000 --kvmem '{"max_tokens":200000,"resident_tokens":16384,"prefill_tokens":8192,"host_mib":921600}'` 允许每请求达到 20 万 token，但共享 52,428,800 token 的物理池，不为每路分别预留 20 万。

调度按物理页引用计费，共享前缀不重复占用预算。接纳请求时预留其输入与剩余输出所需的页，输出未设上限时预留到单请求上下文上限；这会使实际可接纳并发受输出预算影响。预算不足的请求排队，并释放暂时恢复的前缀引用，让活动解码继续执行。历史树中没有活动引用的页仍可由通用页分配器淘汰。此预留只记账，不提前写入 KV 或增加已预分配的总池。

主 KV 由通用 `PagedCacheManager` 管理 DDR 全局页，GPU Indexer 使用同一组 K 页号。完成的请求可以留下页树引用和有上限的 GPU SWA 检查点；多个请求命中相同前缀时共享历史页。共享尾页若需要写入，先复制主 KV 及对应 Indexer 页，再独立追加。前缀淘汰只回收没有活动引用的历史页，SWA 检查点失效时重新 prefill。

等待调度的命中请求只保留共享页引用与 SWA 检查点，不占热槽。每个请求在首次 forward 时领取私有热缓存/SWA 槽位，完成后归还；历史页的生命周期与热槽独立。新请求复用槽位前在所属 GPU worker stream 上重置映射。Graph 的固定地址表同时绑定主 KV 与独立 Indexer，解码时使用动态有效长度和 TopK，换请求或改变顺序不修改 Graph。未覆盖的形状仍走正确的 eager 路径。

Prefill 的多个 query 不能简单共用一个解码 TopK 热缓存。本适配器把当前请求、当前 DSA 层的有效历史临时搬到一份 GPU staging，再执行原有因果稀疏 attention；各层和各请求按 stream 顺序复用这份缓冲。此项临时空间按单层最大上下文预算，未复制整模型 KV。注意其主机传输可能使 prefill 变慢。

模型级回归入口：

```sh
ctest --test-dir BUILD -R '^naive_n05_kvmem_' --output-on-failure
```

回归覆盖同 TP 配置的开关 logits、BF16/FP8 KV、独立缩放 FP8 Indexer、短上下文、超出热缓存容量的 TopK、Graph/eager 切换、请求顺序变化、槽位复用及混合长度 batched prefill。`naive_n05_kvmem_prefix_*` 还覆盖重复输入、并发共享前缀分叉、排队请求超过热槽数、页/SWA 淘汰回退、恢复错误隔离和不重捕获 Graph。
