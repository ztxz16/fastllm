# KVMem（实验性，默认关闭）

KVMem 用固定大小的 GPU KV 池和按需分配的主机备份保存长上下文。超过 GPU 常驻预算后，注意力只访问 sink、最近上下文和检索到的旧页。这会改变模型输出；KV 字节的无损换页不代表稀疏注意力与完整注意力等价。

当前接入 Qwen3 和 Qwen3.5 的单卡 CUDA 普通解码，以及 Qwen3.5 的 DFlash/MTP 推测解码。支持 head_dim 为 128/256 的 FP16/BF16 MHA/GQA，计算与 KV 类型须一致。启用后目标模型使用单请求 eager 路径并关闭历史/prefix 缓存；尚不支持 TP、多模态、滑窗注意力或带位置偏置的目标注意力。未传入配置时，原有缓存、CUDA Graph、融合算子和推测解码路径保持原样。

## 使用

在 `warmup()` 或第一次推理之前配置，配置后不要修改设备、计算/KV 类型或并发数：

```python
model.set_kvmem(
    max_tokens=32768,        # 逻辑上下文上限，不会扩展模型 RoPE 上限
    resident_tokens=8192,   # 每个完整注意力层的 GPU KV 槽位数
    sink_tokens=128,
    recent_tokens=2048,
    retrieval_tokens=4096,
    prefill_tokens=512,
    host_mib=8192,          # 整个请求所有完整注意力层的主机 KV 预算
    retrieval_interval=64,  # 推测验证的检索刷新间隔；1 表示逐轮刷新
)
model.warmup()
```

CLI 对应一个 JSON 配置参数，不需要环境变量：

```bash
ftllm server MODEL --device cuda --max_batch 1 \
  --atype float16 --kv_cache_dtype float16 --speculative_algorithm off \
  --kvmem '{"max_tokens":32768,"resident_tokens":8192,"host_mib":8192}'
```

Qwen3.5 推测解码沿用现有参数，将上述 `--speculative_algorithm off` 替换为：

```bash
# DFlash2
--speculative_algorithm dflash --draft /path/to/DFlash2 --draft_tokens 3
# 内置 MTP 权重
--speculative_algorithm mtp --draft_tokens 3
```

`prefill_tokens` 必须至少容纳 `draft_tokens + 1` 行验证输入。目标 full-attention KV 使用 KVMem；DFlash 草稿保持原有滑窗缓存，MTP 草稿保持原有完整分页缓存。因此 `resident_tokens` 不限制草稿模型的显存；MTP 还需额外分配逻辑上下文上限对应的草稿 KV 和 lookahead 页。

`retrieval_interval` 默认为 64，控制 DFlash/MTP 目标验证的检索评分复用：已提交 token 位置跨过对应间隔时刷新。它是正整数，不要求按物理 KV 页对齐，也不改变常驻或检索页数。需要更短的复用间隔时可设为 32，需要逐轮刷新时设为 1；Python 和 CLI 共用此参数。例如在已有 `--kvmem` JSON 中加入 `"retrieval_interval":32`，或向 `model.set_kvmem` 传入 `retrieval_interval=32`。更长间隔减少检索和换页开销，但查询变化时可能继续使用旧评分并影响输出质量。普通解码仍按 KV 页边界刷新。

除 `max_tokens` 外，token 预算必须按 KV 页长对齐（默认 128）。常驻预算至少覆盖 `sink + recent + retrieval + prefill + 一页`，最后一页用于未对齐边界。主机预算须能容纳逻辑上限下所有完整注意力层的 KV（向上取整到完整页）；内存只在冷页首次换出时实际分配。预算不足、模型不支持或配置组合不支持时会报错。

GPU KV 大小约为 `resident_tokens × 全注意力层数 × KV heads × head_dim × 4` 字节；此外还有 FP32 mean-K 索引、当前分块的 Q/K 临时张量、GDN 状态和模型工作区。`host_mib` 限制 KV 页备份，不包含模型其他 CPU 内存。`--kv_cache_limit` 与此模式不兼容；显式 `--chunked_prefill_size` 可设置默认 `prefill_tokens`，JSON 中的值优先。

冷页备份使用普通主机内存。CUDA 换页批量提交 K/V 拷贝，每批完成后统一同步，避免逐页显式同步；不增加额外的 pinned 中转副本。所有旧页备份完成后才复用槽位，恢复完成后才发布新页表。

物理驻留与注意力选页分开管理：本轮未选中的完整旧页可以留在空闲槽位中，只有新增页或检索缺页需要空间时，才从未选中的驻留页中按最近使用时间淘汰。缓存命中不会增加注意力可见页，也不改变选页顺序或检索评分；常驻上限和 GPU KV 池大小保持不变。`residentPages` 统计全部物理驻留页，可能大于当前注意力视图的页数。

## 超出原生上下文

KVMem 只改变 KV 存储和注意力选页，扩大 RoPE 范围仍须在模型构造时显式配置。HF 和 GGUF 可使用公共 `--max_context_length`、`--rope_scaling` 参数；GGUF 的默认加载路径不变。Qwen3.5 架构、原生 262144 且符合已知 M-RoPE 配方的模型可用 `yarn` 简写，目标 524288 自动采用 factor=2，目标 1000000 采用 factor=4：

```bash
ftllm server /path/to/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --device cuda --cuda_embedding --max_batch 1 \
  --atype float16 --kv_cache_dtype float16 --tokens 1000000 \
  --max_context_length 1000000 --rope_scaling yarn \
  --prefix_cache false --cache_history false --speculative_algorithm off \
  --kvmem '{"max_tokens":1000000,"resident_tokens":8192,"sink_tokens":128,"recent_tokens":2048,"retrieval_tokens":4096,"prefill_tokens":512,"host_mib":65536}'
```

该模型的目标 KV 约为每 token 64 KiB，因此 512K / 100 万 token 需要约 32 / 61 GiB 主机 KV 预算，另加权重等 CPU 内存。8192 常驻槽位约占 512 MiB GPU KV，但索引、工作区和草稿缓存另计。支持内置 MTP 共用目标 YaRN 配置（`--speculative_algorithm mtp --draft_tokens 3`），其额外 GPU KV 在这两个长度下约为 2 / 3.8 GiB。

Qwen3.5 目标也可与全部采用滑窗注意力、原生 default RoPE 的 DFlash2 同用：将上述 `off` 替换为 `dflash --draft /path/to/DFlash2 --draft_tokens 7`。草稿保留自身的 RoPE 参数和滑窗，主模型独立使用 YaRN 验证；草稿 Q/K 不与主模型 KV 直接计算注意力，因此两者不要求相同的旋转维度或缩放系数。加载时检查每层布局、窗口及 RoPE 类型，拒绝完整注意力、缩放 RoPE、partial rotary 或 M-RoPE 的草稿。草稿缓存有界，旋转按需计算绝对位置；候选的接受率仍可能随目标 YaRN 和稀疏选页而变化，不能由短上下文加速比推算长上下文速度。

扩展后的容量、位置编码和任务质量需要分别验证。YaRN 和稀疏选页均可能改变输出，成功回答单个长文检索问题不代表通用长文理解质量得到保证。

## 模型接入接口

- `KvMemPages`（`include/kvmem.h`、`src/kvmem.cpp`）只负责逻辑页、槽位、选页、预算和主机备份。批量传输回调与 CUDA/模型解耦，回调须在返回前完成整批传输，包括异常路径。每个请求的每层单独拥有实例。
- `KvMemAppend`（`src/devices/cuda/fastllm-kvmem.cu`）接收归一化的 RoPE 前 Q/K `[1,T,H,D]` 及 RoPE 后 K/V `[H,T,D]`，增量更新 FP32 key-sum 索引，再写入固定 GPU 池。输出 `Data.pageIndex` 是按逻辑顺序排列的借用视图，`Data.dims[1]` 保留完整逻辑长度。
- 模型覆盖 `SupportsKvMem()` 和返回实际 KV 字节数的 `KvMemLayerBytesPerToken(layer)`，如有混合层则覆盖 `KvMemLayerEligible()`，模型特有约束可通过 `ValidateKvMemModel(config)` 校验本次候选配置，在单卡前向入口调用 `PrepareKvMemCaches()`。完整注意力层在 RoPE 之前保留 Q/K，RoPE 之后调用 `KvMemAppend()`，再使用现有 causal paged attention。位置编码仍使用绝对 token 位置。
- 推测验证在每层 `KvMemCache` 上调用 `BeginTransaction(tokens)`，执行一次指定行数的前向，再调用 `FinishTransaction(acceptedTokens, keyCache, valueCache)`。参数是保留的输入前缀行数，包含验证首 token；传 0 表示完全回滚，也可在尚未执行 Append 时取消。禁止嵌套事务、多次 Append 和越界提交。模型还须恢复其他循环状态：Qwen3.5 验证使用独立 GDN 状态，提交相应前缀快照；无可用快照时回滚 KV，再重放已接受输入。
- 调度器释放缓存时调用 `ReleaseKvMemCache()`，不能将借用的 `pageIndex` 返还给全局分页池。禁止隐式 `Data::CopyFrom` 克隆 KVMem 缓存。事务不提供任意历史快照或 prefix 共享；其他模型必须显式接入状态提交与异常清理。

## 检索与边界

索引保存每个逻辑页、每个 KV head 的 FP32 key sum，查询时除以实际 token 数得到 mean-K。对每个 Q head 计算缩放点积，对所有已提交页做 softmax 后平均各 head 的概率。当前按层独立选页，与 NInfer 的跨层聚合策略不同。

分块 prefill 用首行 query 查询此前已提交的页，避免未来 token 影响之前行的选页；当前 query 所在的全部尾页始终保留，因此压缩页表后的 causal mask 与绝对时间顺序一致。每个 prefill 分块、分块后的首个单 token，以及 decode 的页边界会刷新检索。最后一个 prompt token 独立执行，使用最后的提示 query 刷新检索再采样。

完整冷页首次换出后保持不可变，重复换出复用主机副本。正在写入的半页和整个验证尾部不会被换出；传输完成后才复用槽位和发布新页表。验证只查询此前已提交的索引，新 raw K 暂存至提交时，仅将接受前缀加入 key sum。拒绝尾部的物理页与逻辑长度同步裁剪，后续追加覆盖半页里的无效尾部；拒绝 token 不进入主机备份或索引。

普通解码在已提交的 query 位置跨 KV 页时刷新检索，页内复用评分；推测验证默认每跨 64 个已提交 token 的区间刷新，区间内复用评分，一次验证共享首行 query 的选择。`retrieval_interval=1` 恢复逐轮刷新。刷新间隔以验证前的已提交位置计算，未接受的草稿不推进区间；从普通解码切换到不同的验证间隔或切回时也重新评分。物理驻留缓存与检索刷新间隔分别管理。拒绝尾部不进入索引，整轮取消会使缓存评分失效，避免被取消的首行 query 影响重试。prefill 按分块刷新，分块后的单 token query 也重新评分。这仍是稀疏目标模型的近似策略，不承诺与逐 token 重新检索或完整 KV 的输出逐 token 一致。尚未实现异步预取或目标 graph 捕获。

## 验证入口

`kvmem_pages_test` 覆盖换页、预算、刷新间隔配置校验、因果边界、事务与失败状态；`cuda_kvmem_test` 覆盖 FP16/BF16、MHA/GQA、不同 head_dim、检索恢复、大批量换页、稀疏注意力的 CPU 数值参考，以及拒绝/部分接受/全部接受后的 KV 和检索索引一致性、共享缓存释放及模型配置失败的状态保留。检索测试包含默认 64、显式 32 和逐轮刷新，验证区间内复用、跨界刷新、取消重试及普通/推测模式切换。两者都接入 CTest。CUDA 测试需要可用 GPU。
