# FastLLM DFlash2 部署改动备忘（双卡均衡 / 视觉 / reasoning / 调参）

> 一次性改动记录，供下次借鉴。含：DFlash2 双卡显存均衡、视觉识图（能力上报 + 投机/思考组合）、
> reasoning 分离、与 vLLM 的差异、以及调参结论。
> 改动文件：
> - `src/models/qwen3_5.cpp`、`include/models/qwen3_5.h`（显存均衡 §1–§7；含图请求禁用投机 §8.3）
> - `tools/fastllm_pytools/openai_server/fastllm_completion.py`（reasoning 分离，见 §8.1）
> - `tools/fastllm_pytools/openai_server/fastllm_model.py`（视觉能力上报，见 §8.2）
> - `start.sh`（默认参数，见 §9）
> 验证环境：2 × RTX 2080Ti 22G（NVLink，sm_75），Qwen3.8-27B-NVFP4/W4A16 + DFlash2，262144 tokens。

---

## 1. 问题现象

启动日志里两张卡的 `servingReserve` / 空闲显存差约 1 GB：

```
AutoWarmup GPU 0: servingReserve=2377.26 MB, availForKV=5.93 GB
AutoWarmup GPU 1: servingReserve=1347.42 MB, availForKV=7.02 GB
```

nvidia-smi 最终 `18007 / 16684 MiB`（差 1323 MiB），GPU1 有约 1 GB 用不上。

## 2. 根因

fastllm 的 DFlash backbone TP **只切了 fused MLP**（`*.mlp.gateup_proj` + 配对的 `down_proj`），
其余 draft 权重整块放在 root（`devices.front()`）：

| draft 权重 | 大小(BF16) | 原行为 | vLLM / SGLang |
|---|---:|---|---|
| `self_attn.q/k/v/o_proj` | ~500 MiB | 全在 root | **TP 分片** |
| `attention_conv/mlp_conv.kernel_projection` | ~125 MiB | 全在 root | 复制到各 rank |
| `dflash.fc` | ~250 MiB | 全在 root | 复制到各 rank |
| `dflash.all_kv`（fused 视图） | ~105 MiB | 全在 root | 分片 |
| norm / base_kernel / hidden_projection | 极小 | 全在 root | 复制 |

关键差异：vLLM（`qwen3_dflash.py` + `qwen3_dflash2.py`）和 SGLang（`models/dflash.py`）
都把**整个 draft 放在同一 TP 组**里，attention 用 `QKVParallelLinear`/`RowParallelLinear` 头切分。
fastllm 的注释写得很直白：attention/selector/context 投影“stay on the root until separately validated”。

## 3. 方案：output-gather TP

draft 的 attention 需要在 root 上跑（KV cache 也在 root），不便直接做“输入切分 + all-reduce”。
但上述权重**都对输入通道做规约**，因此可以：

1. 把权重按**输出行**切到各 rank（`BuildMultiCudaRowSplitScheme` + `SplitMultiCudaWeight`，axis=0）；
2. 每个 rank 用**完整复制的输入**算自己那部分输出行；
3. 把各 rank 的输出行 **gather 回 root**，拼成完整输出。

算术与单卡完全一致，只改变权重的物理分布 → 省 root 显存、不动计算图。

复用既有设施：
- `Executor` TP：`Qwen35DFlashTpExecutor` / `Qwen35DFlashTpDeviceSpec`；
- 算子：`MultiCudaLinearOp` 的 `forceOutputGather=1` → `RunMultiCudaOutputGatherLinear`
  （原本只有“没有配对 down 的 gateup”会用到，实际从未触发）。

## 4. 具体改动

### 4.1 新增 eligibility
`Qwen35DFlashTpOutputGatherEligible(name, data)`：匹配
`dflash.fc.weight`、`dflash.all_kv.weight`、
`*.self_attn.mergeqkv.weight`、`*.self_attn.o_proj.weight`、
`*.attention_conv.kernel_projection.weight`、`*.mlp_conv.kernel_projection.weight`。
阈值用新环境变量 `FASTLLM_CUDA_DFLASH_TP_MIN_GATHER_MB`（默认 4 MB，conv 投影 ~13MB 也能切）。

### 4.2 权重准备 `PrepareDFlashWeightsForDevice`
- 把 eligible 权重加入 `deferredTpLinearWeights`，让 `moveLinear` 跳过（直接从 CPU/mmap 源切分）。
- **fused `dflash.fused_kv_qkv.weight` 在 TP 下不再搬到 root**：
  在 host 上物化 `all_kv` + 每层 `mergeqkv` 为**独立 owned 权重**（`target.CopyFrom(view)`），
  物化完 `fusedWeight.FreeSpace()` 释放 staging buffer。
  > 非 TP 路径保持原来的 `FakeFrom` 零拷贝，行为不变。

### 4.3 切分 `PrepareDFlashBackboneTensorParallelWeights`
新增一个循环：对 `Qwen35DFlashTpOutputGatherEligible` 的权重用
`BuildMultiCudaRowSplitScheme` + `SplitMultiCudaWeight(..., axis=0, true, true)` 切分，
再 `convertTpShardsToFp16`（pre-Ampere 转 FP16）。

### 4.4 前向调用
把 draft 前向里这些 `Linear(...)` 换成 `RunDFlashTpLinear(device, input, weight, output)`
（`Qwen35DFlashTpLinear` 内部：有 TP 分片就走 `forceOutputGather`，否则回退普通 `Linear`）：
- `mergeqkv`、`o_proj`、两个 `kernel_projection`（单条 `RunDFlashDraft` + 批量 `RunDFlashDraftBatch`）；
- `dflash.fc`、`dflash.all_kv`（`AppendDFlashTargetHidden` context 路径）。
> `RunDFlashGateupLinear` 因此更名为通用的 `RunDFlashTpLinear`。

### 4.5 KV sizing reserve
`GetAutoWarmupCudaServingReserveBytes` 里判定 `sharded` 时，加上
`Qwen35DFlashTpOutputGatherEligible(...)` 和 `dflash.fused_kv_qkv.weight`，
否则 root 仍会被按“整块”预留（约多留 500 MB，`availForKV` 被压低）。

## 5. 踩过的两个坑（重点）

1. **输入激活被复用**：`RunMultiCudaOutputGatherLinear` 会 `EnsureReplicatedMultiCudaTensor(input)`，
   把输入就地变成 multi-cuda 复制态。`attention_conv/mlp_conv.kernel_projection` 的输入
   `normalized` 之后还要喂给 `dynamicConvolve`，于是先 `Copy(normalized, convInput)` 再传入。
   → 现象：首个真实请求报 `DFlash selector candidate id is out of range`，
   并因 `ErrorInFastLLM` 里 `exit()` 触发 `Qwen3_5Model::~Qwen3_5Model → ShutdownRuntime`
   在当前 worker 线程里 `join` 自己 → `terminate: Resource deadlock avoided`。

2. **allocator 高水位**：把 fused 权重搬到 root 再切片释放，虽然逻辑释放了，但
   `FastllmCudaFree` 的内存仍留在 allocator pool 里，nvidia-smi 一直算“已用”，
   root 凭空多 ~0.7 GB。改为 **host 物化**（和 paired MLP 的 `moveLinear` 注释同一个道理）后消失。

## 6. 实测结果

| 指标 | 改动前 | 改动后 |
|---|---:|---:|
| idle 显存 GPU0/GPU1 | 18007 / 16684 MiB | **17323 / 17190 MiB** |
| 显存差 | 1323 MiB | **133 MiB** |
| servingReserve | 2377 / 1347 MB | **1884 / 1876 MB** |
| `TP prepared` 日志 | 0 output-gather weights | **22 output-gather weights** |
| dflash 权重按卡统计 | — | 1765.6 / 1762.5 MiB（+host 242.5 codebook） |

剩余 133 MiB 是 target 本身的不对称：单独跑 `SPEC_ALGO=off` 也是 133 MiB，与 draft 无关。
功能验证：短问答、256 token 生成、20k token 长 prompt（走 `all_kv` context）均正常；
DFlash 接受率与改动前持平（pos0 ~36–38%）。

## 7. 下次复用清单

- 加新 draft 权重分片：只要满足“对输入通道规约”，直接在 `Qwen35DFlashTpOutputGatherEligible`
  的 suffix 列表里加名字即可；阈值调 `FASTLLM_CUDA_DFLASH_TP_MIN_GATHER_MB`。
- 任何要接 output-gather 的算子：**检查输入是否在调用后被复用**，是则先 `Copy`。
- 物化 weight 视图时优先留在 host 切分，别先搬 root 再释放（allocator 高水位）。
- 调试开关（本次临时加、已移除）：按权重名子集开启 output-gather 做二分定位；
  `FastLLM Error` 消息因 stdout 缓冲可能在 abort 时丢失，跑服务时用 `stdbuf -o0 -e0` 才能看到。

---

## 8. 配套的服务端修复（Python 层）

### 8.1 自定义 `--chat_template` 下也能分离 `reasoning_content`

**现象**：开了思考（`--enable_thinking true`）后，思考内容泄漏进 `message.content`，
`reasoning_content` 为 `None`，`</think>` 混在正文里。

**根因**：`--chat_template` 非空时 `llm.py` 会置 `force_chat_template=True`，
而 `_is_qwen3_5_reasoning_response` 里有 `and not force_chat_template` 判断，
于是禁用了思考解析；prompt 里确实进了思考模式，输出却没做拆分。

**修复**：`fastllm_completion.py` 的 `_is_qwen3_5_reasoning_response` 去掉该限制
（Qwen3.5 官方/sharp 模板本来就输出标准 `<think>...</think>`，拆分是安全的）。

**验证**：`reasoning_content='用户问…答案是7。'` / `content='7'`。

### 8.2 `/v1/models` 正确上报图片能力（修 opencode 不识图）

**现象**：同一个 opencode 配置，连 vLLM 能识图，连 fastllm 不行；模型只看到 `[Image 1]` 占位符。

**根因**：`fastllm_model.py:28` 把 `input_modalities` 只按 `mmproj_path` 判断
（那是 GGUF 专用），HF 内置视觉模型因此被报成 `["text"]`。
opencode 读 `/v1/models` 的该字段，判定不支持图 → 在客户端就把图片替换成文本占位符
（请求根本没带图到服务端）。vLLM 报 image，所以正常。

**修复**：新增 `_supports_image_input(model)`：GGUF 看 `mmproj_path`；
HF 看 `config.language_model_only != true` 且 `vision_config` 非空
（或 `vision_start_token_id` / `image_token_id` 存在）。

**验证**：`GET /v1/models → input_modalities=['text','image']`；
本机 `opencode run -m ... -f big.png`（配置**只**声明 text）也能正确识别图。

**排查提示**：客户端不传图时，先看服务端 `/v1/models` 的 `input_modalities`，
再看服务端日志有没有 `[Vision] ...`；`[Vision]` 出现说明图到了、编码了。
带不带 tools/流式/长 system/前缀缓存都不影响识图（均实测过）。

### 8.3 DFlash + 思考 + 图片 会看错图（重要）

> **一句话备忘**：`DFlash投机 × 视觉` 不稳 → 含图请求**必须关投机**（C++ 守卫）；
> 而 `视觉 × 强制关思考` 会引发“会话历史污染”（模型照抄上文的错误答案）→ 含图**必须保留思考**。
> 合起来：**含图 = 关投机 + 开思考**，与 vLLM 行为一致。这是个很隐蔽的组合 bug。

**现象**：`input_modalities` 修好、图确实送达并编码后，仍出现“乱识别”
（绿圆答成“蓝圆/红方”，推理内容也自洽但完全是错的）。逐一消融后定位为**三者叠加**才触发
（图=绿圆+黑方）：

| 配置 | 结果 |
|---|---|
| DFlash 开 + 思考关 | ✅ 绿/黑 |
| DFlash 开 + 思考**开** | ❌ 蓝/红（稳定复现） |
| DFlash **关** + 思考开 | ✅ 绿/黑 |

即 `--speculative_algorithm dflash` × `enable_thinking=true` × 含图请求。
vLLM 无此问题，所以同一 opencode 在 vLLM 下正常。

**定位手法（可复用）**：`opencode` 的请求用本地反向代理（记录 body 后转发）抓下来，
再原样重放 + 逐项消融（去 tools / 去 system / 改 thinking / 改 SPEC_ALGO）。
注意 `enable_thinking=true` 会用掉 token，`max_tokens` 要放大，否则 content 为空被误判。

**状态相关**：干净启动时 `dflash+思考+图` 反而正常（系统+tools、图在中间、红蓝交替都试过），
反复请求/缓存累积后才出错，很难稳定复现。

**vLLM 为什么没这问题**：`DFlashSpeculator.supports_mm_inputs=False`（`dflash/speculator.py:43`），
`model_runner` 因此不收集/不喂 `mm_embeddings` 给草稿；草稿只提议、**目标验证兜底**，
所以图文最差只是变慢。`llm_base_proposer` 还明确告警 "Proceeding with text-only speculative decoding"。

**最终方案（只加 C++ 守卫，保留思考）**：

- **C++：含图请求绕过投机、目标单跑**（对齐 vLLM 的“目标权威”）
  - `Qwen35MTPForward` / `Qwen35MTPBatchForward` 入口：
    `if (!context->multimodalInput.empty()) return false;`（`qwen3_5.cpp`）。
    返回 false 即回退普通目标前向，本请求全程不做投机。
  - `ResponseContext::multimodalInput` 在请求生命周期内一直非空，覆盖所有 decode 步。
  - 多模态本身只支持单 prompt（`qwen3_5.cpp:32210`），批量入口守卫是冗余保险。
- **思考保留**（`fastllm_completion.py` 不做处理）。曾经试过“含图强制关思考”，
  反而更糟——见下面的“会话历史污染”。

**为什么必须保留思考**（和 vLLM 对齐的关键，实测）：

| 场景 | 关思考 | 开思考 |
|---|---|---|
| 同一张图 + 上文里一条**错误**的 assistant 描述 | ❌ 照抄错误答案 | ✅ **重新看图并纠正** |

vLLM 默认 `enable_thinking:true`，所以它“没有污染问题”本质是**靠思考每轮重新识图**；
fastllm 若把含图思考关掉，模型就会顺着上文自洽地一路错下去（用户遇到的“马里奥→RTX 5090”）。
> 注意：`enable_thinking` 会消耗 token，`max_tokens` 太小时 content 会为空（`finish=length`），
> 属正常，给足预算即可（opencode 默认 32k）。

**验证**（最终配置：含图关投机 + 保留思考）：
- 简单图（绿圆）：4.4s、finish=stop、正确。
- 同图 + 矛盾历史（上文写“RTX 5090”）+ 800 token：**13.2s、finish=stop、纠正为绿圆**，不循环。
- 日志：`Multimodal request: images=1 ... (speculation is skipped ...)` +
  `Multimodal detail: effort=medium ... enable_thinking=True image_sizes=WxH msgs=...`。
- 文本计数请求 decode **191.2 tok/s**（投机生效）；文本 `3+4=?` 仍有 `reasoning_content`。

> 诊断日志（保留，便于复现）：`Multimodal request: images=N ...` 与
> `Multimodal detail: effort=... temp=... top_k=... enable_thinking=... image_sizes=... msgs=...`。
> 若 `images=0` 说明客户端没发图（多为客户端未重启刷新 `/v1/models` 能力）。
> `FASTLLM_VISION_DUMP_DIR=/tmp/vision_dump` 可把收到的图存盘（调试用）。

**opencode 侧两种贴图方式**（实测，`1.18.31`）：
- **剪贴板贴图**：✅ 可用（服务端 `images=1`，识别正确）。
- **传本地文件路径**：❌ opencode 自身 `POST http://opencode.internal/session/.../message`
  返回 **400**，请求根本没发到模型服务（服务端无 `Multimodal request` 日志）。
  属客户端问题；本机 CLI `opencode run -f /abs/path.png` 正常，故建议用剪贴板或 TUI 附件。

**“乱识别”最终定因：会话历史污染 + 关思考叠加**
用 `FASTLLM_VISION_DUMP_DIR` 抓下用户完整请求（17 条消息、4 张同图）后原样重放：
- 截断到第一张图（msg0–10）→ **3/3 正确识别马里奥**；
- 只留最后一条 user+图 → **正确**；
- 带完整历史（msg11/13/15 的 assistant 已被写成 “RTX 5090 D 规格”）→ **继续答 RTX 5090**。

机制：opencode 每轮把整段历史重发；**同一张图 + 上文已有（错误）描述**时，模型倾向沿用上文。
此时**思考开才能纠正**（vLLM 同模型实测：关思考❌照抄 / 开思考✅重看纠正）。
所以早期为了绕 DFlash 而“含图强制关思考”反而放大了这个问题。

**处理**：
1. 最终方案已回到“含图关投机 + **保留思考**”（见上），新会话即正确；
2. 已被污染的旧会话需**换新会话**或清空/compact（历史里那条错误答案会一直被沿用）；
3. 排查手法：`FASTLLM_VISION_DUMP_DIR` 存图 + `req_*.json`，再按消息前缀截断重放定位。

---

## 9. start.sh 参数与调参结论

### 9.1 当前默认（对齐参考 vLLM 部署 start_fp4_df2.sh）

| 参数 | 值 | 说明 |
|---|---|---|
| `TEMPERATURE` | `0.6` | vLLM 用 `OVERRIDE_GENERATION_CONFIG={"temperature":0.6}`；模型自带 `generation_config.json` 是 1.0，偏高 |
| `ENABLE_THINKING` | `true` | 对齐 vLLM 的 `DEFAULT_CHAT_TEMPLATE_KWARGS.enable_thinking` |
| `VISION_DEVICE` | `cuda:1` | 编码器只能整卡放一张卡（占用 ~0.9GB），放非 root 让 root 轻载 |
| `IMAGE_EMBEDDING_CACHE` | `512m` | 图片 embedding 的 **CPU** 缓存，重复图片不重编码 |
| `MAX_BATCH` | `4` | 见 9.2 |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | 与 vLLM 的 `fp8` 同为 e4m3 |
| `CHUNKED_PREFILL_SIZE` | `4096` | 见 9.2 |
| `LOW_GPU_MEM` | 默认 `1` | 本套 DFlash 配置下开/关显存与速度都几乎无差 |

### 9.2 性能实验（显存有富余后能开哪些）

| 尝试 | 结论 | 代价 |
|---|---|---|
| `--max_batch` 2→4→8 | **有效**：日志聚合 33→48→50 tok/s | +~0.3GB/卡 |
| `FASTLLM_CUDA_GRAPH=1`（强制开图） | **无效且更慢**（计数 187 vs 192、聊天 37 vs 40），sm75 本自动禁用 | +1.3~2.7GB/卡；不配 low_mem 会装不下 256k |
| `--chunked_prefill_size` 4096→8192→16384 | **无效**：100k prefill 809→791→781 tok/s | 峰值不变 |
| `--vision_device cuda:0/cuda:1` | 总量一样，只是换卡承重 | — |

**单流 decode 提不上去的原因**：受显存带宽 + 草稿接受率限制（同模型/草稿/硬件，
vLLM 与 fastllm 实测基本打平：计数 ~188/192、数学 ~117/117、聊天 ~39/40，
100k prefill fastllm 略快 ~8%，decode 相近）。加显存不改变这一点。

### 9.3 256k 长上下文

- DFlash 权重分片后，`low_gpu_mem` 关闭时 256k 实测峰值 **18649 / 17662 MiB**，余量 ~3.9GB，不 OOM。
- 加载视觉后 hosting 卡再少 ~0.9GB，仍能装下 256k KV（4.2GB/卡）。

## 10. 与 vLLM 的部署差异速查

| | fastllm `start.sh` | vLLM `start_fp4_df2.sh` |
|---|---|---|
| 显存/卡 | ~17.5GB（余量大） | ~20.2GB（GPU_UTIL 0.96） |
| 并发 | `max_batch=4` | `max_num_seqs=1` |
| CUDA graph | 关（实测关更快） | 开（piecewise） |
| 视觉 | HF 内置，`--vision_device` 选卡 | 原生 |
| 工具调用 | 自动判定（`qwen3_coder`）；可 `--tool_call_parser qwen3_xml` 对齐 | `qwen3_xml` + auto tool choice |
| reasoning 分离 | 需 §8.1 修复 | 原生 `qwen3` parser |
| 启动 | ~40s | ~2–4min |
