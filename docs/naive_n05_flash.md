# Naive-N0.5-Flash-FP8 混合推理

## MiMo-V2 DSA checkpoint

`model_type=mimo_v2`、`enable_dsa=true` 的 split-QKV checkpoint 复用
`NaiveN05FlashModel`。原始 `config.json` 和 safetensors 无需改名或转换。
模型工厂同时识别 `MiMoV2ForCausalLM` / `MiMoV2DSAForCausalLM` 架构名；
运行时仍使用 `naive_n05_flash` 分派 TP、KV 缓存、思考与工具解析。
非 DSA、融合 QKV、全局 attention sink 或带共享专家的 MiMo 配置会明确报错。

已对照的 MiMo checkpoint 为 48 层、hidden size 4096，含 9 个 DSA 全局层和
39 个滑窗层；Q/K 为 192 维、V 为 128 维，前 64 维使用 RoPE。全局/SWA
分别有 4/8 个 KV heads，滑窗为 128。首层为 dense MLP，其余 47 层每层有
256 个专家、每 token 选 8 个；索引器为 16 heads × 128 维、top-k 2048。
这些结构和权重名称与 Naive 的现有实现一致，FP8 专家权重可直接复用加载路径。
`attention_value_scale=0.707`、不同的全局/SWA RoPE theta 和滑窗 sink
均从原始配置或权重读取。配置未给出的索引器尺寸使用上述默认值，并已核对权重形状。

8 张 H20（SM90）部署示例：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=90
FASTLLM_CUDA_GRAPH=1 ftllm server /path/to/MiMo-V2-DSA \
  --model_name mimo-fastllm --tp 8 --dtype auto --atype bfloat16 \
  --cuda_embedding --max_batch 32 --tokens 1048576 --threads 16 \
  --cache_history false --enable_thinking false --port 8000
```

### 启动时预分配 KV

普通 BF16 KV 的多卡 TP 路径在预热结束时为每张卡建立一个固定容量的 KV 池，
不依赖 Hopper、EP8 或 CUDA Graph，也不需要额外环境变量。`--tokens` 控制全局层
的总 token 预算；`--max_batch` 与 `--chunked_prefill_size` 控制滑窗层及分块
prefill 的容量预算。未指定 `--tokens` 时取各 TP 卡中最小空闲显存的 75% 为预算，
扣除滑窗及对齐空间后计算 token 容量；剩余显存留给工作区。未指定 `--max_batch`
时按 512 路预留，不会因显存不足自动下调。配置超出预算会在启动时报错。

请求的 K/V 仍是互不重叠的连续视图，直接复用现有 DSA/SWA 算子。首次请求、
长上下文扩容及 CPU 历史恢复都从池内领取空间，结束后归还并合并空闲区间；
运行期间不为这些 KV 调用 `cudaMalloc`。各 TP rank 在进入计算和通信前完成申请；
池耗尽时终止本轮选中的请求并返回容量错误，后续请求仍可处理，不向池外分配。
池内采用连续区间，未改为分页 attention；极端混合长度仍可能造成池内碎片。
CUDA Graph、采样和临时算子工作区有各自的分配策略，不属于这项 KV 保证。
CPU、按层串行、非 BF16 KV、草稿模型及跳过预热的直接调用保留原路径。

### EP8 与内置 MegaMoE

MiMo FP8 在 8 张 H20 上可直接使用 `--tp 8 --ep 8`：Attention 保持 TP8，
MoE 每卡放置 32 个完整专家，默认调用内置 MegaMoE W8A8，无需额外的
MegaMoE 开关或 `.so` 路径。`--ep 1` 使用原来的 MoE TP 路径；省略 `--ep`
时默认也是该路径（显式环境配置除外）。启动 JSON 支持 `"ep": 8`。

```bash
FASTLLM_CUDA_GRAPH=1 ftllm server /path/to/MiMo-V2-DSA \
  --tp 8 --ep 8 --dtype fp8_e4m3 --atype bfloat16 \
  --cuda_embedding --max_batch 4 --threads 16 --cache_history false
```

SM90 构建默认将 kernel 编入 FastLLM 主库。源码改编自 DeepGEMM，固定版本的
头文件、MIT 许可证和原始文件哈希位于 `third_party/deep_gemm_ep8`；构建和
推理均不需要安装 DeepGEMM Python 包或加载独立适配器。明确请求 `--ep 8`
但卡数、硬件或构建不支持时会报错。EP 必须在加载模型前选定。

内核沿用已有的权重布局，支持 1～4 路解码 CUDA Graph。大于 64 行的 prefill
在 TP/EP 边界按 token 均衡分成 8 份，每张卡每轮最多提交 64 个不同 token。
各分片长度至多相差 1；空分片以 0 行参与同一轮专家计算和同步。各卡完成自己的
输出分片后，通过 NVLink peer copy 直接汇集到原输出张量。主机同步保证拷贝前
分片已完成、所有读取完成前源张量不会复用；无需填充输出或增加 GPU 缓冲区。
此优化只接入上述 H20 EP8 后端，不会自动用于其他模型、GPU 架构或 EP 卡数。
32 个本地专家共享按路由总数和对齐余量计算的 6080 行中间池，所有层复用
43.92 MiB/卡持久工作区，比原 8 行版本增加
29.82 MiB/卡，不增加专家权重副本。工作区硬上限为 48 MiB。
合批 Graph 沿用每个请求独立的 KV 缓存；请求数量、顺序、缓存地址/容量或
Attention 区间改变时重新校验并捕获，不为每个 batch size 保留一套 Graph。
至多 64 行的输入仍由每张卡提交一份 hidden state，避免给短 decode/verify 添加
额外汇集通信；router 投影也保持复制执行。超过 4 路及 verify 保留 eager。
W8A8 与原 W8A16 并非数值等价，可能改变 logits。详细限制和独立校验命令见
[`USAGE.txt`](../tools/mega_moe_sm90/USAGE.txt)。

激活和 KV cache 使用 BF16，专家保留 checkpoint 的 FP8。CUDA 13.0 的
低比特 prefill 编译单元单独使用 C++17，规避 NVCC 与 libstdc++ 13 在
`std::destroy_at` 上的 C++20 内部编译错误，其余 CUDA 单元保持原有标准。
加载器还修复了并行合并专家权重时对 `unordered_map` 的无锁插入/删除：
读取数据保持并行，权重发布、合并、注册与删除共用互斥锁，避免偶发丢失
`gateup_proj.weight`。`hf_parallel_merge` 回归会连续加载并核对所有合并内容。

回归覆盖 MiMo 模型识别、配置拒绝、CLI 的 BF16/TP 默认值、注意力和 8 卡
计算；完整模型验证覆盖算术、排序、代码生成及 4377/9957-token 的长文检索。
生成的质数函数另通过 2023 个输入用例；OpenAI 兼容 API 的 8 路并发对话、
工具调用及工具结果续写、流式输出等 12 项检查通过。
这些测试未覆盖 checkpoint 配置中的完整 1M 上下文。

### MiMo 在 8×H20-3e 上的实测

在上述 TP8 / FP8 专家 / BF16 激活与 KV 配置下，完整 checkpoint 加载约
145 秒。原生接口使用 2048 个随机输入 token、256 个固定输出 token，关闭
历史缓存，结果如下。总吞吐包含 prefill；解码速度为请求级中位数。

| 同时提交请求数 | 总输出吞吐（token/s） | 解码速度（token/s/请求） | 首 token 中位数（秒） |
| --- | --- | --- | --- |
| 1（两次） | 70.5–81.0 | 103.0–109.3 | 0.68–1.30 |
| 8 | 107.1 | 18.19 | 4.27 |
| 32 | 110.6 | 4.79 | 16.76 |

`max_batch=200` 的同类测试在生成完成前发生 `cudaErrorMemoryAllocation`，
当时 device 0 仍报告约 90 GiB 空闲；尚未确认根因，不能将它解释为显存容量
耗尽，也不能报告为完成的 200 并发结果。部署示例因此使用 `max_batch=32`。
原生测速为保证长度使用 `min_length=256`，会走全词表采样路径。

另一次 HTTP 排队测试使用 `max_batch=32`，同时提交 200 个不同的数字串
聊天请求，每请求实际输入 3097 token、固定输出 32 token，200/200 成功。
整轮耗时 368.17 秒，总输出 6400 token（17.38 token/s，包含输入处理与排队），
请求延迟中位数 242.42 秒。8 卡整轮平均 GPU 忙碌率 69.49%、SM 活跃度
37.42%、Tensor Core 活跃度 1.43%、显存带宽活跃度 1.72%；每卡平均
42.87 GiB 显存、201.79 W。它验证的是限流后的 200 路 HTTP 服务，不代表
200 个请求同时在 GPU 解码；这些硬件活动指标也不等于模型 FLOPS 利用率。

### MiMo FP8 BS1 CUDA Graph

`FASTLLM_CUDA_GRAPH=1` 现在允许 GPU 常驻的 FP8 E4M3 专家进入单行 TP
解码图路径，保留 FP16 专家激活、BF16 attention/KV 的现有算术。专家指针表
在预热阶段初始化；若 MoE 退回需要 CPU 读取路由的实现，则禁用该缓存的图捕获。
FP8 多行合批和 speculative verification 仍使用原有 eager 路径。

8×H20 的完整 MiMo 对照覆盖 127、254、2047、2048、2049-token 输入，
共 128 个输出位置，每个位置比较全部 152576 维 FP32 logits。Graph 与 eager
逐字节一致，最大绝对误差及 RMSE 均为 0；另有 6 次 256-token 测速请求的
输出 token 序列一致。

Linux CTest `naive_n05_tp_graph`、`naive_n05_tp_fp8_graph` 覆盖长度边界、
请求缓存复用、采样模式切换、capture begin / instantiate 失败回退。
两个测试固定 `FASTLLM_CUDA_CUSTOM_ALLREDUCE=1`，使跨进程比较采用相同的
通信归约顺序；未固定时自动调优可能选择不同后端，连 prefill logits 也会出现
小幅差异。正常服务继续保留原有自动调优设置。

Nsight 在每张卡上确认每 token 一次 `cudaGraphLaunch`。GPU 0 的每步
kernel 数从此前的 947 减至 849（946 是普通 kernel 发射 API 次数），专家 gate/up+SwiGLU 与 down+reduce 仍为
约 1.258 / 1.453 ms/token，与原 eager 的 1.263 / 1.458 ms 接近。
原始波形、logits、测速脚本及结果保存在 `/root/fastllm/run/cuda_graph/`；
本地结果目录为 `mimo_results/cuda_graph/`。

同一进程内切换 Graph 开关，保持输入、权重、精度和通信后端一致，预热后
按 eager / graph / graph / eager / eager / graph 顺序各测 3 次。
完全不加载 Nsight，2048 输入、强制 256 输出，排除首 token 的结果如下：

| 路径 | 三次解码 token/s | 中位数 token/s | 中位数 ms/token |
| --- | --- | ---: | ---: |
| eager | 108.07 / 112.96 / 111.90 | 111.90 | 8.937 |
| CUDA Graph | 133.38 / 140.61 / 135.74 | 135.74 | 7.367 |

本组完整解码段吞吐提升 21.3%；排除前 16 个输出 token 后，Graph 稳定段
中位数为 140.61 token/s。

另一次已注入 Nsight、尚未开启采集的进程测得 eager / graph 中位数为
97.38 / 74.48 token/s，且通信自动调优结果也发生变化；不能把这组数据当作
无 profiler 的吞吐。正式性能结论采用 `cuda_graph/plain/results.json`；
逐 kernel 采集数据用于确认图回放和归因，不用其墙钟时间推算生产速度。

## 编译与启动

```bash
bash install.sh
FT_NUMAS=1 numactl -C 0-31 -m 0 \
  ftllm server ~/hfmodels/Naive-N0.5-Flash-FP8 \
  --device cuda --moe_device numa
```

`-C` 指定 CPU 编号；小写 `-c` 指定 NUMA 节点，不能用 `-c 0-31`。
默认线程数会考虑 `FT_NUMAS` 和进程 CPU affinity，并给推理控制线程预留 CPU。
在 32 个可用逻辑 CPU 的配置中自动选择 28 个 NUMA 工作线程，也可以显式指定 `--threads 28`。
不要让忙等工作线程占满 `-C` 指定的全部 CPU；工作线程占满可用 CPU 时可能造成调度争抢。
保持 `--dtype auto`，保留 checkpoint 的 FP8 专家权重。激活和 KV cache 默认使用 BF16。
完整的 `model-00001-of-00049.safetensors` 等分片可以直接加载，无需手动补索引文件。
长 prefill 默认自动分配 CPU/GPU 专家；如需关闭 GPU 专家 prefill，在启动前设置
`FT_GPU_PREFILL=0`。此开关在进程初始化时读取。


长请求可按显存容量设置 `--chunked_prefill_size 8192`；默认分块为 2048。
`--device cudapp=2` 将主干按层分到两卡，长 prefill 的专家可同时由 CPU 和两张 GPU 执行。
实际线程数、分块和并行度需按硬件调整，测量结果见[Benchmark](benchmarks/naive_n05.md)。

## 实现范围

- CUDA 执行稠密层、路由、注意力及输出投影；专家权重保存在 NUMA。decode 专家由 CPU 执行，长 prefill 按路由负载分给 CPU 与各辅助 GPU 并行执行。
- 支持 Q/K 192 维、V 128 维、前 64 维的部分 GPT-NeoX RoPE，以及不同全局/滑窗 RoPE 参数。
- 滑窗注意力保留 127 个历史 token，计入 attention sink 和 value scaling。
- 全局层支持 FP8 indexer、因果约束及稳定的 top-2048 选择；索引键保存在对应请求的 KV cache 中。
- NUMA 专家使用动态 W8A8：每 128 个激活使用独立 FP32 缩放系数，在 GEMM 中应用激活和权重的缩放，并对齐原始实现的 BF16 舍入顺序。参考 FP8 内核使用的 FP32 → FP16 向零截断 → FP8 舍入也在 NUMA 路径中保留。
- 默认以 2048 token 分块 prefill，可用 `--chunked_prefill_size` 覆盖；每次 Forward 处理一条序列。跨请求前缀复用使用模型专用的 CPU KV 归档，HTTP 请求由现有调度器处理。

稀疏索引和稳定 Top-K 按形状选择 CUDA 或 CPU 路径。已记录的完整模型验证最长输入为 32727 token；模型配置中的 1M 上下文尚无完整验收结果。

## 工具调用与跨请求前缀缓存

默认启动自动根据 Naive 的 XML 模板选择 `qwen3_coder` 工具解析器，支持 `tools`、
工具结果消息和流式增量；`tool_choice=required` 或指定函数时复用 XML 工具选择引导。
也可以通过已有的 `--tool_call_parser` 显式覆盖解析器。
Python 接口也会把工具历史中的 JSON 参数字符串转换为模板所需的对象。
请求通过 `chat_template_kwargs.enable_thinking=true` 开启思考时，思考内容放入
`reasoning_content`，工具调用仍放入 `tool_calls`；默认启动不开启思考。

MoE 启动默认启用 `--cache_history true`；可用 `--cache_history false` 关闭。
完成或中止请求时，仅记录已经执行 Forward 的 token，排除尚未计算 KV 的最后一个输出。
后续请求按 token ID 匹配最长公共前缀，支持相同请求、追加工具结果、较短请求和分支请求。
相同请求仍保留一个输入 token 重新计算 logits。

归档在 CPU 分块保存滑窗裁剪前的 K/V，以及全局层打包的索引键；命中时全局层恢复完整
前缀，滑窗层只恢复边界前所需的 127 个 token，并恢复绝对位置和调度状态。
不可变归档块可在分支请求之间共享，当前请求的 KV 独立分配。
恢复过程只操作 CPU 内存，不等待其他请求的 GPU Forward。

归档最多保留 5 条已完成记录，按最近使用顺序淘汰。已完成、活动和待恢复请求共享
主机内存预算，共享块只计一次，最后一个引用释放时归还额度。
预算按 serving token pool（未设置时使用模型最大上下文）和每 token 归档大小估算，
目标至少为 8 GiB，再受可用主机/容器内存四分之一的限制；无法读取内存信息时以
40 GiB 为上限。预算在首次记录时确定，仅按实际保存的 token 分配，无新增环境变量。
长会话可以使用其他记录释放的额度；预算不足时淘汰已完成记录，仍不足则保留已缓存
前缀，其余输入正常计算。活动请求持有的块不会因淘汰而失效。
该模型的归档始终使用 CPU 内存，不受 `--cache_fast` 影响，不额外常驻 GPU KV 副本。
API 的 `usage.prompt_tokens_details.cached_tokens` 表示本次命中的输入 token 数。

### 本地 Codex CLI

`/v1/models` 的 Naive 模型目录提供 `none`（关闭思考）和 `low`（开启思考），默认 `none`。
Naive 的 `low` 是思考开关，没有较短的思考 token 预算；Codex 自动生成会话标题时也会
继承此设置。交互速度优先时保留 `none`。首次长提示仍需完整 prefill，后续请求通过
历史缓存复用前缀。

Codex 的自动标题请求有 30 秒超时，长 prefill 时排队生成可能超时并在下一轮重试。
以下命令通过 `X-FastLLM-Codex-Title-Mode: local` 显式启用本地标题：服务端仅匹配
Codex 的标题提示和对应 JSON schema，从用户请求提取不超过 36 字的标题，直接返回，
不进入模型队列。标题是请求文本的摘要标签；普通对话、工具调用和其他结构化输出仍由
模型生成。不带此请求头时保留模型生成标题的行为。

获取当前服务的模型目录并启动客户端：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8080/v1/models -o /tmp/naive-n05-models.json &&
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
codex --no-daemon -m naive-n05 \
  -c model_provider=fastllm \
  -c model_catalog_json=/tmp/naive-n05-models.json \
  -c model_reasoning_effort=none \
  -c cli_auth_credentials_store=ephemeral \
  -c features.plugins=false -c features.remote_plugin=false -c features.apps=false \
  -c web_search=disabled -c check_for_update_on_startup=false \
  -c analytics.enabled=false \
  -c model_providers.fastllm.name=FastLLM \
  -c model_providers.fastllm.base_url=http://127.0.0.1:8080/v1 \
  -c model_providers.fastllm.wire_api=responses \
  -c model_providers.fastllm.requires_openai_auth=false \
  -c 'model_providers.fastllm.http_headers={"X-FastLLM-Codex-Title-Mode"="local"}' \
  -c model_providers.fastllm.supports_websockets=false \
  -c model_providers.fastllm.request_max_retries=0 \
  -c model_providers.fastllm.stream_max_retries=0
```

仅关闭 `remote_plugin` 不会关闭插件目录的其他远程查询；此命令关闭整个插件功能。
`cli_auth_credentials_store=ephemeral` 让这个本地客户端不读取持久化 ChatGPT 登录凭据，
避免发送前等待远程用户设置查询超时；不会删除已有登录、技能或会话记录。
`NO_PROXY` 与 `no_proxy` 保证本地 API 直连。代码或配置更新后应重启服务和 Codex，
并重新获取模型目录；旧进程与旧目录文件不会自动加载修改。


`tools/naive_n05_flash_bench.py` 默认关闭历史缓存；加 `--cache_history true` 可测前缀复用，并通过 `cached_input_tokens` 查看命中。
缓存只减少命中前缀的 prefill。恢复边界可能改变计算批量和浮点累加顺序，不保证任意边界的 logits 逐位相同。

## DSpark 推测解码

支持配套的 `Naive-N0.5-Flash-FP8-Draft` checkpoint，沿用现有命令行参数：

```bash
FT_NUMAS=1 numactl -C 0-31 -m 0 \
  ftllm server ~/hfmodels/Naive-N0.5-Flash-FP8 \
  --device cuda --moe_device numa --threads 28 \
  --draft ~/hfmodels/Naive-N0.5-Flash-FP8-Draft --draft_tokens 7
```

Draft 使用 5 层 BF16 Qwen3 骨干、1024-token 滑窗、学习的 mask embedding 和
vanilla Markov head。目标模型的 8 个中间层输出经投影成为 draft 上下文；目标模型的
embedding 和输出头共享。DSpark 从 anchor 所在的第 0 个位置开始预测，7 个位置均可
作为候选，与仅使用后续 mask 位置的 DFlash 不同。
默认置信度阈值为 0.5，可用已有的 `--speculative_dspark_confidence_threshold` 调整。

随机采样采用[标准拒绝采样](https://arxiv.org/abs/2211.17192)：候选来自完整归一化的
draft 分布 `q`，以 `min(1, p(token)/q(token))` 接受；拒绝时从归一化的
`max(p-q, 0)` 采样，全部接受则额外从目标模型分布采样一个 token。
目标和 draft 均应用温度、top-k、top-p、重复惩罚及最低生成长度限制。
贪心分布是此算法的退化情形。拒绝后的 KV 和中间特征按实际接受长度回退。

验证阶段的线性层使用与逐 token decode 相同的归约顺序，避免 BF16 舍入改变路由并
逐层放大。历史缓存同时归档 draft 的投影特征，命中后只恢复最后 1023 个位置；
请求提前结束时不把未输出的候选发布到前缀缓存。
需要工具名称/参数约束或工具正文采样的请求，以及要求返回 logits 的请求，使用逐 token
目标推理；这些请求仍可使用 draft 上下文归档和前缀缓存。

NUMA FP8 W8A8 内核按行数选择小块及打包路径，小块共享最多 8 行的权重解码，
同时复用多列的激活读取；较大块复用 8 行 × 32 列的 BF16 点积。
不均匀专家组使用现有分组任务队列并优先处理较大的组。优化覆盖不同 token 行数和尾块，
不依赖固定的候选数量。

### 复现与验证

开启 `UNIT_TEST` 后编译以下目标。配对基准在同一个进程内加载同一组目标和 draft 权重，
每题分别完整生成普通和推测回答预热，再交替测量两条路径，关闭历史缓存，记录各 token
时间、候选接受率和完整输出。生成后的代码在独立子进程执行测试。
生成结果写到仓库外的报告目录，再交给代码检查工具复验。

```bash
cmake -S . -B build-fastllm -DUNIT_TEST=ON
cmake --build build-fastllm -j16 --target naive_n05_speculative_bench \
  numas_fp8_moe_bench speculative_sampling_test naive_n05_draft_test
python3 tools/naive_n05_speculative_check.py \
  --model ~/hfmodels/Naive-N0.5-Flash-FP8 --write-cases /tmp/naive-cases.json
FT_NUMAS=1 FT_THREADS=28 numactl -C 0-31 -m 0 \
  build-fastllm/naive_n05_speculative_bench \
  ~/hfmodels/Naive-N0.5-Flash-FP8 ~/hfmodels/Naive-N0.5-Flash-FP8-Draft \
  /tmp/naive-cases.json /tmp/naive-results.json
python3 tools/naive_n05_speculative_check.py \
  --model ~/hfmodels/Naive-N0.5-Flash-FP8 \
  --report /tmp/naive-results.json --output /tmp/naive-code-checks

# 不均匀路由：7 行、64 个专家、每行选 8 个，其中 4 个为公共专家。
FT_NUMAS=1 FT_THREADS=28 FT_GPU_PREFILL=0 numactl -C 0-31 -m 0 \
  build-fastllm/numas_fp8_moe_bench 7 40 64 4

# 使用原 checkpoint 自带的 Torch 实现导出 draft 骨干参考。
python3 tools/naive_n05_draft_fixture.py \
  --draft ~/hfmodels/Naive-N0.5-Flash-FP8-Draft --output /tmp/naive-draft-fixture
build-fastllm/naive_n05_draft_test /tmp/naive-draft-fixture
ctest --test-dir build-fastllm --output-on-failure \
  -R '^(speculative_sampling|naive_n05_history|naive_n05_attention|numas_fp8_eager_moe(_avx2|_hybrid)?)$'
```

## Logits 对照

工具 `tools/naive_n05_flash_logits.py` 导出未经采样处理的 logits，按相同 token 前缀比较。
`reference` 使用 checkpoint 自带的 Transformers 模型代码和原始 FP8 eager experts；专家逐层搬到 GPU，适合显存不足以容纳整个模型的机器。
参考环境验证使用 Transformers 5.17、PyTorch 2.10.0+cu128、kernels 0.16.0 和 accelerate。

先用模型 tokenizer 生成一份输入 token ID JSON，例如 `[151644, ...]`，两个后端使用同一文件：

```bash
FT_NUMAS=1 numactl -C 0-31 -m 0 \
  python3 tools/naive_n05_flash_logits.py fastllm \
  ~/hfmodels/Naive-N0.5-Flash-FP8 --device cuda --moe_device numa \
  --input-ids input.json --output fastllm.npz --steps 3

python3 tools/naive_n05_flash_logits.py reference \
  --model ~/hfmodels/Naive-N0.5-Flash-FP8 --device cuda:0 \
  --input-ids input.json --output reference.npz --steps 3 \
  --teacher-forcing fastllm.npz

python3 tools/naive_n05_flash_logits.py compare reference.npz fastllm.npz
```

两个采集命令均可增加 `--repeat-to 2057`，验证跨过 2048 个索引候选后的稀疏注意力。
比较输出包括 MAE、RMSE、最大绝对误差、余弦相似度、概率分布 KL、top-1 和 top-10 重合数。
如果生成 token 已分叉，必须使用 `--teacher-forcing` 对齐后续输入，工具会拒绝比较不同前缀的 logits。

## 回归测试

开启 CMake `UNIT_TEST` 后：

```bash
cmake -S . -B build-fastllm -DUNIT_TEST=ON
cmake --build build-fastllm --target naive_n05_attention_test naive_n05_history_test numas_fp8_eager_moe_test -j16
ctest --test-dir build-fastllm -R 'naive_n05_(attention|history)|numas_fp8_eager_moe' --output-on-failure
```

CUDA 测试覆盖部分 RoPE、GQA、滑窗边界、attention sink、稀疏因果掩码、FP8/BF16 indexer 与分数相同时的稳定选择。
NUMA 测试以独立标量参考覆盖 W8A8 量化、FP8 次正规数、BF16 舍入、专家分组和请求规模变化，同时运行 AVX512 与 AVX2 路径。

新增的 `naive_n05_history_test` 验证真实请求初始化和缓存钩子，覆盖精确重复、短请求、
任意公共前缀、127/128 滑窗边界、跨块恢复、分支互不影响、多模态隔离、关闭清空、
命中与重复记录时的 LRU 更新，以及 CPU 恢复不等待 Forward 锁。
工具解析回归可运行 `python3 -m unittest discover -s test/toolcall -p 'test_qwen*.py'`。
Naive 思考与工具混合输出回归：
`python3 -m unittest discover -s test/api -p 'test_naive_n05_tools.py'`。


## 仅量化 MoE 专家的 NVFP4 导出

`tools/naive_n05_export_nvfp4.py` 将原始 FP8 专家反量化后，导出为 E2M1 权重、
每 16 列一个 E4M3 缩放值和每矩阵一个 FP32 缩放值。采用就近舍入，平局取偶数；
该导出未做激活校准或量化感知训练。推理使用 BF16 激活。
注意力、路由、首层稠密 MLP、词嵌入及输出头原样复制。

导出需要有 CUDA 的 PyTorch 和 safetensors，输出目录必须不存在：

```bash
python tools/naive_n05_export_nvfp4.py \
  --model ~/hfmodels/Naive-N0.5-Flash-FP8 \
  --output ~/hfmodels/Naive-N0.5-Flash-MoE-NVFP4

FT_NUMAS=2 numactl -C 0-63 -m 0,1 \
  ftllm server ~/hfmodels/Naive-N0.5-Flash-MoE-NVFP4 \
  --device cuda --moe_device numa --threads 40
```

运行时自动识别 safetensors 中的 NVFP4 格式，使用与 Qwen4 相同的紧凑 E4M3 NUMA
布局；无需设置新的环境变量。`nvfp4_export.json` 记录量化方法、采样权重误差和全部
非专家张量的 SHA256。已验证的导出含 36,096 个专家矩阵；472 个非专家张量的字节校验
全部通过。权重分片从 293.465 GiB 缩至 170.025 GiB，减少 42.1%。

导出器的小规模回归无需加载完整模型，覆盖舍入、全零专家、分块缩放和非专家
字节一致性。使用已安装 PyTorch 和 safetensors 的 Python 运行：

```bash
python -m unittest discover -s test -p test_naive_n05_export_nvfp4.py -v
```

### 八卡 CUDA 专家推理

CUDA 专家使用现有的紧凑 `NVFP4_BLOCK_16_E4M3_PACKED` 行布局：每行保存
4 字节原始 FP32 全局缩放，每 16 个权重保存 8 字节 FP4 数据和 1 字节原始 E4M3
分块缩放，行末对齐到 4 字节。保留一份 GPU 专家权重，模型文件无需重新导出。
CPU/NUMA 专家继续使用原始紧凑布局。

Gate 和 Up 各行保留自己的全局缩放，避免归一化到同一最小值造成溢出。
CUDA MoE 支持默认 BF16 和 FP16 激活，沿用原生 FP32 分块缩放路径的点积顺序、
归约、SwiGLU 和路由舍入。内核按需解码 FP8 并乘行全局缩放，不缓存展开的 FP32
分块缩放。测试用两种布局的同一权重检查输出逐 bit 一致。
8 张 32 GiB RTX 5090 可以使用普通设备映射把 48 层分配到八卡：

```bash
ftllm server ~/hfmodels/Naive-N0.5-Flash-MoE-NVFP4 \
  --device '["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4","cuda:5","cuda:6","cuda:7"]' \
  --moe_device '["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4","cuda:5","cuda:6","cuda:7"]' \
  --max_batch 1 --mtp 0 --tokens 8192 --chunked_prefill_size 512 \
  --cuda_slab 288 --cache_history false
```

这是按层分配，不是张量并行。对该 checkpoint 的 3028 亿个专家参数，紧凑专家数据
连同行全局缩放约占 158.992 GiB；相比每块保存 FP32 缩放的 12 字节布局，
减少约 52.508 GiB。
实际进程显存还包含稠密权重、KV cache、工作区和分配器开销。

`cuda_nvfp4_compact_moe_test` 用独立 FP64 参考检查原始 FP4/FP8 解码、不同 Gate/Up
全局缩放、零与最大分块缩放，以及 BF16/FP16 的多种路由批次，并与 FP32 分块
缩放布局逐 bit 对比。开启 `UNIT_TEST`
后可编译并运行：

```bash
cmake --build build-fastllm --target cuda_nvfp4_compact_moe_test -j16
ctest --test-dir build-fastllm --output-on-failure -R '^cuda_nvfp4_compact_moe$'
```


## 多卡专家辅助

长 prefill 复用 `GetNumasMoeCudaAssistDevices()`，识别 TP、`cudapp` 和 `FT_MOE_ASSIST_DEVICES` 的设备列表。
调度器根据 CPU 和各 GPU 的预计完成时间分配专家；各 GPU 使用独立 worker、stream、cuBLAS handle 和工作区锁。
保留 FP8 激活量化、BF16 舍入、独立 route 写回及按专家 ID 的归约顺序；GPU worker 失败时先等待全部任务结束，再执行完整 CPU 回退。
此路径的 decode 专家仍在 CPU，辅助卡主要用于长 prefill。

## KV 容量与数值边界

全局层按 `input_token_length + output_token_limit - 1` 预留 KV，受模型和请求上限约束，并按 128 token 对齐。
SWA 只预留窗口和当前 chunk 所需空间。预留是物理容量提示，不改变逻辑长度；无输出上限或缺少输入长度元数据时保留增量扩容。
RoPE 与 AttentionValues 按实际维度计算；Indexer 要求量化块宽为 128，不支持的布局、类型和设备明确报错。
Grouped Marlin 的工作区同时覆盖按专家分组和按 route 填充的布局，不依赖固定专家数。

## TP 候选输出与小批 MoE

普通 TP 解码和 DSpark verify 会自动在每卡选择局部候选，在 CPU 合并全局候选，
减少完整词表的回传。Greedy 保留原 CUDA Top1 的并列分数顺序；有限 Top-k 采样按
分数降序、token ID 升序。普通采样在选择前缩放温度，DSpark 使用原始分数选择，
随后沿用原概率构造、接受/拒绝采样及 KV 提交规则。

紧凑采样支持 K=2..64、每卡词表不超过 262144、全局词表不超过 2^24；这些是算子
工作区和 FP32 token ID 格式的边界，不依赖 RTX 5090、TP8 或固定模型词表。
Greedy 不受上述每卡词表大小限制。要求完整 logits、重复惩罚、最短输出长度、
词表约束或超出紧凑算子边界时，保留原完整 logits 路径。普通串行推理沿用原实现。
Graph 复用同时校验候选模式、K、温度缩放以及已有的缓存地址/容量和通信 generation。

BF16/NVFP4 小批 MoE 对 2..8 行、top-k<=16 按专家稳定分组，保留原 route ID 和
每个 token 的加权累加顺序。每个输出 tile 在一个 CTA 内完成 K 归约，避免后续行
的路由选择改变已有前缀结果。与逐行计算比较使用既有数值容差；相同路径的 eager、
Graph replay 和后缀路由变化下的前缀比较仍要求逐 bit 一致。其他数据类型和尺寸
继续走已有调度，跨设备 scratch 迁移也保留。

这些路径没有额外环境开关。`FASTLLM_TP`、`FASTLLM_DSPARK_TOKENS` 和
`FASTLLM_DSPARK_CONFIDENCE_THRESHOLD` 是现有 CLI 的有效配置接口，继续保留。

启用 `UNIT_TEST` 和 `USE_CUDA` 后，相关 CTest 为 `naive_n05_logits`、
`naive_n05_verify_selection_{2,4,8}`、`naive_n05_verify_selection_graph`（Linux）、
`naive_n05_tp_graph`（Linux）、`cuda_nvfp4_marlin_grouped_rows` 和
`cuda_nvfp4_marlin_cross_device_rows`。`speculative_sampling` 还覆盖 CPU 概率与残差采样。


## 草稿投影融合与独立 TP

草稿 Q/K/V 与 Gate/Up 在模型加载阶段分别合并，沿用 WeightMergeRule 的所有权规则，
合并成功后删除原条目。上下文 K/V 从合并 QKV 建立只读行视图，不另存合并权重缓存。
BF16 草稿支持 Q/K RMSNorm、RoPE 和 KV 写入融合；SwiGLU 保留先将 SiLU 舍入为
BF16 再相乘的数值语义。布局不满足融合条件时保留原算子路径，Graph 和 eager 均可运行。

草稿并行只使用一个配置项 `FASTLLM_DSPARK_TP`：

| 值 | 执行方式 |
|---|---|
| 不设置或 `1` | 单卡草稿，兼容原默认行为 |
| `2`、`4` 等整数 N | QKV/Attention/O 与 MLP 均按 N 卡切分 |
| `mlp:2`、`mlp:4` 等 `mlp:N` | 每卡计算完整 Attention 并持有完整 KV，只有 MLP 切分 |

配置在草稿第一次使用多卡时固定，不支持加载后切换。Target 已配置 TP 时，草稿采用其
设备列表的前 N 张卡；否则从草稿当前设备开始，按可见 CUDA 设备编号补齐。不会把
target/head 的并行度改成 N。上下文 FC 和 Markov proposal 继续在草稿首卡运行。
仅 MLP TP 会产生计算所需的 Attention 副本；分片与副本都释放父投影的原始存储，
不额外保留未分片大矩阵。

并行实现按实际设备数量处理，不限定为双卡或四卡。N 不能超过可用设备数或 target TP
设备数，MLP 中间维度必须能被 N 整除；完整 TP 还要求 Q/KV 头数能被 N 整除。
仅 MLP TP 支持单 KV 头。当前路径要求 CUDA embedding、BF16 稠密投影、peer access，
不支持 low-memory 模式；head_dim 为 4 的倍数且不超过 256，草稿 block 在 2..31 范围。
不符合要求时明确报错，不在已分片权重上静默退回单卡。

多卡上下文投影使用逐层 K/V 行视图和 batched GEMM；工作区只保留激活与小指针表，
不缓存权重副本。草稿拥有独立通信组、持久工作线程和 Executor，不替换 target 的通信组。
所有从卡提交完成后首卡才允许复用输入存储。Graph 捕获失败时统一释放各卡 graph 并回退
到相同 TP 策略的 eager 路径。未为上下文更新新增 CUDA Graph。

Linux CUDA CTest 中，`naive_n05_draft_fusion` 与 `naive_n05_draft_kv` 覆盖融合数值、
布局拒绝、请求复用及跨窗口更新；`naive_n05_draft_tp_*`、`naive_n05_draft_mlp_tp_*`
覆盖完整/仅 MLP TP、MQA、2/4/8 卡、Graph/eager、捕获失败回退和非零首卡。
TP 改变 GEMM 与跨卡归约顺序，跨策略使用既有数值容差；同一策略 Graph/eager 要求逐位一致。

草稿并行度需按实际硬件和请求测量；默认不自动提高，不承诺线性加速。


## 普通 TP 解码合批

未配置 DSpark 草稿模型时，Naive TP 支持服务端按 `--max_batch` 合并活跃请求。
例如保留现有 TP8 启动参数，设置 `--max_batch 2 --mtp 0`，并移除
`--speculative_algorithm`、`--draft` 等草稿参数。`--mtp 0` 本身不会关闭显式启用的 DSpark。

多个请求共享一次 TP worker 派发，QKV、输出投影、路由、稠密 MLP 与词表投影按多行执行；
Attention/Indexer 使用各请求独立的 KV、位置及长度。为保持普通解码的专家归约顺序，
MoE 专家投影暂按请求执行，其余投影和层间通信仍按多行执行。权重沿用原有分片，不复制 KV
或增加合并权重缓存。采样配置、最短输出限制与历史 token 仍按请求分别处理。
当前服务调度器仍逐请求 prefill，再对活跃请求合批 decode；指针式 ForwardBatch
也接受按请求顺序打包、无 padding 的变长输入。

合批 Graph 状态与单请求 decode、DSpark verify 分开。满足已有 Graph 后端约束时，
最多八个单 token 请求复用每卡一个 Graph；请求顺序、数量、KV 地址/容量、
Attention 区间或候选选择配置改变时重新检查。更大 batch、prefill 和 Graph 捕获失败
使用同一合批前向的 eager 路径。没有新增环境变量，融合和 Graph 仍遵循现有设置。

本路径仅为普通 CUDA TP 开启合批能力。草稿解码和非 TP 模式仍使用现有单请求路径；
它不是两个独立 GPU stream 同时跑两份模型。多行投影共享权重读取，具体吞吐收益
取决于各请求的专家重合度、上下文长度与硬件，需要端到端测量。

### TP 跨请求历史前缀复用

Naive CUDA TP 支持 `--cache_history true`，普通解码、普通合批和 DSpark 共用已有的
CPU 历史归档。按完整 token 前缀匹配，可复用重复、追加、缩短和中途分叉的请求。
SWA 归档保留已移出活动窗口的历史行，DSA 包含 index key，DSpark 同时保存已提交
hidden features；未接受的草稿以及结束/取消后未发出的后缀不会作为命中前缀发布。

归档只保存一份逻辑 KV；TP 恢复时重新分片，并向共享 KV 头的 rank 复制对应头。
请求创建仅恢复私有 CPU 状态，首次前向由各 rank 并行上传 CUDA，避免与其他请求的前向工作区竞争。
单个存活请求可复用上一请求的空闲 GPU 分配，恢复内容后再执行；有其他存活请求时不转移 KV 所有权。
Graph replay 完成后记录新行，每卡批量暂存后统一下载，不将 CPU 下载放入 Graph 捕获，保留已有融合路径。
需要单 token target 前向并收集草稿 hidden states 时，也可使用单行 Graph；
恢复完草稿上下文后直接开始推测的请求继续走原有 Verify Graph。
缓存使用上述共享主机预算，最多保留 5 条已完成记录，不新增配置环境变量。
`--prefix_cache` 的通用 paged-cache 开关不代替这个模型的 `--cache_history`。

历史归档及恢复包含 CPU/GPU 传输，有固定开销；命中 token 数不等于端到端收益。
短前缀可能不比重新 prefill 更快，应按实际上下文和命中率测量首 token 延迟及生成速度。
