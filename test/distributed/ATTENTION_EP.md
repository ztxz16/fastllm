# Attention DP + EP：阶段 3～6

实现范围是本机 `qwen2_moe`、FP16 权重/激活/KV、`attention_dp=ep=1/2/4`、`dp=tp=1`。
真实 Qwen1.5-MoE-A2.7B-Chat 使用四张 V100；单卡/双卡只验收缩小模型。

代码已接入；本地禁止编译，GPU 功能和性能必须以服务器本次报告为准。
下面的命令由用户在 **Vast 的 Linux 终端**执行，不在 Windows PowerShell 执行。

2026-11-17起，真实模型的默认`real`改为下文“生成级HF对齐”，默认Top-5、
候选门槛容差0.1。此前全量logits和逐层/路由的严格检查保留为`real-numeric`，
阈值与历史FAIL不改写。本文历史实验中的“阈值保持不变”仍指严格数值诊断。
上述生成级HF对齐调整只涉及测试及报告，不代表真实GPU验收已经通过。
需要一次查看四项结论时使用`real-all`：exact、严格Top-5和Top-5+0.1共用一次生成矩阵，
随后独立运行numeric。旧的单策略入口及默认值保持不变。

## 批次不变性计划：先修复 MoE，再逐步验证全链路（实施中，待 GPU 验收）

2026-11-19 用户确认实施顺序：**先解决已定位的 MoE 差异，再逐步完成全链路
不变性验证。** 全链路是最终验证范围，不意味着首轮就替换所有计算环节。

1. 优先处理已定位的专家 Gate/Up、Down 矩阵乘差异，使用相同输入验证跨批次
   结果及数值正确性，再通过正式推理入口复验原始并发失败用例。
2. 在局部修复基础上逐步验证路由、共享专家、Q/K/V、Attention、输出投影和
   最终词表投影等环节。已有实现满足不变性时继续复用；发现差异并定位后再修改。
3. 保留完整生成 Token 序列一致的最终目标，逐步覆盖并发 1/4/8、不同准入顺序、
   Eager/Graph 和前缀缓存关闭、冷、热状态。局部通过不代表全链路验收通过。

实施约束：复用现有 CUDA Graph 捕获与回放流程。新的 MoE 专家 GEMM 应兼容现有
缓冲区生命周期、设备端路由和调用结构；仅在发现具体不兼容时做必要调整。

### 已批准的计算与接口计划

- 首版限定 V100（SM70）、Qwen1.5-MoE-A2.7B-Chat 的 FP16 执行路径。可选模式默认
  关闭；相同硬件、程序版本、权重、EP 拓扑及采样设置下，要求完整生成跨批次一致。
  一致性为硬门槛；性能实测并报告，不预设退化上限。HF 对齐独立判定，不要求复制旧路径输出。
- 专家 GEMM 采用 CUTLASS SM70：FP16 输入/权重、FP32 累加、FP16/FP32 输出；候选
  线程块 64×128×32、warp 32×64×32、指令块 8×8×4、两阶段流水、Split-K=1。
  这只是待验证的起点。不同 M 使用相同配置，每个输出块完成全部 K，边界用掩码处理。
  支持连续且满足对齐条件的矩阵、单列输出，以及最终类型转换前加入 bias。
- 首阶段只接入专家 Gate/Up、Down，保持布局、激活和舍入边界。其他投影和 Attention、
  路由、有序专家求和先复用并验证；若全链路仍有差异，继续定位后修复，不能把局部通过
  当作全链路完成。首版不做在线调优、按 M 切换算法或 grouped GEMM，不改第三方源码。
- 正式入口提供 `--batch-invariant`，默认关闭，沿 Python 会话、协调器、worker 传至
  执行器。测试入口使用 `ATTENTION_EP_BATCH_INVARIANT=1`，不能另走测试专用后端。
- 模式在进程组创建时固定，核对各 worker 实际模式及配置。缓存继续归属进程组，切换模式
  时重建进程组及缓存。不支持的硬件、模型、精度、布局或资源明确拒绝；失败沿用既有
  进程组清理，不静默切换到未经批次不变性验证的路径。关闭模式保留原有计算。

### 验证与交付计划

1. 从保存的正式专家输入验证共同输入行在不同行数、排列、分块位置、尾部和空专家下
   逐位一致，并用独立高精度参考检查正确性；覆盖 bias、两种输出类型、非法输入与拒绝路径。
   缺失快照时通过正式诊断入口重采，不能以随机小矩阵代替真实差异的复验。
2. 完整推理固定 Prompt Token ID、贪心采样、128 个输出 Token 和 EOS 策略，逐步覆盖
   输入长度 512/1024/1792、并发 1/4/8、Eager/Graph、缓存关闭/冷/热、同时准入、
   两路后两路及混合 Prefill/Decode，每组合至少 5 轮，与同模式单请求基准比较。
3. 保留第 79 个 Token 首差复现，记录 Token、logits、路由、rank、实际路径；检查首次
   Graph 捕获、回放、空闲 rank、缓存清理和资源释放，失败时保存部分汇总。
4. 性能独立于 HF 对齐，统一 warmup、迭代、同步及缓存条件，至少 5 轮，记录 Median、
   P95、CV、吞吐、显存和原始样本。旧/新路径生成历史不同则不直接计算加速比。
5. 报告包括 JSON、Markdown、CSV、XLSX，以及环境、Git 提交、完整命令和实际后端。
   任一验收失败返回非零；HF 历史失败独立保留。仅在正式推理矩阵通过、实际性能代价报告
   齐备后验收完成。本地不编译；编译与 GPU 验证在服务器执行。

### 当前代码状态与服务器验证

第一阶段候选已接入：只替换专家 Gate/Up、Down，使用固定 CUTLASS 配置；Decode 在原
工作区上按设备计数聚合有效行、掩码填零及写回，沿用既有 Graph 捕获/回放、路由和清理。
当前未替换 router、共享专家、Q/K/V、Attention 输出投影或词表投影。
启动回执的 `scope=moe_experts` 表示实现范围，`full_model_validated=false` 表示它不是验收
结论。请求状态中的 `expert_gemm_submissions` 记录实际主机提交（包括捕获，不冒充回放
kernel 次数），`expert_gemm_backend` 用于核验进入目标路径。

本地仅进行 Python 静态/CPU 回归和差异检查，尚未编译、运行 GPU 或测量性能。
后续必须根据服务器结果推进：专家失败先修专家；专家通过后检查完整生成，剩余差异由
逐层诊断定位，不预先替换全部投影。现阶段不能宣称完整批次不变性已经实现并验收。

将本次代码同步到服务器后，在原 V100 环境设置下先执行以下两步；第二步是独立算子
验证/微基准，调用正式 GEMM 实现，但不能代替完整推理性能结果：

```bash
bash test/distributed/run_attention_ep_v100.sh build
bash test/distributed/run_attention_ep_v100.sh invariant-gemm
```

`invariant-gemm` 默认验证边界用例后，通过正式观察器重采旧模式两/四请求的第0层专家
快照并复验差异。若保留原 `moe_block_batches` 结果，可设置 `ATTENTION_EP_MOE_BASELINE`
为该结果目录，复用已保存输入并检查模型来源。真实矩阵共同输入行必须逐位一致；独立
FP64 参考检查固定抽样的 8 行全部输出列，抽样索引写入报告，边界小矩阵则全量检查。
单独运行 `run_sm70_expert_gemm.py --synthetic-only` 仅代表边界验证，不能作为真实差异已修复。

算子检查通过后，分别运行下列入口；入口之间不会自动串联，也不附带 HF 对齐：

```bash
# 正式会话完整生成矩阵：512/1024/1792，1/4/8 并发，Graph/缓存/准入组合，每组合 5 轮。
bash test/distributed/run_attention_ep_v100.sh invariant

# 原第79个Token相关的固定准入诊断；保持正式原生执行，不能替代公开队列验收。
bash test/distributed/run_attention_ep_v100.sh invariant-admission

# 完整生成仍有差异时，采集候选的逐层及第0层 MoE 内部快照定位首差。
bash test/distributed/run_attention_ep_v100.sh invariant-trace

# 独立端到端性能对照：旧路径/候选分别运行，输出历史与缓存工作量一致才计算加速比。
bash test/distributed/run_attention_ep_v100.sh invariant-performance
```

`ATTENTION_EP_INVARIANT_PROMPT_LENGTHS=1024` 可定向复验完整生成，报告会标明未覆盖全部计划
矩阵。`invariant-performance` 可用既有 `ATTENTION_EP_PERF_PROMPT_LENGTHS`、
`ATTENTION_EP_PERF_OUTPUT_TOKENS`、`ATTENTION_EP_PERF_CACHE_STATES`、
`ATTENTION_EP_PERF_CONCURRENCIES` 缩小性能矩阵。普通 `performance`、`http` 和生成对齐入口
可通过 `ATTENTION_EP_BATCH_INVARIANT=1` 使用候选；未设置时仍使用旧路径。

每次新建结果目录并生成 JSON、Markdown、CSV、XLSX。任一失败或本次选定矩阵未执行完整
返回非零，保留已完成报告。主动缩小矩阵可得到局部 PASS，但报告会标记
`full_planned_matrix=false`，不能当作完整验收。旧路径已有失败不能被候选 PASS 覆盖；
完整正式推理矩阵与实际性能报告尚待服务器。

## Attention DP/EP 后续优化：保持批次不变性的 Split-K（待实现）

2026-11-19 记录：针对 Attention DP/EP 的 MoE 专家 GEMM，后续评估固定 Split-K，
在 Token 行数较少、K 维较长时增加 GPU 线程块并行度。当前仅记录优化方向，尚未实现、
启用或取得性能结果。计划中的 V100 / CUTLASS SM70 可选严格模式首版先不启用 Split-K，
用于建立可验证的基线；这不是永久禁用 Split-K，也不是批次不变性的必要条件。

Split-K 将同一个输出块的 K 维交给多个线程块计算部分结果，再合并这些结果。
它与单个线程块内部按固定 K 步长循环、使用两阶段流水是不同的机制。
后续候选必须满足以下约束：

- 同一设备、权重形状和数据类型使用固定的拆分数量、K 边界及计算配置，不随输入
  Token 行数、请求准入顺序或专家路由负载改变。
- 固定部分结果的累加精度、存储精度和最终合并顺序；候选采用 FP32 部分和，
  不使用依赖线程块完成先后顺序的浮点原子累加。bias、类型转换及激活的舍入边界
  必须明确，不能因切换 Prefill、Decode 或 Eager/Graph 而变化。
- 工作区计入正式推理预算，Graph 捕获前完成分配；严格模式资源不足或执行失败时
  明确报错，不能因本次批次大小静默切换另一种归约方式。
- 先用相同输入验证共同 Token 在不同行数、排列、分块边界和 padding 下逐位一致，
  并独立校验数值正确性；再通过正式入口验证并发 1/4/8、混合 Prefill/Decode、
  Eager/Graph 及前缀缓存冷/热下的完整生成一致性。不同归约方案之间的数值误差
  单独记录，不能将两种方案之间允许的误差用于放宽各自的跨批次一致性要求。
- 与无 Split-K 基线公平比较，计入部分和写回、合并 kernel 和工作区开销；独立微基准
  与正式推理性能分别报告。统一输入、精度、warmup、迭代、同步和缓存条件，至少
  5 轮，记录 Median/P95 延迟、CV、吞吐、显存、实际路径及原始样本。生成历史不同
  时不直接计算端到端加速比；通过一致性验收并证实收益后，再考虑纳入严格模式。

## 实现与生命周期

- Python 单个调度线程负责队列和 rank 准入，一个原生协调器驱动整个 EP 组。
  每 rank 最多一个活动请求；活动、空闲、开始、结束都进入同一通信轮次。
- 混合 Prefill/Decode 使用 Eager。没有新 Prefill 且存在 Decode 的轮次整体使用 Graph。
  GPU 元数据包含历史长度、位置和活动掩码；空闲 rank 继续执行专家通信。
- MoE Graph 固定地址跨请求保留。请求结束清空活动状态，模型关闭释放 Graph 与 KV 预留。
  每 rank 预留 2048 Token KV；活动请求字节数和预留字节数分别报告。
- KV、前缀快照、工作区各有 512 MiB 上限；工作区预算包括 Graph 与临时缓冲的总和。
  前缀按 rank 独立存储，最多 8 条，LRU 淘汰。缓存清理暂停准入并等待活动请求结束。
- 原生采样保留种子、temperature、top-k/top-p 语义。EOS、取消与超时在轮次边界释放 rank。
  每请求 token 队列上限 64；消费者落后导致该请求取消，不阻塞其他 rank。
- 任意 worker 故障使全组不可复用，回收整个进程树；不自动重试已执行请求。
- MoE专家输出与路由权重相乘后先转FP16，每个Top-K槽位单独保存，通过一次FP16
  ReduceScatter返回请求所属rank；每槽仅一个专家owner写入，其他rank为零。
  返回后按全局专家ID升序逐次FP16相加，保留HF `index_add_` 的逐专家舍入边界。
  Prefill、Decode和Graph共用该合并路径，Graph每次在设备读取并排序本轮专家ID。
  禁止先按rank累加再规约，也不能按Top-K槽位顺序累加；两者都会改变数值语义。
  诊断中的routed_output仍传FP32，但其值为最终FP16累加值的无损提升。
  贡献缓冲为FP16 `[capacity*ep,topk,hidden]`，接收缓冲为 `[capacity,topk,hidden]`，
  继续计入512 MiB预算。相对旧FP32聚合通信，每Token元素载荷由 `4*hidden` 增为
  `2*topk*hidden` 字节（Top-K=4时为两倍）；尚未测量性能，不能据此宣称优化吞吐。
- MoE Attention保留HF FP16 Eager的中间张量精度：QK点积先转FP16，缩放后再转FP16，
  softmax以FP32计算后转FP16再与V相乘。单层、Eager及Graph使用同一路径；dense分支保持原语义。
- MoE一般Linear使用`CUBLAS_COMPUTE_32F + CUBLAS_GEMM_DEFAULT_TENSOR_OP`，覆盖
  无偏置QKV、O投影、Router、使用cuBLAS的专家投影和LM Head；沿用现有输出类型及舍入边界。
  Dense Qwen3仍使用PEDANTIC。该选择与PyTorch 2.7.1常规FP16 GEMM配置对齐，
  不保证不同矩阵形状、融合布局或自定义Decode kernel与HF逐位一致，必须重跑完整模型验收。
- MoE有偏置QKV采用三次独立`cublasLtMatmul`：FP16输入/输出、FP32计算、`BIAS` epilogue、
  `beta=0`和1 MiB工作区。Q/K/V分别使用原始投影宽度及连续输出，之后仅拷贝到原Attention布局。
  合并权重不变；偏置地址每层更新，算法按形状和实际地址对齐复用。对齐偏好与
  [PyTorch 2.7.1 gemm_and_bias](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CUDABlas.cpp#L1189)一致。
  临时与Graph资源分别管理；warmup完成算法选择，捕获期间只提交已有计划，缺计划或Lt失败
  直接报层号/投影/形状，不fallback。新增投影缓冲和Lt工作区均计入原有512 MiB总预算。
  本改动必须重新编译，CMake显式链接同一CUDA工具链的cuBLASLt。

20261115T035744Z的独立QKV诊断在真实层13、4、7的同一HF输入上，只有`lt_separate_qkv`
同时得到Q/K/V、Attention输出、下游router logits与专家选择的零差异；合并Lt、分离GemmEx、
合并PyTorch Linear与FP32输出后转FP16均有差异。该证据定位了本轮修复，仍不代表原生完整层、
完整模型或Graph/Prefix通过；独立诊断中的其他计算使用HF操作，正式验收阈值保持不变。

20261115T153521Z的完整前向独立实验通过归因门禁：仅将HF专家累加换成FP32后，
与保存的原生logits比较由18个超差变成0个，但候选仍有分层超差，且相对原HF有16个
logits超差。因此该实验用于定位正式累加语义，不能替换HF验收参考。上述正式合并修复
仍须重新编译并执行synthetic、regression和real；真实阈值继续为atol=0.05、rtol=0.01。
synthetic新增 `expert_accumulation`：加权贡献按专家ID为 `[2048,1,-2048,1]`，正确
FP16串行结果为1，FP32或Top-K顺序 `[2,0,3,1]` 均为2。该用例通过正式单层及完整
生成入口覆盖EP=1/2/4、空闲rank、Prefill/Decode、Graph/Prefix及冷热复用。

## 服务器执行顺序

本地修复推送到 `sk`，由用户在远端合并到 `master` 后，服务器再更新 `master`。
保留服务器自己的未提交改动；如 Git 拒绝更新，先贴出报错，不执行强制重置。

```bash
cd /home/project/vllm_fastllm
git status --short
git fetch origin
git switch master
git pull --ff-only origin master
```

固定环境并编译。本脚本使用新构建目录，避免与 CUDA 13 或其他旧 CMake 缓存混用。

```bash
export ATTENTION_EP_PYTHON=/venv/main/bin/python3
export ATTENTION_EP_REFERENCE_PYTHON=/venv/fastllm-reference-cu126/bin/python3
export ATTENTION_EP_CUDA_HOME=/usr/local/cuda-12.6
export ATTENTION_EP_BUILD=build-attention-ep-cu126
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash test/distributed/run_attention_ep_v100.sh build
```

构建成功后，按顺序执行。任一步失败先停止并保留报告路径。

```bash
bash test/distributed/run_attention_ep_v100.sh synthetic
bash test/distributed/run_attention_ep_v100.sh regression
bash test/distributed/run_attention_ep_v100.sh real
bash test/distributed/run_attention_ep_v100.sh http
bash test/distributed/run_attention_ep_v100.sh performance
```

默认模型路径 `/home/models/Qwen1.5-MoE-A2.7B-Chat`，可用 `ATTENTION_EP_MODEL` 覆盖。
真实模型参考在独立进程加载 Transformers FP16 权重，按层分布至四卡；不会与 FastLLM 同时驻留。
参考进程需要已安装支持该模型的 PyTorch、Transformers、Accelerate；脚本不安装、不下载。
如另有参考虚拟环境，设置 `ATTENTION_EP_REFERENCE_PYTHON=/该环境/bin/python3`。
`synthetic` 阶段的 Python 回归检查也优先使用该参考环境，以满足其中 CPU PyTorch
参考用例的依赖；变量未设置或为空时使用主 Python 环境。后续 GPU 验收仍使用主环境。
参考模型按连续层显式分配到四卡，避免自动放置把权重全部放在一张 32 GiB 卡上。
Embedding与RoPE在GPU 0，最终Norm在GPU 3，绑定的LM Head与Embedding同卡。
设备映射不包含根模块兜底，首次前向前检查全部参数、buffer和子模块执行hook是否同卡。
服务器参考环境当前为 `/venv/fastllm-reference-cu126/bin/python3`：PyTorch 2.7.1+cu126、
Transformers 4.57.6、Accelerate 1.12.0。主环境仅用于分词，不能用其Transformers版本替代参考环境。
`reference.log` 记录设备检查和逐历史进度；参考失败时终端显示退出码与日志路径，
`setup.traceback.log` 保存主验收脚本的异常栈。仅修改这些Python脚本时无需重新编译原生产物。

`regression` 会在独立结果目录生成稠密小模型和 tokenizer，然后调用原有 DP2×TP1、DP4×TP1、
DP2×TP2 验收以及 TP Decode Eager/Graph 验收，不需要另下载稠密模型。

## 生成级HF对齐（2026-11-17）

`real`默认采用`--alignment-policy topk --alignment-topk 5 --near-tie-tol 0.1`。
规则参考vLLM的[生成模型测试说明](https://docs.vllm.ai/en/latest/contributing/model/tests/)
及`tests/models/utils.py`中生成Token的双向候选检查。上游已查阅文件的Git blob为
`0bec9ca9bec88f10d61770d821143f3528dbfdfe`，通用测试文件blob为
`07585525e9151e815106ea27bc2b4c55cfee0075`；这不是所有vLLM模型统一的精度保证。

- 同一份本地模型、相同Prompt Token ID、FP16权重/KV、贪心、无采样过滤或惩罚、
  相同每请求生成长度、禁用EOS提前停止。HF直接使用保存的Token ID，不重新套聊天模板。
- HF通过公开`generate`独立生成，动态KV、Eager、逐请求运行；FT通过正式会话独立生成。
  显式设置`use_model_defaults=False`，防止模型保存的采样/惩罚默认值覆盖验收参数。
  主比较不使用教师强制。两个后端分开驻留，参考进程不安装MoE观察器或更改HF计算。
- `exact`要求完整生成Token和解码文本均一致；`topk`默认在首次Token分歧处双向检查，
  每个选中Token须在对方Top-K，或满足`本侧logprob >= 对方Top-K末位logprob - 容差`。
  容差为0时只允许集合互包含。首次分歧之后不沿不同历史比较logits。
- 这个跨侧门槛不约束同一Token两侧的logprob差值，也不证明只是微小浮点扰动；
  两边选词明显不同时仍可能通过。因此候选级PASS不能标成全量数值对齐或严格输出一致。
  判定器双方使用相同的Python float64稳定log-softmax，Top-K同分按Token ID升序，
  不宣称逐位复刻上游GPU归一化/Top-K的同分次序。保存原始logits供复算。
- 共同配置单独标记`shared_eager`：FT/HF均Eager、无Prefix、串行请求；FT的
  Graph、Prefix及并发配置单独标记`ft_production_vs_hf_eager`，仍核对实际路径和命中。
  Prefix路由检查当前请求实际复用的Token数，不能用历史累计命中替代本次命中。
  后者是生产配置的输出验收，不是相同运行模式的速度对比。
- 每行报告记录选定策略、`exact_match`、`candidate_match`、首次分歧、比较步数与
  `uncompared_steps`。`failed_elements`表示当前生成门禁失败；同历史全词表的
  max/mean误差、`logit_failed_elements`与`logit_diagnostic_status`另外保留，绝不清零。
  数值超差在报告和终端中只提示WARNING，不影响生成验收判定；终端保留最大/平均误差
  与超差元素数。任一生成case或执行路径失败仍返回非零。
- 启动前登记预期用例数，只有全部完成且通过才汇总为PASS；中途报告包含`complete=false`。
  对齐与性能分别登记、分别汇总；对齐报告不包含性能用例。
- 严格数值验收独立运行，继续检查教师强制完整历史的每步logits与实际专家集合，
  失败仍返回非零。合成小模型、算子、生命周期以及性能公平性检查不放宽。

保持前文参考环境设置，使用已有原生产物即可；这些命令不会编译：

```bash
# 默认候选级验收：Top-5，边界容差0.1
bash test/distributed/run_attention_ep_v100.sh real
# 严格Token和文本一致
bash test/distributed/run_attention_ep_v100.sh real-exact
# 只允许互在Top-5，不启用额外边界容差
ATTENTION_EP_NEAR_TIE_TOL=0 bash test/distributed/run_attention_ep_v100.sh real
# 原来的逐层/路由/全量logits严格数值诊断
bash test/distributed/run_attention_ep_v100.sh real-numeric
# 一套自由生成同时判定三种规则，再单独运行numeric；不编译
bash test/distributed/run_attention_ep_v100.sh real-all
```

各命令新建独立目录并生成JSON/Markdown/CSV/XLSX。历史报告继续按原规则解释，
不得把旧FAIL改写成PASS；新规则需重新运行并标明策略后才能得出新结论。

### 独立性能测试

`performance`只运行性能矩阵，不运行HF生成对齐、numeric或完整合成验收，也不读取
对齐请求集或使用参考Python环境。`real`、`real-exact`、`real-numeric`和`real-all`
继续独立进行对齐；`http`只验证服务功能。已经完成的阶段无需因测速再次执行。

```bash
bash test/distributed/run_attention_ep_v100.sh performance
```

默认性能矩阵为Eager/Graph × Prefix off/cold/hot × 并发1/4/8，共18项。保持原有
32个Prompt Token、16个输出Token、禁用EOS、一次warmup和5轮计时；hot模式另在
计时前提交4个请求准备缓存。输出历史、缓存复用及Graph路径检查在计时区外执行，
检查失败仍返回非零。加速比是同配置Eager耗时/当前模式耗时，不是相对HF或vLLM。

默认每个输入长度单独登记18项，保存原始样本及JSON/Markdown/CSV/XLSX汇总；中途停止保留
部分结果并标记未完成。目录仍使用`时间戳_moe_full_随机号/`，不会覆盖历史结果。
直接使用Python入口的`--performance`具有相同语义；省略`--model`时只准备一个
FP16合成小模型供测速，不附带完整合成验收。非默认对齐策略或候选阈值不能与测速混用。

长上下文使用同一个性能入口，显式设置输入长度列表及固定输出长度：

```bash
ATTENTION_EP_PERF_PROMPT_LENGTHS="512 1024 1792" \
ATTENTION_EP_PERF_OUTPUT_TOKENS=128 \
bash test/distributed/run_attention_ep_v100.sh performance
```

该命令运行54项，每项5轮，仅做长上下文性能与自身输出/路径检查，不自动运行持续压测
或HF对齐。仍使用固定的`1,2,3,4`循环Token序列控制请求形状和缓存；实际权重由
`ATTENTION_EP_MODEL`指定。输入长度、输出长度及原始Token记录在报告中；吞吐量按
实际配置的输出长度计算，Eager/Graph只比较同长度、同缓存和同并发的结果。
非默认长度的case名称包含`prompt长度_output长度`，目录命名与历史结果保持一致。
Python入口对应`--performance-prompt-lengths 512 1024 1792 --performance-output-tokens 128`。
输入长度必须为正且不重复，输出至少2个Token；输入长度+输出长度-1必须不超过正式
入口的2048容量限制。未满足这些条件会在加载模型前拒绝。

若性能用例提示`benchmark modes used different output histories`，比较的是FastLLM
Eager与Graph的实际生成Token，不是HF对齐，也不能直接认定为可接受的近似同分。
失败用例仍保存`case名称.json`：包含已完成轮次的原始Token、每Token执行路径、rank、
耗时样本及错误原因；`comparison.first_divergence`定位轮次/提交序号/Token位置（均从1开始）。
`eager_reference_file`指向同配置Eager证据，`output_variant_count`和
`within_mode_vs_first_request`观察相同请求在本模式内是否产生不同输出，仅作诊断。
分歧后的不同Token位置也计入数量，不代表逐位置同历史数值差异。
原始证据在计时结束或异常清理后落盘，不启用计时区内trace；校验失败仍返回非零，
不输出该失败case的有效加速比。旧版本未保存的失败输出无法事后恢复，原FAIL报告保留。

可先定向复测一组配置，自动包含对应的Eager与Graph各一项、各5轮：

```bash
ATTENTION_EP_PERF_PROMPT_LENGTHS=1024 \
ATTENTION_EP_PERF_OUTPUT_TOKENS=128 \
ATTENTION_EP_PERF_CACHE_STATES=off \
ATTENTION_EP_PERF_CONCURRENCIES=4 \
bash test/distributed/run_attention_ep_v100.sh performance
```

缓存筛选默认`off cold hot`，并发筛选默认`1 4 8`；Python入口对应
`--performance-cache-states off --performance-concurrencies 4`。报告登记所选范围和预期
用例数，`performance_matrix_complete=false`表示仅覆盖子集；子集PASS不能替代完整矩阵，
复测未重现也不能覆盖历史失败。复测保持原请求、调度、缓存和路径规则，不自动重试。

### 并发输出波动的独立诊断

若Eager和Graph各自重复运行也出现不同Token序列，不能只凭跨模式FAIL归因于Graph。
下面入口单独调用正式会话，沿用性能case的循环Prompt Token、禁用EOS、贪心采样、
关闭Prefix Cache、轮转路由和线程池并发提交方式；不运行HF、教师强制或性能计时：

```bash
"$ATTENTION_EP_PYTHON" test/distributed/probe_qwen_moe_concurrency.py \
  --build "$ATTENTION_EP_BUILD" --model "$ATTENTION_EP_MODEL" \
  --prompt-length 1024 --output-tokens 128 --concurrency 4 \
  --repeats 5 --trace-tokens 78 79 80
```

每种模式独立加载一次模型、warmup一个并发批次，再运行5轮。选择性trace只在所列
1-based输出位置取回完整logits，各活动轮次仍记录实际联合动作、提交Token数、
请求ID、输出位置和输入历史长度。原生协议已有trace开关，本次Python诊断不要求重编译。
诊断取回、Python记录和落盘可能影响运行时序；相同准入代码不保证复现原执行时间表。

结果目录为`时间戳_moe_concurrency_probe_随机号/`，包含：

- `eager/`与`graph/`中的启动、每请求原始Token/选择性logits/完整调度及关闭记录。
- `*_vs_first.json`：各请求与本模式首个正式请求比较；`*_vs_eager.json`：Graph与同轮次、
  同提交序号Eager比较。不同输出变体数量相同不豁免逐请求失败。
- `summary.json/md/csv/xlsx`：默认60项比较，失败及中断仍保存部分汇总并返回非零。
  `failed_elements`表示不同Token位置数量；max/mean绝对误差仅覆盖同输入历史的已采集logits，
  不设新的数值阈值，分叉后的logits明确跳过数值比较。候选分数与分差只作为诊断证据。

首次Token分歧在采集窗口内时打印双方Top-5、选词分差及执行/调度证据；不在窗口内则
明确报告未采集，下一次调整`--trace-tokens`，不推断为近似同分。`OBSERVED_TOKEN_DIVERGENCE`
表示本次观察到输出差异；`NOT_OBSERVED`仅表示本次未观察到，不能宣称原问题已修复。
V编号只在本次运行内比较。换服务器后先检查GPU/CPU、编译缓存和二进制SHA256等元数据，
新机器上的诊断不能代替旧机器的故障复现。本报告不输出吞吐或加速比，不覆盖历史验收结果。

### 固定准入轮次的归因诊断

若额外Eager步骤与输出变体相关，但随机入队难以稳定复现，可固定联合轮次：

```bash
"$ATTENTION_EP_PYTHON" test/distributed/probe_qwen_moe_admission.py \
  --build "$ATTENTION_EP_BUILD" --model "$ATTENTION_EP_MODEL" \
  --prompt-length 1024 --output-tokens 128 --first-wave 4 2 \
  --repeats 5 --trace-tokens 2 79
```

`first-wave=4`让四rank同时Begin；`first-wave=2`让rank 0、1先Begin，下一轮
它们Decode而rank 2、3 Begin。也可显式选择`--first-wave 4 1 2 3`补充1+3和3+1。
这是直接使用正式`RoundTransport`协议的独立归因诊断，仅规定准入轮次，不运行Python
异步准入队列，不能作为正式队列/HTTP/性能验收。原生前向、算子选择、KV生命周期、
通信同步、Graph回退和实际采样沿用正式实现；每步输入实际采样Token，无教师强制。
诊断不改变生产调度、不合入性能入口、不要求重编译，也不自动放宽原有失败标准。

每种模式/组合独立加载，warmup一批，再执行5批四请求。默认共80个请求比较项，分别
比较同组合/同rank首次结果、同模式全4准入结果、Graph对应的Eager结果；全4准入还
比较同批rank 0，避免漏掉rank之间的差异。完整logits只在指定位置保存，数值误差只
比较相同输入历史；第79步分歧之前的logits差异只能缩小定位范围，不能单独证明根因。

结果保存在`时间戳_moe_admission_probe_随机号/`：每批含`rounds.jsonl`与`rank0.json`
等实际输出，根目录含逐请求比较证据和`summary.json/md/csv/xlsx`。任一Token差异、
缺失logits、错误路径、Graph计数不符或KV未释放均返回非零；中断保留部分结果。
`NOT_OBSERVED`仅表示本次固定组合未复现，不会覆盖历史性能FAIL。

### Prefill组合的逐层定位

若固定准入报告已采集第1步logits，且两请求与四请求在Prefill结束时已不同，使用
`probe_qwen_moe_prefill_batches.py`比较rank 0、1的完整Prompt中间张量。该入口只比较
首轮`[Begin,Begin,Idle,Idle]`与`[Begin,Begin,Begin,Begin]`，不包含后加入的rank 2、3，
也不包含混合Decode轮次；不能把结果推广为后加入请求的完整根因。

需要先在GPU服务器重编译worker；保留已有CMake缓存、CPU/CUDA架构与依赖配置：

```bash
cmake --build "$ATTENTION_EP_BUILD" --target fastllm_process_tp_package -j 4
"$ATTENTION_EP_PYTHON" test/distributed/probe_qwen_moe_prefill_batches.py \
  --build "$ATTENTION_EP_BUILD" --model "$ATTENTION_EP_MODEL" \
  --baseline /绝对路径/时间戳_moe_admission_probe_随机号
```

每组合独立加载模型，warmup一次Prefill，再采集一次Prefill；两次都End并核验活动KV
释放。Prompt、采样参数、capacity沿用基准报告，后续128步生成不执行。每个活动rank的
采集logits和首Token必须精确复现基准`eager_firstN_thenM/repeat01/rankR.json`；不一致时
报告`trace_attribution_valid=false`并拒绝逐层归因，不调整容差。通过后比较embedding、
每层Attention/MoE输出、final norm和logits，记录完整张量与末Token各自的max/mean误差、
不同元素数及首个差异位置。最早不同阶段是观测到的计算边界，不等于已证明具体算子错误。

完整FP16轨迹默认仍限制32 MiB。本入口只向子进程设置`FASTLLM_MOE_TRACE_MAX_MIB=256`，
该变量只接受32或256，且仍计入原有512 MiB总工作区预算。长Prompt的二进制文件按块比较，
不展开为全部Python浮点对象。以1024-token、2048隐藏宽度、24层为例，每rank每次快照约
196 MiB；两组合含warmup共约2.3 GiB文件。仅诊断需要这些快照，不输出性能结果。

结果保存在`时间戳_moe_prefill_batches_随机号/`，含原始快照、协议结果、SHA256、执行命令、
构建缓存、硬件信息和四格式汇总。零容差阶段FAIL表示定位到数值差异；原有生成、numeric与
性能验收结论不变。异常或中断仍保存部分报告并返回非零。

若首个差异已经定位到`layer_0_moe`，使用下一步内部诊断：

```bash
cmake --build "$ATTENTION_EP_BUILD" --target fastllm_process_tp_package -j 4
"$ATTENTION_EP_PYTHON" test/distributed/probe_qwen_moe_block_batches.py \
  --build "$ATTENTION_EP_BUILD" --model "$ATTENTION_EP_MODEL" \
  --baseline /绝对路径/时间戳_moe_prefill_batches_随机号
```

基准必须是已完成、归因有效且rank 0/1首个差异都在第0层MoE的完整逐层报告。
入口检查模型config/index及历史快照SHA256，沿用原Prompt、capacity、采样和正式
Eager/Prefix-off路径。两请求和四请求各做一次Prefill warmup与一次正式采集，
每次End并核验KV释放，不执行后续Decode、HF对齐或性能测试。

只向本次子进程设置`FASTLLM_MOE_BLOCK_TRACE_DIR`，与完整逐层快照互斥。
观察器在第0层原有路由计数同步后，按实际专家行数预建空间；GPU快照最多256 MiB，
仍计入原512 MiB总工作区预算。之后只在原stream追加D2D复制，最终统一读回；
不更换GEMM、路由、累加精度、NCCL调用或同步位置。含warmup共16份rank快照，
实际文件量取决于路由行数，可能达到数GiB；使用独立结果目录，不覆盖旧基准。

四个rank都记录内部张量：无本地请求的rank也负责自己的专家，不能省略。阶段包含
输入、归一化、router logits、专家ID/权重、全组AllGather、每专家实际槽位索引、
Pack、Gate/Up、激活、Down、ReduceScatter后的逐槽贡献、有序累加、共享专家及
最终输出。原始FP16/FP32/int32值按连续小端二进制保存；warmup和正式快照均校验
布局、完整性和有限值，正式轮次还检查跨rank通信、padding、专家计数及Pack槽位。

必须先通过18项精确重放：6个活动请求的完整logits/采样Token、第0层输入和输出。
失败时`trace_attribution_valid=false`，拒绝将观察器造成的输出变化用于原问题归因。
通过后按相同请求/Token/路由槽位配对专家矩阵；记录两边完整矩阵行数，但只比较
共同请求对应的行。路由ID不同则停止专家内部配对，保留最早路由差异，不误比不同专家。
最终报告`first_different_stage`与每个专家的首个差异，并保存JSON/Markdown/CSV/XLSX。
零容差FAIL表示观测到差异，不等于已证明算子错误；该诊断不自动放宽原验收标准。

若共同输入和路由完全相同、首差位于专家Gate/Up或Down，可进一步做独立cuBLAS归约实验：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$ATTENTION_EP_REFERENCE_PYTHON" \
  test/distributed/probe_qwen_moe_gemm_reduction.py \
  --build "$ATTENTION_EP_BUILD" --model "$ATTENTION_EP_MODEL" \
  --cuda-home "$ATTENTION_EP_CUDA_HOME" \
  --baseline /绝对路径/时间戳_moe_block_batches_随机号
```

使用已有CUDA PyTorch与safetensors的reference环境，不编译、不重建worker。
入口校验基准有效性、原worker、模型config/index及内部快照SHA256；按每个专家的
首差选择Gate/Up或Down，并再次要求共同输入行完全一致。Down上游激活已经不同则
拒绝归因。两边保留各自完整矩阵行数执行，只在结果比较时按原路由槽位抽取共同请求。

从原模型分片读取对应专家权重，Gate在前、Up在后拼接，沿用FP16输入/输出、
`CUBLAS_COMPUTE_32F`和`CUBLAS_GEMM_DEFAULT_TENSOR_OP`，在原owner GPU上
使用现有worker链接的cuBLAS库。每模式每专家每矩阵warmup一次，正式调用至少5轮；
必须全部精确重放原始输出，才进入候选配置。库符号来源和math mode均运行时核验。

候选仅给私有handle增加`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`，
该标志要求使用计算类型进行归约，见[NVIDIA数值行为说明](https://docs.nvidia.com/cuda/archive/12.6.2/cublas/index.html#gemm-algorithms-numerical-behavior)。
它不保证不同矩阵形状逐位相同；对照结果也不能单独证明具体算法或归约步骤。
本入口是独立数值实验，不修改正式后端，不运行完整生成、HF对齐或性能计时。

报告目录为`时间戳_moe_gemm_reduction_随机号/`，保存原始输出、每轮样本引用和摘要、
权重来源、GPU/库/构建信息及JSON/Markdown/CSV/XLSX。相同重复输出复用已保存文件。
重点读取`baseline_replay_passed`和`candidate_batch_invariant`，后者还要求各矩阵重复
运行稳定；`candidate_targets`列出各专家结果。旧配置跨批次差异仍记FAIL，因此即使候选
全部通过，总报告仍FAIL并返回非零。基准重放失败或诊断异常时，候选结论为null并保留
部分报告。候选通过后仍需另行修改正式路径并验证完整模型，不能直接视为问题已修复。

### 四项联合验收

`real-all`等价于真实模型入口的`--alignment-policy all`。固定比较exact、K=5/容差0、
K=5/容差0.1和numeric；若环境覆盖了K或宽松容差，入口会拒绝运行，避免报告名称与实际
规则不符。该模式不与`--performance`混用；性能测试继续使用独立入口。

- HF自由生成参考只运行一次；FT只运行原有Graph/Prefix、cold/warm及前缀路由矩阵。
  每个请求的三种判定共用同一份Token、文本和原始logits，不为切换规则重新生成。
- 三种结论分别保存。exact失败或严格Top-5失败后，仍完成其他请求和numeric。
  模型执行、输入验证、Graph/Prefix路径检查失败会使对应生成策略全部失败。
- numeric使用独立参考进程及教师强制历史，继续执行逐层/专家路由和四组Graph/Prefix
  数值检查；不能把生成报告中的数值WARNING当作numeric PASS。
- 普通阶段异常会保存部分报告并继续下一阶段；Ctrl+C保存部分报告后停止，不启动下一阶段。
  阶段缺失、少跑用例或中途退出均标记`complete=false`。四项任一失败时最终返回非零。

每次运行创建新的`时间戳_moe_full_随机号/`目录：

- 根目录`summary.json/md/csv/xlsx`：exact、top5、top5_near_tie、numeric四行独立结论，
  包含阶段完成状态及明细报告路径。根报告失败数统计失败case及未完成标记；最大误差取
  阶段case最大值，平均误差不跨不同大小的张量直接平均，应查看逐case报告。
- `generation/summary.json/md/csv/xlsx`：逐请求的三种状态、Token/文本一致性、首次分歧、
  已比较与未比较步数、数值误差及WARNING。每个case JSON仅保存一份FT原始生成数据，
  三套候选/严格判定放在`comparison.comparisons`，HF原始结果保存在同目录参考文件中。
- `numeric/summary.json/md/csv/xlsx`：原严格数值检查的独立明细与失败原因。

JSON保留原始数值精度，其余表格浮点显示三位小数。根报告只有四项全部完成并通过才为PASS，
不能用宽松策略的通过覆盖严格策略的失败；已有历史报告不修改。

## 失败历史的分层定位

固定短序列的逐层检查通过，不保证中文请求的长历史没有数值差异。可从已保存的请求中
自动提取首次logits超差的真实Token历史，分别检查全部层的Attention、独立MoE、完整层。
独立MoE使用HF实际Attention残差输出作为双方共同输入；不累积FastLLM上游层误差。
该诊断仅激活失败请求所在DP rank，其他rank空闲但参与EP通信；不重放原始并发调度，
不验证Graph/Prefix/最终Norm与LM Head，也不能替代完整模型验收。

更新Python脚本后可直接运行已有服务器产物，不需要重新编译。保持前文CUDA环境设置：

```bash
/venv/main/bin/python3 test/distributed/run_qwen_moe_precision_diagnostic.py \
  --result-dir test/distributed/results/20261114T151715Z_moe_full_f844ff43 \
  --request session-0002/request-000001.json \
  --model /home/models/Qwen1.5-MoE-A2.7B-Chat \
  --build build-attention-ep-cu126 \
  --reference-python /venv/fastllm-reference-cu126/bin/python3
```

HF和FastLLM进程分开加载，任务按已有checker文件大小与数量限制分块，原始日志保留。
逐层数值失败继续记录剩余任务；进程或参考生成失败时保存已完成部分。每字段记录
max/mean绝对误差、失败数、最坏元素及实际/参考值；阈值保持`atol=0.05, rtol=0.01`。
四种报告均包含`saved_full_model_logits`原失败，所以总体FAIL/非零退出是预期的，
不能将局部子层PASS解释为原问题已修复。诊断目录不会覆盖原验收报告。

### Router一个FP16间隔差异的独立定位

针对已保存诊断中的第1、9、14、17、22层，可仅读取这些层的Norm/Gate权重，
对同一输入比较HF归一化顺序与原生256线程归约顺序的张量模拟，再分别调用
PyTorch Linear和使用已有worker链接的cuBLAS库、修复前参数的FP32_PEDANTIC GEMM。
第22层额外使用已保存的原生Attention输出，分离上游输入变化的影响。
这是独立数值实验，不加载完整模型、不编译，不代表正式后端验收或性能结果。

```bash
/venv/fastllm-reference-cu126/bin/python3 test/distributed/probe_qwen_moe_router_precision.py \
  --diagnostic test/distributed/results/20261114T154834Z_moe_precision_b7cc418d \
  --model /home/models/Qwen1.5-MoE-A2.7B-Chat \
  --worker build-attention-ep-cu126/fastllm_tp_worker
```

每组显示Norm差异数、四种组合与已保存HF/原生router输出的精确差异数和最大误差。
零容差用于定位一个FP16间隔的差异，不改变完整模型验收阈值；存在差异时总体FAIL
和非零退出是预期现象。`raw.json`保存全部模拟中间量和分数，四种summary报告保存
比较结果及库版本。归约模拟未直接采集原生Norm输出；若模拟组合无法复现原生日志，
不能据此断言原生Norm已定位或排除，还需采集真实中间量。PyTorch对照也须先检查
能否复现已保存HF输出，不能把库版本或执行环境差异归因给单个生产算子。

服务器历史诊断`20261114T162200Z_router_probe_21e073d6`显示：五个相同HF输入的
重点层均有`norm_different=0`，PyTorch GEMM逐位复现已保存HF分数，旧PEDANTIC
配置逐位复现已保存FastLLM分数。第22层换成原生Attention输入后仍存在上游差异。
该证据支持修复MoE GEMM配置，不能视为修复后验收。上述探针刻意保留旧PEDANTIC
路径用于历史归因；新二进制应通过`synthetic`、`regression`、`real`验证实际输出。

### Attention投影的独立定位

若同输入独立MoE已通过，而完整层因Attention残差变化发生路由差异，可使用保存的
分层诊断输入，仅加载目标层Attention权重，对照以下实验：

- `hf_replay`：真实HF eager Attention模块及RMSNorm，先要求精确复现保存的HF残差。
  无法复现即停止该次归因，保留报告，不能把环境差异归因于原生算子。
- `split_bias`：Q/K/V分别执行PyTorch无bias Linear，FP16落盘后再加bias。
- `merged_qkv`：使用已有worker链接的cuBLAS与当前MoE的32F/TENSOR_OP参数，
  合并QKV GEMM、输出FP16后加bias；该组同时包含矩阵合并和cuBLAS调用策略的变化。
- `fp32_o`：仅将输出投影替换成当前MoE参数的FP32输出GEMM，再转FP16加残差。
- `native_norm`：仅替换输入归一化为256线程归约顺序的张量模拟。
- `combined_projections`：组合上述原生归约顺序模拟、合并QKV和FP32输出投影。
- `fused_qkv`：合并QKV并将bias广播到输出，再用`beta=1`进行GEMM，
  对照当前MoE一次FP16舍入的投影配置；归一化和输出投影保留HF实现。
- `current_projections`：组合归约顺序模拟、融合bias的QKV和FP32输出投影。

所有实验仍使用HF的RoPE、QK、softmax和PV，**组合实验也不等于原生Attention重放**。
输出更接近原生日志只能作为下一步定位线索，不能单凭此结果宣布原生根因或模型修复。
这是独立数值实验，无性能计时，不修改正式推理权重、后端或容差，不编译，不加载完整模型。

```bash
/venv/fastllm-reference-cu126/bin/python3 test/distributed/probe_qwen_moe_attention_precision.py \
  --diagnostic test/distributed/results/20261115T011823Z_moe_precision_49509d28 \
  --model /home/models/Qwen1.5-MoE-A2.7B-Chat \
  --worker build-attention-ep-cu126/fastllm_tp_worker
```

默认检查第6、9、16、18、22、23层，可用`--layers`指定唯一层号。脚本限定已记录的
Transformers 4.57.6接口和默认RoPE、无滑窗、无缓存的单请求prefill。每组打印对已保存
HF/原生残差的精确差异数和最大误差；零容差FAIL表示数值不同，并非新的完整模型失败。
原验收`0.05 + 0.01 * abs(reference)`的失败数量另外保存在JSON中。每层原始JSON保存
归一化、Q/K/V、Attention context、输出投影和残差；summary生成MD/CSV/JSON/XLSX。

服务器诊断`20261115T020759Z_attention_probe_e3838df2`中，六层`hf_replay`
均精确复现HF；单独改成QKV点积先转FP16再加bias，就重现了大部分原生差异。
第18层相对原生的差异数从21001降至866，再结合归约顺序模拟降至219。
`fp32_o`在这六组输入中均与HF相同。此证据支持修正QKV的bias舍入边界，
但剩余误差和完整模型是否通过仍须由正式验收确认。

第一轮舍入修正曾先把bias广播到现有输出缓冲，再调用`beta=1`的GEMM，执行
`FP16(dot + bias)`，避免旧的`FP16(FP16(dot) + bias)`。每次调用都重新初始化
缓冲，单层、prefill、decode和Graph走同一入口，不增加GPU工作区或同步。
语义依据见[cuBLAS GEMM公式](https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html#cublasgemmex)。
上述探针的`split_bias`、`merged_qkv`、`combined_projections`刻意保留旧的
单独bias操作作对照，不代表当前完整原生执行路径。后续已按本页开头说明改为
三个独立Lt投影；有bias时由epilogue处理，无bias路径不变。

`synthetic`增加`qkv_bias_rounding`用例：解析构造bias抵消，正确Attention残差
首通道为1.03125，旧路径为1.0，在原阈值下必须失败；同时通过正式会话入口覆盖
EP1/2/4、Eager/Graph、Prefix开关和完整logits。CPU检查用独立PyTorch Linear
核对期望值，不能代替GPU验收。此修正涉及CUDA代码，需要先重新构建，再执行
`synthetic`、`regression`和`real`，误差容差保持不变。

新诊断`20261115T032151Z_moe_precision_6950e33c`使用失败step=1的26个Token。
全部24层的独立Attention和独立MoE均在原阈值内，但完整层4、7出现非同分专家换序，
完整层13的token=11把专家8换成24，随后路由输出7个、层输出5个元素超差。
独立MoE使用HF边界时通过，只能说明该共同输入下通过，不能排除累积误差或证明
最终logits的全部误差都由第13层导致。

可打开`--router-replay`，检查每种Attention投影输出如何影响下游专家选择：

```bash
/venv/fastllm-reference-cu126/bin/python3 test/distributed/probe_qwen_moe_attention_precision.py \
  --diagnostic test/distributed/results/20261115T032151Z_moe_precision_6950e33c \
  --model /home/models/Qwen1.5-MoE-A2.7B-Chat \
  --worker build-attention-ep-cu126/fastllm_tp_worker \
  --layers 4 7 13 --router-replay
```

路由重放固定使用HF的post-Attention RMSNorm、Gate、FP32 softmax和TopK。
首先要求保存的HF Attention边界能精确复现HF完整层router logits，专家选择只允许
两侧都同分的换序；不满足就停止归因并保留报告。之后将保存的原生Attention输出、
各投影实验输出分别送入同一个HF路由，记录分数、专家ID、路由权重及失败Token的
候选专家分数。保存的原生完整层路由用于额外对照，不被替换。

这只是改变Attention边界输入的独立对照，后半段使用HF实现，不能视为原生MoE重放。
每层JSON保存原始结果，四格式summary将精确分数差异和专家选择差异分别列项；
选择差异的`failed_elements`统计Token数，不计算专家ID的数值误差。仍不修改验收容差，
不重新编译，也不以这些独立实验代替完整模型验收。

诊断`20261115T034534Z_attention_probe_0ca04f43`进一步显示：三层保存的原生
Attention输出接HF路由后，均精确复现保存的原生完整层router logits。仅替换成
合并QKV的GemmEx/beta=1投影，也能复现第4层的换序、第13层的专家8→24；
第7层尚有未复现的换序，不能把全部差异归给同一个投影因素。

下一组仅改变QKV计算，使用`--modes`指定实验，必须以`hf_replay`开始且不重复。
`--qkv-report`额外记录并打印每个Q/K/V投影对HF的精确差异，保留所有原始中间量。

- `torch_merged_qkv`：PyTorch公开Linear一次计算合并QKV及bias，用于区分矩阵合并与接口变化。
- `separate_fused_qkv`：三个独立GemmEx/beta=1投影，与HF保持相同Q/K/V矩阵尺寸。
- `fp32_qkv`：合并GemmEx/beta=1输出FP32，再统一转FP16；其余Attention不变。
- `lt_qkv`、`lt_separate_qkv`：显式调用已有原生cuBLAS所依赖的Lt库，分别合并或分开
  QKV；使用32F计算、FP16输入输出、beta=0和BIAS epilogue，不回退到其他接口。

PyTorch 2.7.1的[带bias Linear实现](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/Linear.cpp#L53)
可进入[cuBLASLt分支](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/Blas.cpp#L335)，
其[gemm_and_bias配置](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CUDABlas.cpp#L1189)
使用epilogue处理bias。因此仅有相同数学公式和FP16输出类型，不能认定与GemmEx逐位相同。
探针的PyTorch模式保留自身调度，只有`lt_*`模式声明实际强制使用Lt接口；它们仍是独立
实验，不代表正式后端已实现Lt。调用依据为[CUDA 12.6 Lt接口](https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html#cublasltmatmul)。

```bash
/venv/fastllm-reference-cu126/bin/python3 test/distributed/probe_qwen_moe_attention_precision.py \
  --diagnostic test/distributed/results/20261115T032151Z_moe_precision_6950e33c \
  --model /home/models/Qwen1.5-MoE-A2.7B-Chat \
  --worker build-attention-ep-cu126/fastllm_tp_worker \
  --layers 13 4 7 --router-replay --qkv-report \
  --modes hf_replay fused_qkv torch_merged_qkv separate_fused_qkv fp32_qkv lt_separate_qkv lt_qkv
```

Lt按当前stream执行，私有handle与临时描述在成功及失败路径均释放。使用首个合法heuristic，
报告保存矩阵尺寸、算法原始字、实际库版本与路径、工作区限制及需求；无算法或执行失败
直接报错。默认工作区1 MiB；若设置`CUBLASLT_WORKSPACE_SIZE`，按PyTorch的KiB语义读取。
不修改HF全局后端、计算精度开关或原验收阈值。

## 失败Decode的缓存与完整历史对照

若真实验收首次超差发生在step大于0的位置，可使用已有构建产物对同一实际Token历史
进行定向对照。`eager`只表示没有使用CUDA Graph；该轮仍可能是缓存Decode，或同时
有其他rank的Begin。正式trace现在记录`joint_round.actions`（0空闲、1开始、2解码、
3结束）和`submitted_token_counts`；后者是提交长度，尚未扣除原生Prefix命中。

```bash
/venv/main/bin/python3 test/distributed/probe_qwen_moe_decode_precision.py \
  --result-dir test/distributed/results/20261115T023144Z_moe_full_acb8e3ac \
  --model /home/models/Qwen1.5-MoE-A2.7B-Chat \
  --build build-attention-ep-cu126 \
  --reference-python /venv/fastllm-reference-cu126/bin/python3
```

探针从默认`session-0002/request-000001.json`恢复首次失败历史（可用`--request`指定
其他请求），保持教师强制Token、模型和原来的`atol=0.05, rtol=0.01`。HF四卡参考先
完整重算该历史，再从空KV开始运行原prompt并逐Token追加；缓存实现使用
[Transformers 4.57.6 Qwen2MoE公开forward](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py)，
每步检查KV长度。参考子进程退出后，通过正式`QwenMoeProcessSession.generate`分别在
两个新会话中执行缓存Decode和完整历史Prefill，禁用Graph/Prefix，核验实际联合轮次。

这是仅rank0接收请求、其他rank参加EP的受控对照，**不重放原来的并发准入顺序**；
Prompt长度、专家GEMM形状和HF缓存/完整计算的舍入差异均需结合结果分析，不能仅凭
缓存对照有差异就断言KV存储错误，也不能用新增HF缓存参考替换正式验收参考。
本探针不解释独立层测试的非同分专家排序失败，该项仍需按原标准验收。

输出目录保留HF与原生完整logits、缓存长度、联合轮次、原构建缓存和二进制SHA-256，
以及JSON/Markdown/CSV/XLSX汇总。JSON另存全部真实超差词表项及各自阈值；终端只打印
最严重的五项。最大绝对误差项可能仍在相对容差内，不等同于实际超差项。
即使受控对照通过，报告也保留原失败并返回非零，不能当成完整模型验收通过。
此探针与trace记录仅涉及Python，更新后无须重新编译。

## 正式完整前向轨迹与最终Norm/LM Head定位

独立子层检查会在每层重新使用HF输入，不能观察原生误差沿24层传播。
`20261115T044828Z_decode_probe_f3327e03`中，原生缓存和完整重算都相对HF超差，
而两种原生模式之间在原阈值内。因此不能仅靠切换缓存参考解释失败，下一步采集
正式`StepMoe`完整历史Eager前向中的Embedding、每层Attention残差、MoE残差，
以及最终Norm的末Token输出。输入仍为原失败的实际Token历史。

采集默认关闭，仅诊断子进程设置`FASTLLM_MOE_TRACE_DIR`。它要求Prefix/Graph关闭、
位置零的Begin；Decode和不支持模式明确拒绝，不静默回退。快照区最多32 MiB，计入
原有512 MiB工作区预算；源张量、算子选择、stream及计算顺序不变。每阶段只在原
stream复制快照，无逐层同步；一次整体读回后在原有最终同步之后落盘。临时快照
随工作区释放，异常走原worker失败清理。不允许用采集运行做性能比较。

需要用户在服务器重新构建一次，然后用已有参考环境执行：

```bash
bash test/distributed/run_attention_ep_v100.sh build
/venv/fastllm-reference-cu126/bin/python3 test/distributed/probe_qwen_moe_forward_precision.py \
  --diagnostic test/distributed/results/20261115T043656Z_moe_precision_78444795 \
  --model /home/models/Qwen1.5-MoE-A2.7B-Chat --build build-attention-ep-cu126
```

脚本依次运行两次普通前向及一次采集前向，三次都是正式四rank会话且仅rank0有请求。
`plain_repeat_exact`检查跨会话稳定性，`trace_vs_plain_exact`检查采集有无扰动；
任一失败都将`trace_attribution_valid`置为false，此时仍保存观察，但不能将轨迹
直接归因到原失败。探针不重放原混合请求调度，也不替代原Graph/Prefix验收矩阵。

HF轨迹直接来自上一轮完整HF前向的hook记录，并检查Token历史、阶段完整性和各层
输入连续性。所有阶段报告最大/平均绝对误差、超差数、精确差异数以及末Token指标，
原数值阈值保持`0.05 + 0.01 * abs(reference)`。阶段首次超差有助于缩小范围，
不等于该阶段的算子必然错误；前层传播的输入差异仍须结合后续同输入对照分析。

原生会话退出后，只读取最终Norm和LM Head权重，在参考环境进行独立输出头重放：

- HF全行输出头必须精确复现已保存HF logits，避免把错误参考用于归因。
- 原生GemmEx/FP32输出头重放必须精确复现正式logits的JSON传输值。正式通道使用
  json11的`%.12f`，只对重放值使用同样编码再零容差比较，保留编码前误差。还要求
  相邻FP32数值编码后可区分；接近零导致编码歧义时门禁仍失败，不据此作精确归因。
- 同一原生末层输入比较实际Norm和HF Norm；同一Norm输出比较两种输出头。
- 单独比较HF全Token矩阵与最后一行矩阵，保留GEMM形状差异。

这些独立重放只用于诊断，不修改正式后端。输出头门禁和采集门禁均通过才标记
`head_attribution_valid=true`。旧独立子层非PASS项也打印并保存在metadata中。
结果保留原始FP16二进制、描述与SHA-256、完整logits、输出头候选、构建缓存和
二进制哈希，汇总生成JSON/Markdown/CSV/XLSX；每次使用新目录。保留原失败或任一
精确门禁失败时返回非零，不能用本诊断的局部PASS宣布完整模型通过。

已有完整轨迹和`head-replay.json`时，可以只读复核，**无需重新编译或运行GPU**：

```bash
/venv/main/bin/python3 test/distributed/recheck_qwen_moe_forward_precision.py \
  --probe test/distributed/results/20261115T132323Z_forward_probe_c0356506
```

原诊断目录及其引用的历史参考目录必须仍在。复核校验FP16轨迹哈希、输入历史、
HF边界连续性、输出头元数据，并要求除修正的传输门禁外，重新计算的指标与原报告
一致。任一不一致或资料缺失都停止归因并输出失败报告；不会把缺失数据当成PASS。
结果写入新的`*_forward_recheck_*`目录，原始报告和日志保持原样。复核报告中
`command/git/gpu`等字段保留原采集信息，`recheck_command/recheck_git`记录本轮CPU分析。

主报告仍保留真实模型的超差；`tokens/summary.{json,md,csv,xlsx}`额外记录每层每个
Token的累积误差，终端列出每个Token首次超出原阈值的位置，层号和Token位置从0开始。
`native_head_reproduces_traced_logits.raw_numeric_comparison`保留未编码误差和精确差异数。
仅输出头门禁恢复PASS不代表完整模型通过；只要原数值失败仍存在，复核就返回非零。

### 首次超差MoE的双输入检查

`20261115T134038Z_forward_recheck_6cbcdd37`的轨迹门禁通过，最早超差为第13层MoE、
Token索引11；第15层MoE的Token索引6也明显超差。层号和Token位置均从0开始。
这些位置只说明首次越过原阈值，不能排除更早层的输入误差。本轮输出头的传输值
一致，但1个近零FP32值存在编码歧义，输出头精确归因仍不可用。

下面的独立诊断保留整段Token形状，分别将HF和原生的真实Attention后输入送给双方
MoE；仅加载指定层的HF Norm/MoE权重，原生复用已有正式层检查入口。**无需重新编译**：

```bash
/venv/fastllm-reference-cu126/bin/python3 test/distributed/probe_qwen_moe_stage_inputs.py \
  --probe test/distributed/results/20261115T132323Z_forward_probe_c0356506 \
  --model /home/models/Qwen1.5-MoE-A2.7B-Chat --build build-attention-ep-cu126 --layers 13 15
```

`--probe`指原始轨迹目录而非复核目录。脚本验证历史哈希/轨迹门禁、当前worker哈希、
参考环境版本，并使用原HF层所在设备。每层HF重放须精确复现已保存的所有字段，原生
重放须精确复现正式轨迹的FP16层输出；任一不一致将该层`attribution_valid`置false。
对FP16原生层输出仅恢复可唯一确定的JSON传输值，不对FP32输出头使用这种恢复方法。

报告分别保存输入差异、HF对输入扰动的传播、同一原生输入下的实现差异、总输出差异，
以及双方实际router、专家、权重、路由和共享专家输出。不能相加这些最大误差或失败
元素数量；它们可能出现在不同元素上。`tokens/summary.{json,md,csv,xlsx}`保存逐Token
结果；终端打印原首次超差Token的四组指标。所有历史失败和输出头不确定性继续保留，
不改变原阈值；即使两层重放成功，仍会返回非零，不能代表完整模型通过。

### 第0层路由专家贡献的精度重放

`20261115T145219Z_stage_inputs_e7c9fd36`的双输入门禁通过。Token 6进入第0层MoE
前输入完全一致，实际router logits、专家ID/权重和共享专家输出也一致；路由贡献
最大差异约`3.43322756e-05`，残差输出最大差异约`6.10351562e-05`。这将该Token
的首个差异缩小到路由专家分支，但尚未区分专家GEMM与累加舍入。

独立脚本复用双输入检查保存的原生结果，只在参考环境加载指定层并观察HF公开
MoE前向。记录每个实际专家返回值、加权贡献及真实Token行顺序；HF原始前向不
修改算子或精度。随后在CPU对同一组贡献按实际专家调用顺序分别做FP16和FP32
逐次累加，按原FP16边界加共享专家和残差。无需编译或重新启动原生四rank进程：

```bash
/venv/fastllm-reference-cu126/bin/python3 test/distributed/probe_qwen_moe_reduction_precision.py \
  --pair test/distributed/results/layer0-pair.yHrjDl/20261115T145219Z_stage_inputs_e7c9fd36 \
  --model /home/models/Qwen1.5-MoE-A2.7B-Chat --layer 0 --tokens 6 11
```

开始前重新校验原始轨迹、双输入任务和原生输出门禁，拒绝历史证据变化、遗漏任务或
输入形状改变。新增采集必须精确复现保存的HF所有字段；FP16贡献重放必须精确复现
HF路由及最终层输出。任一失败时仍生成已执行部分的报告，不能据此作数值归因。
实际专家覆盖、Token索引、乘权重舍入及有限值均校验，采集失败也移除hook和Mode。

`fp16`/`fp32`行均比较已保存的原生输出，报告原阈值下的max/mean/failed以及
`exact_different_elements`。终端同时列出指定Token的路由和共享专家是否完全一致。
`routed_output`原生记录为FP32的12位小数JSON，数值不等不能直接解释为位不等；
`output`为可唯一恢复的FP16值，可查看精确差异数。所有原始贡献、重放结果及
JSON/Markdown/CSV/XLSX汇总写入新目录，历史失败保持FAIL和非零返回码。

**FP32重放是独立数值候选**：它不模拟原生专家GEMM、rank分组或NCCL归约树，也未
观察原生各专家的独立输出。即使明显更接近原生结果，也只能支持累加精度这一解释，
不能单独证明原生专家计算全部一致或宣布完整模型通过。Token 11在第0层Attention
已经存在的输入差异需要另外处理；本检查不会消除或豁免这部分误差。

## 完整前向的独立FP32专家累加实验

`20261115T151031Z_moe_reduction_fd1c4eed`通过所有采集与FP16贡献重放门禁。
第0层Token 6/11在两种输入下的FP32重放最终输出均与原生FP16值完全一致。
整段HF输入的最终输出平均差异从`2.20054062e-6`降至`1.81339251e-8`，
原生轨迹输入从`2.26194523e-6`降至`8.37026164e-8`；仍分别有256/986个
输出元素数值不等，均未超原阈值。路由或共享分支不完全一致的Token分别为
`[21]`和`[2,13,21]`。这支持第0层主要差异来自累加舍入，不证明所有层或完整模型通过。

下一实验加载完整HF模型，先用原版公开入口复现相同完整历史，再观察实际专家
贡献。原版logits、完整隐藏状态轨迹和路由记录必须精确复现历史，观察前后logits
必须完全相同。门禁通过后，只在独立候选前向中用FP32累加每层实际FP16加权贡献，
在加共享专家前转回FP16，再由原HF层执行残差及后续层。每层候选也必须先证明
捕获的贡献可精确重放该次输入下的原FP16累加；原路由、专家GEMM、共享专家和
Attention不替换。候选结束后卸载所有hook，并再次验证原版HF logits未被改变。

```bash
/venv/fastllm-reference-cu126/bin/python3 test/distributed/probe_qwen_moe_accumulation_forward.py \
  --probe test/distributed/results/20261115T132323Z_forward_probe_c0356506 \
  --model /home/models/Qwen1.5-MoE-A2.7B-Chat
```

该命令只运行参考模型，不重新启动FastLLM、不编译。保持原完整Token ID、eager、
use_cache=False和原四卡按层分配，拒绝参考版本不一致或历史证据变化。
在新建目录保存`baseline.json`/`candidate.json`完整中间结果，以及JSON/Markdown/
CSV/XLSX汇总；终端打印原HF与候选相对历史原生轨迹、最终logits的max/mean/failed。
逐层记录候选相对原HF的专家集合变化Token；这不是与原生专家选择的直接比较。

**候选是改变累加语义的因果实验，不是新的验收参考。** 它沿HF真实专家调用顺序
累加，不复刻原生专家批次、GEMM或NCCL归约树，仍可有其他差异。不得将候选PASS
替代原HF验收，或据此放宽阈值；完整报告保留历史FAIL及非零退出码。实验使用历史
完整原生轨迹比较，只支持该历史的结论，不覆盖原调度、KV Cache或CUDA Graph验收。

## 启动阶段异常的定向复现

若`ready.json`尚未产生、请求数为0，故障发生在进程组初始化或模型加载阶段，
不能仅凭case名称中的Graph标记归因于图执行。退出监控会在统一回收前读取
原始`exit_code`或`signal`，同时保存PID和事务sequence；读取状态不会提前回收
进程。worker异常会先写stderr，再发送协议错误；父进程/控制连接守护退出也会
记录父PID和poll事件。其他worker的守护退出可能只是整组清理的后果，应先看
协调器记录的首个故障rank。`signal=9`本身不能证明发生了OOM。

重新构建后可只重复失败的小模型case，默认10轮：

```bash
bash test/distributed/run_attention_ep_v100.sh reproduce f16_bias_ep4_graph1_prefix0 10
```

每轮新建正式会话并保留单独的启动/请求/关闭日志，沿用完整logits误差检查、
冷热两次请求及Graph捕获/重放检查。任一轮失败，四格式汇总和进程退出码仍为
失败，后续成功不会覆盖原失败。报告标记`focused_synthetic`，不代表跨模式、
完整小模型矩阵或真实模型验收通过。定向通过后仍需运行完整验收。

## 正式运行示例

```bash
export PYTHONPATH=/home/project/vllm_fastllm/tools
/venv/main/bin/python3 -m fastllm_pytools.cli run \
  /home/models/Qwen1.5-MoE-A2.7B-Chat \
  --tp-backend process --dp 1 --tp 1 --attention-dp 4 --ep 4 \
  --tp-binary build-attention-ep-cu126/fastllm_moe_generate \
  --tp-worker build-attention-ep-cu126/fastllm_tp_worker \
  --cuda-graph --prefix-cache true --dp-routing prefix \
  --tp-timeout 1800 --prompt '请用一句话介绍北京。' --max-new-tokens 32
```

HTTP 将 `run` 改为 `serve`，去掉 `--prompt`、`--max-new-tokens`，添加
`--host 0.0.0.0 --port 8080 --model_name qwen-moe`。

## 验收与报告边界

### 分批后台验收入口（2026-11-21）

当前优先补齐 HTTP、生命周期、小模型、TP/DP 回归、独立 HF 对齐与当前版本性能报告。
前缀缓存命中/未命中不一致仍是独立待办；本轮结果不能替代完整批次不变性验收。
本地没有服务器 GPU、模型与编译产物时，只能检查脚本，不能把脚本就绪记为 GPU PASS。

代码同步到四卡 V100 服务器、已有编译产物可用后执行：

```bash
(
set -eo pipefail
source /workspace/fastllm-v100-setup/env.sh
cd /home/project/vllm_fastllm
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash test/distributed/run_attention_ep_acceptance_v100.sh
)
```

启动器立即返回 PID 和独立日志目录，后台顺序运行 29 份独立 `run.sh`；不并行争用 GPU。
每份脚本最多 48 项，基准、参数化组合以及 HF 策略/性能比较汇总均保守计入上限：

- 单层：EP=1/2/4 各一份，每份 37 项，合计 111 项。
- 完整小模型：两份各 48 项，保留全部 8 种夹具的 EP/Graph/Prefix 组合。
- 生命周期：小模型旧路径、真实模型旧路径、真实模型批次不变模式，各 2 项。
- HTTP：真实模型旧路径与批次不变模式各 6 项。
- 稠密 Decode：Eager/Graph 各 32 项；公开 DP/TP：四种配置各 12 项。
  Decode 的页池边界通过正式 `--page-size 4 --tokens 8` 配置，分别验证请求容量
  7/8 成功、9 明确拒绝；核对各 rank 实际页池配置，并保留数值与资源清理检查。
  此项替换旧连续 KV 的 `reject_kv_limit`（256 MiB）预期，不修改推理容量限制。
- HF：全部 8 条中文请求拆成两份，每份分别跑旧路径和批次不变模式。
  每份包含 38 项生成明细、5 项 numeric 和 4 项策略汇总，共 47 项；
  exact、Top-5、Top-5/0.1、numeric 各自保留结论，任一失败仍返回非零。
- 性能：512/1024/1792 输入、128 输出，按 off/cold/hot 分成 9 份。
  每份包含 6 项旧路径、6 项候选及 6 项比较，共 18 项，全部至少 5 轮。

小模型夹具包含不满足 SM70 专家对齐要求的维度及多种源权重类型，继续作为既有路径回归。
SM70 模式通过真实模型的生命周期、HTTP、生成/numeric 与性能入口验证。
numeric 中独立的单层检查器继续使用既有路径，不能把该项 PASS 解释成候选专家后端的单层验收。

只运行选定范围也使用同一后台启动器，例如：

```bash
bash test/distributed/run_attention_ep_acceptance_v100.sh --groups validation
bash test/distributed/run_attention_ep_acceptance_v100.sh --only http_candidate
bash test/distributed/run_attention_ep_acceptance_v100.sh --groups performance
```

上述是可选的独立启动方式，不要同时执行多个 GPU 验收任务。完整批次清单见本次 `plan.md`，
每批有独立命令、PID、stdout、stderr、退出码和原始报告。普通用例失败保留证据后继续其他批次，
最终所选范围任一失败或缺少报告都会返回非零。后台运行时不要更新源码或替换二进制，
准备阶段保存的摘要会在每批开始前核对；摘要只能固定本次产物，不能证明产物由当前 Git 编译。

汇总位于 `test/distributed/results/YYYYMMDDTHHMMSSZ_attention_ep_acceptance_<8位随机ID>/`：
`summary.json/md/csv/xlsx`、`delivery.md`、`exit.code`；性能另在 `performance/` 聚合。
所选范围 PASS 与 `final_delivery_eligible=false` 同时保留，避免覆盖已知缓存一致性阻塞。
性能报告包括原始证据链接、模型、并行场景、输入输出长度、缓存/Graph 模式、延迟、吞吐和可比性；
输出历史或缓存行为不一致时保留耗时，加速比为空。

本次性能还会保存到 `docs/performance/runs/<运行目录名>/`，提供 JSON、Markdown、CSV。
原始结果目录仍生成 XLSX；持续性能汇总目录遵循用户要求不增加 Excel。
本轮性能属于当前版本结果；修复缓存问题后仍需按影响范围复测，不能提前标记最终交付。

- 小模型包含 FP16/BF16/FP32 源权重、有无 bias、归一化路由、绑定输出层、无本地专家和热点专家。
  单层验收覆盖 Router、专家与层输出；全模型逐步比较独立 CPU 重算的完整 logits。
  `attention_fp16_rounding`通过正式单层入口覆盖放大的有限投影、变长/空rank与完整层；
  同一用例的CPU PyTorch检查证明缺失Attention中间FP16舍入会超过原有误差阈值。
  `qkv_bias_rounding`继续拒绝点积先转FP16再加bias的重复舍入；`mixed_qkv_bias`首层无QKV bias、
  第二层有bias，通过正式单层与EP1/2/4、Graph/Prefix组合检查逐层后端选择、GQA不等宽布局、
  变长/空rank及缓存复用。CPU检查只验证用例的参考语义和检错能力，不代替服务器GPU执行。
- 真实权重默认使用固定中文JSONL、独立贪心生成及上述候选级对齐；严格Token/文本结果另列。
  `real-numeric`保留独立Transformers FP16重算完整历史末位置logits的旧数值验收。
  另外使用四个固定短序列记录全部层的真实输入、Router、共享/路由专家贡献和层输出，
  经正式单层接口进行四 rank 对照；单层检查关闭权重落盘，避免再次输出几十 GB 权重。
  专家ID、路由权重和路由累加值直接观察HF实际Top-K、dtype转换与index_add_的返回值，
  不在CPU重新选专家，也不将HF的FP16累加改成FP32。支持Linear gate与ModuleList专家；
  算子序列不符或中间量无法精确还原HF MoE输出时失败，退出时清理全部观察器与hook。
  专家集合必须完全相同且无重复；仅当置换槽位在双方各自router logits中都严格同分，
  才按专家ID对齐权重作比较。不同集合（包括Top-K截断点同分时）、非同分换序仍失败。
  原始路由证据不变，同分置换槽位数单独记录；专家ID不计入浮点张量max/mean误差。
  初始阈值保持小模型 `atol=rtol=0.008`，真实模型 `atol=0.05, rtol=0.01`。
  不会因失败自动放宽阈值。
  logits失败信息包含当步max/mean绝对误差、最大误差词表ID及实际/参考值；贪心Token相同不豁免。
- 跨模式诊断使用相同教师强制 Token 历史。该参数只在 trace 会话开放，不向 HTTP 暴露，
  不改变权重、算子、KV、Graph 或通信路径。实际采样差异另计数，不沿着分叉历史误比。
- 生命周期覆盖变长、EOS、取消、非法输入、2048 边界、部分前缀、LRU、清空与 worker 故障。
  HTTP 通过真实 CLI/模型工厂运行中文请求、SSE、并发、满队列、断连、截止时间和 SIGTERM。
- 性能比较同输入、同输出 Token 历史的 **FastLLM Eager 与 Graph**，不宣称是 Transformers 性能对比。
  并发 1/4/8，缓存关闭/批次起始冷/热，warmup 1 次、每轮 1 个并发批次、独立 5 轮。
  批次计时包括准入、前向、采样、结果和 KV 活动状态释放；不包含权重加载、校验、日志落盘。
  冷批次内部的新请求可以命中同批次先完成请求的快照，这是正式服务的缓存行为。
- 输出 Median/P95、CV、吞吐、TTFT、ITL、排队时间、显存快照和真实路径/计数。
  加速比为同缓存/并发配置 `Eager Median / Graph Median`。
  CUDA stream 在每轮结果返回前同步，两个模式使用同样边界。
- 每组结果写入唯一时间戳目录，生成 `summary.md/csv/json/xlsx`，失败返回非零。
  JSON 保留原始浮点精度，表格浮点显示 3 位。原始 trace、采样和命令同目录保存。

阶段6尚需服务器报告才能验收完成。生成级报告只承诺所选策略及实际配置覆盖；
`real-numeric`中的逐层中间量使用固定短序列，中文JSONL诊断比较每个生成位置完整logits。
生成级PASS、严格输出一致和内部数值诊断须分别报告。
