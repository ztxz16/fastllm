# Attention DP + EP：阶段 3～6

实现范围是本机 `qwen2_moe`、FP16 权重/激活/KV、`attention_dp=ep=1/2/4`、`dp=tp=1`。
真实 Qwen1.5-MoE-A2.7B-Chat 使用四张 V100；单卡/双卡只验收缩小模型。

代码已接入；本地禁止编译，GPU 功能和性能必须以服务器本次报告为准。
下面的命令由用户在 **Vast 的 Linux 终端**执行，不在 Windows PowerShell 执行。

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
参考模型按连续层显式分配到四卡，避免自动放置把权重全部放在一张 32 GiB 卡上。
Embedding与RoPE在GPU 0，最终Norm在GPU 3，绑定的LM Head与Embedding同卡。
设备映射不包含根模块兜底，首次前向前检查全部参数、buffer和子模块执行hook是否同卡。
服务器参考环境当前为 `/venv/fastllm-reference-cu126/bin/python3`：PyTorch 2.7.1+cu126、
Transformers 4.57.6、Accelerate 1.12.0。主环境仅用于分词，不能用其Transformers版本替代参考环境。
`reference.log` 记录设备检查和逐历史进度；参考失败时终端显示退出码与日志路径，
`setup.traceback.log` 保存主验收脚本的异常栈。仅修改这些Python脚本时无需重新编译原生产物。

`regression` 会在独立结果目录生成稠密小模型和 tokenizer，然后调用原有 DP2×TP1、DP4×TP1、
DP2×TP2 验收以及 TP Decode Eager/Graph 验收，不需要另下载稠密模型。

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

MoE正式QKV现在先把bias广播到现有输出缓冲，再调用`beta=1`的GEMM，执行
`FP16(dot + bias)`，避免旧的`FP16(FP16(dot) + bias)`。每次调用都重新初始化
缓冲，单层、prefill、decode和Graph走同一入口，不增加GPU工作区或同步。
语义依据见[cuBLAS GEMM公式](https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html#cublasgemmex)。
上述探针的`split_bias`、`merged_qkv`、`combined_projections`刻意保留旧的
单独bias操作作对照，不代表修正后的完整原生执行路径。

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

- 小模型包含 FP16/BF16/FP32 源权重、有无 bias、归一化路由、绑定输出层、无本地专家和热点专家。
  单层验收覆盖 Router、专家与层输出；全模型逐步比较独立 CPU 重算的完整 logits。
  `attention_fp16_rounding`通过正式单层入口覆盖放大的有限投影、变长/空rank与完整层；
  同一用例的CPU PyTorch检查证明缺失Attention中间FP16舍入会超过原有误差阈值。
  `qkv_bias_rounding`继续拒绝点积先转FP16再加bias的重复舍入；`mixed_qkv_bias`首层无QKV bias、
  第二层有bias，通过正式单层与EP1/2/4、Graph/Prefix组合检查逐层后端选择、GQA不等宽布局、
  变长/空rank及缓存复用。CPU检查只验证用例的参考语义和检错能力，不代替服务器GPU执行。
- 真实权重使用固定中文 JSONL 请求，独立 Transformers FP16 重算完整历史的末位置 logits。
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

阶段 6 尚需服务器报告才能验收完成。逐层中间量检查使用固定短序列；中文 JSONL 全模型
检查比较每个生成位置的完整 logits。两者应分别报告覆盖范围。
