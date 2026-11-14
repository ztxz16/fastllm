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
- MoE共享Linear使用`CUBLAS_COMPUTE_32F + CUBLAS_GEMM_DEFAULT_TENSOR_OP`，覆盖
  Attention投影、Router、使用cuBLAS的专家投影和LM Head；沿用现有输出类型及舍入边界。
  Dense Qwen3仍使用PEDANTIC。该选择与PyTorch 2.7.1常规FP16 GEMM配置对齐，
  不保证不同矩阵形状、融合布局或自定义Decode kernel与HF逐位一致，必须重跑完整模型验收。

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
