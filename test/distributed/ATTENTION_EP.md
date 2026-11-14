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

## 服务器执行顺序

先更新 `sk`，保留服务器自己的未提交改动；如 Git 拒绝更新，先贴出报错，不执行强制重置。

```bash
cd /home/project/vllm_fastllm
git status --short
git fetch origin
git switch sk
git pull --ff-only origin sk
```

固定环境并编译。本脚本使用新构建目录，避免与 CUDA 13 或其他旧 CMake 缓存混用。

```bash
export ATTENTION_EP_PYTHON=/venv/main/bin/python3
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

`regression` 会在独立结果目录生成稠密小模型和 tokenizer，然后调用原有 DP2×TP1、DP4×TP1、
DP2×TP2 验收以及 TP Decode Eager/Graph 验收，不需要另下载稠密模型。

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
- 真实权重使用固定中文 JSONL 请求，独立 Transformers FP16 重算完整历史的末位置 logits。
  另外使用四个固定短序列记录全部层的真实输入、Router、共享/路由专家贡献和层输出，
  经正式单层接口进行四 rank 对照；单层检查关闭权重落盘，避免再次输出几十 GB 权重。
  初始阈值保持小模型 `atol=rtol=0.008`，真实模型 `atol=0.05, rtol=0.01`。
  不会因失败自动放宽阈值。
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
