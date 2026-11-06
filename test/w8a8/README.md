# CUDA W8A8 验证

结论：测试只在对应GPU机器运行。`build`负责编译安装；其他suite把每条
命令、UTC时间和完整输出写入`test/w8a8/logs/<timestamp>/`。

SM90/SM120标准FP8 W8A8采用固定后端生命周期：每份权重在每张GPU上只进行
一次真实CUTLASS GEMM选择。首次同步验证成功后固定为CUTLASS；失败则固定
交给Legacy，正式推理期间不再混切。FP8权重本身已经是CUTLASS可消费格式，
不会复制第二份权重；设备scale、类型化bias、量化激活和workspace临时区会
缓存复用。scale、bias与输出类型转换已融合进CUTLASS epilogue，不再分配
完整的FP32中间矩阵或额外启动缩放kernel。

## 严格调用条件

| 路径 | 条件 |
|---|---|
| SM90 INT8 dense | 编译含 `90a`，CUDA 12.3+；运行时精确 SM90；FP16/BF16 输入输出同 dtype；2D signed-I8 权重；权重必须 symmetric、无 zero/min；每输出通道一个 FP32 scale；激活支持动态 per-token symmetric/asymmetric以及静态 tensorwise symmetric/asymmetric INT8；非对称路径在CUTLASS epilogue融合AZP修正；K%16=0、N%8=0；bias 为空或 FP32[N] |
| SM90 standard FP8 dense | 编译含 `90a`，CUDA 12.3+；运行时精确 SM90；FP16/BF16；FP8 E4M3 权重；weight `blockK=1, blockM=K` 且 scale 数为N（per-channel）或1（tensorwise）；激活动态 per-token；K%16=0、N%8=0；bias 为空或 FP32[N]；CUTLASS使用FP8 FastAccum与Persistent调度，小M按vLLM策略走SwapAB |
| SM90 FP8 blockwise dense | 编译含 `90a`，CUDA 12.3+；运行时精确 SM90；FP16/BF16；FP8 E4M3 权重；activation/weight block 都是 128x128；M/N/K 正数，K/N 为 128 倍数；bias 为空或 FP32[N] |
| SM90 FP8 grouped MoE | 运行时精确 SM90；FP16/BF16输入输出同 dtype；SwiGLU；routed expert使用CUTLASS grouped GEMM，shared expert复用正式Dense FP8 W8A8路径后加权合并；expert为standard tensorwise/per-channel FP8 E4M3（不是blockwise）；合法route index；hidden/inter为16倍数；按vLLM对齐M<=4、M<=64、N>=8192、K>=8192和默认五档调度；不满足直接回现有MoE路径 |
| SM90 FP8 Block128 MoE | 运行时精确 SM90；FP16/BF16输入输出同 dtype；SwiGLU；权重为FastLLM交错Block128格式，每128个FP8值后跟一个FP32 scale；激活动态量化为1x128；hidden/inter均为128倍数；同时覆盖独立expert的MergeMOE与3D gate/up/down的FusedMOE，二者复用Triton grouped路由、GEMM与归并；shared expert复用正式Dense Block128路径；失败回退已有CUDA实现 |
| SM120 standard FP8 dense | 编译含 `120a`/`120f`，CUDA 12.9+；运行时精确 SM120，不含 SM121；FP16/BF16；FP8 E4M3 权重；weight `blockK=1, blockM=K` 且 scale 数为N（per-channel）或1（tensorwise）；激活动态 per-token；K%16=0、N%8=0；bias 为空或 FP32[N] |

## 命令

编译安装：

```bash
W8A8_NVCC=/usr/local/cuda-13.0/bin/nvcc \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm120-build" \
  test/w8a8/run_w8a8_tests.sh build
```

算子功能（FastLLM分支、固定后端生命周期及vLLM官方形状/scale组合）：

```bash
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm120-function" \
  test/w8a8/run_w8a8_tests.sh ops-functional
```

SM90 INT8 AZP可单独执行，不混入FP8 Dense、Block128或MoE case：

```bash
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm90-int8-azp-function" \
  test/w8a8/run_w8a8_tests.sh ops-int8-azp-functional
```

SM90动态per-token对称INT8 W8A8也可单独执行：

```bash
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm90-int8-symmetric-function" \
  test/w8a8/run_w8a8_tests.sh ops-int8-symmetric-functional
```

SM90标准FP8 W8A8 grouped MoE可单独执行；该入口覆盖五档CUTLASS调度、
FP16/BF16、tensorwise/per-channel Scale、Block128 Triton grouped路径和
shared expert，并生成XLSX汇总：

```bash
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm90-fp8-moe-function" \
  test/w8a8/run_w8a8_tests.sh ops-moe-functional
```

功能套件结束后会在`W8A8_LOG_DIR`直接生成
`operator-functional-summary.{md,csv,json,xlsx}`。表格逐项记录后端、类别、
M/N/K、输入类型、权重scale布局、激活量化、实际后端路径、bias、
检查编号、误差和PASS/FAIL结果；同时保存GPU、CC、CUDA、编译架构、
Git提交、完整命令及warmup/iters；
若某个case失败导致套件提前退出，退出钩子仍会汇总已经执行的case及失败项。

其中SM120生命周期包含四类用例：首次GEMM故障后固定为`Rejected`且
不再重试；已固定为`Cutlass`后的运行故障必须抛错而不能fallback；权重
`Data`析构后必须删除以其地址为键的后端状态，防止新对象继承陈旧记录；
权重从GPU0迁移到GPU1时必须清理旧卡状态，并从主机scale源在新卡重建缓存。
故障通过测试专用环境变量注入，但调用的仍是正式W8A8 CUTLASS入口和状态机。

算子性能（直接生成FastLLM/vLLM Markdown、CSV、JSON和XLSX对比）：

```bash
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm120-perf" \
  test/w8a8/run_w8a8_tests.sh ops-performance
```

SM90 INT8 AZP独立性能对比：

```bash
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm90-int8-azp-perf" \
  test/w8a8/run_w8a8_tests.sh ops-int8-azp-performance
```

SM90对称INT8独立性能对比；FastLLM与vLLM使用完全相同的BF16激活、
INT8权重和per-channel权重scale，并都把动态per-token激活量化计入延迟：

```bash
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm90-int8-symmetric-perf" \
  test/w8a8/run_w8a8_tests.sh ops-int8-symmetric-performance
```

性能报告中的`fastllm_speedup_vs_vllm_x`按
`vLLM耗时 / FastLLM耗时`计算：大于1表示FastLLM更快，小于1表示
FastLLM更慢。每个形状默认独立执行5轮，报告同时给出Median、P95、CV、
Effective TOPS和实际后端路径；原始轮次样本保存在JSON/CSV的
`*_samples_ms`字段中。Effective TOPS按`2*M*N*K/Median延迟`计算，
属于有效计算吞吐；CV越小表示测试越稳定。MD、CSV和XLSX中的浮点展示值
统一保留小数点后3位，JSON保留原始精度供复算。成功完成双方测试且
FastLLM严格目标路径生效的case标记为`PASS`；性能高低本身不决定PASS/FAIL。

forward_check：

默认执行功能判定：输入Prompt和首Token必须一致，Top10重合率必须不低于
0.8。首Token logprob差仍会打印，但超过0.1时只警告，不会使功能测试失败。
结果目录生成`forward-check-summary.{md,csv,json,xlsx}`，明确记录功能判定、
严格对齐诊断项及最终PASS/FAIL。

```bash
W8A8_MODEL=/root/autodl-tmp/neuralmagic/Qwen2___5-7B-FP8-dynamic \
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm120-forward" \
  test/w8a8/run_w8a8_tests.sh forward
```

标准Llama的KV布局回归使用单个正式生成请求，依次验证`continuous+auto`、
`paged+native_paged+strict`和`paged+flashinfer_paged+strict`。该套件专门防止
batch=1通过旧`ForwardBatch`静默退回Continuous；显式Paged未出现在KV布局
Trace中或指定Backend未被确认时必须失败。InternLM2、Phi3和MiniCPM当前只
声明Continuous能力，不纳入Paged组合。

```bash
W8A8_MODEL=/root/autodl-tmp/neuralmagic/Meta-Llama-3___1-8B-Instruct-quantized___w8a8 \
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/llama-attention-layouts" \
  test/w8a8/run_w8a8_tests.sh forward-attention-layouts
```

需要严格数值对齐时，直接执行共享入口并增加
`--strict-alignment`；此时首Token logprob差超过0.1会使测试失败。

整体模型性能（相同HTTP接口、相同Token ID，生成FastLLM/vLLM对比表）：

```bash
W8A8_MODEL=/root/autodl-tmp/neuralmagic/Qwen2___5-7B-FP8-dynamic \
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/sm120-model" \
  test/w8a8/run_w8a8_tests.sh model-performance
```

结果目录生成`model-performance-compare.{md,csv,json,xlsx}`。成功完成全部
请求并得到完整对应记录的case标记为`PASS`；速度比不作为正确性判定。

整体性能入口统一使用 `--batch-sizes 1,2,4,8,16,32 --input-tokens 512 --output-tokens 64`，
删除旧的Prefill/Decode专用Batch、输入和输出参数。每个Batch只执行一组
完整请求；Prefill图（TTFT、Prefill吞吐、E2EL）和Decode图（TPOT、ITL、
Output吞吐、E2EL）共用测量结果。Prefill吞吐按总Prompt Token数除以
批次开始至最后一个首Token到达的时间计算，包含缓存复用Token及排队、
传输时间，不表示纯Prefill kernel吞吐。

标准Llama需要横向比较KV布局和Paged Backend时，使用矩阵入口。它依次运行
`continuous+auto`、`paged+native_paged+strict`和
`paged+flashinfer_paged+strict`，每组仍完整覆盖Eager/Best、Cold/Cache Hit及
统一请求batch=1/2/4/8/16/32。各组报告分别保存，同时在矩阵根目录生成合并的
`model-performance-attention-layout-matrix.{md,csv,json,xlsx}`和跨布局趋势图。

```bash
W8A8_MODEL=/root/autodl-tmp/neuralmagic/Meta-Llama-3___1-8B-Instruct-quantized___w8a8 \
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/llama-attention-model-performance" \
  test/w8a8/run_w8a8_tests.sh model-performance-attention-layouts
```

Decode性能归因使用独立的`decode_nsys_compare.py`。它让两个后端分时加载
同一模型；每个Batch先用同一Prompt完成CUDA Graph、shape和Prefix Cache
warmup，再由Nsight Systems只采集一轮BEST模式HTTP请求。默认覆盖
batch=1/2/4/8/16/32，并输出GPU kernel、CUDA API、显存操作、按名称推断的
算子类别以及Top Kernel表。报告同时提供请求P50/P95、Batch吞吐、按稳定
Decode步归一化的GPU时间和kernel数，以及GPU事件跨度内的空闲时间；未返回
缓存统计时明确标记为“未报告”。Nsight会扰动绝对延迟，因此报告用于解释
时间构成，正式TPOT仍以`model-performance`结果为准。
每个成功完成采集和稳定Decode边界校验的后端/Batch记录标记为`PASS`。

```bash
python test/w8a8/decode_nsys_compare.py \
  --model /root/autodl-tmp/neuralmagic/Qwen2___5-7B-FP8-dynamic \
  --result-dir test/w8a8/logs/decode-nsys \
  --vllm-python /root/miniconda3/bin/python \
  --fastllm-python /root/miniconda3/bin/python \
  --batch-sizes 1,2,4,8,16,32 \
  --prompt-tokens 512 \
  --output-tokens 64
```

标准Llama的Nsight矩阵使用同样三种组合，并为每种组合覆盖
batch=1/2/4/8/16/32。默认开启`--cpu-trace`，采集汇总完成后删除大体积
`.nsys-rep`、SQLite和导出CSV；设置`W8A8_KEEP_NSYS_REPORTS=1`才会保留。
矩阵根目录生成`decode-nsys-attention-layout-matrix.{md,csv,json,xlsx}`和跨布局
GPU/CPU提交趋势图。

```bash
W8A8_MODEL=/root/autodl-tmp/neuralmagic/Meta-Llama-3___1-8B-Instruct-quantized___w8a8 \
W8A8_VLLM_PYTHON=/root/miniconda3/bin/python \
W8A8_LOG_DIR="$PWD/test/w8a8/logs/llama-attention-decode-nsys" \
  test/w8a8/run_w8a8_tests.sh decode-nsys-attention-layouts
```

若Nsight Systems已经把Attention差距定位到FastLLM的
`PersistentVariableLengthMergeStatesKernel`和vLLM的
`flash_fwd_splitkv_combine_kernel`，可再用独立的Nsight Compute入口做
同Batch、同Prompt的单Kernel对比。双方保持BEST模式CUDA Graph。脚本先从
进程启动执行一次轻量NCU发现，确认目标Kernel可见；再用完整生命周期Nsys
统计模型启动、Graph warmup、显式warmup及正式请求的调用数；最后从进程
启动执行正式NCU，只采集首个稳定Decode合并Kernel。

```bash
python test/w8a8/attention_ncu_compare.py \
  --model /root/autodl-tmp/neuralmagic/Qwen2___5-7B-FP8-dynamic \
  --nsys-result-dir test/w8a8/logs/decode-nsys \
  --result-dir test/w8a8/logs/attention-ncu \
  --vllm-python /root/miniconda3/bin/python \
  --fastllm-python /root/miniconda3/bin/python \
  --batch 1 \
  --prompt-tokens 512 \
  --output-tokens 64
```

输出`attention-ncu-compare.{md,json,xlsx}`及双方原始`.ncu-rep`。报告比较
Duration、DRAM读写、带宽、Occupancy、寄存器、共享内存和Warp Stall。
NCU不直接提供逻辑Split数量，不能用Grid大小代替Split数量。NCU会重放
Kernel，因此该结果只用于解释合并Kernel为何有差距，不用于计算TPOT。
每个成功命中目标Kernel并完成NCU指标解析的后端记录标记为`PASS`。
每个后端会依次生成`01-discovery`、`02-startup-calibration`和`03-profile`
目录。重复测试且已经确认NCU能看到目标Kernel时，可用`--skip-discovery`
省略第一阶段；不得跳过完整生命周期校准。

需要进一步区分GPU执行和CPU提交/调度影响时，在同一命令末尾增加
`--cpu-trace`。脚本会在同一次采集中增加OS Runtime、逐次CUDA API和
Kernel提交/排队数据，并在Markdown与XLSX中新增`CPU调度汇总`、
`CPU-GPU联合分析`、`CUDA API分类`和`Top CUDA APIs`。其中
`TPOT-GPU跨度`只是请求排队、CPU调度、同步和响应等未解释时间的综合差值，
不能当成某个CPU函数的独占耗时。

为避免Nsight importer被FastLLM使用的系统`libstdc++`干扰，脚本不会把
`LD_PRELOAD`传给Nsight控制进程，只会把它传给被采集的模型服务。如果某个
Nsight安装仍只生成`.qdstrm`而没有`.nsys-rep`，脚本会停止并给出明确错误，
不会用不完整数据生成对比表。

## 模型（魔塔优先）

| 用途 | ModelScope 模型 |
|---|---|
| SM90 INT8 dense，最小 forward_check | `neuralmagic/Qwen2.5-0.5B-Instruct-quantized.w8a8` |
| SM90/SM120 FP8 blockwise dense | `Qwen/Qwen3-30B-A3B-FP8`（block=128；同时可测 MoE 现有 blockwise 路径） |
| SM90 standard FP8 grouped MoE | ModelScope 未找到可直接匹配 FastLLM 独立 expert 布局的模型；HF fallback：`nm-testing/Mixtral-8x7B-Instruct-v0.1-FP8-Dynamic` |
| SM90/SM120 standard FP8 dense，较小 | `neuralmagic/Qwen2.5-7B-FP8-dynamic` |

注意：SM90 grouped CUTLASS 的目标格式是 standard per-token/per-channel FP8；Qwen3 官方 FP8 是 128x128 blockwise，不能拿它证明 standard grouped kernel 命中。
HF Mixtral fallback 的 `config.json` 是 compressed-tensors `float-quantized`：FP8 weight 为 static symmetric channel，activation 为 dynamic symmetric token。

开关：`FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90=0` 可单独关闭SM90 grouped CUTLASS，便于和原grouped CUDA fallback做A/B性能对比；`FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90_STRICT=1` 会在目标CUTLASS提交失败时直接报错，避免功能或性能测试静默回退。
