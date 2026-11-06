# NVFP4 验证

本目录增加 NVFP4 专用的 vLLM 对照测试入口，不修改通用 `forward_check.py`、
`prefill.py`、`decode.py`。
运行脚本会把命令、UTC 时间、工作目录和完整输出一并写入 `test/nvfp4/logs/`。

## 调用条件

| 路径 | 精确条件 |
|---|---|
| dense CUTLASS W4A4 | CUDA 12.8+；CUDA 12.x 编译目标 `100a/120a/121a`，CUDA 13.x 为 `100f/120f/121f`；当前运行时 SM100-SM119 或 SM120-SM129；权重 `NVFP4_BLOCK_16`、blockM=16；输入/输出同为 FP16 或 BF16；支持空 bias 或 FP32 bias；原始 K 为 16 的倍数；K/N 不足 32 时内部补零并裁剪；M>=1 |
| 动态激活 FP4 | SM100 或 SM120-SM129；FP16/BF16 输入；列数为 16 的倍数；dense 正式入口还受上述 K/N、M 条件约束 |
| SwiGLU+FP4 融合 | SM100 或 SM120-SM129；FP16/BF16 gate/up；hidden 为 16 的倍数；已接入 Dense `SwigluLinearAdd` 和 grouped MoE |
| grouped MoE W4A4 | CUDA 12.8+；SM100-SM119支持FP16/BF16，SM120-SM129支持BF16；SwiGLU/GeGLU；仅 `NVFP4_BLOCK_16` routed experts；hidden/inter 为32的倍数；expert数1-256；batch>=1默认优先使用grouped路径；shared expert由正式Dense Linear路径单独计算后累加；不支持EP |
| dense Marlin W4A16 | SM75+；FP16 输入/输出；权重 `NVFP4_BLOCK_16`、blockM=16；无 bias；M>=1；K/N 为 64 的倍数 |

环境变量：

动态激活量化与 vLLM 保持一致：CUDA 12.8 使用 PACK8/128-bit load，CUDA
12.9+ 使用 PACK16/256-bit load；两条路径均由编译器版本自动选择，无需手动宏。

- `FASTLLM_CUDA_NVFP4_W4A4=0`：关闭 dense W4A4。
- `FASTLLM_CUDA_NVFP4_W4A4_STRICT=1`：dense W4A4必须命中CUTLASS；预重排或
  首次GEMM验证失败时直接报错，禁止Legacy/Marlin fallback。
- `FASTLLM_CUDA_NVFP4_SWIGLU_QUANT=0`：关闭 Dense SwiGLU+FP4 量化融合。
- `FASTLLM_CUDA_NVFP4_TRACE=1`：在 log 中打印命中或回退原因；该模式会
  插入CUDA流同步，只用于功能诊断，不用于正式性能测试。
- `FASTLLM_CUDA_MOE_NVFP4_W4A4=0`：关闭 grouped MoE W4A4。
- MoE grouped W4A4后端选择不使用batch/M门槛；首次选择后固定为CUTLASS
  或Legacy，正式推理期间不再切换；
  decode的batch=1仍保留专用路径，batch>1默认优先使用原生grouped W4A4。

## 权重显存生命周期

每个NVFP4 `Data + device` 在权重进入GPU时选择候选后端。dense CUTLASS候选会立即
完成重排并释放原始 `cudaData`，状态记为Prepared；首个真实GEMM同步验证成功后固定
为Cutlass。验证失败则销毁候选cache、从CPU/mmap恢复原始权重，并固定使用legacy。
正式推理开始后不再通过修改环境变量切换已固定后端，运行阶段kernel失败直接报错。

dense W4A4 的首次后端选择不设置 M 门槛。M 只用于选择 CUTLASS 内部 tile/config，
不会导致CUTLASS与legacy在不同batch之间切换。上传阶段负责重排并释放原始CUDA
权重；首个真实GEMM仍负责验证激活量化、GEMM、后处理和异步CUDA状态。

SwiGLU 融合能力单独记录：融合 warmup 失败只关闭融合，随后执行普通 SwiGLU，再由
Linear 独立选择 CUTLASS/Marlin/legacy。它不会把 Linear 一起降级。

CPU/mmap 原始数据继续保留，用于目标GPU重建。NVFP4权重进入GPU时逐个完成上传、
预重排和原始CUDA表示释放，首个真实Linear只负责同步验证GEMM并固定后端。权重从
GPU0移动到GPU1时先销毁GPU0 cache，再从host源直接上传到GPU1并立即预重排，
不再执行“host→GPU0原始权重→GPU1”的中转。

`ops-functional` 会检查：上传预重排后立即释放原始CUDA权重、正式推理不切换后端、
首次GEMM验证失败后恢复并固定legacy、融合拒绝不影响Linear；存在两张GPU时还检查
跨GPU直接重建及GPU0/GPU1输出数值。SwiGLU用“普通SwiGLU+独立NVFP4量化”
作为融合量化的对照；激活量化覆盖真实K=4096/7168/14336。SM100和SM120的
grouped MoE覆盖M=2/64/224、topk=1/8、experts=40/64。

`ops-performance` 的dense W4A4覆盖M=1/16/64/127/128/129/255/256/257/512/1024，
精确验证激活scale的128行边界和SM120的M=256 tile分派边界；另测FP16输出及
Qwen3-8B的4096x6144、4096x24576、12288x4096真实Linear形状。grouped MoE
使用同一组M/topk/experts矩阵。测试开启strict检查：CUTLASS失败时直接报错，
不允许legacy fallback伪装成CUTLASS成绩。执行结束后自动生成
`ops-performance.md`和`ops-performance.csv`，表格包含参数、延迟、带宽和算力。

`ops-compare`保持上述FastLLM算子测试不变，再由独立的
`operator_performance_compare.py`分进程运行相同shape、dtype、warmup和iters的
vLLM算子。vLLM侧借鉴其`benchmark_nvfp4_gemm.py`、
`benchmark_nvfp4_quant.py`和`benchmark_cutlass_moe_nvfp4.py`，覆盖Dense W4A4、
SwiGLU+FP4量化和Grouped MoE W4A4；两边都在框架调用边界计时并在末尾同步GPU。
结果按每个case交替显示FastLLM/vLLM，
并计算`FastLLM speedup = vLLM latency / FastLLM latency`：

```bash
NVFP4_VLLM_PYTHON=/path/to/vllm/python \
  test/nvfp4/run_nvfp4_tests.sh ops-compare
```

输出文件为`operator-performance-compare.{md,csv,json}`；vLLM子进程命令和输出
保存在`operator-performance-compare-vllm.log`。

SwiGLU+NVFP4还提供同一二进制的多版本选优测试。`baseline`保留优化前固定256线程、
M补齐128的kernel；`optimized`使用当前按GPU占用率设置grid且只处理真实M的kernel；
`cached`复用`optimized`的device kernel，并缓存每块GPU的SM启动属性，同时把launch
错误检查移出稳定热路径。三个版本都先执行功能检查，再按rows=1/16/32/64/128和
hidden=4096/7168/14336逐case选出FastLLM最优版本；vLLM只作为固定reference：

```bash
export NVFP4_VLLM_PYTHON=/path/to/python-with-vllm
export REPORT_DIR="$PWD/test/nvfp4/logs/swiglu-versions-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$REPORT_DIR"

python3 test/benchmark/operator_compare/operator_benchmark.py \
  --config test/nvfp4/swiglu_versions_compare.json \
  --output-prefix "$REPORT_DIR/swiglu-versions-compare" \
  2>&1 | tee "$REPORT_DIR/suite-output.log"
```

简化入口：

```bash
NVFP4_VLLM_PYTHON=/path/to/python-with-vllm \
NVFP4_LOG_DIR="$PWD/test/nvfp4/logs/swiglu-versions-$(date -u +%Y%m%dT%H%M%SZ)" \
  test/nvfp4/run_nvfp4_tests.sh swiglu-versions
```

结果包含`swiglu-versions-compare.{md,json}`、原始结果CSV、候选版本横向CSV、
最终选优及vLLM对比CSV和每次执行的完整日志。

## 模型和显卡

- dense W4A4 首选：`nm-testing/TinyLlama-1.1B-Chat-v1.0-NVFP4`；B200/B300（SM100）或 RTX PRO 6000 Blackwell（SM120，96GB）。
- grouped MoE W4A4：`RedHatAI/Qwen3-30B-A3B-NVFP4`；模型卡明确给出vLLM
  离线用法，Qwen3-MoE也由FastLLM支持。单卡约18GB权重，优先在RTX 5090
  （SM120）验证，再在B200/B300（SM100）做架构交叉验证。
- Marlin W4A16：`nm-testing/fp4_nvfp4a16-e2e`；SM75及以上GPU。

先由有对应 GPU 的机器完成构建。本次代码迁移按要求不在本机编译。

## 命令

以下命令都把原始命令和完整输出保存到 `NVFP4_LOG_DIR`。建议同一次验证固定同一个目录。

远端编译。脚本通过 `nvidia-smi` 自动识别 B200/B300、RTX 5090/RTX PRO 6000
等 GPU，把实际算力显式传给 `CUDA_ARCH`，并通过仓库现有的 `install.sh`
编译 FastLLM 和 `optest`，避免旧版 CMake 产生无效的 `compute_native`：

```bash
export NVFP4_LOG_DIR="$PWD/test/nvfp4/logs/remote-sm120-$(date -u +%Y%m%dT%H%M%SZ)"
test/nvfp4/run_nvfp4_tests.sh build
```

算子功能测试：

```bash
test/nvfp4/run_nvfp4_tests.sh ops-functional
```

算子性能测试：

```bash
test/nvfp4/run_nvfp4_tests.sh ops-performance
```

Blackwell 算子功能与性能：

```bash
NVFP4_GPU_PROFILE=sm100 test/nvfp4/run_nvfp4_tests.sh ops
NVFP4_GPU_PROFILE=sm120 test/nvfp4/run_nvfp4_tests.sh ops
```

Marlin 算子功能与性能：

```bash
NVFP4_GPU_PROFILE=preblackwell test/nvfp4/run_nvfp4_tests.sh ops
```

forward_check 只使用 vLLM 作为参考实现。vLLM 和 FastLLM 在两个独立进程中依次
加载同一个 NVFP4 模型，避免两份模型同时占用显存。默认功能模式要求输入
token、首个生成 token 一致且 top-k 重合率达标；首 token logprob 差仍会输出，
但超阈值只记录 `WARN`。显式传入 `--strict-alignment` 时，才要求首 token
logprob 差不超过 `--max-first-logprob-diff`（默认 0.1）。运行环境必须能
`import vllm`。若 vLLM 安装在另一个
Python 环境，用 `NVFP4_VLLM_PYTHON` 指定：

```bash
NVFP4_MODEL=/models/TinyLlama-1.1B-Chat-v1.0-NVFP4 \
NVFP4_VLLM_PYTHON=/path/to/vllm/python \
  test/nvfp4/run_nvfp4_tests.sh forward
```

整体模型对比由独立的`model_performance_compare.py`执行，不修改FastLLM原有
`test/benchmark/prefill.py`和`decode.py`。脚本只用AutoTokenizer构造共享Token
ID：统一使用 `--batch-sizes 1,2,4,8,16,32 --input-tokens 512 --output-tokens 64`。
每个Batch只运行一组完整请求，两张Prefill/Decode图共享同一次测量数据。
旧的Prefill/Decode专用输入、输出和Batch参数已删除。FastLLM和vLLM均通过HTTP
`/v1/completions`直接消费Token ID并回传校验，不再次套用各自的Chat Template。
两个框架分时加载模型，使用相同的多线程HTTP客户端、Greedy采样并忽略EOS，
默认覆盖并发请求数batch=1/2/4/8/16/32；实际GPU Batch由服务端调度决定。
Prefill吞吐为总Prompt Token数除以批次开始至最后一个首Token到达的时间，
包含缓存复用及排队、传输时间，不表示纯Prefill kernel吞吐。

测试同时覆盖`eager/best`和`cold/cache_hit`四种组合。两个后端都启用Prefix
Cache；Cold的warmup及5轮正式测试使用开头不同的Prompt，避免命中已有缓存，
Cache Hit则重复使用相同Prompt。每个Case默认测5轮并逐项取中位数。vLLM的Eager
模式增加`--enforce-eager`，Best模式使用其默认CUDA Graph策略；FastLLM的Eager
模式设置`FASTLLM_CUDA_GRAPH=0`，Best模式设置`FASTLLM_CUDA_GRAPH=1`。结果写入
`model-performance-results/model-performance-compare.{md,csv,json}`，包含TTFT、
TPOT、ITL、E2EL、Prefill/Output吞吐和FastLLM/vLLM比值：

```bash
NVFP4_MODEL=/models/TinyLlama-1.1B-Chat-v1.0-NVFP4 \
  test/nvfp4/run_nvfp4_tests.sh model-performance
```

需要定位Decode差距来自GPU算子还是引擎调度时，使用独立的Nsight Systems
入口。每个Batch先完成相同Prompt的warmup和Prefix Cache填充，再采集一轮
BEST模式请求；模型加载和Cold Prefill不进入报告。默认输出各Batch的TPOT、
请求P50/P95、Batch吞吐、按Decode步归一化的GPU kernel时间与数量、GPU时间线
内部空隙、显存操作、算子类别和Top Kernel。未返回缓存统计的后端会标为
“未报告”，不会误写成缓存未命中：

```bash
python test/nvfp4/decode_nsys_compare.py \
  --model /models/TinyLlama-1.1B-Chat-v1.0-NVFP4 \
  --quantization nvfp4 \
  --result-dir test/nvfp4/logs/decode-nsys \
  --vllm-python /root/miniconda3/bin/python \
  --fastllm-python /root/miniconda3/bin/python \
  --batch-sizes 1,2,4,8,16,32
```

在命令末尾增加`--cpu-trace`，可在同一次Nsight采集中补充稳定Decode窗口的
CUDA API、Kernel提交、GPU排队和`TPOT-GPU跨度`综合差值。Markdown及XLSX
会新增CPU调度与CPU-GPU联合分析；该综合差值包含排队、同步和响应等时间，
不是某个CPU函数的独占耗时。

Nsight结果用于性能归因，不替代未启用Profiler时的正式TPOT结果。脚本会
隔离Nsight importer与目标服务的`LD_PRELOAD`，避免只产生无法解析的
`.qdstrm`文件。

当Nsight Systems已定位到Attention合并Kernel后，可使用独立的Nsight
Compute入口对比FastLLM的FlashInfer merge与vLLM的split-KV combine。该
入口保持双方BEST模式CUDA Graph，并分三阶段执行：轻量NCU发现目标、完整
生命周期Nsys校准、正式NCU采集。最终只剖析首个稳定Decode目标Kernel：

```bash
python test/nvfp4/attention_ncu_compare.py \
  --model /models/TinyLlama-1.1B-Chat-v1.0-NVFP4 \
  --quantization nvfp4 \
  --nsys-result-dir test/nvfp4/logs/decode-nsys \
  --result-dir test/nvfp4/logs/attention-ncu \
  --vllm-python /root/miniconda3/bin/python \
  --fastllm-python /root/miniconda3/bin/python \
  --batch 1
```

结果包含`.md/.json/.xlsx`和双方原始`.ncu-rep`。NCU会重放Kernel，报告
只用于分析工作量、内存效率、占用率和Warp Stall，不能替代正式TPOT。
NCU不直接暴露逻辑Split数量，脚本不会把Grid大小误写成Split数量。每个
后端输出`01-discovery`、`02-startup-calibration`和`03-profile`。重复测试
可用`--skip-discovery`省略发现阶段，但完整生命周期校准始终执行。

一次执行全部：

```bash
NVFP4_MODEL=/models/TinyLlama-1.1B-Chat-v1.0-NVFP4 \
  test/nvfp4/run_nvfp4_tests.sh all
```

主要日志文件：

```text
environment.log
nvcc_version.log
install.log
op_dense_w4a4_decode_padding_bias.log
op_dense_w4a4_decode_perf.log
op_dense_w4a4_prefill_perf.log
op_swiglu_fp4_quant.log
op_swiglu_fp4_quant_fp16.log
op_swiglu_fp4_quant_perf.log
forward_check_vllm.log
forward-vllm-results/vllm.json
forward-vllm-results/fastllm.json
ops-performance.md
ops-performance.csv
operator_performance_compare.log
operator-performance-compare-cases.json
operator-performance-compare-vllm.log
operator-performance-compare-vllm.json
operator-performance-compare.md
operator-performance-compare.csv
operator-performance-compare.json
model_performance_compare.log
model-performance-results/shared-prompt-token-ids.json
model-performance-results/fastllm-eager-server.log
model-performance-results/fastllm-eager-results.json
model-performance-results/vllm-eager-server.log
model-performance-results/vllm-eager-results.json
model-performance-results/fastllm-best-server.log
model-performance-results/fastllm-best-results.json
model-performance-results/vllm-best-server.log
model-performance-results/vllm-best-results.json
model-performance-results/model-performance-compare.md
model-performance-results/model-performance-compare.csv
model-performance-results/model-performance-compare.json
```

指定已有 `optest` 或日志目录：

```bash
NVFP4_OPTEST=/opt/fastllm/optest \
NVFP4_LOG_DIR=/logs/nvfp4-run-001 \
  test/nvfp4/run_nvfp4_tests.sh ops
```

一般不需要设置 `NVFP4_GPU_PROFILE` 或 `NVFP4_CUDA_ARCH`。前者只在自动识别错误或
需要强制测试某条路径时覆盖；后者只在需要交叉编译或指定多个架构时传给 `install.sh`。
