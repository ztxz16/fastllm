# 多进程 TP 底座验收

本目录按阶段提供 Linux 单机进程底座、Qwen3 分片/GPU 权重加载、单层前向，
以及完整 Prefill、请求 KV/Decode 和文末新增的生成验收。通信底座包括独立 C++ worker、Unix socket、
NCCL 初始化和标量 SUM 自检、心跳、超时、退出与故障回收。
**Qwen3 稠密模型的多进程 TP 生成、常驻请求、正式模型工厂和 Decode CUDA Graph
已完成用户服务器验收。本轮新增双卡 CUDA IPC 自定义 AllReduce，GPU 验收待执行。**
默认 `thread` 保持原行为；进程后端使用 C++/CUDA + NCCL，只支持未量化稠密 Qwen3。
跨请求前缀缓存仍未实现。文中各阶段说明是历史边界，
当前入口及限制以文末“正式入口接入”“Decode CUDA Graph”和“自定义 AllReduce”各节为准。
构建选项 `BUILD_PROCESS_TP` 默认关闭，显式开启后才构建进程产物。

## 运行条件

- Linux 5.3+，内核与容器策略允许 `pidfd_open`、`pidfd_send_signal`。
- 底座脚本 Python 3.9+；模型工厂和 HTTP/SSE 工具需要 Python 3.10+。部分 Conda Python 没有暴露 pidfd 接口，验收器在 Linux x86_64/aarch64
  上通过系统调用提供同等语义；仍需内核和容器允许 pidfd，不退回裸 PID 发信号。
- 验收器须允许 `prctl(PR_SET_CHILD_SUBREAPER)`，以回收协调器死亡后的孤儿 worker。
- CUDA/NCCL 检查需要 NVIDIA GPU、CUDA 开发环境及 NCCL 头文件和库。
- 设备编号是 `CUDA_VISIBLE_DEVICES` 生效后的逻辑编号；不能重复指定同一 GPU。
- 控制面测试不初始化 CUDA/NCCL，也可使用 CPU 构建产物。
- 此脚本不编译。下方构建命令只在用户的服务器执行，本地工作区禁止编译。

## 服务器构建与检查

在仓库根目录执行，使用独立构建目录，避免覆盖已有构建：

```bash
cmake -S . -B build-process-tp -DUSE_CUDA=ON -DBUILD_PROCESS_TP=ON -DCUDA_ARCH=native
cmake --build build-process-tp --target fastllm_tp_worker fastllm_tp_check -j 4

python3 test/distributed/run_process_tp.py \
  --check build-process-tp/fastllm_tp_check \
  --worker build-process-tp/fastllm_tp_worker \
  --devices 0,1 --cuda \
  --build-cache build-process-tp/CMakeCache.txt
```

沿用服务器原有 CUDA/NCCL 路径配置。编译器不支持 `native` 时，将 `CUDA_ARCH`
改为服务器 GPU 对应架构；不要复制其他 GPU 的架构值。独立 worker 通过执行文件
启动，不继承父进程已加载的 CUDA 库，因此还须保证其运行时库可被动态链接器找到。
打包时会携带 worker 和检查程序，并为常见 pip NVIDIA 库目录提供相对 RPATH。

只在服务器验证控制面时：

```bash
cmake -S . -B build-process-tp-cpu -DUSE_CUDA=OFF -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON
cmake --build build-process-tp-cpu --target fastllm_tp_worker fastllm_tp_check -j 4
python3 test/distributed/run_process_tp.py \
  --check build-process-tp-cpu/fastllm_tp_check \
  --worker build-process-tp-cpu/fastllm_tp_worker \
  --devices 0,1 --build-cache build-process-tp-cpu/CMakeCache.txt
```

仅运行 Python 参数与进程清理检查，不导入推理动态库、不编译、不需要 GPU：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 test/distributed/run_process_tp.py --frontend-only
```

验证容器 PID 1 可以作为协调器时，在服务器的验收命令上加 `--pid-namespace`。
该项通过 `unshare --user --map-root-user --pid --fork --kill-child=SIGKILL` 启动真实检查程序，
要求允许创建 user/PID namespace，且已安装 util-linux。检查程序必须确认自身 PID 为 1，
再通过公开进程底座完成 worker 启动、Ping 和正常退出。缺少命令或权限时记 FAIL，不静默跳过。
该项默认不启用，不能把普通控制面通过当成 PID 1 场景已验证。

## 检查内容与报告

- 通过公开 `ProcessGroup` 接口验证正常启停、三次重启和幂等 Stop。
- 在临时线程内启动进程组，线程退出后继续 Ping，检查 worker 生命周期不依赖启动线程。
- 启动失败、握手超时、协议版本错误、会话错误、重复设备、worker 崩溃和 SIGSTOP 卡死。
- 父进程被 SIGKILL 后，使用 pidfd 确认 worker 退出，避免检查复用的裸 PID。
- 每个验收 case 使用独立 session/进程组；超时、异常、中断以及正常退出均清理整组，
  并限时回收被接管的孤儿进程。worker 须保留继承的进程组，不自行脱离 session。
- Python 回归实际创建 SIGSTOP 子进程，检查超时、错误输出、非零退出、Ctrl+C、父进程
  死亡及启动失败后的回收和日志；同时确认不会清理或回收其他 case 的进程。
- 父进程死亡测试先只杀协调器，观察 worker 自行退出；兜底清理不能替代这项断言。
- 报告读取支持 json11 的多行 JSON 和前置日志；父进程死亡握手等待完整 JSON，
  同时保留截止时间、输出大小限制及失败时的部分日志。
- CUDA 路径每卡提交 FP32 标量 `rank + 1`，检查真实 NCCL SUM 输出。
  例如两卡每卡均须返回 `3`，四卡均须返回 `10`；记录最大/平均绝对误差和失败元素数。
- CUDA 检查没有 CPU fallback，缺少 CUDA 或 NCCL 时失败。

每次运行新建带 UTC 时间和随机后缀的目录，保存原始日志及 Markdown、CSV、JSON、XLSX。
生命周期和参数检查没有数值张量输出，误差字段标为 `n/a`。CUDA/NCCL 版本与 GPU 名称
由实际 worker 返回，架构配置保存于 CMakeCache。JSON 保留原始数值，其他表格浮点保留三位。
任一 case 失败返回非零，尽可能保存已经完成的结果。

这是功能验收，不是性能基准；不报告吞吐或加速比。没有真实 C++/GPU 运行结果时，
只能声称 Python 参数、进程清理回归或静态检查通过。

## 底座边界

- 协议使用定长小端头部、版本、会话代次、单调命令序号和有界 payload，不传裸指针。
- 一次组操作先向全部 rank 发送再接收，共用截止时间；当前控制面串行化所有命令。
- 任一 rank 故障使整组进入 Failed，终止其他 rank，拒绝后续命令。
- 失败后需显式 Stop 才能重启；不自动重试可能已经执行的 CUDA 操作。
- 普通超时会发送 SIGKILL 并限时回收。内核不可中断状态等异常情况下仍可能无法及时回收；
  此时保留错误并阻止重启，不承诺强制信号能解除驱动或内核挂死。
- Ready 仅表示底座就绪，不代表模型可接收请求。

## Qwen3 按 rank 读取权重（独立组件，尚未接入推理）

新增 `Qwen3ShardLoader` 公共 C++ 接口。构造时只读 HF 配置、索引和 safetensors
文件头；`LoadTensor` 直接读取本 rank 的行/列区间，合并到最终本地缓冲区。
这一步没有把权重装入 GPU，也没有启用 `--tp-backend process` 推理。

支持范围：

- 标准 HF `model_type=qwen3` 稠密模型，单文件或 `weight_map` 多文件 safetensors。
- F16、BF16、F32 原样读取，不进行精度转换；融合的来源张量必须 dtype 相同。
- QKV 按头切分、Gate/Up 成对切分、O/Down 按列切分、LM head 按词表切分。
- Embedding 和 Norm 复制；tied embedding 缺少独立 LM head 时从 embedding 读取词表分片。
- 维度非整除时均匀分配余数；TP 大于 KV 头数时复制 KV，保持 Q/O 分片对应。
- 可选完整 Q/K/V bias；O/Down bias 仅 rank 0 读取，其他 rank 输出零，避免归约重复相加。
- 未量化权重之外的 dtype、预融合源权重、Gate/Up bias、未知权重与非法布局明确拒绝。
  尚不支持设备比例配置或线程后端的逐层分片尺寸轮换。
- 输入文件在读取器生命周期内必须保持不变。读取器检查文件大小变化与短读，
  不保证检测同尺寸的外部篡改。调用方提供每个目标张量的分配上限。
- 相对模型路径在构造时固定，后续改变工作目录不会切换来源。配置、索引和
  safetensors 文件头的重复 JSON 字段（含转义后的同名键）明确拒绝，不采用覆盖值。

在**服务器**编译并运行（这些命令不会由验收脚本自动执行）：

```bash
cmake -S . -B build-process-tp-04d4599f \
  -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON -DCUDA_ARCH=native
cmake --build build-process-tp-04d4599f --target fastllm_tp_shard_check -j 4
python3 test/distributed/run_qwen3_shards.py \
  --check build-process-tp-04d4599f/fastllm_tp_shard_check \
  --build-cache build-process-tp-04d4599f/CMakeCache.txt
```

该检查是纯 CPU 文件读取测试，不需要下载模型或占用 GPU。也可以在服务器使用
`-DUSE_CUDA=OFF` 的独立构建目录。它不代表 CUDA 前向或进程推理已经通过。

首个真实模型选用官方 [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
未量化 safetensors 版本，保留 config.json、权重及 tokenizer 相关文件。
官方配置为 28 层、16 个 Q 头、8 个 KV 头、共享 embedding，权重 dtype 为 BF16；
这些维度适合先验证 TP=2。当前 CPU 读取器原样读取 BF16，不要求预先转 FP16。
不要选 GGUF、AWQ、GPTQ、FP8 或 MoE 变体；这些格式尚未接入此读取器。
后续 V100 GPU 前向验证计划使用 FP16，精度转换还需在接入前向时实现与验证。
若导出真实模型分片，embedding 的单张量大小超过默认 512 MiB，应添加
`--max-bytes 1073741824`；仅检查元数据不分配该权重缓冲区。

验收器保留固定小型输入、逐 rank 原始输出、完整命令与日志，验证每个目标张量的
dtype/shape、逐元素数值和原始字节，并将分片重组后与原始权重比较；覆盖三种 dtype、
TP=1/2/3/4/8、MHA/GQA、不均匀 KV 分组、共享权重、偏置、多文件和拒绝路径。
还覆盖构造后切换工作目录、重复 JSON 字段和包含转义的合法 JSON。
结果保存为 `summary.md/csv/json/xlsx`，任一 case 失败退出非零。全部预定用例完成前
汇总不能为 PASS，JSON 记录 `complete` 与完成数量；异常退出不能充当正常拒绝。
超时和中断也保留已写出的 stdout/stderr。禁止 `python -O`。

报告中的 `payload_bytes_read` 是请求读取的 payload 字节数，不是真实磁盘带宽或
页缓存读取量；`max_tensor_buffer_bytes` 是最大单个返回缓冲区，不是测得的进程 RSS。
正式 worker 持有的权重总量可由 `tensor_bytes` 估计，仍需后续接入后的真实内存验证。

仅检查真实模型元数据，不读取权重 payload：

```bash
build-process-tp-04d4599f/fastllm_tp_shard_check \
  --model /path/to/Qwen3 --rank 0 --world-size 2
```

带 `--output NEW_DIR` 会逐张量读取并导出原始字节；目录必须不存在，失败时保留已导出部分。
`--max-bytes` 默认 512 MiB，限制单个目标张量；大模型 embedding 可能需要显式提高此值。
检查程序的 `--change-cwd DIR` 仅用于构造后切换目录的回归，仍调用相同公共读取接口。

仅运行验收器自身的 Python 回归（不验证 C++，不编译）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s test/distributed -p 'test_qwen3_shard_runner.py' -v
```

后续继续接入 Qwen3 rank-local 前向、KV 生命周期与统一采样。
CUDA Graph、自定义 AllReduce、跨请求前缀缓存仍需后续逐项适配。

## Qwen3 多进程 GPU 权重驻留（待服务器验收）

新增 `ProcessGroup::LoadQwen3 / ReadQwen3Weight / UnloadQwen3`。每个 worker
自行调用原分片读取器，CPU 同时只保留单个源分片和至多 1 MiB 转换缓冲，
全部权重转换为行主序 FP16 后驻留当前 GPU。BF16 转换在 CPU 上完成，因此
不要求 V100 原生支持 BF16。CPU 源字节上限与 GPU 权重总字节上限独立；
在读取 payload/分配 GPU 前检查配额。拒绝 NaN、Inf 和绝对值大于 65504 的值。

权重阶段只提供权重存储；Norm 权重也按 FP16 保存，
单层前向见下节，仍须服务器验证数值精度和算子布局。量化、设备迁移、CUDA Graph、
自定义 AllReduce、KV/前缀缓存均未在此阶段实现。原线程 TP 路径未改变，
Python `--tp_backend process` 仍保留未实现保护，不得据此宣称模型生成已完成。

生命周期：同一 worker 执行 stream 上同步上传和读回；加载成功后才发布权重。
重复加载必须先卸载；卸载同步释放，重复卸载安全。加载或 CUDA 失败时不尝试
继续复用部分权重，由控制面终止并回收整组 worker。正常停止先释放权重再销毁
通信组。GPU 指针只在对应 worker 内有效，不经控制协议传输。

在**服务器**构建（V100、CUDA 12.6、CUDA 12.6 版 NCCL）：

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1

cmake -S . -B build-process-tp-cu126 \
  -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
cmake --build build-process-tp-cu126 \
  --target fastllm_tp_worker fastllm_tp_check fastllm_tp_model_check -j 4

# 先确认原通信/生命周期行为未回归。
python3 test/distributed/run_process_tp.py \
  --check build-process-tp-cu126/fastllm_tp_check \
  --worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1 --cuda \
  --build-cache build-process-tp-cu126/CMakeCache.txt

# 真实 worker 加载小型确定性权重；独立 Python 参考验证全部 GPU 读回字节。
python3 test/distributed/run_qwen3_device_weights.py \
  --check build-process-tp-cu126/fastllm_tp_model_check \
  --worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1 \
  --build-cache build-process-tp-cu126/CMakeCache.txt
```

GPU 验收生成 Markdown/CSV/JSON/XLSX 报告、完整命令、原始输出、固定输入及
GPU 读回文件。逐元素报告最大/平均绝对误差、失败元素数；逐字节验证包含负零。
重载用例的第二轮使用同形状但数值不同的模型，检查旧权重是否残留；检查器启动
失败、超时或中断时，stderr 保留失败原因及完整命令。
用例含 F16/BF16/F32、全部有限 FP16 值、F32 ties-to-even/次正规舍入、1 MiB
分块边界、配额边界、重复加载、卸载后读、非法读取范围，以及只在最后 rank
出现的非有限值/溢出。拒绝用例必须同时验证整组 Failed 且 worker 已回收。
不是性能测试，不报告吞吐或加速比。运行 `--devices 0` 可补单 rank 验证。

真实 Qwen3-1.7B 可通过同一检查程序进行加载/读回冒烟，约额外写入 4.4 GiB；
该命令本身只检查状态及字节计数，不提供真实模型的独立数值或推理正确性结论：

```bash
TP_DEVICE_RUN=$(mktemp -d "$PWD/test/distributed/results/qwen3-device-real-XXXXXX")
build-process-tp-cu126/fastllm_tp_model_check \
  --worker build-process-tp-cu126/fastllm_tp_worker \
  --model /home/models/Qwen3-1.7B --devices 0,1 \
  --output "$TP_DEVICE_RUN/weights" \
  >"$TP_DEVICE_RUN/report.json" 2>"$TP_DEVICE_RUN/stderr.log"
```

本地仅允许不触发编译的 Python 回归：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s test/distributed -p 'test_*.py'
```

## Qwen3 单层 TP 前向

用户回传的双 V100 日志显示提交 `135809d7` 编译成功，并通过通信底座、GPU
权重和单层验收。单层报告在服务器 `test/distributed/results/20261106T082732Z_layer_02b42bad`。
后续完整 Prefill 改动复用并调整单层工作区，须重新执行以下单层回归。

`ProcessGroup::ForwardQwen3Layer` 调用正式 worker 持有的 GPU 权重和执行 stream，
使用 cuBLAS GEMM、CUDA 算子及 NCCL；验收工具不另建测试专用计算路径。
支持单请求、连续位置、无历史 KV 的完整 causal prefill，执行一个 Decoder 层：
Input RMSNorm → QKV → Q/K RMSNorm → split-half RoPE → causal GQA → O projection
→ NCCL SUM → 残差 → Post RMSNorm → Gate/Up → SiLU*Up → Down → NCCL SUM → 残差。
支持可选 QKV/O/Down bias；O/Down bias 由原加载器仅在 rank 0 保留。
KV 头复制及非均匀头/MLP 分片沿用同一分片计划。

该入口提供 `Full`、`Attention`、`Mlp` 三个阶段，独立阶段仍使用完整层中的同一
算子与通信逻辑。输入/结果经控制协议以 FP32 传输，内部激活/权重为 FP16，
GEMM 用 `CUBLAS_COMPUTE_32F_PEDANTIC` 累加；O/Down 的 rank-local 结果以
FP32 规约，整体转 FP16 后与残差相加。RMSNorm 归一化后转 FP16 再乘 scale；
RoPE 使用 FP16 sin/cos 与乘法舍入；Attention 的 score/softmax/V 累加为 FP32。
这是明确的 eager 正确性基线，不是速度或完整 HF 模型一致性的验收结论。

边界：token 数 1..128、hidden<=65536、偶数 head_dim<=256、本地融合 QKV
宽度<=65536、本地 MLP 宽度<=262144，临时缓冲总上限 256 MiB；单个请求仍受
1 MiB 控制帧限制（24 字节任务头 + T*H*4 字节输入）。连续位置起点<=1048576，
只影响 RoPE，不意味着 KV 已存在。非 SiLU、rope scaling/rope_parameters、
部分旋转或 sliding attention 配置明确拒绝。输入/结果非有限值拒绝，绝无 CPU fallback。
本单层接口不提供持久 KV、Decode、全模型 logits/采样、CUDA Graph 或前缀缓存；
Python `--tp_backend process` 仍保持未实现保护。

在上述 CUDA 12.6 环境下，于**服务器**构建和验收：

```bash
cmake -S . -B build-process-tp-cu126 \
  -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -DCUDA_ARCH=70
cmake --build build-process-tp-cu126 \
  --target fastllm_tp_worker fastllm_tp_check fastllm_tp_model_check fastllm_tp_layer_check -j 4

python3 test/distributed/run_process_tp.py \
  --check build-process-tp-cu126/fastllm_tp_check \
  --worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1 --cuda \
  --build-cache build-process-tp-cu126/CMakeCache.txt
python3 test/distributed/run_qwen3_device_weights.py \
  --check build-process-tp-cu126/fastllm_tp_model_check \
  --worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1 \
  --build-cache build-process-tp-cu126/CMakeCache.txt
python3 test/distributed/run_qwen3_layer.py \
  --check build-process-tp-cu126/fastllm_tp_layer_check \
  --worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1 \
  --build-cache build-process-tp-cu126/CMakeCache.txt
```

单层验收用相同的固定输入、源精度和模型配置对比独立 CPU 数学参考、单 GPU
进程入口和双 GPU TP 的每个 rank。逐元素阈值为 `0.004 + 0.004*abs(reference)`；
rank 之间以及改变尾 token 前后的前缀必须完全相同。覆盖三种源精度、偏置、
非均匀 KV 分片、KV 复制、单 token/128 token、head_dim=128/256、位置边界、
层号/形状/阶段错误、卸载后访问及不支持配置，共 25 个用例。
实际值或参考值出现 NaN/Inf 均失败；预期拒绝用例还必须确认退出清理成功。
保存完整实际输出、参考数据、命令、GPU/CUDA/NCCL/
构建信息及 Markdown/CSV/JSON/XLSX 汇总。任何失败均退出非零；中断也保留部分报告。

检查程序 `--tasks` 接收有界 JSON 数组，每个任务包含 `name, layer, tokens,
hidden, position_begin, stage, input`，其中 input 为行主序 FP32 数组，stage 为
0/1/2。可指向真实模型做单层冒烟，但本轮默认验收使用小型完整模型，不能替代
Qwen3-1.7B 全模型 logits/生成验证。

## Qwen3 完整 Prefill

2026-11-06 用户反馈双 V100、CUDA 12.6 上五组验收全部 PASS；完整 Prefill 报告为
`20261106T092424Z_prefill_1e980cbd`（小模型）和
`20261106T092512Z_prefill_b21e42fb`（真实 Qwen3-1.7B）。这些结果来自下述 Decode
改动之前，更新代码后仍需回归；不代表新 KV/Decode 功能已通过 GPU 验收。

`ProcessGroup::PrefillQwen3` 接收单请求 token ID，执行 Embedding → 全部 Decoder
层 → 最终 RMSNorm → 分片 lm_head。Embedding 为每卡复制，隐藏状态为 FP16，
所有层在 GPU 同一 stream 上执行；每层的 Attention/MLP 分别进行一次 NCCL FP32
SUM，复用单层入口的全部计算与通信逻辑。工作区在请求内跨层复用，成功后释放。
失败时整组退出，卸载时先销毁执行器再释放权重，重载后不得复用旧指针。

本轮仅返回**最后一个输入 token 的完整词表 FP32 logits**，不返回所有 token。
各卡按已有词表分片计算 logits，协调进程验证分片连续、总长度一致和数值有限后
按词表序拼接。结果是下一 token 的未归一化分数；不采样，不推进 Decode，不保存 KV。
在该历史阶段，Python `--tp_backend process` 仍拒绝启动服务；最新接入状态见文末。

支持未量化 F16/BF16/F32 源权重，GPU 统一 FP16；支持 tied embedding、非均匀
词表分片及已有 QKV/O/Down bias。序列长度 1..128，position 起点 0..1048576；
head/hidden 等限制沿用单层入口。总词表上限 1048576，每 rank 至多 262138 个词表
元素（24 字节头和 FP32 logits 必须放入 1 MiB 帧），工作区总上限 256 MiB。
不支持量化、RoPE scaling、滑窗及非 SiLU 配置，明确拒绝，不做 CPU fallback。

默认验收有 21 个小模型 case：三种源精度、bias、tied embedding、KV 复制、不同
层数、257 维 RMSNorm、128 维头、token/位置边界、重复请求、变更权重后的重载、
卸载后访问、非法 token ID 与不支持配置。每个正例比较双卡 TP、单卡和独立 CPU
完整权重数学参考；逐元素容差 `0.008 + 0.008*abs(reference)`，重复请求必须精确相等。

`--model` 模式直接读取现有真实模型，固定相同 token ID、FP16 激活、Eager、无 KV/
前缀缓存、无采样、输出 token 数为 0，比较单卡与双卡的末 token logits，容差
`0.05 + 0.01*abs(single)`。它验证多进程 TP 与单卡路径一致性，**不能排除两条路径
共用的数学错误，也不替代 HF/原线程后端的独立全模型验证**。报告记录 argmax，
但通过条件为逐元素误差和重复请求一致性，不保证近似平分时 argmax 一致。

所有报告保存原始完整 logits、固定 token ID、误差统计、GPU/CUDA/NCCL 信息、Git
提交和构建配置，以及 Markdown/CSV/JSON/XLSX 汇总；失败或中断保留部分结果并非零退出。
汇总错误只保存有界诊断字段，完整 logits（包括失败前已完成任务）保留在各 case 的
`tp.stdout.log` / `single.stdout.log`，避免词表向量挤入汇总表格。
功能检查不含性能计时，warmup=0、iterations=1、repeats=1。

在**服务器**沿用 CUDA 12.6、CUDA 12 版 NCCL 环境执行；首次先运行上节的通信、
权重、单层回归，再运行新增两组验收：

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1

cmake -S . -B build-process-tp-cu126 \
  -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
cmake --build build-process-tp-cu126 \
  --target fastllm_tp_worker fastllm_tp_check fastllm_tp_model_check \
           fastllm_tp_layer_check fastllm_tp_prefill_check -j 4

python3 test/distributed/run_qwen3_prefill.py \
  --check build-process-tp-cu126/fastllm_tp_prefill_check \
  --worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1 \
  --build-cache build-process-tp-cu126/CMakeCache.txt

python3 test/distributed/run_qwen3_prefill.py \
  --check build-process-tp-cu126/fastllm_tp_prefill_check \
  --worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1 \
  --build-cache build-process-tp-cu126/CMakeCache.txt \
  --model /home/models/Qwen3-1.7B
```

底层检查程序的 `--tasks` 是 JSON 数组，元素为 `{name, token_ids, position_begin}`，
不是自然语言文本；如需自行选择 prompt，先用该模型的 tokenizer 得到 token ID。
本轮默认真实模型用确定性 ID 序列检验数值，不以生成文本作为正确性依据。

现有双 V100 服务器也可一键构建并依次执行通信、GPU 权重、单层、小模型完整
Prefill、真实模型 Prefill 五组验收；失败立即停止，无安装或模型下载：

```bash
bash test/distributed/run_qwen3_prefill_v100.sh /home/models/Qwen3-1.7B
```

该脚本固定 CUDA 12.6、架构 70、逻辑 GPU 0/1，**会触发 C++/CUDA 编译，仅在服务器运行**。
模型目录可传绝对路径，也可传相对于调用者当前目录的路径。

## 请求内 KV Cache 与逐 token Decode

2026-11-06 用户回传双 V100、CUDA 12.6 的七组 PASS 日志。前五组为
`20261106T100805Z_0311b33e`、`20261106T100835Z_device_weights_5b466a58`、
`20261106T100859Z_layer_2ac7efb3`、`20261106T100938Z_prefill_aae18589`、
`20261106T101019Z_prefill_2b546b24`；新增 Decode 为
`20261106T101047Z_decode_f46b3b89`（小模型）和
`20261106T101153Z_decode_af11a7f8`（真实模型）。这是用户提供的服务器结果，
本地未运行 CUDA；不代表下述新增生成入口已经通过 GPU 验收。

新增公开入口 `BeginQwen3Request`、`DecodeQwen3`、`EndQwen3Request` 和
`Qwen3Requests`。每个进程组只允许一个活动请求，所有 rank 使用同一 Prompt 和
后续 token ID。Begin 执行 Prefill 并保留各层旋转后的 K 和原始 V；Decode 每次
只计算新 token，Attention 读取历史 KV，成功同步后才增加有效长度。任一 worker
失败会终止整组，不能复用部分更新的缓存。

GPU 缓存为每层 FP16 `[capacity, 2, local_kv_heads, head_dim]`，第二维依次为 K、V。
KV 按现有 Attention 分片保存，TP 大于 KV 头数时对应 KV 头复制。分配字节为
`layers * capacity * 2 * local_kv_heads * head_dim * sizeof(FP16)`，每 rank 上限
256 MiB，独立于 256 MiB 临时工作区。缓存跨前向调用存活，临时工作区仍每次释放；
End 同步并释放 KV，可以重复调用；卸载模型和正常 Stop 也释放 KV。

本轮边界：

- Prompt 长度 1..128；请求总容量 1..2048，包含 Prompt 和后续送入的 token。
- `positionBegin + capacity - 1 <= 1048576`；Decode 的位置由已缓存长度自动推进。
- 返回最后输入 token 对应的完整 FP32 词表 logits；调用者选择下一个输入 token。
  不含 tokenizer、采样、EOS 停止、连续批处理或完整聊天服务。
- 单个活动请求，无分页和跨请求前缀共享；Eager + NCCL，CUDA Graph、自定义
  AllReduce 和前缀缓存仍是后续必需适配项目，本轮未实现。
- 无状态 `PrefillQwen3` 保持原语义，可在请求活动时用于完整前缀重算，不读写请求 KV。
- Python 服务端 `--tp_backend process` 仍未开放，不能把检查器当作完整服务后端。

默认数值验收包含 30 个 case：F16/BF16/F32 源权重（GPU 均为 FP16）、bias、
tied embedding、KV 头复制、128 维头、位置上限、容量边界、不同 Prompt、重复
请求、活动请求卸载、变更权重后的重载、无请求 Decode、重复 Begin、无效 ID、
容量耗尽、KV 配额超限和报告容量超限。所有正例均逐步对比单卡与双卡完整 logits，前缀不超过
128 时还通过公开无状态 Prefill 入口重算；小模型对照独立未分片 CPU 数学参考。
最大上下文用例从 128 个 ID=14 的 Prompt token 继续送入 1920 个 ID=16，达到
2048；固定的小模型已检查该输入能识别丢弃早期 128 个 KV 的错误。所有步
比较 TP/单卡，CPU 重算选取有效长度 128、129、256、2048。该 case 包含较多
同步调用和 Python 参考计算，属于正确性测试，不用于测速。

每步还检查每个 rank 的实际 KV 分配字节、有效长度、容量、RoPE 起点，验证
无状态重算不会破坏缓存，以及 End/Unload 后全零。重复请求输出须精确一致。
小模型逐元素容差为 `0.008 + 0.008*abs(reference)`；真实模型模式为
`0.05 + 0.01*abs(single)`，固定相同 token ID（5 个 Prompt + 3 步 Decode，
重复两轮），不比较自行采样得到的不同文本。真实模型没有独立 HF 对照，因此
仍不能排除所有共同实现错误，也不代表吞吐、延迟或正式生成服务已验收。

验收报告包含 JSON、Markdown、CSV、XLSX，JSON 和原始 stdout 保留完整精度；
记录实际 GPU/CUDA/NCCL、编译架构、Git、完整命令、固定 ID、cache/执行模式。
原始逐步 logits 与 KV 状态在 `tp.stdout.log`、`single.stdout.log`；独立 CPU
参考在 `reference.json`，数值比较失败或中断时仍保存已算出的参考值。
检查器累计最多保存 8388608 个 logit 元素，可用 `--max-logit-elements` 降低上限；
超过时返回失败、保留已完成步骤并清理 worker，避免长序列的大词表报告撑爆主机内存。
这仅限制验收报告，不限制正式公开 Decode 接口。失败记录具体操作、形状和原因，
保留部分报告并非零退出。

纯 Python 判定回归，不触发编译：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s test/distributed -p 'test_*.py'
```

**双 V100 服务器**一键编译、回归之前的五组测试，再运行小模型和真实模型 Decode：

```bash
bash test/distributed/run_qwen3_decode_v100.sh /home/models/Qwen3-1.7B
```

脚本使用 CUDA 12.6、架构 70、逻辑 GPU 0/1，**会触发编译，禁止在本地无授权运行**。
成功时打印七个 PASS 目录及“七组验收均通过”。仅单独运行新验收时，在服务器
完成对应构建后执行：

```bash
cmake --build build-process-tp-cu126 \
  --target fastllm_tp_worker fastllm_tp_decode_check -j 4

python3 test/distributed/run_qwen3_decode.py \
  --check build-process-tp-cu126/fastllm_tp_decode_check \
  --worker build-process-tp-cu126/fastllm_tp_worker \
  --devices 0,1 --build-cache build-process-tp-cu126/CMakeCache.txt

python3 test/distributed/run_qwen3_decode.py \
  --check build-process-tp-cu126/fastllm_tp_decode_check \
  --worker build-process-tp-cu126/fastllm_tp_worker \
  --devices 0,1 --build-cache build-process-tp-cu126/CMakeCache.txt \
  --model /home/models/Qwen3-1.7B
```

## 文本入口与协调器生成循环（单请求已完成服务器验收）

用户提供的双 V100 服务器日志确认两组生成验收通过：
`20261106T105443Z_generation_aed9accf`、`20261106T105519Z_generation_ebb16515`。
修复 tokenizer 返回格式后，真实中文问答在
`text-generation/20261106T110219Z_c6a8caac` 正常生成 17 token，以 EOS 结束。
这些是服务器回传结果，本地未重新执行 GPU 验收。

本轮新增独立 `fastllm_tp_generate` 程序和 `tools/qwen3_process_chat.py` 文本前端。
正式公开接口为 `GenerateQwen3`，直接复用 `BeginQwen3Request`、`DecodeQwen3` 和
`EndQwen3Request`；没有第二套前向或 KV 实现。文本前端仅用本地 Transformers
AutoTokenizer 分词和解码，不加载 PyTorch 模型；worker 各自加载权重。
原有 Python 服务的 `--tp_backend process` 仍未开放。

行为和边界：

- 模型限已有稠密 Qwen3；源 F16/BF16/F32 统一按既有 FP16 GPU 路径执行。
- 使用模型自带聊天模板，默认 `enable_thinking=False`。`--thinking` 显式打开；
  `--raw` 跳过模板。模板展开后的 Prompt 仍限 1..128 token，超长报错，不截断。
- 协调器只采样一次并向所有 rank 发送同一 token。temperature=0 为贪心，
  同分取最小 token ID；正温度按 temperature → top-k → top-p → 重归一化采样。
  top-k=0 不截断，top-p=1 不截断；默认 seed=0，使用明确的 53-bit mt19937_64 随机数。
- 默认只沿用模型 EOS 配置：优先 generation_config 的 eos_token_id，缺省用
  config，再合并 tokenizer.eos_token_id。不自动沿用 generation_config 的采样参数。
  `--eos-token-id` 可重复指定覆盖值；`--ignore-eos` 明确关闭 EOS 停止。
- 输出 ID 包含实际采样出的 EOS，显示文本会跳过特殊 token；停止原因区分 eos/length。
  最后一个输出不再送入 Decode，所需容量为 `Prompt + max_new_tokens - 1 <= 2048`。
  默认最多输出 32 token；公开接口拒绝覆盖已有活动请求。
- 单次模式处理一个请求，文本在结束后完整显示；新增常驻串行模式见下节。
  不支持流式 HTTP、连续批处理或跨请求缓存。
  无 CUDA Graph、自定义 AllReduce、重复惩罚、min-p 和 stop-string。
- 默认不存储历史 logits；原生 `--trace-logits 1` 用于诊断，累计最多 8388608 个元素。
  诊断只观察正式生成结果，不改变采样、权重布局、CUDA stream 或同步策略。
  EOS/长度结束释放 KV，异常也尝试释放；worker 失败仍整组终止。
- 文本入口和验收器共用 Linux subreaper/进程组清理：超时或中断回收协调器与孤儿
  worker，清理前保留组长 PID，避免 PID/PGID 复用；不依赖容器 PID 1。
  文本输出失败也返回非零，并在 response.json 中记录 FAIL，不会误报成功。

验收使用同一个生成程序。小模型套件包含 24 个原生采样黄金边界检查和 17 个生成
case：源 dtype、bias、tied embedding、KV 复制、长度 1、Prompt 长度 128、首个 EOS、
固定 seed、top-k=1、无效长度/采样参数/EOS。每步沿实际输出 token 序列在单卡重放，
统一 Prompt 和后续 ID；小模型再和独立 CPU 完整前缀计算逐元素比较。单卡使用相同
FP16 KV 类型、Eager、无 prefix cache，不能让两个后端各自采样不同前缀后比较 logits。
真实模型套件包含采样检查和固定 Prompt ID 的真实自回归生成/单卡重放。
同 seed 重复采样要求输出 ID 完全一致；不要求不同 TP 数下随机输出文本逐字相同。
报告保存 JSON、Markdown、CSV、XLSX、原始 logits、生成 ID、EOS/采样配置和 CPU 参考。
真实文本示例另走 AutoTokenizer 全链路，输出 `response.json` 和原生进程日志；
它只验证路径完成，不自动判定回答质量，不替代独立 HF 端到端对照或性能测试。

服务器安装分词依赖（仅需 tokenizer，不需安装 torch），并沿用 CUDA 12.6/NCCL 环境：

```bash
python3 -m pip install transformers jinja2
bash test/distributed/run_qwen3_generation_v100.sh /home/models/Qwen3-1.7B
```

脚本会重编译、回归之前七组验收、执行新增两组验收，然后运行真实中文问答。
共九个 `PASS` 目录；文本结果另在 `test/distributed/results/text-generation/` 的唯一子目录。
本地只允许 Python/文本检查，不能执行该编译脚本。

只运行问答时（服务器已构建 `fastllm_tp_generate` 和更新后的 worker）：

```bash
export CUDA_VISIBLE_DEVICES=0,1
python3 tools/qwen3_process_chat.py \
  --model /home/models/Qwen3-1.7B \
  --prompt '请用一句话介绍你自己。' \
  --max-new-tokens 64

# 随机采样；默认关闭 thinking，避免短输出预算全部用于思考。
python3 tools/qwen3_process_chat.py \
  --model /home/models/Qwen3-1.7B \
  --prompt '给一只橘猫起一个名字。' \
  --temperature 0.7 --top-k 20 --top-p 0.9 --seed 123 \
  --max-new-tokens 32
```

原生接口接受 `--request` JSON 文件，字段为 `token_ids`、`max_new_tokens`、
`temperature`、`top_k`、`top_p`、`seed` 和 `eos_token_ids`。默认值与上文相同；
只有 `token_ids` 必填。原生程序不读取 tokenizer/EOS 配置，省略 EOS 数组即仅按
长度停止，文本前端会显式传入解析后的 EOS 配置。成功 stdout 是最终 JSON，
可能带 NCCL 前置日志；任何失败返回非零，保留部分 steps 和清理状态。

## 常驻模型与连续请求（基础已服务器验收，取消扩展待验收）

`fastllm_tp_generate --session-fd FD` 接受父进程传入的独立 stream socket；
Python 公开客户端为 `tools/qwen3_process_session.py` 中的 `Qwen3ProcessSession`。
模型和 NCCL 组只初始化一次，各请求串行调用与单次入口共用的 `GenerateQwen3`。
每次请求都新建并释放独立 KV；不保留前一请求的上下文，不实现 prefix cache。

- 控制消息为网络字节序 uint32 长度 + UTF-8 JSON；与 stdout/stderr 日志隔离。
  请求最多 1 MiB，响应最多 512 MiB；trace 仍限制累计 8M logits 元素。
- ready 消息确认模型已加载；generate 的 id 从 1 连续递增，响应包含原请求、
  worker 身份、输出 ID、结束原因和 KV 状态。客户端串行化并发调用；FIFO 排队由下文
  HTTP 服务负责。cancel_event 通过独立数据报按请求 ID 取消；on_token 支持逐 Token 回调，尚无连续批处理。
- 参数拒绝且确认 KV 为空时允许继续；worker 失败、超时、协议错乱或资源异常
  终止并回收整组，后续调用失败，不会自动重试或切换线程后端。
- shutdown/EOF 卸载模型并退出；Python 在正常退出及异常时都使用已有的进程组
  清理工具。客户端超时包含一次请求的控制通信和生成，不是单个 token 的耗时。
  只有整组回收成功后才写入成功退出记录；清理异常和已失败会话不能被汇总为 PASS。
- 该阶段仅开放独立 Qwen3 HTTP 入口；最新的通用 CLI 接入见文末。
  不宣称多请求并行计算、CUDA Graph、自定义 AllReduce 或跨请求前缀缓存已完成。

服务器一键构建并执行四组检查（已有生成两组、新增常驻两组），再连续回答两条问题：

```bash
bash test/distributed/run_qwen3_session_v100.sh /home/models/Qwen3-1.7B
```

已有七组底座验收不在此脚本重复执行。新检查使用正式会话客户端，覆盖 F16/BF16/F32、
bias、tied embedding、KV 复制，逐组验证 A→B→A、空 Prompt/零输出长度拒绝后恢复、EOS 首步停止、
固定 seed 重复采样、worker PID 不变和最终清理。小模型逐步对照 CPU，真实模型
将相同输出前缀在单卡重放；输出最大/平均绝对误差及失败元素数。每组生成唯一目录，
保存 JSON/Markdown/CSV/XLSX、完整命令、编译配置、GPU/CUDA 信息和逐请求原始输出。
任意用例失败或执行不完整均返回非零并保存部分报告。这些是正确性检查，不是性能结论。

仅运行常驻交互入口（需要已重新编译的生成程序）：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export USE_TORCH=0
python3 tools/qwen3_process_chat.py \
  --model /home/models/Qwen3-1.7B --interactive --max-new-tokens 64 \
  --output test/distributed/results/text-session
```

每行独立提问，Ctrl-D 退出；每条原始问题和回答保存到 `text-NNNNNN.json`。
某条输入非法时显示错误并继续接受下一条；只要出现过失败，最终退出码就是非零。
`session/ready.json`、逐请求结果、`shutdown.json` 记录权重驻留与清理情况。


2026-11-06，用户在双 V100 服务器完成原常驻版验收：

- 生成：`20261106T112843Z_generation_a2674039`、`20261106T112920Z_generation_716ee10e`。
- 常驻：`20261106T112946Z_session_ab3de6f2`、`20261106T112954Z_session_15c54f1f`。
- 文本：`text-session/20261106T113158Z_069d4baf`，两请求分别 17/38 token，均 EOS，failed=0。

取消扩展加入“Prefill 前取消 → 同一 worker 再次生成 A”；本次再加入流式贪心/采样对照，常驻验收变为每模型 14 项；
上面的历史结果不能代替新增取消功能的服务器验收。

## Qwen3 HTTP/SSE 服务与协作取消（待服务器验收）

该阶段入口为 `tools/qwen3_process_server.py`，仅依赖标准库 HTTP 和 tokenizer。
本轮已将其实现移入 FastLLM 包，旧脚本保留兼容转发；通用 CLI 接入见文末。

- 模型/两 rank 权重加载一次；一个消费线程执行 FIFO 队列，每次只有一个活动请求。
  `--queue-size` 是等待槽数，默认 8，满时 HTTP 429；重复的活动 `X-Request-ID` 返回 409。
- `GET /health` 提供就绪、活动 ID、等待 ID、worker 身份；`GET /v1/models` 返回模型名。
- `POST /v1/chat/completions` 接受 `model`、`messages`、`max_tokens`、`temperature`、
  `top_p`、`top_k`、`seed`、`stream`，以及流式专用 `stream_options.include_usage`。模型默认名 `qwen3`；采样默认贪心，关闭 thinking。
  文本 messages 可以带首条 system，然后 user/assistant 交替且以 user 结束；完整历史
  每次通过同一聊天模板编码，不自动保存服务端对话历史。
- Prompt 含模板及历史仍限制 1..128 token；`prompt+max_tokens-1<=2048`。超限拒绝，
  不静默截断。请求体上限 64 KiB；未知字段、工具、多模态、chunked 请求均拒绝。
  每次响应后关闭连接；不支持 HTTP 请求流水化。连接线程总数也有上限。
- 原生 `--cancel-fd` 为独立 SOCK_DGRAM 通道，网络字节序 uint32 对应 session 请求 ID。
  在 Prefill 前、Prefill 返回后及 Decode token 边界检查；不会强行中断 CUDA/NCCL。
  取消后调用正式 End 路径释放请求 KV，检查为空后才确认 reusable；迟到取消不能误伤下一请求。
- 客户端断连、请求超时、`DELETE /v1/requests/{X-Request-ID}` 都触发同一取消路径。
  排队任务直接移除，活动任务等待原生取消确认。DELETE 的 202 仅代表已提交取消，
  不是 GPU 已停；非流式请求如已在原生完成则完成结果优先。流式取消期间可能丢弃增量，
  即使原生同时完成也不会报告成功。显式取消状态 499，超时状态 504；流开始后的错误见下文。
- `--request-timeout` 默认 120 秒，包含排队和生成；取消确认可能等待当前 GPU 步骤。
  `--session-timeout` 默认 180 秒，原生故障/无法响应时回收整组并将服务标记不可用，
  不自动重试、不重载模型、不 fallback。取消并清理成功后后续请求才继续。
- SIGINT/SIGTERM 停止接收新请求、取消排队与活动任务、卸载/回收进程组；成功才写 shutdown.json。
- 默认监听 `127.0.0.1`，无认证和 TLS；定位是本机/SSH 隧道验证入口。日志目录包含
  `ready.json`、每请求 `http-*.json`、原生 session 日志/请求/输出及最终退出记录；目录唯一。

双 V100 服务器一键编译和验收（流式复用原生成入口，故同时回归常驻数值对照）：

```bash
bash test/distributed/run_qwen3_http_v100.sh /home/models/Qwen3-1.7B
```

共三组：合成模型常驻/取消/Token 流 84 项、真实 Qwen3 常驻/取消/Token 流 14 项、真实 HTTP/SSE 服务 17 项。
前两组沿用 CPU / 单卡 logits 对照；HTTP 使用正式 HTTP 类和同一会话后端，比较与
相同 Prompt ID、FP16 KV、EOS/采样参数、Eager 模式单次入口的精确 Token ID、文本、
用量和结束原因。还验证有界队列、排队/活动取消、断连、超时、复用与退出。
SSE 验证逐段文本拼接、角色/结束帧、可选 usage、唯一 DONE、流中取消/断连/超时及恢复；
原生 Token 回调在贪心和固定 seed 采样两种配置下必须与非流式结果精确一致。
HTTP 的误差字段指 Token ID 精确一致性，不是 logits 误差；生命周期用例为 n/a。
任何失败或少执行用例都返回非零，保留 JSON/Markdown/CSV/XLSX 和原始报告。
这些检查不构成性能结论，也不代替独立框架的模型质量对照。

验收后启动服务（前台运行，Ctrl-C 退出）：

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1
export USE_TORCH=0
python3 tools/qwen3_process_server.py --model /home/models/Qwen3-1.7B \
  --host 127.0.0.1 --port 8000 --queue-size 8
```

另一个服务器终端发送请求：

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Request-ID: demo-1' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"请用一句话介绍你自己。"}],"max_tokens":64,"stream":false}'
```

需要取消尚未结束的 demo-1 时，在第三个终端执行：

```bash
curl -sS -X DELETE http://127.0.0.1:8000/v1/requests/demo-1
```

## SSE 流式语义（待服务器验收）

`stream=true` 使用与非流式相同的模板、采样、GPU 后端和 KV 生命周期；原生每采样出
一个 Token，立即通过会话 socket 发送独立事件（请求 ID、连续 index、整数 Token ID）。
客户端逐条校验，最终还会核对完整 Token ID 列表；乱序、缺失、多余或不匹配的事件会使整组失败。
ready 中的 token_events 能力必须为 true；Python 服务拒绝未重新编译的旧协调器。

- HTTP 返回 `text/event-stream`：角色帧 → 文本增量 → finish_reason 帧 → 可选 usage 帧 → `[DONE]`。
  `stream_options: {"include_usage": true}` 才发送独立 usage 帧，其 choices 为空。
- Tokenizer 在同一个消费线程执行，流式与非流式均显式关闭空格清理。
  UTF-8 未完整时产生的尾部替换字符暂缓输出，结束时保留真正的替换字符；已经发送的文本
  前缀不能被修改。不满足前缀稳定约束的自定义 tokenizer 会明确报错，不输出拼接错误的文本。
- 只有原生最终结果通过、请求 KV 确认释放、HTTP 审计落盘之后，才能发送成功结束帧和 DONE。
  首个内容之前的输入错误仍返回普通 HTTP 错误 JSON；已经发送 SSE 响应头之后，错误以
  `data: {"error":{"message":...,"type":"request_error","code":...}}` 返回并关闭连接。
  失败流没有成功 finish_reason、usage 或 DONE；客户端不能把 EOF 当成成功。
- 每请求最多缓冲 64 条文本增量，回调不阻塞原生接收线程；队列满时取消当前请求并返回
  503 错误。客户端断开则取消当前请求；同一模型继续服务后续请求。响应写入也有 socket 超时。
- 回调异常先发取消，再读完当前原生结果并验证 KV 清理，最后抛 GenerationObserverError；
  这样单请求解码失败不会错位消费下一请求的消息。协议错误仍回收整组。
- 旧 HTTP 连接/定时器的内部取消同时匹配 Job 身份，避免已完成 ID 被复用后误取消新请求。
- 验收生成的 `*.sse.json` 保存收到的事件；会话检查保存 `token_events.json`，所有报告
  仍包含 JSON、Markdown、CSV、XLSX。功能检查不是性能测速。

按上文启动服务后，在另一个服务器终端运行（`-N` 关闭 curl 自身的输出缓冲）：

```bash
curl -N -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Request-ID: stream-demo-1' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"请用一句话介绍你自己。"}],"max_tokens":64,"stream":true,"stream_options":{"include_usage":true}}'
```

本轮仍不包含连续批处理、CUDA Graph、自定义 AllReduce 或跨请求 Prefix Cache。


## 正式入口接入（服务器验收已通过）

`make_normal_llm_model(args)` 在 `tp_backend=process` 时返回 `ProcessTPModel`，
常驻一组 C++ worker。`response` / `stream_response` / 已编码请求的 `generate` 使用
同一原生 `GenerateQwen3`，采样、Decode 循环、GPU 算子和 NCCL 均没有移到 Python。
请通过上下文管理器或 `close()` / `release_memory()` 显式释放模型。
提前关闭流式迭代器会取消请求、排空原生结果、清理 KV；关闭模型会取消在途请求并回收进程。

正式 CLI 已支持 `ftllm chat/run/server/serve ... --tp_backend process --tp 2`。
`chat/run` 支持 `--prompt` 单次输出以及交互历史、`clear`、`stop`、EOF；历史每次重新编码，
不截断、不复用跨请求 KV。`server/serve` 使用同一个模型工厂和已有的受限 HTTP/SSE 实现，
不是线程后端的完整 FastAPI 服务：仅文本 chat completions、models、health、取消接口，
不支持 Responses、Anthropic、工具、多模态、嵌入或 API Key。设置不支持的参数会在加载前失败。
`llm.model(...)`、异步 handle API、webui/export/benchmark 尚不属于这个进程后端适配接口。

- 仅 Linux + NVIDIA CUDA，本轮验收模型为官方未量化 `Qwen/Qwen3-1.7B`。
  源权重 F16/BF16/F32 由既有读取器转换为 FP16，KV 也是 FP16；不支持 BF16 计算或量化权重。
- Prompt（含模板及历史）1..128 token，`prompt + max_new_tokens - 1 <= 2048`。
  单活动请求，HTTP 有界排队，无连续批处理。文本生成默认关闭 thinking，最大输出默认 32。
- `--tp 2` 表示可见设备中的前两张卡；`--tp 0,1` 为逻辑编号；`--tp 0` 为单卡 0。
  `--tp auto` 要求显式设置 `CUDA_VISIBLE_DEVICES`。不做卡数静默裁剪，不接受线程 TP 的比例配置。
- 已有 `--dtype auto/float16`、`--atype auto/float16`、`--kv_cache_dtype auto/float16` 可用。
  不支持的显式选项（如 `--api_key`、量化、paged KV、prefix cache、LoRA）明确拒绝。
- 默认从 Python 包目录或 PATH 查找原生程序，也可传 `--tp-binary` / `--tp-worker`。
  不自动编译或下载模型。标准 wheel 也包含进程公共模块，并在开启构建时打包生成协调器。
- `fastllm_process_tp_package` 仅构建进程产物并生成 `build-process-tp-cu126/process-tp-python/ftllm`，
  不要求编译完整旧线程后端。该目录是用于服务器验收的进程专用包，不含线程后端共享库。

服务器（双 V100，CUDA 12.6，master 已合并 sk）执行：

```bash
bash test/distributed/run_qwen3_entrypoint_v100.sh /home/models/Qwen3-1.7B
```

预计三组 PASS：合成模型常驻/取消/流式 84 项、真实模型对应 14 项、正式入口 12 项。
最后一组实际启动 `python3 -m ftllm.cli` 子进程，验证模型工厂复用、非流式/流式文本、
非法输入后的恢复、HTTP/SSE、SIGTERM 清理，以及不支持 API Key 时必须拒绝启动。
与原生入口统一 Prompt ID、采样/EOS、FP16 KV、Eager 和空前缀缓存，并比较实际 Token ID、
文本、finish reason、usage 及 worker 身份；底层数学回归由前两组承担。
报告包含 JSON/Markdown/CSV/XLSX，失败保留部分结果并返回非零。

一键验收成功后，手动运行正式 CLI（使用本次构建出的包，不依赖旧版已安装 ftllm）：

```bash
export PYTHONPATH="$PWD/build-process-tp-cu126/process-tp-python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1
export USE_TORCH=0
python3 -m ftllm.cli run /home/models/Qwen3-1.7B \
  --tp_backend process --tp 2 --prompt '请用一句话介绍你自己。' --max-new-tokens 32
python3 -m ftllm.cli server /home/models/Qwen3-1.7B \
  --tp_backend process --tp 2 --host 127.0.0.1 --port 8000 --model_name qwen3
```

这里的模块调用与已安装 `ftllm` 命令使用相同的 `ftllm.cli.main`；不添加另一条推理实现。
用户已确认这一步的 84 / 14 / 12 项服务器验收全部 PASS；下一项为以下 Decode CUDA Graph。


## Decode CUDA Graph（第 2 项，用户服务器验收已通过）

正式模型工厂和 `ftllm chat/run/server/serve --tp_backend process` 增加 `--cuda-graph`。
默认关闭。开启后 Prefill 仍为 Eager，单 token Decode 捕获 embedding、全部 decoder 层、
两次/层 NCCL AllReduce、最终 Norm 和分片 lm_head；CPU 采样、控制通信和 logits 读回在图外。
图路径与 Eager 共用算子、权重布局、FP16 KV、stream 和同步边界，不新增测试专用推理路径。

- 每个请求第一次 Decode 分配固定地址工作区、执行一次 warmup，再捕获并重放。
  每次更新 GPU token ID、历史长度和绝对 RoPE 位置，重放时从 GPU 读取；不会固定在首步位置。
  warmup 只覆盖待写入的当前 KV 槽位，不推进有效长度，实际重放后长度仅增加一次。
- cuBLAS 提供 4 MiB 用户工作区，避免捕获期间临时分配；请求 Graph 工作区上限 256 MiB。
  它与无状态 Prefill 的临时工作区分离，交错重算不会释放图引用的指针。
- 请求结束/取消先同步，销毁图，解除 cuBLAS 用户工作区，再释放工作区和 KV。
  模型权重及 worker 保留。新请求重新捕获；本阶段不跨请求复用图或 KV。
- 所有 rank 使用统一开关。捕获、实例化、重放或 NCCL 失败均报错并终止整组，不静默回退。
  `controlOnly + cudaGraph` 拒绝。该阶段协议为 v2；后续自定义规约升级为 v3，协调器/worker 必须一起重建。
- 原生 `fastllm_tp_generate` / `fastllm_tp_decode_check` 使用 `--cuda-graph 0|1`；
  Python 正式入口为无值开关。线程后端的 `FASTLLM_CUDA_GRAPH` 环境变量不控制此进程实现，
  开启该环境变量时正式进程入口明确拒绝，避免两套设置混用。
- `ready.json`、每请求原生报告包含 `execution` 与各 rank `graph`；后者记录
  enabled、active、captures、replays、warmups、eager_decodes、workspace_bytes。
  结束后 active=false、workspace_bytes=0；计数保留至卸载。开启 Graph 时 eager_decodes 必须为 0。
  生成 N 个 token 实际需要 N-1 次 Decode，首 token EOS 的请求不必捕获。

服务器双 V100、CUDA 12.6、NCCL 2.22.3，master 合并本轮 sk 后执行：

```bash
bash test/distributed/run_qwen3_cuda_graph_v100.sh /home/models/Qwen3-1.7B
```

七组验收：通信/协议/进程清理底座；Eager 常驻回归 84；Graph 合成 Decode 30、真实 Decode 1；Graph 合成常驻 96、
真实常驻 16；Graph 正式入口 12。逐步比较 Graph TP / Eager TP / Eager 单卡 logits，
合成模型还有独立 CPU 参考；覆盖源 F16/BF16/F32、bias、KV 复制、位置/容量边界、
2048 长度、交错 Prefill、卸载重载、请求复用、Graph 后取消恢复及拒绝路径。
正式入口比较 Token ID/文本、HTTP/SSE 和清理，客户端强制检查实际 Graph 次数。
所有组输出 JSON/Markdown/CSV/XLSX，失败返回非零并保留部分报告。
这是功能验收，不测性能，也不宣称加速；每请求捕获有额外开销。

单次正式文本生成（完成上面构建后）：

```bash
export PYTHONPATH="$PWD/build-process-tp-cu126/process-tp-python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1
export USE_TORCH=0
python3 -m ftllm.cli run /home/models/Qwen3-1.7B \
  --tp_backend process --tp 2 --cuda-graph \
  --prompt '请用一句话介绍你自己。' --max-new-tokens 32
```

用户已提供七组 PASS（2026-11-06 14:55—15:01 UTC，末组 `20261106T150149Z_entry_236769fc`）。
本地未编译、未运行 GPU；下一项自定义 AllReduce 见下节，跨请求 Prefix Cache 在其后。
捕获设计依据：[NCCL 2.22.3 CUDA Graph](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2223/user-guide/doc/usage/cudagraph.html)、
[cuBLAS 12.6.3 CUDA Graph](https://docs.nvidia.com/cuda/archive/12.6.3/cublas/index.html#cuda-graphs-support)。

## 自定义 AllReduce（第 3 项，待服务器验收）

正式工厂/CLI 增加 `--custom-allreduce`，可以与 `--cuda-graph` 组合。
默认仍使用 NCCL。当前自定义路径限定单机双进程、两张 SM70+ GPU（各至少 16 SM），
要求双方设备可见、双向 P2P、原生跨卡原子能力；能力不足或 IPC 打开失败直接终止整组。
NCCL 模式保留原来的卡数范围。此处未扩展为多机或 4/8 卡自定义规约。

- C++/CUDA `ProcessAllReduce::Sum` 接入每层 Attention 输出、MLP 输出的 FP32 规约。
  数据连续，FP32 输入/输出，SUM；并不提供 BF16/FP16 的自定义规约核。
  源模型权重 F16/BF16/F32 仍使用既有 FP16 计算路径，层内规约是 FP32。
- 每卡独占一块 2 MiB IPC 分配（信号和 payload），另有 4 字节关闭屏障缓冲。
  NCCL 只在启动交换 IPC 句柄/UUID、较大消息和关闭屏障中使用。
  不能在进程间直接传裸指针；导入后的地址由各进程自己取得。
- `count <= 65536`（含边界，最多 256 KiB）使用固定 staging copy 加一阶段 peer-read kernel；
  更大消息明确使用 NCCL。这个容量阈值不声称是实测性能最优阈值。
  Prefill/Decode 都按同一消息大小规则选择；不是所有 Prefill 都必然使用 NCCL。
- 系统作用域 acquire/release 握手；固定 16 blocks，各自维护 GPU 序号。
  写入下一轮 payload 前，必须等所有 block 的对端读取结束。
  Graph 记录 copy/kernel 节点，序号在 GPU 随实际执行推进，捕获本身不增加规约次数。
- IPC 属于通信组：请求结束释放 KV/Graph，卸载模型保留 IPC；组关闭先完成所有 stream，
  关闭对端映射，再经过集体屏障释放所有者分配。失败不做单 rank 重试，沿用整组终止/回收。
  模型重载、交错无状态 Prefill 和新请求共用该生命周期。
- 原生 `fastllm_tp_generate`、`fastllm_tp_decode_check`、`fastllm_tp_check` 接受 `--custom-allreduce 0|1`。
  正式 Python CLI 为无值开关。旧线程环境变量 `FASTLLM_CUDA_CUSTOM_ALLREDUCE` 不控制本实现。
  控制协议 v3，新旧 worker/协调器混用明确拒绝；本轮脚本会一起重建所需产物。
- 每个 rank 的 `graph` 诊断新增 `allreduce`：enabled、active、custom_calls、nccl_calls、workspace_bytes。
  即使 Eager 也有这份诊断。custom_calls/nccl_calls 来自 GPU，包含 Graph warmup/replay 的实际操作，
  不包含 IPC 引导或关闭屏障；模型卸载不清零，组重建才清零。active 表示 IPC 映射，不是请求活动状态。
  为兼容既有客户端，模型报告的 `backend` 标识保持不变，规约路径以该诊断的实际计数为准。

服务器 master 合并 sk 后执行：

```bash
bash test/distributed/run_qwen3_custom_allreduce_v100.sh /home/models/Qwen3-1.7B
```

该脚本重建所需原生产物和正式包，先跑原 NCCL 七组，再跑自定义规约七组：

1. 通信底座：27 次变长 FP32 数值检查，覆盖 1/31/255/256/4095/4096/65535/65536/65537 元素，
   验证实际 custom/NCCL 次数；非法输入、TP=1 拒绝、重启、崩溃、停顿、对端退出后的整组回收。
2. Eager 合成 Decode 30 项；逐步对照 NCCL TP、单卡和独立 CPU 参考。
3. Graph 合成 Decode 30 项；严格校验每层两次规约及 warmup、replay 的实际次数。
4. Graph 真实模型 Decode 1 项。
5. Graph 合成常驻会话 96 项，包含流式、请求复用、取消恢复。
6. Graph 真实模型常驻会话 16 项。
7. Graph 正式工厂/CLI/HTTP/SSE 12 项，与 NCCL 基线对照 Token ID 和文本。

报告目录不覆盖历史结果，输出 JSON/Markdown/CSV/XLSX；任一组失败返回非零。
原始逐任务计数保存在各用例 JSON，数值报告包含最大/平均绝对误差、失败元素数。
这是功能验收，尚无延迟/吞吐基准，不能据此宣称快于 NCCL。
本地只做纯 Python 回归和静态 review，未编译或运行 GPU。服务器全部通过后，再进行跨请求 Prefix Cache。

单次正式入口（上面脚本完成构建，并使用上一节 CUDA/PYTHONPATH 环境后）：

```bash
python3 -m ftllm.cli run /home/models/Qwen3-1.7B \
  --tp_backend process --tp 2 --cuda-graph --custom-allreduce \
  --prompt '请用一句话介绍你自己。' --max-new-tokens 32
```

设计依据：[CUDA 12.6.3 IPC](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#interprocess-communication)、
[PTX 系统作用域内存序](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model)。
