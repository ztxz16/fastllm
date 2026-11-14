# 多进程 TP 底座验收

本目录按阶段提供 Linux 单机进程底座、Qwen3 分片/GPU 权重加载、单层前向，
以及完整 Prefill、请求 KV/Decode 和文末新增的生成验收。通信底座包括独立 C++ worker、Unix socket、
NCCL 初始化和标量 SUM 自检、心跳、超时、退出与故障回收。
**Qwen3 稠密模型的多进程 TP 生成、常驻请求、正式模型工厂、Decode CUDA Graph
和双卡 CUDA IPC 自定义 AllReduce 已完成用户服务器验收。本轮新增跨请求前缀缓存，GPU 验收待执行。**
默认 `thread` 保持原行为；进程后端使用 C++/CUDA + NCCL，只支持未量化稠密 Qwen3。
文中各阶段说明是历史边界，当前入口及限制以文末“正式入口接入”“Decode CUDA Graph”、
“自定义 AllReduce”和“跨请求前缀缓存”各节为准。
构建选项 `BUILD_PROCESS_TP` 默认关闭，显式开启后才构建进程产物。

新增的 **完整模型 DP 已接入正式工厂与 CLI**，支持纯 DP 和副本内部 TP；
设备预览与实际执行分别验收，详见文末“DP 参数与 GPU 分配”和“DP 多副本执行”。

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

### 双 V100 退出监控完整验收

在 CUDA 12.6 服务器上运行以下脚本；它会构建所需产物，再运行五组 NCCL 验收：
进程退出/通信、Eager/Graph Decode，以及 Eager/Graph 正式入口（Graph 同时开启 Prefix Cache）。
所有默认路径均使用 NCCL，不要求自定义规约所需的双向 P2P/native atomics。

```bash
bash test/distributed/run_process_tp_exit_v100.sh /home/models/Qwen3-1.7B
```

脚本保留已有的 `CUDA_VISIBLE_DEVICES`，未设置时使用 `0,1`；TP 始终选可见列表中的
前两张卡。V100 型号本身不能保证所选 GPU 对满足自定义规约条件。只有需要额外验收
自定义通信时，才设置 `TP_EXIT_CUSTOM_ALLREDUCE=1`，追加通信底座、Graph Decode 和
Graph 正式入口三组测试。该模式仍严格检查实际路径，能力不足时失败，不静默回退：

```bash
TP_EXIT_CUSTOM_ALLREDUCE=1 bash test/distributed/run_process_tp_exit_v100.sh /home/models/Qwen3-1.7B
```

默认成功仅代表 NCCL 配置通过，自定义 AllReduce 会明确显示为“未启用、未验收”。

## 检查内容与报告

### 空闲退出事件与心跳回归

空闲监控现在同时等待 worker 的 pidfd、停止管道和下一次心跳到期。退出事件可直接
唤醒监控线程；心跳到期仍通过正式 Ping/Pong 检查进程响应。停止或前台失败关闭唤醒
管道写端；监控线程持有独立 pidfd 副本，避免前台清理关闭原句柄时发生复用竞态。
正常 Stop 会先停止并 join 监控线程，再发送 Shutdown，正常 worker 退出不应误报故障。

以下 case 全部调用正式 `ProcessGroup` 接口。`control_` 表示纯控制面，`cuda_` 表示
真实 CUDA/NCCL 或显式开启的自定义规约配置；CUDA 场景在故障前及重启后检查实际 SUM 输出。

| 新增 case 后缀 | 执行范围 | 验证内容 |
| --- | --- | --- |
| `monitor_idle_exit_first` | control / cuda | 60 秒心跳下杀掉首 rank；不再发前台命令，5 秒内失败并回收整组，报告正确 rank；重启可工作 |
| `monitor_idle_exit_last` | control / cuda | 同上，杀掉末 rank，确认监听整组而非仅第一个 worker |
| `monitor_heartbeat_timeout` | control / cuda | SIGSTOP 后进程仍存活；100 毫秒心跳及 1 秒操作超时仍能发现不响应并清理 |
| `monitor_stop_restart` | control / cuda | 60 秒心跳下连续启停 8 次，含启动后立即停止；Stop 不等心跳，幂等停止，无句柄泄漏 |
| `monitor_destructor` | control / cuda | 60 秒心跳下由析构关闭，线程及时结束，worker 已回收，句柄数量恢复 |
| `monitor_group_isolation` | control | 同一父进程管理两组；一组崩溃及重启不影响另一组的 Ping/关闭 |
| `monitor_heartbeat_healthy` | control 夹具 | 两个 worker 实际收到至少 3 轮后台心跳并回复；前台 Ping 和正常关闭继续成功 |
| `monitor_stop_heartbeat` | control 夹具 | 两个 worker 已收到后台心跳但不回复时调用 Stop；按原操作超时完成失败清理，关闭不死锁 |

已有 `watch_*` 场景继续覆盖前台等待、半包、完整/过期错误、启动失败和正常退出顺序，
并增加前台失败后 Stop 的返回耗时断言。测试记录实际故障原因、rank、完成耗时、句柄数和
回收状态；JSON 保留原始精度，Markdown/CSV/XLSX 浮点值保留 3 位。5 秒上限包含调度和
整组回收，只是功能断言，不是纯退出检测延迟或性能比较。原生 case 需要服务器重新编译后验收。

服务器双 V100 / CUDA 12.6 的无模型验收（会编译；不依赖 PyTorch）：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash test/distributed/run_process_tp_monitor_v100.sh
```

脚本会检查最终 CMake 缓存中的 CUDA、进程 TP、NUMA 和 SM70 选项。旧构建目录切换
CUDA 编译器可能导致 CMake 清空缓存并恢复默认选项；检测到这种情况后会重新应用一次
完整配置，仍不符合要求则停止，不继续构建缺失的目标。配置命令本身失败时直接退出。

所选 GPU 对具备双向 P2P/native atomics 时，可设置 `TP_MONITOR_CUSTOM=1`，在 NCCL 验收后
追加自定义规约配置。需要正式模型 Eager/Graph 回归时，继续使用上面的
`run_process_tp_exit_v100.sh`；它也会运行本节新增 case。

### 既有验收范围

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

边界：token 数 1..2048、hidden<=65536、偶数 head_dim<=256、本地融合 QKV
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
词表分片及已有 QKV/O/Down bias。序列长度 1..2048，position 起点 0..1048576；
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

- Prompt 长度 1..2048；请求总容量 1..2048，包含 Prompt 和后续送入的 token。
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
  `--raw` 跳过模板。模板展开后的 Prompt 仍限 1..2048 token，超长报错，不截断。
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
- Prompt 含模板及历史仍限制 1..2048 token；`prompt+max_tokens-1<=2048`。超限拒绝，
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
不截断；跨请求 KV 复用需显式开启文末的前缀缓存。`server/serve` 使用同一个模型工厂和已有的受限 HTTP/SSE 实现，
不是线程后端的完整 FastAPI 服务：仅文本 chat completions、models、health、取消接口，
不支持 Responses、Anthropic、工具、多模态、嵌入或 API Key。设置不支持的参数会在加载前失败。
`llm.model(...)`、异步 handle API、webui/export/benchmark 尚不属于这个进程后端适配接口。

- 仅 Linux + NVIDIA CUDA，本轮验收模型为官方未量化 `Qwen/Qwen3-1.7B`。
  源权重 F16/BF16/F32 由既有读取器转换为 FP16，KV 也是 FP16；不支持 BF16 计算或量化权重。
- Prompt（含模板及历史）1..2048 token，`prompt + max_new_tokens - 1 <= 2048`。
  单活动请求，HTTP 有界排队，无连续批处理。文本生成默认关闭 thinking，最大输出默认 32。
- `--tp 2` 表示可见设备中的前两张卡；`--tp 0,1` 为逻辑编号；`--tp 0` 为单卡 0。
  `--tp auto` 要求显式设置 `CUDA_VISIBLE_DEVICES`。不做卡数静默裁剪，不接受线程 TP 的比例配置。
- 已有 `--dtype auto/float16`、`--atype auto/float16`、`--kv_cache_dtype auto/float16` 可用。
  不支持的显式选项（如 `--api_key`、量化、paged KV、LoRA）明确拒绝。
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

## 自定义 AllReduce（第 3 项，服务器验收已通过）

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
  该阶段控制协议 v3；前缀缓存阶段升级为 v4。新旧 worker/协调器混用明确拒绝，脚本一起重建所需产物。
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
本地只做纯 Python 回归和静态 review，未编译或运行 GPU。用户已确认十四组服务器验收全部 PASS。

单次正式入口（上面脚本完成构建，并使用上一节 CUDA/PYTHONPATH 环境后）：

```bash
python3 -m ftllm.cli run /home/models/Qwen3-1.7B \
  --tp_backend process --tp 2 --cuda-graph --custom-allreduce \
  --prompt '请用一句话介绍你自己。' --max-new-tokens 32
```

设计依据：[CUDA 12.6.3 IPC](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#interprocess-communication)、
[PTX 系统作用域内存序](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model)。

## 跨请求前缀缓存（第 4 项，待服务器验收）

正式工厂/CLI 使用 `--prefix-cache true`，默认关闭；可以同时开启 `--cuda-graph --custom-allreduce`。
原生生成/Decode 检查器使用 `--prefix-cache 0|1`，Python `Qwen3ProcessSession` 使用布尔参数 `prefix_cache`。
前缀匹配、LRU、KV 复制/释放和后缀计算均在 C++/CUDA worker，Python 只传参并检查实际执行证据。
旧线程环境变量 `FASTLLM_PREFIX_CACHE` 及分页快照参数不控制此实现，显式使用会被拒绝。

- 模型执行器内按 token ID 的最长公共前缀匹配，同时要求相同 RoPE 起点。
  模型、rank、FP16 KV 布局由执行器生命周期隔离；卸载/重载清空，不能复用旧模型快照。
- 完全相同、扩展、缩短或部分相同的 Prompt 都可命中，但至少重算最后一个 token 取得 logits。
  单 token Prompt 不复用。无状态 `PrefillQwen3` 不读写跨请求缓存。
- 成功 Prefill 后复制全部 Prompt 的各层 FP16 KV，生成的 Decode token 不写入快照。
  快照不可变，命中后拷贝到独立请求 KV；Graph 仍只绑定本请求的 KV/工作区。
  请求结束释放请求 KV/Graph，保留快照；清快照不改变活动请求 KV/Graph。
- 最多 8 条 LRU 记录；采用全局 KV 头数计算 64 MiB 的保守预算，确保各 rank 淘汰决策一致。
  实际单卡占用不超过该预算，TP 分片可能让实际占用更低。单条超过预算就不插入，下一次仍全量计算。
  内存不足或 CUDA/NCCL 失败终止整组，不单 rank 降级。进入后缀计算前通过 NCCL 核对各 rank 复用长度。
- 取消发生在 Prefill 完成之后时，已完成的 Prompt 快照保留；取消前生成的 token 不会成为新快照。
  未执行 Prefill 的取消/非法长度不会改变缓存。非法 token ID 仍先验证，不能因命中跳过检查。
- 正式模型 `model.clear_prefix_cache()` 或会话同名方法等待当前请求完成后清空快照，权重继续驻留。
  原生 `ProcessGroup::ClearQwen3PrefixCache()` 也支持活动请求期间清理，因快照与请求内存互不别名。
  正常模型关闭释放快照、请求资源及权重；worker 失败由整组进程回收 CUDA 上下文。
- 每 rank 的 `graph` 诊断新增 `prefix_cache`：enabled、entries、device_bytes、lookups、hits、
  reused_tokens、last_reused_tokens、last_computed_tokens、evictions、clears。计数是整数；
  显式清缓存只增加 clears 并清空资源，不清零历史计数；卸载模型全部清零。
  各卡 KV 头数可以不同，因此资源字节不强制相等，匹配/淘汰/执行长度必须一致。
- 原限制保持：未量化稠密 Qwen3，FP16 计算/KV，Prompt 最长 2048 token，总请求容量最多 2048，
  单活动请求；没有分页缓存、跨模型/跨进程组共享或连续批处理。命中只减少 Prefill 后缀工作量，
  复制及一致性检查有开销；尚未做吞吐/延迟基准，不据此宣称加速比。

控制协议升级 **v4**，请一起重建 coordinator、worker、检查器及正式 Python 包。
在远端 master 合并 sk 后，仅在服务器执行：

```bash
(
set -euo pipefail
cd /home/project/vllm_fastllm
git fetch origin
git switch master
git pull --ff-only origin master
bash test/distributed/run_qwen3_prefix_cache_v100.sh /home/models/Qwen3-1.7B
)
```

脚本沿用 CUDA 12.6 / SM70 / 双 V100，先重建并跑原十四组，再跑十一组新增验收，总计 **25 组**：

1. Eager/Graph × NCCL/自定义规约四种组合，每种各跑合成 10 项、真实模型 1 项，共八组。
   覆盖 F16/BF16/F32 源权重、bias、tied embedding、KV 头复制和不均匀分片；逐步对照
   关闭前缀缓存的 NCCL Eager TP、单卡，以及六种普通合成配置的独立 CPU 数学参考。
   验证冷/完全/部分/更短/扩展命中、不同 RoPE 起点、8 条 LRU 淘汰、活动 Graph 中清理、
   无状态 Prefill 交错、1/128 token 边界、模型重载隔离及缓存驻留时非法后缀导致整组回收。
   三个较大合成配置另验 64 MiB 保守预算下/上/超界、单条超预算不插入，以及命中后规约从
   大消息 NCCL 转入小消息自定义路径；这三项用 NCCL TP/单卡数值参考，不执行昂贵 CPU 重算。
2. 前缀缓存 + Graph + 自定义规约的合成常驻会话 96 项、真实模型 16 项，共两组；含拒绝、EOS、
   采样、流式、提前取消和 Graph 运行后取消恢复，检查真实 logits 和 Token ID。
3. 同样组合的正式工厂/CLI/HTTP/SSE 13 项，包含实际前缀命中、公开清缓存接口和清理后冷启动。

每组唯一结果目录在 `test/distributed/results`，包含 JSON/Markdown/CSV/XLSX。
报告记录最大/平均绝对误差、失败元素、完整命令、GPU/CUDA/架构/Git、逐任务复用长度和显存字节；
功能测试 warmup/iterations/repeats 为 0/1/1（Graph 另有每请求一次内部 warmup），不包含性能计时。
任何失败返回非零且保留已经完成的报告。本地只做纯 Python/静态检查，未编译、未运行 GPU。

服务器脚本完成构建后，正式入口示例：

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1 USE_TORCH=0
export PYTHONPATH="$PWD/build-process-tp-cu126/process-tp-python${PYTHONPATH:+:$PYTHONPATH}"
python3 -m ftllm.cli chat /home/models/Qwen3-1.7B \
  --tp_backend process --tp 2 --prefix-cache true --cuda-graph --custom-allreduce \
  --max-new-tokens 32
```

## DP 参数与 GPU 分配（第一步）

新增 `--dp`（完整模型副本数，默认 1）和 `--parallel-plan`（仅输出设备分配 JSON）。
预览不要求模型目录、原生可执行文件、PyTorch 或 Transformers，不加载模型、不启动推理 worker。
真实枚举通过 CUDA Driver API 查询可见逻辑设备、UUID 和名称，不创建 CUDA context，
也不使用 `nvidia-smi` 的物理序号推算分配。需要 Linux 和可加载的 NVIDIA 驱动，不需要编译。

- `DP>1` 必须指定 `--tp-backend process`；此时 `--tp` 为每副本卡数，省略代表 1，范围 1..64。
  纯 DP 为 `--dp 2 --tp 1`，DP+TP 为 `--dp 2 --tp 2`，分别需要 2、4 张 GPU。
- 按 CUDA 可见逻辑顺序连续分配，副本间不重复用卡；多余 GPU 不参与分配。
  用 `CUDA_VISIBLE_DEVICES` 选择或排列设备；多副本暂不接受 `--tp auto`、设备列表或加权配置。
  卡数不足、重复设备、与 `--device`/`--moe_device` 冲突均报错，不裁剪、不回退。
  显式可见列表的条目数与驱动实际枚举不一致时也拒绝，防止无效或重复选择器导致静默截断。
- `DP=1` 的正常执行保持旧行为，不增加驱动查询；原有线程/进程 TP 参数继续有效。
  单副本预览支持裸卡数、简单编号列表和 `auto`，进程后端沿用原有显式 `--tp` 要求；
  线程后端的加权设备配置不在本次预览范围内，且预览不会像旧线程路径那样静默裁剪卡数。
- `--custom-allreduce` 在预览中只允许每副本 `TP=2`，这一步只校验配置，不验证 P2P 连通性。
- CLI 在 `chat/run/server/serve` 中输出预览后退出。JSON 配置支持 `dp` 和布尔型
  `parallel_plan`，也接受 `parallel-plan` 别名，但不能同时提供两个别名；配置文件保持原有覆盖 CLI 的顺序。
  JSON 中 `dp: true`、浮点数或字符串都拒绝，不能绕过正整数校验。
- 没有 `--parallel-plan` 时，`DP>1` 由正式模型工厂进入多副本池；直接构造单副本
  `ProcessTPModel` 仍拒绝 `DP>1`，不能静默启动一个副本。
- 返回的 JSON 包含 `dp_size`、`tp_size`、`total_gpus`、`visible_device_count` 和 `replicas`。
  每个副本包含 `dp_rank` 和设备的 `logical_id`、`uuid`、`name`。
  `driver_cuda_version` 是驱动支持的 CUDA 版本编码，不是所安装 Toolkit 的版本；
  `preview_only: true` 表示本次调用只预览，`multi_replica_execution_available: true` 表示支持另行执行。

以下命令在仓库根目录直接使用当前源码的正式 CLI，无需更新旧构建包或重新编译。
先在两卡服务器检查纯 DP：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD/tools${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m fastllm_pytools.cli serve \
  --tp-backend process --dp 2 --tp 1 --parallel-plan
```

若有四张 GPU，再预览 DP+TP（两张卡不能通过这项实际设备检查）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$PWD/tools${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m fastllm_pytools.cli serve \
  --tp-backend process --dp 2 --tp 2 --parallel-plan
```

带报告的验收入口：

```bash
# 无 GPU 的 Python 回归：CUDA 查询和原生进程使用测试替身；不会编译。
python3 test/distributed/run_parallel_config.py

# 两卡服务器：上述回归，加上正式 CLI 的真实 CUDA 设备预览。
CUDA_VISIBLE_DEVICES=0,1 python3 test/distributed/run_parallel_config.py --real-cuda --dp 2 --tp 1

# 四卡服务器：上述回归，加上 DP+TP 的实际设备预览。
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test/distributed/run_parallel_config.py --real-cuda --dp 2 --tp 2
```

每次生成独立的 `test/distributed/results/*_parallel_config_*` 目录，包含
`summary.json`、`summary.md`、`summary.csv`、`summary.xlsx` 和原始测试日志；
实际 GPU 用例另存完整 CLI 命令及 stdout/stderr。任一失败返回非零，中途失败保留已执行部分。
报告区分模拟与真实驱动路径，记录 Git/命令/设备及 CUDA 来源；未执行真实查询时 GPU/CUDA 为 null，
编译架构为不适用。此阶段不加载真实模型，不包含模型精度或吞吐性能结论。

## DP 多副本执行（第二步）

正式模型工厂及 `ftllm run/chat/server/serve` 支持完整模型副本 DP：

| 参数 | 执行方式 | GPU 分配 |
| --- | --- | --- |
| `--dp 2` | 纯 DP，省略 TP 等价于每副本一张卡 | `[0]`、`[1]` |
| `--dp 4 --tp 1` | 四个完整模型副本 | `[0]`、`[1]`、`[2]`、`[3]` |
| `--dp 2 --tp 2` | 两个完整模型副本，每副本内部 TP2 | `[0,1]`、`[2,3]` |

以上均要求 `--tp-backend process`。每个副本启动独立 C++ 协调器和 TP worker，
只在自己的 GPU 集合内做 NCCL/可选自定义 AllReduce。权重、前向、采样、Graph 与 KV
仍由既有 C++/CUDA 路径执行；Python 仅负责 tokenizer、生命周期和分发。
GPU 编号是 CUDA 可见逻辑编号；子进程继承启动时的环境快照，不修改父进程的可见设备列表。

- 全部副本就绪后才接受请求，启动期间任一失败都会回收已成功启动的副本。
- 默认按空闲副本轮转，一次请求只提交到一个副本，不重试、不复制请求。
  每副本一次处理一个请求，多个副本可以同时运行；不包含 continuous batching。
  可选的前缀感知策略见文末“DP 空闲优先与前缀缓存路由”。
- 模型工厂的 `generate` 返回 `dp_rank` 和池内唯一的 `dp_request_id`；原生 `id` 仍属于该副本自己的会话。
  `response`/`stream_response` 保持文本接口。HTTP 消费线程数等于 DP 副本数，队列容量仍是全局等待请求数。
  `/health` 增加 `active_request_ids`、`dp_size` 和 `replicas`，`workers` 中包含 `dp_rank`。
- 取消、消费者提前关闭文本流、超时、原生协议错误均复用 TP 清理语义；故障副本不能继续分配。
  本阶段发生致命副本故障时停止整个池的新分配，由关闭流程回收所有副本；尚无自动重启或降级服务。
- Prefix Cache/Graph/自定义规约的资源和计数各副本独立；清空前缀缓存会暂停新分配、等待旧调用结束，
  再清理全部副本。没有跨副本 KV 迁移或共享缓存。
- 正常退出、Ctrl-C、CLI SIGTERM 和启动失败均经过回收路径。程序接口应使用上下文管理器或显式 `close()`。
  关闭会取消原生在途请求，并回收全部进程，不依赖客户端继续读取暂停的文本迭代器。
- 支持范围沿用进程 TP：Linux NVIDIA CUDA、本地未量化稠密 Qwen3、FP16 执行、Prompt 最多 2048 token。
  纯 DP 可使用 Graph 和 Prefix Cache；自定义 AllReduce 只允许副本内部 TP2，必须满足 GPU P2P 条件。

示例（在仓库根目录，使用已有产物）：

```bash
export USE_TORCH=0
export PYTHONPATH="$PWD/tools${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0,1 python3 -m fastllm_pytools.cli run /home/models/Qwen3-1.7B \
  --tp-backend process --dp 2 \
  --tp-binary build-process-tp-cu126/fastllm_tp_generate \
  --tp-worker build-process-tp-cu126/fastllm_tp_worker \
  --prompt '请用一句话介绍你自己。' --max-new-tokens 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m fastllm_pytools.cli serve /home/models/Qwen3-1.7B \
  --tp-backend process --dp 2 --tp 2 \
  --tp-binary build-process-tp-cu126/fastllm_tp_generate \
  --tp-worker build-process-tp-cu126/fastllm_tp_worker \
  --host 127.0.0.1 --port 8000
```

本地仅 Python 检查（CUDA/原生协议使用测试替身，不编译）：

```bash
python3 test/distributed/run_parallel_config.py --execution-tests
```

覆盖 DP1 兼容、DP2/4 单卡副本、DP2×TP2、真实子进程协议并发、启动回滚、运行故障、取消、
流式关闭、SIGTERM 清理、HTTP 多请求、Graph/IPC/Prefix 组合，以及验收器成功与部分失败报告。

新的 Ubuntu 24.04 / 四卡 V100 服务器（Vast 镜像已配置 NVIDIA apt 仓库）执行：

```bash
# 首次准备：并行安装 CUDA 12.6 工具链，切换并固定配套 NCCL，创建 Python venv，下载模型。
# 此步骤会安装系统依赖并下载文件；不会编译项目。已有完整环境可跳过。
bash test/distributed/prepare_process_dp_v100.sh

# 服务器构建及五组验收；默认使用 CUDA_VISIBLE_DEVICES=0,1,2,3。
bash test/distributed/run_qwen3_dp_v100.sh /home/models/Qwen3-1.7B
```

五组包含：Python/真实设备回归；DP2×TP1；DP4×TP1 Graph+Prefix；DP2×TP2；
DP2×TP2 Graph+Prefix+IPC。后四组每组 10 个功能 case，使用正式工厂做同 TP 大小的 DP=1 基线，
检查精确 Token ID、请求归属、并发、取消、缓存清空、文本流、HTTP 和退出回收。
CLI case 在同一验收进程调用正式 `cli.main`，保留等价终端命令与原始输出；不会绕过模型工厂。
严格检查实际 Graph/规约/Prefix 路径；不支持的 P2P 或执行设置直接 FAIL，不静默回退。
每组独立保存 Markdown、CSV、JSON、XLSX 和原始日志，失败返回非零且保留部分报告。
这轮验证功能正确性，不包含吞吐提升、延迟或调度最优性的结论。

2026-11-12 用户四卡 V100 / Qwen3-1.7B 实测：四种模型配置均 PASS。
前三份服务器报告为 `20261112T030604Z_dp_bfd599a5`、`20261112T030733Z_dp_8ff6997e`、
`20261112T031013Z_dp_d9a32c84`。组合配置先因 GPU 配对不满足双向 P2P/原生原子检查失败，
按 UUID 将可见顺序调整为物理 GPU 0、3、1、2 后，通过报告为 `20261112T032047Z_dp_4a333ee6`。
这台机器的 0+3 是 NV2，1+2 是 NV1；此配对不适用于其它机器，运行前须检查各自拓扑。

## DP 吞吐与延迟对照（第三步）

`benchmark_qwen3_dp.py` 使用当前源码的正式 `make_normal_llm_model` 与 `generate`，
只运行已编译的协调器和 worker，不构建、不下载、不加载旧 Python 推理库。
默认比较 DP1×TP1、DP2×TP1、DP4×TP1、DP1×TP2、DP2×TP2；每种分别测并发 1 和 4。
每轮固定 8 个请求，每请求输出 32 token，关闭 EOS 提前结束，使用同一份实际 Prompt Token ID、
贪心采样参数与 seed=0；Prompt 仍受正式后端最多 2048 token 限制。
每个 case 预热 1 轮，正式执行 5 次独立工作轮，模型权重始终驻留。optimized 在预热前额外通过公开接口
清空前缀，并串行执行 DP 个冷请求，确认所有副本都已建立相同前缀；这一步不计入预热轮数或正式计时。
默认共 20 个 case。

两个 profile 分别报告：

- `common`：所有配置都使用 Eager、关闭 Prefix Cache；TP2 使用 NCCL，TP1 无跨卡规约。
- `optimized`：所有配置都使用 Decode Graph、预热相同 Prompt 的 Prefix Cache；TP2 使用已经验收的
  自定义 AllReduce。每个副本都必须完成预热，正式请求必须实际复用整个可复用前缀。
  这是当前已支持优化的统一组合，不表示已搜索出每种硬件/负载下的最佳配置。

每种配置按顺序执行，避免不同模型争抢 GPU。并发是固定数量的客户端线程，每个客户端结束一个请求后
才提交下一个；这是有限工作量的闭环测试，不代表无限队列饱和吞吐或开放到达率下的服务容量。
并发 1 时 DP 通常不能提高吞吐；并发 4 用于观察多个副本同时处理请求的效果。
单副本内部仍为单请求执行，没有 continuous batching。

计时范围与报告口径：

- 模型加载、分词、测试线程及计时缓冲区创建、测试预热、输出对照、报告/样本保存都在计时外。
  计时包含公开模型入口的排队、协议校验、IPC、前向、采样、流式事件以及正式 KV 生命周期。
  正式路径每请求的 KV/Graph 分配、捕获、warmup 和释放仍然保留；不会为测速单独省略。
  所有配置都以同步生成接口完成为边界，不额外插入逐 kernel CUDA 同步。
- 新的公开参数 `--tp-request-log on|off` 默认 `on`，JSON 配置同样校验。
  `off` 关闭逐请求输入/结果 JSON 及 DP dispatch 文件；仍保留启动、退出、分配和原生错误日志，
  不改变校验、取消、缓存、调度或故障处理。测速使用 `off`，返回的原始结果在停止计时后写到 `raw/`。
  该参数仅适用于进程后端；线程后端拒绝 `off`。
- 每次重复的 batch 时间是全部固定请求完成的墙钟时间；吞吐为实际固定输出 token 总量除以该时间。
  `--iterations N` 会在一次重复内执行 N 轮，累计各轮有效计时，不计轮间检查和落盘。
  Median/P95 batch 延迟和 CV 使用各次重复样本；请求延迟/TTFT 的 Median/P95 使用所有请求样本。
  TTFT 从调用公开生成接口前开始，包含等待可用副本；ITL 是客户端相邻 Token 事件间隔，包含 IPC。
  这不是单个 GPU kernel 的延迟，也不包含 HTTP 网络层。
- `speedup_vs_dp1` 仅为同 profile、同 TP 大小、同并发的 DP1 Median batch 时间除以当前时间。
  增加 GPU 的扩展收益与每卡效率分别通过总吞吐和 `output_tokens_per_s_per_gpu` 展示。
  不跨 profile、不同缓存状态或不同 TP 计算 DP 加速比。
- 每轮停止计时后检查实际执行证据，以及同 profile/同 TP/同缓存状态的 DP1 Token ID 精确一致性。
  optimized 的冷请求对照 DP1 冷基准，预热及正式请求对照 DP1 热基准；只有 DP1 能建立基准，多副本
  不能用自己的首个输出作为基准。建立前缀的串行请求和预热轮都必须覆盖全部副本。
  每个请求必须实际处于指定的冷/热状态，部分前缀复用不允许冒充完整的热缓存。
  JSON 保存两种 Token 基准和前缀建立记录，MD/CSV/XLSX/JSON 记录 `cold_warm_token_mismatches` 与
  `cold_warm_first_divergence_token_1based`；这些字段描述冷热差异，不判断两种路径的数值等价性。
  最大/平均绝对误差和失败元素数作用于 Token ID，不能当作独立数学精度结论。
  任一输出错误、冷热状态不一致、路径回退、请求失败或中断都返回非零，保留已完成样本与部分报告。

服务器已完成第二步构建时，更新源码后直接运行（无需重新编译）：

```bash
# 按实际机器的设备身份/拓扑设置；本例对应已验收的四卡 V100。
export CUDA_VISIBLE_DEVICES="GPU-a88c9669-989e-a8dd-22a0-94a45b467c49,"\
"GPU-0310107f-031e-6fcd-eb65-6dfa45981711,"\
"GPU-f0a99453-777f-8b16-89a3-bab78019f23d,"\
"GPU-081484a6-9730-453b-59a2-32bd75ab6c2c"
bash test/distributed/benchmark_qwen3_dp_v100.sh /home/models/Qwen3-1.7B

# 可选：只跑公共配置；其它参数原样传给 Python 入口。
bash test/distributed/benchmark_qwen3_dp_v100.sh /home/models/Qwen3-1.7B --profiles common
```

每次创建唯一的 `test/distributed/results/*_dp_perf_*` 目录，包含 `summary.md`、`summary.csv`、
`summary.json`、`summary.xlsx`、`samples.csv`、输入 Token ID、原始逐轮响应和各模型生命周期日志。
XLSX 包含 Cases/Samples/Metadata；展示浮点数统一三位，JSON 保存原始精度和全部 Token 时间戳。
报告记录 GPU/UUID/拓扑、CUDA、编译架构、Git、完整命令、模型参数、预热/迭代/重复次数和实际分配。
服务器吞吐数值须以实测报告为准。

无 GPU 的检查命令（协议使用替身，结果不属于真实性能报告）：

```bash
python3 test/distributed/run_parallel_config.py --execution-tests --benchmark-tests
```

### 冷/热前缀输出分歧诊断

若 optimized 在预热时出现 Token 不一致，先保留失败报告。`diagnose_qwen3_prefix.py` 读取该报告的
`request.json`，原样保留 Prompt、输出长度、EOS 和采样配置；通过正式 `Qwen3ProcessSession` 在单卡上
依次运行 Eager 和 Decode Graph，各自检查冷请求、热请求、热请求重复，以及清空前缀后的冷请求。
只使用已有 generate/worker，启用原生 logits trace，不编译或下载。该工具限贪心、无 EOS 的请求，
Prompt/输出长度至少为 2，完整 trace 不得超过原生 8M logits 上限；超限报错，不截断输入。

```bash
# 在已配置 CUDA/Python 环境的服务器上运行；替换为实际失败报告目录。
python3 test/distributed/diagnose_qwen3_prefix.py \
  --model /home/models/Qwen3-1.7B \
  --generate build-process-tp-cu126/fastllm_tp_generate \
  --worker build-process-tp-cu126/fastllm_tp_worker \
  --build-cache build-process-tp-cu126/CMakeCache.txt \
  --request-file test/distributed/results/实际失败报告目录/request.json
```

诊断只比较输入历史相同的步骤，包含首次选词分歧的那一步；后续输入已经不同，不能再直接比较 logits。
沿用真实模型前缀验收的 `atol=0.05, rtol=0.01`，报告首次分歧位置及两侧前两名 Token、分数和分差。
相同缓存状态的重复请求还必须输出完全相同的 Token。生成完整的 MD/CSV/JSON/XLSX 汇总和原始响应，
任一数值超差、路径错误或重复不一致返回非零，失败也保留报告。
PASS 仅说明所比较步骤在既有容差内；它不表示冷热输出完全相同、分歧后数值已验证或独立数学正确性。
这是定位工具，不计时、不提供性能结论，也不会修改或放宽性能验收标准。

### TP并发下降的交替诊断

`diagnose_qwen3_tp_concurrency.py` 专门检查同一DP1模型在并发1/4下的吞吐差异，默认TP2、optimized。
读取已有性能报告的 `request.json`，原样使用Prompt Token、输出长度、EOS和采样配置。
只加载一个正式模型，先建立前缀并分别预热，再以ABBA（1、4、4、1）、BAAB（4、1、1、4）交替顺序
执行5组；每个并发共10个正式样本，每轮仍为相同的8个请求。模型、设备、热缓存和执行后端保持相同，
不通过重启模型或修改频率来制造差异；计时、流式事件和严格Token检查复用正式DP测速入口。

```bash
# 使用已经验收的双向P2P/原生原子GPU对；本机原GPU0与GPU3。
export CUDA_VISIBLE_DEVICES="GPU-a88c9669-989e-a8dd-22a0-94a45b467c49,GPU-0310107f-031e-6fcd-eb65-6dfa45981711"
bash test/distributed/diagnose_qwen3_tp_concurrency_v100.sh \
  /home/models/Qwen3-1.7B test/distributed/results/实际性能报告目录/request.json
```

默认每秒由独立Python进程只读采集GPU UUID、频率、功耗、温度、利用率和时钟限制原因，以及测试进程
及后代的累计CPU时间、CPU使用率、亲和性和可用的cgroup v2 CPU计数。采样进程不导入模型或创建CUDA
上下文；测试失败或中断时回收。时钟原因字段不支持时明确记录降级，基础GPU查询失败或观察进程提前退出
会使诊断失败。cgroup路径和不可用信息原样保存，不能把缺失信息理解为没有CPU配额或没有限流。
GPU字段含义见 [NVIDIA nvidia-smi文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。

报告包含 `summary.md/csv/json/xlsx`、全部原始生成结果、`telemetry.jsonl`、`timeline.json` 和
`observations.xlsx`（Timeline/GPU/CPU/Cgroup四张表及对应CSV）。主线程的采样校验和报告IO在计时外；
观察进程与生成同时运行，可能带来观察开销。CPU使用率100%表示占用一个逻辑CPU，进程总量可以超过100%。
CPU表的 `cpu_interval_start/end` 标出两次CPU快照之间的实际区间，`cpu_overlapping_sequences` 标出
该区间涉及的测量轮次；首次快照没有CPU使用率，不能把累计区间误当成当前GPU查询窗口。
GPU利用率有自己的采样周期，时间窗口重叠和相关性不能单独证明因果。GPU汇总只纳入采集窗口完整落在
对应计时内的目标UUID样本，边界样本仍保留在原始表格中。所有浮点展示保留三位，JSON保留原始精度。

`throughput_ratio_vs_c1` 是同一TP/profile下C1 Median批耗时除以当前并发Median批耗时；
`within_block_c4_over_c1_ratios` 用各组同等工作量的累计耗时计算。它们用于分析并发变化，不是DP扩展收益。
可追加 `--no-telemetry` 检查观察开销，或 `--tp 1` / `--profile common` 做单独控制实验；不同profile
或不同采样状态的结果必须分别标识，不能混算加速比。脚本不安装依赖、不重新编译、不改变GPU或CPU设置。

## DP 空闲优先与前缀缓存路由（第四步）

新增 `--dp-routing round_robin|prefix`，默认仍为 `round_robin`。`prefix` 要求
`DP>1`、`--tp-backend process` 和 `--prefix-cache true`；CLI、JSON 配置和模型工厂统一校验。
纯 DP 和 DP+TP 使用同一分发规则：

1. 只比较当前空闲副本中可复用的 Prompt Token 数，选择最长匹配。
2. 匹配长度相同或全部未命中时，保持轮转顺序；至少保留最后一个 Prompt Token 重算。
3. 持有缓存的副本正忙时，使用其他空闲副本；全部忙时沿用可取消、可超时的等待。

路由依据 worker 实际保留的快照目录，不把历史请求当作仍存在的缓存。原生反馈新增
`prefix_cache.directory_version=1` 和 `snapshots`；每个快照只有 RoPE 起点与 Token ID，
没有 GPU 指针。协调器检查各 TP rank 的目录一致性，Python 在请求完成、取消及清理后更新记录。
取消可能保留已完成 Prefill 的有效快照，因此由原生目录决定能否复用。淘汰、清空、关闭与故障均会
更新或作废对应记录；部分副本清缓存失败会停止新分发，等待超时且尚未开始清理时保留原状态。

文本和流式入口在分配前分词一次，复用正式 TP 请求构造与执行流程；HTTP 走同一模型工厂。
`ready`、设备预览及 `/health` 包含策略名称，生成结果与 dispatch 日志中的 `dp_routing` 包含
`policy`、分配时的 `idle_ranks`、`estimated_reused_tokens`、`wait_ms`。轮转模式不估算前缀，
对应估算字段为 `null`。保留每副本单活动请求、独立权重和 KV 的限制，不迁移或共享跨副本 KV。

需要同时重新编译协调器与 worker；混用新旧原生产物明确失败。新 Python 配合成套旧原生产物时，
默认轮转仍兼容，但启用 `prefix` 会因缺少目录明确拒绝，不会静默回退。

更新产物后，可在前述纯 DP 或 DP+TP 的 `run/chat/serve` 命令中追加：

```bash
--prefix-cache true --dp-routing prefix
```

### 混合请求验收

`benchmark_qwen3_dp_routing.py` 通过正式工厂对比两种路由，分别测并发 1 和 4。每轮 24 个请求，
包含六种互不共享首 Token 的 64-token Prompt，每种重复四次，输出长度依次为 8、32、8、32，
共 480 个输出 Token。固定 Token ID、贪心采样、seed=0、空 EOS 集合、FP16 KV 和 Decode Graph；
TP2 使用已验收 GPU 对的自定义 AllReduce。

两种策略在每个预热及正式轮开始前调用公开清缓存接口并验证目录为空；随后缓存如何演化是策略的
比较对象。同 TP 的 DP1 独立建立每个请求的冷、热输出基准，按实际复用长度验证，不能将冷、热
输出不同直接判为错误。另以相同请求历史的 DP1 基准验证部分命中、八条目上限后的淘汰和清空，
并验证取消后有效缓存复用及忙副本避让。所有功能 case 均检查实际输出。

默认一轮预热、至少五轮正式测量。计时只含正式入口排队、IPC、生成、KV 生命周期和生产协议校验；
清缓存、测试校验、原始样本及汇总文件写入均在计时外，逐请求落盘使用正式参数关闭。
报告包含 Median/P95 批耗时、请求延迟、TTFT、ITL、CV、吞吐、实际执行路径、命中率与复用 Token 数。
`speedup_vs_round_robin = 同 DP/TP/并发的轮转 Median 批耗时 / 当前策略 Median 批耗时`。
路由策略依次运行，报告记录顺序；应结合波动判断结果，不能仅凭一次加速比断言稳定收益。

每次生成独立的 `*_dp_routing_*` 目录，保存 `requests.json`、`references.json`、全部原始轮次，
以及性能和 `functional/` 各自的 `summary.md/csv/json/xlsx`。任意 case 失败返回非零，并保存部分报告。
本地仅运行 Python 子进程协议替身与报告回归，不据此宣称 GPU 功能或性能已验收。

在 sk 合入远端 master 后，四卡 V100 服务器执行下列命令。脚本重建正式产物，先验证默认轮转，
再分别验收 DP4×TP1、DP2×TP2 的新路由；不重新下载模型。UUID 顺序对应本机已验收的物理卡 0、3、1、2。

```bash
(
set -euo pipefail
cd /home/project/vllm_fastllm
git switch master
git pull --ff-only origin master
export CUDA_VISIBLE_DEVICES="GPU-a88c9669-989e-a8dd-22a0-94a45b467c49,GPU-0310107f-031e-6fcd-eb65-6dfa45981711,GPU-f0a99453-777f-8b16-89a3-bab78019f23d,GPU-081484a6-9730-453b-59a2-32bd75ab6c2c"
bash test/distributed/run_qwen3_dp_routing_v100.sh \
  /home/models/Qwen3-1.7B \
  test/distributed/results/20261112T065746Z_dp_perf_ab619d18/request.json
)
```

## DP HTTP/SSE 持续稳定性验收（第五步）

`run_qwen3_dp_service.py` 在独立 Linux Python 子进程中调用正式 `ftllm serve` CLI，使用端口 0，
从正式 ready 文件获取监听地址。客户端通过真实 loopback HTTP/1.1 socket 请求服务；模型工厂、
tokenizer、队列、DP 路由、取消、KV 和停服流程均使用正式实现。服务器运行需要第四步已经构建的
原生产物，本步骤不改变原生协议，不执行编译或安装。

每组固定同一中文 Prompt，混合普通 HTTP 与 SSE、8/32 个输出 Token 上限；聊天模板、seed=0、
贪心采样及模型 EOS 策略保持一致，实际输出可能因 EOS 提前结束。每个请求验证原生 Token ID、
最终文本、请求归属、usage、结束原因，以及 SSE 的角色、增量拼接、唯一终帧与 `[DONE]`。
同 TP 的 DP1 独立建立冷、热输出基准，按原生实际复用长度选择基准，不把冷热输出差异判为错误。
严格检查实际 Graph、自定义 AllReduce（TP2）及 Prefix Cache 路径。

默认并发数和等待队列容量均为 `2 × DP`，先做一轮预热，再连续执行五轮，每轮至少 30 秒，轮间
只校验与保存证据，不清空缓存或重启模型。每轮保存所有 HTTP/SSE 响应，验证 worker 身份和模型
加载次数不变、请求 KV 已释放、队列和活动请求归零，并只读记录本服务进程树的 RSS、线程数和 FD 数。
资源样本用于后续观察变化，不将一次采样或有限时长 PASS 解释为不存在任何内存泄漏。

随后使用 1024 Token 输出上限的真实长 SSE 占满副本，收到内容且确认请求仍在活动后执行：

- 队列满时返回 429，重复活动请求 ID 返回 409；被拒绝的请求不能生成审计完成记录。
- 排队取消返回 499，且没有原生请求号；活动取消必须有原生取消确认及空请求 KV。
- 普通 HTTP 主动断连、SSE 收到内容后断连；随后重新请求并校验实际输出。
- 同时存在活动请求、排队请求及慢请求体时发送 SIGTERM。慢请求先等待真实 `100 Continue`，
  确认服务已接收请求头；检查取消审计、原生卸载后的 KV/Prefix Cache、正式关闭报告及所有原进程消失。
  停服时网络连接允许提前结束，原生取消和清理必须有完整证据。

如果模型在制造压力前已 EOS，测试明确失败并记录无法确认重叠，不修改 EOS、不挂起服务执行来制造通过。
普通 HTTP 断连可能发生在正式分发之前，此时记录为分发前取消；活动 SSE 取消单独要求原生确认。

这是功能与持续稳定性验收，保留正式请求日志和 HTTP 审计落盘，**不作为性能基准**。
`elapsed_s` 只记录持续时间，不与第三、四步吞吐混算加速比。各组产生唯一 `*_dp_service_*` 目录，
包含 `summary.md/csv/json/xlsx`、独立基准、服务器命令/输出、全部原始请求响应、每轮资源快照及
正式原生日志。Token 差异记录最大/平均绝对误差和失败元素数量；不适用数值误差的生命周期检查为 n/a。
任意 case 失败或验收器中断返回非零，尽可能保存部分报告并有界回收本次启动的进程。

本地回归仅以 Python 子进程替代原生协议，并替换 CUDA 枚举和 tokenizer；其余 CLI、真实 HTTP/SSE、
取消和停服实现不替换。覆盖纯 DP/轮转、DP+TP/前缀路由的完整流程，以及文本错误、Token 错误、
SSE 截断、验收器 SIGTERM 的失败报告和进程清理，不能替代 GPU 服务器验收。

在 sk 合入 master 后，现有四卡 V100 服务器执行：

```bash
(
set -euo pipefail
cd /home/project/vllm_fastllm
git switch master
git pull --ff-only origin master
export CUDA_VISIBLE_DEVICES="GPU-a88c9669-989e-a8dd-22a0-94a45b467c49,GPU-0310107f-031e-6fcd-eb65-6dfa45981711,GPU-f0a99453-777f-8b16-89a3-bab78019f23d,GPU-081484a6-9730-453b-59a2-32bd75ab6c2c"
bash test/distributed/run_qwen3_dp_service_v100.sh /home/models/Qwen3-1.7B
)
```

脚本先做 Python 回归，再依次运行轮转/前缀路由各自的 DP4×TP1 和 DP2×TP2，共四组。
仅持续负载阶段合计至少 10 分钟，另加模型加载、基准和异常场景时间。无需再次下载模型或重新编译。
每组应完成 12 个 case 并输出 PASS；可直接运行 Python 验收器调整 `--round-seconds` 延长观察，
报告会记录实际时长和请求数。

## DP 长Prompt与混合长度诊断（第六步）

进程后端现在统一接受包含聊天模板在内的 1..2048 个 Prompt Token，并继续要求
`prompt_tokens + max_new_tokens - 1 <= 2048`。2048 Token 输入最多生成一个输出 Token；
最后输出不再送入 KV。Python、协调器、worker、Attention 和 Prefix Cache 目录使用相同上限。
这仍是有限容量的完整 Prefill，不是分块 Prefill 或无限上下文；原有工作区、请求 KV 各
256 MiB 和 Prefix Cache 64 MiB 预算不变，较大模型可能先触发显存预算拒绝。
**必须重新构建协调器、worker 和进程 Python 包。**

`diagnose_qwen3_dp_lengths_v100.sh` 依次执行 Python 回归、服务器编译、三组独立数值/边界验证，
最后运行 DP4×TP1 和 DP2×TP2 的正式 HTTP/SSE 诊断。数值验证包括：127/128/129、512、1024、
2047/2048/2049 的 Prefill 边界，BF16/F32 源权重和 bias，长Prompt初始化 KV 后 Decode、
总容量超限拒绝，以及长前缀目录的命中与清理；合成小模型对照独立 CPU 数学参考和单卡，
保留误差和完整原始数据。CPU 参考计算期间可能暂时没有新输出，开始每个 case 都会打印 RUN。

服务诊断默认构造接近 64/512/1024 Token 的三种文本，经过模型的正式聊天模板，不截断
Token；实际长度与完整 ID 保存到 `inputs.json`。每种输入分别配 16/96 输出上限和 HTTP/SSE，
共 12 个请求，每轮相同工作量、固定 seed=0；模型 EOS 不变，吞吐按实际输出数计算。
每种 Prompt/输出上限独立建立同 TP 的 DP1 精确 Token 基准，校验实际路径、文本、usage、
SSE 完整性和 KV 回收。CUDA Graph 开启；TP2 开启自定义 AllReduce，长Prefill超出该规约容量
时仍使用正式 NCCL 路径。**本轮使用轮转路由并关闭 Prefix Cache**，不把部分前缀复用造成的
数值差异混入长度诊断；不声明已覆盖带缓存的真实模型混合长度负载。

每种并发先预热一轮，再轮换运行 C1、C=`DP` 和 C=`2×DP`，各五轮，保存 Median/P95/CV、实际吞吐及
逐输入类型延迟。`speedup_vs_c1 = 同一DP/TP配置C1 Median轮耗时 / 当前Median轮耗时`，
它是并发工作量的时间比，不是 DP 扩展比或其它框架的加速比。计时包含真实 HTTP、服务分词、
排队、生产日志、推理及响应读取；测试校验、测试报告落盘、模型加载和输入构造均在计时外。
每条正式 HTTP 审计新增单调时钟 `timing`：`queue_ms` 从入队到取出，不包含分词；
`service_ms` 从取出到完成，不含最终 HTTP 审计写盘。SSE 的 `first_content_ms` 是客户端
首个非空文本块延迟，包含排队，不冒充原生首 Token 时间；普通 HTTP 不提供此指标。
各客户端保留固定请求通道：第 lane 个客户端按乱序后的索引 lane、lane+C、lane+2C 顺序提交。
所有输入在整轮开始时均已就绪，因此另外记录 `client_wait_ms`（整轮开始到实际客户端提交）和
`completion_ms`（整轮开始到该请求完整响应接收完毕）。后者包含尚未提交的等待，不能把降低
并发后仅服务端排队减少解读为整体完成时间改善，也不代表线上独立到达请求的响应延迟。

以 0.5 秒周期通过 nvidia-smi 查询整卡显存，并在每轮结束保存本服务进程 RSS/FD/线程数。
显存统计按 UUID 保存到 `memory.json` 和汇总 metadata，可能包含其它进程，且会漏掉瞬时峰值；
不得将观测最大值解释为本服务精确峰值，也不能仅凭有限采样断言无内存泄漏。
采样失败、输出错误、请求失败或清理失败均返回非零；保留部分汇总与失败轮原始响应。
每组输出独立 `*_dp_lengths_*` 目录，包含 `summary.md/csv/json/xlsx`、`inputs.csv`、
输入与 DP1 基准、逐轮原始响应和显存采样。XLSX 同时包含逐请求与逐输入类型统计。

sk 合入 master 后，在现有四卡 V100 服务器执行（无需下载模型）：

```bash
(
set -euo pipefail
cd /home/project/vllm_fastllm
git switch master
git pull --ff-only origin master
export CUDA_VISIBLE_DEVICES="GPU-a88c9669-989e-a8dd-22a0-94a45b467c49,GPU-0310107f-031e-6fcd-eb65-6dfa45981711,GPU-f0a99453-777f-8b16-89a3-bab78019f23d,GPU-081484a6-9730-453b-59a2-32bd75ab6c2c"
bash test/distributed/diagnose_qwen3_dp_lengths_v100.sh /home/models/Qwen3-1.7B
)
```

默认应得到六个报告：Python 回归、长Prefill、长生成、长Prefix Cache、DP4×TP1、DP2×TP2。
这些服务器结果尚需实际运行确认；本地 Python 协议替身不能替代 CUDA 编译和 GPU 数值验收。

## 并发数等于副本数的混合长度对照（第七步）

用于已完成第六步长Prompt编译、数值边界及服务验收的实例。增加 C=`DP` 对照后，同一服务、
同一组请求下测三档：DP4×TP1 的 C1/C4/C8，DP2×TP2 的 C1/C2/C4；每档预热一轮、正式五轮。
保留原有固定客户端通道与输入构造方式，正式 HTTP FIFO 调度、轮转路由、缓存状态和数值检查
均不改变。执行顺序逐轮轮换并写入 `metadata.repeat_orders`，减少固定位置造成的时间漂移；
五轮不能做到三种执行位置完全均衡，不将小幅差异直接认定为稳定收益。

重点同时比较吞吐、轮耗时 Median/P95/CV、短输入/短输出请求的队列 P95、提交后响应 P95，
以及包含客户端等待的完成时间 P95。各并发档每轮输入集合、Token ID、输出上限和 EOS 策略
完全相同，但请求提交时序不同；这是同一有限请求集合的并发对照，不是固定到达率的线上
延迟测试，也不是生产调度优化后的性能结果。报告额外保留逐请求 `client_wait_ms` 和
`completion_ms`，在 Markdown/CSV/XLSX 和原始精度 JSON 中汇总；终端直接打印各档 CV 和
最短输入各变体的尾延迟。P95 不能相减来估算各阶段贡献。

已有第六步服务器产物时，sk 合入 master 后执行以下命令，**无需编译或下载模型**：

```bash
(
set -euo pipefail
cd /home/project/vllm_fastllm
git switch master
git pull --ff-only origin master
export CUDA_VISIBLE_DEVICES="GPU-a88c9669-989e-a8dd-22a0-94a45b467c49,GPU-0310107f-031e-6fcd-eb65-6dfa45981711,GPU-f0a99453-777f-8b16-89a3-bab78019f23d,GPU-081484a6-9730-453b-59a2-32bd75ab6c2c"
bash test/distributed/diagnose_qwen3_dp_lengths_v100.sh /home/models/Qwen3-1.7B --reuse-build
)
```

`--reuse-build` 缺少协调器、worker 或编译配置时立即失败，不会隐式编译；不再重复长Prompt
数值边界验收。此模式输出三个独立报告：Python 回归、DP4×TP1、DP2×TP2。后两份报告分别
必须包含三个并发 case 且每个完成五轮才可整体 PASS；失败时保存已完成样本和失败轮原始响应。
新实例或尚未完成长Prompt原生产物验收时，应先运行上一节不带该选项的完整流程。

## 固定服务配置与JSONL请求验证（第八步）

固定两套配置供已有四卡 V100 / Qwen3-1.7B 实例使用：

| profile | DP | TP | 建议客户端并发 | 服务端等待队列容量 |
| --- | --- | --- | --- | --- |
| dp4_tp1 | 4 | 1 | 4 | 8 |
| dp2_tp2 | 2 | 2 | 2 | 4 |

两套均使用正式进程后端、FP16 KV、CUDA Graph、轮转路由、关闭 Prefix Cache、开启请求日志，
TP2 启用自定义 AllReduce，并保留正式路径的 NCCL 使用条件。客户端并发指同一时间在途的请求数，
由调用方控制；等待队列容量不是服务端全局并发上限，也不会约束其它客户端的在途请求数。
生产启动器与请求验证器共用 `fastllm_pytools.dp_baseline` 构造的参数，经过正式 CLI、模型工厂
和 HTTP/SSE 路径，不调用独立的推理实现。

选择依据来自第七步服务器报告 `20261112T133217Z_dp_lengths_728d2163` 和
`20261112T133613Z_dp_lengths_132f3c35`（服务器 Git `2d89fc35`，各五轮）：

纯 DP 的 C4/C8 吞吐分别为 147.945/144.643 Token/s，排队 P95 为 0.720/1243.060 ms；
DP+TP 的 C2/C4 吞吐分别为 105.596/112.963 Token/s，排队 P95 为 0.142/1653.806 ms。
原始全精度 JSON、逐轮样本、XLSX、GPU/CUDA/编译元数据均在上述服务器报告目录；这里仅摘录
配置选择依据，未复制或重新生成原始报告。纯 DP 两档吞吐基本相当；TP2 的 C2 是优先降低提交后
响应延迟的取舍。包含客户端等待的整体完成 P95 并未在 C=DP 下全面更低，不能推广为所有负载的
最优配置。这些记录不是下面通用中文示例的测试结果。

### 先验证请求集

`data/qwen3_requests_example.jsonl` 包含通用中文摘要、信息提取、翻译和多轮问答，共八条请求，
HTTP/SSE 各四条；没有实际业务数据。sk 合入 master 后，在之前的四卡 UUID 顺序下执行：

```bash
(
set -euo pipefail
cd /home/project/vllm_fastllm
git switch master
git pull --ff-only origin master
export CUDA_VISIBLE_DEVICES="GPU-a88c9669-989e-a8dd-22a0-94a45b467c49,GPU-0310107f-031e-6fcd-eb65-6dfa45981711,GPU-f0a99453-777f-8b16-89a3-bab78019f23d,GPU-081484a6-9730-453b-59a2-32bd75ab6c2c"
bash test/distributed/run_qwen3_dp_requests_v100.sh /home/models/Qwen3-1.7B
)
```

脚本不编译、不安装、不下载。先做 Python 回归，再顺序验证两套 profile；每套测 C1 和建议的
C=DP，各预热一轮、正式五轮，启动并在结束后关闭本次正式服务。不要与已有占用同一组 GPU
的推理服务同时运行。每种完整对话及采样参数分别建立同 TP 的 DP1 Token 参考，HTTP/SSE 可
共享参考；校验输出、文本、usage、路径和退出后的 KV 回收，不评价摘要/翻译等回答质量。

预期输出三个报告目录：Python 回归和两份 `*_dp_requests_*`。后两份必须各完成两个 case 才能
整体 PASS；请求、采样或清理失败均返回非零并保留部分报告。报告保留 `summary.md/csv/json/xlsx`、
逐请求和逐输入统计、显存采样、完整响应；新增 `requests.jsonl` 原文及 SHA256、`inputs.json`
正式 Token ID 和 `reference_texts.json` 单副本回答文本，供人工检查质量。所有输入在整轮开始
时就绪，沿用固定客户端通道，输入顺序按 seed=0 打乱；包含客户端等待的完成时间只表示该有限
请求集合的完成情况。吞吐按实际生成 Token 数统计，不固定或忽略 EOS。

### 替换JSONL

第二个位置参数可指定自己的脱敏请求文件，例如：

```bash
bash test/distributed/run_qwen3_dp_requests_v100.sh /home/models/Qwen3-1.7B /home/requests/my_requests.jsonl
```

每行一个对象，外层只允许 `name` 和 `request`：

```json
{"name":"summary_01","request":{"messages":[{"role":"user","content":"请用一句话总结这段文字：……"}],"max_tokens":96,"stream":true}}
```

`name` 必须唯一，且为 1..64 位字母、数字、下划线或连字符；`request` 使用正式聊天请求字段。
默认 `model=qwen3`、`max_tokens=96`、`stream=false`；本轮一致性验证固定 greedy：temperature=0、
top_k=0、top_p=1、seed=0。SSE 默认补充 `stream_options.include_usage=true`，明确指定 false
会被拒绝。支持正式服务已有的 system/user/assistant 文本多轮语义，未知字段、工具、多模态、
不支持的采样设置、重名或空文件均失败。完整聊天模板后的输入和生成容量仍受 2048 Token 上限
约束，不截断、不重复请求来凑负载。两套 profile 一起验证至少需要四条请求。

### 常驻服务

两套配置各使用四张 GPU，应择一运行。确认上述请求验证通过后，可在设置好同样 UUID 顺序的
终端中启动纯 DP：

```bash
bash tools/run_qwen3_dp_baseline_v100.sh /home/models/Qwen3-1.7B dp4_tp1 --port 8080
```

DP+TP 则将 profile 改为 `dp2_tp2`。启动器默认监听 127.0.0.1，输出 ready JSON 后服务可用；
模型 API 名称为 `qwen3`，聊天路径为 `/v1/chat/completions`，健康检查为 `/health`。
使用已有 SSH 8080 端口转发即可从本机连接。Ctrl+C 或 SIGTERM 走正式有界停服路径，取消请求
并回收 worker；常驻服务与短时验证脚本的生命周期不同。每次启动生成新的日志子目录。
追加 `--print-command` 仅打印 profile、建议客户端并发和正式 CLI 命令 JSON，不查询 GPU、
不加载模型、不启动 worker。进程服务仍只支持当前已实现的稠密 Qwen3 路径。

## Attention DP＋EP：配置与设备规划（阶段 1）

本节是新增 Qwen MoE 进程模式的阶段记录。**阶段 1 仅接入参数检查和设备预览，
当时未接入多 rank MoE 推理**。阶段 3～6 已新增联合执行、Graph、前缀缓存与正式服务入口，
当前构建及验收步骤见 [ATTENTION_EP.md](ATTENTION_EP.md)；GPU 验收仍需在服务器实际执行。
后续依次验收权重与单层、Eager 联合推理、Graph/缓存、正式服务、四卡完整模型。
现有稠密 Qwen3 的完整副本 DP/TP 语义保持不变。

- `--attention-dp` 和 `--ep` 默认均为 1；第一版要求两者相等且为 1、2 或 4。
- 新模式要求 `--tp-backend process --dp 1 --tp 1`；省略 TP 视为 1，拒绝
  `auto`、设备列表、`0`、布尔值及小数。设备选择、排序统一用 `CUDA_VISIBLE_DEVICES`。
- Attention rank 与 EP rank 一一对应、共用设备；4/4 只用四卡。
  Attention TP 组分别为 `[0] [1] [2] [3]`，EP 组为 `[0,1,2,3]`。
- 新轴大于 1 时即选择新模式。两轴均为 1 时，进程后端依据本地
  `config.json` 的 `model_type=qwen2_moe` 识别单 rank MoE；仅读取模型元数据。
  没有 MoE 配置时保持旧 DP/TP 模式，因此包含默认值的旧配置导出可以往返加载。
- CLI、JSON、工厂和直接进程入口共用校验。JSON 接受 `attention_dp` 或
  `attention-dp`，但同时出现两种拼写会拒绝；沿用 JSON 覆盖 CLI 的原有加载顺序。
- `--dp-routing prefix` 可以用于 Attention rank，仍要求 `--prefix-cache true`。
  `--custom-allreduce` 被拒绝；新模式的通信基线为 NCCL AllGather/ReduceScatter。
- 预览增加 `mode`、`attention_dp_size`、`ep_size`、`ranks`、`attention_groups`、
  `ep_groups`；`total_gpus` 是 EP 组大小。当前
  `attention_ep_execution_available=false`，不把规划能力标为推理能力。
- 官方模型固定为 `Qwen/Qwen1.5-MoE-A2.7B-Chat` revision
  `ec052fda178e241c7c443468d2fa1db6618996be`。真实模型后续仅做四卡 V100 验收；
  1/2 rank 的数值验收使用缩小模型。本阶段不下载模型，也不编译。

本地仅 Python 检查（CUDA 枚举及原生会话使用替身）：

```bash
python3 test/distributed/run_parallel_config.py --execution-tests --benchmark-tests
```

服务器在 `master` 包含本阶段提交后运行；沿用已修复的 Python 环境。
下面只读 CUDA Driver API，不创建 CUDA context、不加载模型、不启动推理 worker。
每次生成独立的 Markdown、CSV、JSON、XLSX 和原始日志；给出完整 GPU UUID 时，
额外把实际枚举身份与指定顺序逐项比较。脚本默认使用本轮 Vast 的四个 UUID，
已有 `CUDA_VISIBLE_DEVICES` 时保留用户顺序；可通过 `ATTENTION_EP_PYTHON` 替换 Python。
先验收 4/4、2/2，再反转四卡可见顺序验收 4/4，任一失败立即非零退出：

```bash
cd /home/project/vllm_fastllm
bash test/distributed/run_attention_ep_config_v100.sh
```

阶段 1 的配置检查通过不代表 MoE 权重、数值、Graph 或服务通过；服务器真实枚举结果
也必须保留在本阶段报告中，再进入阶段 2 的单层验收。

## Attention DP＋EP：MoE 权重与单层（阶段 2）

本阶段增加公开原生 `ProcessGroup::LoadQwenMoe` 和 `ForwardQwenMoeLayer`，
复用已有 worker、stream、权重读取/FP16 上传及 Attention 算子。**阶段 2 仅交付权重与
无历史 KV 的单层执行；完整生成、联合请求调度、Graph、Prefix Cache 和 HTTP/SSE
在阶段 3～6 接入。** 当前 CLI 的 `attention_ep_execution_available=true` 表示有执行入口，
仍不能把单层验收当成完整模型验收。完整验收见 [ATTENTION_EP.md](ATTENTION_EP.md)。

- 显式选择 `qwen2_moe`，支持 EP=1/2/4，Attention TP=1。Embedding、lm_head、
  Attention、Router、共享专家及 Norm 完整复制；路由专家按 `expert_id % ep_size`
  驻留。检查全部权重元数据，仅读取本 rank 的专家 payload。
- 支持 F16/BF16/F32 未量化源权重，上传为 FP16；支持 QKV bias、无 bias、tied
  embedding 和 safetensors 索引。拒绝量化、混合稠密/MoE 层、非法 Top-K、缺失
  或额外张量、部分 QKV bias、非有限权重和设备/CPU 配额不足。
- Qwen2 MoE Attention 不执行 Q/K Norm，不执行稠密 TP Attention AllReduce。
  Router 使用全部专家上的 FP32 softmax、Top-K，再按 `norm_topk_prob` 决定是否
  重归一化，最后转 FP16；相同概率时以较小专家 ID 优先。共享专家按独立 sigmoid
  门控后与路由贡献相加，最后加残差。
- 每轮按最大本地 Token 数 padding；三次 AllGather 分别交换归一化激活、专家 ID、
  路由概率，ID=-1 是无效行掩码。所有 rank 都执行相同集合通信，即使本地没有请求、
  没有专家或没有命中专家。ReduceScatter 使用 FP32 Sum 返回所属 rank 的贡献。
- Prefill 在设备建立每专家索引，读回计数后只对有效聚合 Token 调用 cuBLAS。
  全组每 rank 至多一个 Token 时使用设备路由索引 CUDA 专家内核，不读回路由计数。
  这只是后续 Decode/Graph 的算子基础，本阶段没有 Graph 捕获或 KV 生命周期。
- 专家输出和路由乘法按 FP16 舍入，跨专家贡献以 FP32 累加/通信，再转 FP16。
  不要求与 Transformers 的逐专家 FP16 累加逐位相等，按固定数值容差验收。
- 每 rank 临时工作区上限 512 MiB；正常单层结束同步释放，模型正常卸载释放全部权重。
  失败通过现有进程组路径终止所有 worker。无历史单层诊断拒绝 Graph 和 Prefix Cache；
  完整 MoE 接口支持这两个选项，仍拒绝 custom AllReduce；稠密生成入口拒绝 MoE 权重。
- 单层诊断接口接收各 rank 独立的 FP32 `[T,hidden]` 输入，T=0..2048。
  控制请求和每 rank 响应分别限 1 MiB；这是诊断传输限制，后续整模型执行应在设备
  保留激活。返回实际输出、Router logits、Top-K ID/权重、路由/共享专家输出、
  本 rank 专家 Token 数和通信计数。该接口有诊断读回，不用于性能结论。

模型语义依据固定模型 revision `ec052fda178e241c7c443468d2fa1db6618996be`，
以及 [Transformers v4.40.0 Qwen2 MoE 实现](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L706)。
首版实际模型仍为 `Qwen/Qwen1.5-MoE-A2.7B-Chat`；默认单层验收使用自动生成的
小模型，GPU 权重导出限每 rank 32 MiB。真实模型的单层比较使用 `--dump-weights 0`
关闭权重落盘，见阶段 3～6 的完整验收脚本。

### 服务器验收

无需下载模型。服务器 `master` 包含本阶段提交后，四张 V100、CUDA 12.6 和 NCCL
环境中执行；**脚本会编译 C++/CUDA，仅供服务器运行**：

```bash
cd /home/project/vllm_fastllm
git switch master
git pull --ff-only origin master
# 沿用已确认的四卡顺序；未设置时脚本默认 0,1,2,3。
# 默认使用当前环境的 python3；清除旧服务器遗留的解释器路径。
unset ATTENTION_EP_PYTHON
# 固定 Toolkit，并为首次切换 CUDA 版本使用新的构建目录。
export ATTENTION_EP_CUDA_HOME=/usr/local/cuda-12.6
export ATTENTION_EP_BUILD=build-process-tp-cu126-v100
bash test/distributed/run_qwen_moe_layer_v100.sh
```

脚本默认从当前 `PATH` 选择 `python3`，兼容已激活的 Conda/venv 环境，不要求
存在 `/home/venvs/fastllm-dp`。需要固定解释器时，在运行前设置
`export ATTENTION_EP_PYTHON=/实际环境/bin/python3`（替换为服务器上的有效路径）。
启动时打印实际解释器并检查 Python 3.10+；显式指定无效路径时直接报错，不自动换环境。

脚本默认选择 `/usr/local/cuda-12.6`，不继承通用 `CUDA_HOME`；其他 Toolkit 安装位置
通过 `ATTENTION_EP_CUDA_HOME` 指定。启动时打印实际 nvcc 路径和版本，并查询
`--list-gpu-arch` 确认包含 `compute_70`。CUDA 13 已移除 Volta 离线编译支持，
本验收使用支持 V100 的 CUDA 12.x；缺少 Toolkit 时需先安装，脚本不会自动安装或修改系统软链接。
参考 [NVIDIA CUDA 13 发行说明](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html#deprecated-architectures)。

已有构建缓存若记录了其他 nvcc 路径，脚本会在编译前拒绝复用，并提示设置
`ATTENTION_EP_BUILD` 为新目录；旧目录和产物保留。目录名字包含 `cu126` 并不能证明
其中缓存的编译器是 CUDA 12.6，不能沿用曾用 CUDA 13 配置的目录。

脚本先检查 Python/CUDA 环境和构建缓存，再运行配置/协议及数学参考的 Python 检查、
四卡实际分配预览，之后编译 worker 和检查器。随后回归稠密 Qwen3 的 CPU 权重读取、
GPU 上传、单层计算，最后执行 MoE
1/2/4 rank 的完整矩阵。`ATTENTION_EP_BUILD` 可指定构建目录，`ATTENTION_EP_JOBS`
可指定构建并发数；不修改用户的 GPU 顺序。

已有正确构建产物时可以单独运行 MoE 检查，不编译：

```bash
"${ATTENTION_EP_PYTHON:-python3}" test/distributed/run_qwen_moe_layer.py \
  --check "${ATTENTION_EP_BUILD:-build-process-tp-cu126}/fastllm_moe_layer_check" \
  --worker "${ATTENTION_EP_BUILD:-build-process-tp-cu126}/fastllm_tp_worker" \
  --devices 0,1,2,3 --build-cache "${ATTENTION_EP_BUILD:-build-process-tp-cu126}/CMakeCache.txt"
```

验收以独立 CPU 完整层数学参考为准，`atol=0.008, rtol=0.008`，不自动放宽。
每项保存实际 GPU 权重二进制、任务输入、原始 stdout/stderr、完整命令；数值成功比较后
保存 CPU 中间量。检查所有 rank 的张量清单与逐元素数值，确认没有加载外部专家，
同时核对 Router、Top-K、专家加权/共享输出、Attention、整层、因果性及重复执行。
覆盖源 dtype、bias、归一化开关、60 专家/Top-4、小于 rank 数的专家、热点、空闲 rank、
整组空闲、变长序列、2048/2049 Token、非法配置、加载配额、卸载后调用、重复加载，
以及成功卸载/重载和有限权重导致的 Router FP16 溢出。默认 33 类用例分别执行
1/2/4 rank，共 99 项。所有路径仍需服务器真实执行，Python 自测不代表 CUDA 已验证。

每次创建唯一报告目录，生成 Markdown、CSV、JSON、XLSX；展示浮点三位小数，JSON
保留原始精度，记录误差、失败元素、GPU/CUDA/NCCL、架构、Git、命令和原始样本。
本阶段是功能检查（warmup=0、iterations=1、repeats=1），不报告性能加速比。
任一 case 失败返回非零，已执行部分仍生成报告。服务器全部通过后才推进阶段 3。
