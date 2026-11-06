# 多进程 TP 底座验收

当前只提供 Linux 单机进程底座：独立 C++ worker、Unix socket 控制协议、
NCCL 初始化和标量 SUM 自检、心跳、超时、退出与故障回收。
**尚未接入模型权重、推理调度、CUDA Graph、自定义 AllReduce 或前缀缓存。**
`--tp-backend process` / `--tp_backend process` 已加入公共参数入口，但会在模型加载前
明确报错；默认 `thread` 保持原行为。禁止把底座检查成功当成模型 TP 已实现。
构建选项 `BUILD_PROCESS_TP` 默认关闭，显式开启后才构建和打包这两个实验程序。

## 运行条件

- Linux 5.3+，内核与容器策略允许 `pidfd_open`、`pidfd_send_signal`。
- Python 3.9+。部分 Conda Python 没有暴露 pidfd 接口，验收器在 Linux x86_64/aarch64
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

这一阶段只提供权重存储，尚未调用正式前向算子；Norm 权重也按 FP16 保存，
前向接入时仍须验证其数值精度和算子布局。量化、设备迁移、CUDA Graph、
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
