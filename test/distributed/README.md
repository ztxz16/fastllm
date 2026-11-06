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
