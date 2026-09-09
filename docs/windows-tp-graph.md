# Windows 双卡 CUDA Graph

Windows 在 `USE_CUDA=ON`、`USE_NCCL=OFF` 时，可以自动使用映射主机内存完成
双卡 CUDA Graph 中的小消息集合通信。该后端用于减少逐 token 解码时的 CPU
同步和提交开销，不要求 CUDA P2P。大消息 prefill 使用下述独立的 eager 优化。

## 使用方式和边界

以支持 TP 和 Graph 的 Qwen 模型为例：

```powershell
$env:FASTLLM_CUDA_GRAPH = "1"
ftllm run D:\models\Qwen3.8-27B-FP8 --tp 2 --dtype auto
```

只沿用已有的 `FASTLLM_CUDA_GRAPH` 开关，通信实现自动选择。设为 `0` 可关闭
Graph。`mapped-host TP2 CUDA Graph collectives: self-test passed` 表示该 GPU
组已通过通信自检，不等同于模型已捕获 Graph；模型的图模式仍受形状等条件限制。

- 仅 Windows NVIDIA CUDA、两个不同 GPU，设备需支持映射主机内存且计算能力
  至少为 7.0；实际拓扑还必须通过启动时的双图重复回放自检。
- 每次集合通信每卡最多 64 KiB，支持 FP16、BF16、FP32、INT8、INT32 的 SUM
  all-reduce / reduce，以及 broadcast。优先保留已有 P2P 自定义 all-reduce。
- eager 执行使用同步主机中转，其中上述五种类型的大块双卡 SUM 可以在 GPU 上计算。
  自检失败、更多 GPU、超限或不支持的数据
  类型不能使用此 Graph 后端；模型需要处理捕获失败并回退，或由调用方关闭 Graph。
- 使用同一通信组的两卡必须按相同顺序提交匹配的通信操作；不支持多个独立执行
  流并发复用该组。等待对端超过 1 秒或通信参数不匹配会终止当前 CUDA 执行，
  避免静默返回错误结果；这类运行时失败后需要重启进程。
- 不改变 Linux、NCCL 或 ROCm 的通信后端，也不承诺不同拓扑的性能收益。

## 实现和资源生命周期

私有实现位于 `src/devices/multicuda/fastllm-host-mapped-collective.cuh`，从无 NCCL
实现中包含，不增加公共接口。每卡一个 CTA 将小消息写入映射主机内存，通过系统
作用域 release/acquire 交换就绪和完成标记；每个标记只有一个 GPU 写入，不依赖
跨 GPU 原子 read-modify-write。

GPU 上的序号随实际执行递增，交替使用两组标记，完成握手后才允许重用消息区。
这使捕获、重复回放和重新捕获使用同一套同步逻辑。启动自检连续原地 SUM 32 次，
检查每次回放产生的新结果确实参与下一次计算。自检失败不会发布该通信状态。

每个有序物理 GPU 对缓存一个进程生命周期的通信状态，约 128 KiB 锁页消息区，
外加标记和设备序号。切换通信组后仍保留已捕获图引用的地址；同一 GPU 对重新
初始化或重新捕获不会再分配一个通信区。

无 NCCL 的同步路径还修正了通信组选择：优先使用线程 TP 已初始化的组，避免
旧 MultiCuda 设备列表为空或只含单卡时，将跨卡 SUM 错误执行为本地复制。

## 大消息 prefill

Windows 无 NCCL 的双卡 eager all-reduce / reduce，支持 FP16、BF16、FP32、
INT8、INT32；消息每卡至少 64 KiB 且支持映射主机内存时，自动复用锁页主机
缓冲区。阈值按字节计算，与元素类型无关。每卡完成输入 D2H 后，
两个 CPU rank 线程会合；GPU 读取本卡原输入和对端的映射缓冲区并计算 SUM。
两卡完成读取后再次会合，才允许下一次通信复用缓冲区。该路径仍是同步通信，
无需 P2P 或额外环境变量，也不受小消息 Graph 后端 64 KiB 上限的约束。

求和保持原 CPU fallback 的 rank 顺序：FP16 沿用软件转换规则，BF16 使用
FP32 累加并按最近偶数舍入，FP32 保持逐步加法舍入，整数使用 INT64 累加后
收窄到原类型。FP32/BF16 的 NaN 仍为 NaN，但不保证保留 CPU 的 NaN 载荷和
符号位。锁页分配失败时，两卡共同回到原 CPU 求和路径；broadcast、更多 GPU、Linux 和 ROCm
仍沿用原路径。每个 rank 线程保留一个按最大已见消息增长的锁页缓冲区，并在线程
退出时释放。它不被 Graph 引用，与上述进程生命周期的小消息区相互独立。

## 回归测试

配置 `UNIT_TEST=ON`、`USE_CUDA=ON`、`USE_NCCL=OFF`，在 Windows 构建并运行：

```powershell
cmake --build build --config Release --target hostCollectiveRegression hostMappedCollectiveRegression
ctest --test-dir build -C Release -R "^cuda_host_.*collective$" --output-on-failure
```

无两卡环境返回跳过码 77。两个测试共享 CTest GPU 资源锁，避免并行运行时相互
争用显卡。`hostCollectiveRegression` 也在 Linux 无 NCCL 构建注册。

- 同步通信回归：空/过期旧设备列表、两种设备顺序、五种数据类型、四种消息
  大小、两种根节点和原地/非原地通信；额外检查五种类型的原始位模式、浮点
  特殊值、整数溢出、各类型的 64 KiB 前后边界、10 MiB 消息，以及同一线程中
  缓冲区扩容和大小交替。除 FP32/BF16 的 NaN 载荷外，逐位对照原 CPU 求和结果。
- Graph 回归：五种类型、四种消息大小（含 64 KiB 边界）、两种根节点、原地/
  非原地通信、每图 32 次不同输入回放、重新捕获、交换设备顺序并切回原组，
  共 61,440 次通信/卡；同时检查超限捕获被拒绝。FP16/BF16 包含舍入输入。

性能数据依赖模型、输入长度和拓扑，应分别报告 prefill 与稳定解码；只比较解码
吞吐不能代表完整请求延迟。
