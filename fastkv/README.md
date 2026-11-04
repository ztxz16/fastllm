# Fast KV Manager

Fast KV Manager 是独立的 C++17 组件，提供有界的状态传输、缓冲区生命周期管理、逻辑分片规划和不可变前缀缓存。它不依赖 FastLLM、Python、Mooncake、UCX、ZeroMQ 或第三方 RPC 框架。

构建要求 CMake 3.18 或更新版本。基础构建只需要 C++ 标准库、线程和 POSIX socket；CUDA 内存与原生 RDMA 分别是可选功能，依赖 CUDA Runtime 和系统 libibverbs。

## 构建与验证

```bash
cmake -S fastkv -B build-fastkv -DFASTKV_TESTS=ON
cmake --build build-fastkv -j
ctest --test-dir build-fastkv --output-on-failure
```

启用 GPU 内存和 RDMA：

```bash
cmake -S fastkv -B build-fastkv \
  -DFASTKV_CUDA=ON -DFASTKV_RDMA=ON -DFASTKV_TESTS=ON
cmake --build build-fastkv -j
```

嵌入其他 CMake 项目时，可以使用 `add_subdirectory(fastkv)`，然后链接 `fastkv` target。公共头文件为 `fastkv/fastkv.h`。

诊断工具会校验每一个 payload 字节。其耗时包含分配、注册、往返以及 CPU 校验，不能直接当作网络带宽：

```bash
# 接收端
build-fastkv/fastkv_probe serve 0.0.0.0:19000 \
  --transport rdma --device 0 --rdma-devices mlx5_0,mlx5_1

# 发送端
build-fastkv/fastkv_probe call receiver:19000 \
  --transport rdma --device 0 --bytes 67108864 --count 10 \
  --rdma-devices mlx5_0,mlx5_1
```

## 接口与约束

| 接口 | 作用 |
|---|---|
| `Buffer` | Host/CUDA 内存、切片、只读视图，以及 allocation lease。地址复用不会复用 allocation ID。 |
| `Message` | 有界元数据和独立 buffer 列表，不含远端指针或本地分配器对象。 |
| `Service` | 同步 `Call`、异步 `Submit`、有界工作队列、超时、停止和统计。 |
| `Budget` | RAII 字节或槽位额度；最后一个 lease 释放后归还额度。 |
| `PlanCopies` | 同一逻辑表示下的多维分片交集，支持源副本去重，拒绝覆盖缺口和歧义。 |
| `ObjectStore` | 完整对象的不可变发布、LRU 淘汰、读取 lease；被读取的对象不会提前释放。 |
| `PrefixCache` | 命名空间隔离的最长前缀查找；hash 仅筛选候选，命中前比较完整 token 序列。 |

调用者负责在借用 GPU buffer 前完成生产者的写入，并提供真实的生命周期 owner。`Read`、`Write`、`CopyFrom` 是同步接口。网络接收使用私有 staging；传输完成且可见后才交付给调用者。向模型的提交和 GPU graph 的执行由模型拥有者负责。

RDMA 使用 RC QP 和多 NIC 分条传输。注册缓存挂在 allocation lease 上；QP/CQ 先销毁，随后释放在途 buffer 和 MR。协议先交换长度并预留接收额度，再传 payload，最后确认完成。请求失败不会将部分 buffer 发布为可用状态。

`rdma` 模式不可静默降级。`auto` 允许握手双方协商 TCP/RDMA，但不会在传输中途重放失败的操作。TCP 同时支持 Host 和 CUDA buffer，CUDA 当前经主存分块搬运。RDMA GPU 内存需要驱动提供可用的 peer-memory/GPUDirect 支持；它不是单纯安装 libibverbs 就能保证的能力。

`Service::Stop()` 应由拥有者线程调用；它中断网络等待并等待 handler 退出，不能从该 Service 自己的 handler 中调用。handler 中的长计算需要自己检查停止标志。元数据、buffer 数量、payload、在途字节、线程和队列都有上限。

## 扩展边界

模型的逻辑状态、目标分片、传输和缓存分别独立。新模型提供状态 adapter；新传输实现内部数据传输接口，不应修改模型的 KV 语义。FP8 scale、压缩 tail、循环窗口、线性注意力递归状态等都应当作为有名称、有版本的状态组件处理。不同表示的转换由显式 codec 完成，`PlanCopies` 不会把不同精度当成普通 memcpy。

当前网络是可信内网协议，没有 TLS 或身份鉴权。当前采用静态成员和请求失败语义，不包含分布式共识、故障后自动重算、NVMe 存储后端或跨机 MoE collective。对象存储是进程内组件，不是持久化分布式存储服务。未来 wide-EP 的计算通信与 KV 状态传输应分别接入。

FastLLM 的模型接入、部署与目前的模型限制见 [PD 部署文档](../docs/fastkv_pd.md)。
