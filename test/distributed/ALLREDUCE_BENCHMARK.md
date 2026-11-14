# AllReduce 算子对比

仅测 FP32 AllReduce，不加载模型。三方为 FT 正式 ProcessAllReduce、vLLM 正式
CustomAllreduce 和原生 NCCL。先建立当前实现基线，不修改通信 kernel。

七个阶段依次为 plan、preflight、smoke、correctness、performance、concurrent、report。
每阶段使用独立目录；已有目录不会被覆盖。plan 的状态为 PLANNED，不表示 GPU 验收通过。

第一步（无需 GPU、不编译）：

```bash
cd /home/project/vllm_fastllm
export AR_RUN="$PWD/test/distributed/results/$(date -u +%Y%m%dT%H%M%SZ)_allreduce"
python3 test/distributed/benchmark_allreduce.py plan --output "$AR_RUN/01-plan"
```

共同范围是 TP2、连续 FP32、每卡 4..65536 个元素且为 4 的倍数。超出共同范围的
case 单独记录实际路径，不能把 FT 的 NCCL 回退计为 FT 自定义 kernel。

计时保留正式算子所需复制和同步，排除输入恢复、初始化、校验、日志及进程控制。
Graph 分为双方都保留 staging copy 的共同配置，以及各自支持的注册输入配置。
所有性能 case 先验证输出和路径，再测至少五轮；加速比定义为对比后端耗时/FT耗时。

第二步（服务器现有 CUDA/vLLM 环境，无安装或编译）：

```bash
export AR_PYTHON="$(command -v python3)" # 若使用 venv，改成该环境 Python 的绝对路径
python3 test/distributed/benchmark_allreduce.py preflight --python "$AR_PYTHON" \
  --devices 0,1,2,3 --output "$AR_RUN/02-preflight"
```

预检执行已有 PyTorch CUDA kernel、加载已有 vLLM 自定义扩展，并记录每张卡的
SM/UVA、每对卡的双向 P2P/native atomics、已加载库路径和版本、vLLM 源码指纹。
候选卡对不等于已通过跨卡算子测试；第三步才验证实际执行。V100 为 SM70，不能
默认任意新版 PyTorch/vLLM 预编译包都能运行；预检失败时先处理兼容环境。

第三步（只在 V100 服务器编译，随后三方各执行两项实际求和）：

```bash
AR_CUDA=/usr/local/cuda-12.6 bash test/distributed/build_allreduce_adapter.sh
python3 test/distributed/benchmark_allreduce.py smoke --python "$AR_PYTHON" \
  --pair 0,1 --output "$AR_RUN/03-smoke"
```

选择预检中的合格卡对；GPU 编号为 CUDA_VISIBLE_DEVICES 下的逻辑编号，两进程应
同时可见两张卡。NCCL 前缀可用 AR_NCCL_ROOT 指定。构建只包含薄 C ABI 和正式
process_allreduce.cu，不构建模型或复制 CUDA kernel。vLLM 使用已有 Python/扩展
正式入口；需要本地对应版本支持 register_graph_buffers 参数。不修改已安装 vLLM。
所有 worker 运行于独立进程组，超时或退出后有界终止和回收整组。

第四步（数值验证，三种模式分别建立通信组）：

```bash
python3 test/distributed/benchmark_allreduce.py correctness --python "$AR_PYTHON" \
  --pair 0,1 --timeout 900 --output "$AR_RUN/04-correctness"
```

输入由同一份纯 CPU 代码按固定种子生成，两后端收到相同 FP32 字节。
普通输入先用 CPU FP64 求和，再舍入为 FP32，逐元素要求相等；报告中的绝对误差
仍相对 FP64 值计算，因此 PASS 时可能有正常的 FP32 舍入误差。NaN/Inf 单独按类别
比较，不进入有限值误差均值。每张卡均检查实际输出，记录输入指纹和首批失败元素。
Graph capture/warmup 位于计数检查区间外；重放更新 GPU 同步序号，不重建通信组。

第五步（先验证报告和产物指纹，再测速）：

```bash
python3 test/distributed/benchmark_allreduce.py performance --python "$AR_PYTHON" \
  --pair 0,1 --correctness "$AR_RUN/04-correctness" --warmup 100 --iterations 1000 \
  --repeats 5 --timeout 900 --output "$AR_RUN/05-performance"
```

每个样本在同一 stream 依次提交「恢复输入、start event、完整算子、end event」。
输入恢复不在事件区间内；所有实现均在批次结束后同步并读取事件，不能逐次同步一方。
保留双方生产 staging copy 和内核同步。Eager 事件区间可能包含主机提交间隙，因此
这是完整算子路径的事件延迟，不是 profiler 中单个 kernel 的硬件持续时间。
校验计时区最后一次调用的真实输出，而不是另跑一次后将结果当作计时区正确性证据。
每次通信取两 rank 时间较大值；逐次原始时间保存在 JSON 中。calls_per_second 是
由算子延迟折算的调用率，不是包含输入准备的应用墙钟吞吐。
