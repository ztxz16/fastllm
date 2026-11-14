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
