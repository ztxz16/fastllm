# DeepSeek V4.1 执行后端

`FASTLLM_EXECUTION_MODE` 是统一的执行模式选择变量，现由 DeepSeek V4.1、Qwen4 与 GLM 5.3 Flash 接入。
完整的共用组件、forward 拆分和模块边界说明见 [执行路径设计](execution_modes.md)。
其他模型保持原有行为。变量不改变设备配置，也不会自动搬迁权重。

| 值 | 选择规则 |
| --- | --- |
| `auto`（默认，也包括未设置或空值） | 按各层最终生效的主计算和路由专家设备自动选择 |
| `gpu` | 主计算和路由专家均配置为 CUDA / multicuda |
| `hybrid` | 主计算和路由专家的设备组合同时包含 GPU 与 CPU / NUMA / 磁盘专家后端 |
| `cpu` | 主计算和路由专家均使用主机后端（CPU / NUMA / disk），保留 CPU 参考路径 |

显式模式与设备配置不符时，初始化报错，避免静默进入另一条路径。支持分层
`--moe_device_layers`、按层设备映射以及 TP 不满足 head 对齐要求时的设备回退。
Engram/embedding 表的存储位置、小路由结果的 CPU 分桶和算子的 dtype fallback
不决定执行模式；`gpu` 表示 GPU 专家后端，并不保证每一个辅助算子都在 GPU 上。

## 启动

保留自动选择，通常无需调整原命令：

```bash
export FASTLLM_EXECUTION_MODE=auto
ftllm server MODEL --device cuda --moe_device numa
# 日志：FASTLLM_EXECUTION_MODE=auto -> hybrid (DeepSeek-V4.1)
```

显式选择纯 GPU 或混合模式：

```bash
FASTLLM_EXECUTION_MODE=gpu ftllm server MODEL --device cuda --moe_device cuda
FASTLLM_EXECUTION_MODE=hybrid ftllm server MODEL --device cuda --moe_device numa
```

多卡纯 GPU 使用现有的 `multicuda:0,1,...` 设备配置，选择规则相同。
`--device cuda --moe_device cuda --ngram_device disk` 仍选择 `gpu`。
`--device cuda --moe_device disk` 则选择 `hybrid`，由磁盘专家后端处理权重读取及
CPU/GPU 执行；这与仅将 Engram 表放在磁盘上的配置不同。
磁盘专家会临时装载和释放权重，目前自动使用 eager 执行，避免 CUDA Graph 的
持久工作区与临时权重生命周期不匹配；即使请求开启 Graph，也会明确记录回退。
仅 Engram 放 disk 不触发该回退，纯 GPU 和 NUMA 专家仍保留原有 Graph 支持。
混合模式的 GPU 专家缓存仍通过 `--moe_cuda_cache` 等参数配置；执行模式与缓存容量、
动态分流策略相互独立。设置 `hybrid` 本身不会开启专家缓存。

Python 启动器为该模型默认设置 `FASTLLM_EXECUTION_MODE=auto`。C++ 初始化在最终设备
回退之后解析并保存模式。不会把进程环境中的 `auto` 改写为推断值，因此
同一进程后续创建不同配置的模型仍能分别选择。修改环境变量不会切换已经加载的模型。

## 代码组织

- `include/utils/execution_mode.h`：与模型无关的模式解析、设备分类和一致性检查。
- `src/models/deepseekv41_gpu.cpp`：GPU 路由专家执行、TP 路由暂存、共享专家合并。
- `src/models/deepseekv41_hybrid.cpp`：CPU/NUMA/磁盘专家、共享专家重叠执行、GPU 专家缓存、
  动态分流、跨 CPU/GPU 数据搬运及输出回传。CPU 参考路径复用此后端的无缓存分支。
- `src/models/deepseekv41_execution.h/.cpp`：共享工作区、MoE 调用上下文和
  TP 感知的张量搬运工具。
- `src/models/deepseekv41.cpp`：公共权重、注意力、KV、路由数学、Graph 分段及采样。
  独立的 GPU / Hybrid forward 通过公共块接口调用这些实现。
- `include/utils/hybrid_moe.h` 与 `src/utils/hybrid_moe.cpp`：接入模型共用的混合 MoE
  步骤生命周期、后端调度与 prefill 预取入口。

两条后端共享底层 CUDA 算子与模型数学，不复制注意力、请求状态和 KV 实现。
DSpark 的主模型验证经过同一后端分派；其草稿模型代码仍在原有独立文件。
新增后端优化时，先在对应后端修改；公共数值或存储修改则放在公共实现。

## 回归检查

```bash
cmake --build build-fastllm --target execution_mode_test deepseekV41TpGraphRegression -j16
build-fastllm/execution_mode_test
python test/basic/test_deepseek_v41_tp_graph.py \
  --binary build-fastllm/deepseekV41TpGraphRegression --gpu-experts
python test/basic/test_deepseek_v41_tp_graph.py \
  --binary build-fastllm/deepseekV41TpGraphRegression
python test/basic/test_deepseek_v41_tp_graph.py \
  --binary build-fastllm/deepseekV41TpGraphRegression --expert-cache --fp8-dense
```

还可运行以下磁盘存储组合，分别验证模式选择、Graph 和 KV 回滚：

```bash
python test/basic/test_deepseek_v41_tp_graph.py \
  --binary build-fastllm/deepseekV41TpGraphRegression --gpu-experts --ngram-device disk
python test/basic/test_deepseek_v41_tp_graph.py \
  --binary build-fastllm/deepseekV41TpGraphRegression --disk-experts
```

TP 测试需要两张 CUDA 卡，覆盖 prefill/decode、Graph 捕获/回放、1–8 个 DSpark
候选的验证、特征与 KV 回滚。`--gpu-experts`、`--disk-experts` 与 `--expert-cache` 三者互斥。
