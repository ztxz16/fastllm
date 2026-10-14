# GPU 与混合推理执行路径

`FASTLLM_EXECUTION_MODE=auto|gpu|hybrid|cpu` 提供统一的模式选择接口。
模型逐个接入；未接入的模型保持原行为。解析依据是各层最终主干与路由专家设备，
不根据 Engram、PLE、embedding 的存储位置选择，也不搬迁权重。
显式模式与放置冲突时初始化报错。

`include/utils/hybrid_moe.h` 和 `src/utils/hybrid_moe.cpp` 提供混合 MoE 的
步骤生命周期、按设备惰性启动的策略组、共享专家回调及 NUMA prefill 预取入口。
它们复用现有格式后端；不支持的格式返回原有执行路径。
