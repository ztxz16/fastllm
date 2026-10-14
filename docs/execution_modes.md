# GPU 与混合推理执行路径

`FASTLLM_EXECUTION_MODE=auto|gpu|hybrid|cpu` 是统一选择器，支持
Qwen4（包括 `qwen4_exp` / Qwen3.8-Flash-Next）、DeepSeek V4.1 与
GLM 5.3 Flash（`glm5_next`）。
默认 `auto` 根据每层最终生效的主干和路由专家设备选择；显式模式与设备不符时报错。
环境变量不搬迁权重，也不改变已经创建的模型。

| 配置 | 执行路径 |
| --- | --- |
| 主干与路由专家均在 CUDA / multicuda | GPU |
| 主干和专家的有效配置同时包含 CUDA 与 CPU / NUMA / disk | Hybrid |
| 主干与专家均在 CPU / NUMA / disk | CPU，使用保留的主机专家实现 |

Engram、PLE、embedding 的存储位置与主干执行模式独立。仅将这些表放在 CPU
或磁盘上仍可选择 GPU forward。路由结果的 CPU 分桶也不改变模式。

## 分层方式

两条 forward 独立维护执行顺序：

- `src/models/qwen4_exp_gpu.cpp` 与 `qwen4_exp_hybrid.cpp`；
- `src/models/deepseekv41_gpu.cpp` 与 `deepseekv41_hybrid.cpp`；
- `src/models/glm5_next_gpu.cpp` 与 `glm5_next_hybrid.cpp`。

公共的 `ForwardTarget` / `ForwardSegments` / `ForwardLayers` 只派发到所选 forward，不再持有共同的
主循环。GPU forward 调用 GPU 专家计算；Hybrid forward 自己管理混合 MoE 步骤、
预取和专家执行。CPU 路径使用 Hybrid 文件中的主机后端分支，不启动 CUDA 缓存策略。

公共模型代码保留权重加载、数学语义、Attention、KV 和采样；Graph 策略与共享机制的文件边界见下文。
Qwen / DeepSeek 的 `ForwardWorkspace` 持有一次前向的状态，`ForwardState` 提供块接口。
GLM 保留栈上的逐层临时张量，通过 Attention、FFN 前后处理块共享数学实现。
Attention/GDN/QSA、Hyper-Connection、共享专家与现有 CUDA 融合算子继续共用，
不复制数值实现，不使用 `Forward<isHybrid>` 模板包装同一条主循环。

张量生命周期也是块接口的一部分：跨块张量归 workspace 所有，CUDA Graph 的
持久地址仍由原图工作区持有。Qwen 每层临时量保持原来的释放顺序，避免改变临时
显存池的复用行为。DeepSeek 的 TP 异步作用域覆盖整次前向，并在释放图锁前结束。
Qwen / DeepSeek 混合策略的启动仍位于 embedding 之前；准备输入是显式的公共块，避免拆文件时
改变缓存准入等待、设备选择与输入计算的先后顺序。

MTP / DSpark 的请求管理、接受/拒绝与回滚由公共逻辑维护；目标模型验证进入所选 forward。

## 复用混合 MoE

通用入口在 `include/utils/hybrid_moe.h` 与 `src/utils/hybrid_moe.cpp`：

1. 使用现有格式描述 `FastllmCudaMoeCacheLayer` 注册专家表，保留原权重表的身份与生命周期。
2. 在支持动态分流的完整 decode 步骤外建立 `HybridMoeDecodeStep`，调用 `Begin`。
   `Finish` / 析构负责结束策略与缓存准入，异常路径也会收尾；不能在每个专家或每层重新开始。
3. 路由完成后调用 `RunHybridMoe`，传入归一化输入、专家索引、分数、权重表和层号。
   可传入独立共享专家的回调，让后端安排 CPU 专家与 GPU 计算重叠。
4. 分块流水 prefill 可使用 `HybridMoePrefetch`。它检查层范围、行数和 NUMA 放置，
   支持的格式沿用原后端预取；其他情况保持原路径。
5. 多设备分层使用 `HybridMoeDecodeGroup`：在每张卡第一次到达适用 MoE 层时启动，
   每个设备每步只启动一次；不适用的层不占据条目。退出时按设备顺序收尾。
   TP 主机专家使用 `RunHybridMoeOnDevices`，保留设备组、共享专家回调和原格式适配。

动态分流的成本模型、热点统计、缓存准入/驱逐、异步专家上传和格式专用计算继续由
`src/devices/cuda/moe/fastllm-moe-cache.cu` 及其格式适配层统一维护。
Qwen4、DeepSeek V4.1 与 GLM 5.3 Flash 通过上述入口使用同一后端，无需把这些策略复制到每个模型。

格式相关数学不能混用：DeepSeek V4.1 的 block-32 激活量化、路由分数乘法位置、
BF16 舍入与 SwiGLU clamp 继续由其适配层负责。新模型必须先验证该格式契约。
不支持的格式或批量返回原专家后端；选择 Hybrid 不等于启用 GPU 专家缓存。

TP 副本暂存、跨设备输出归并以及 Qwen 的双卡流水/EP 协作仍保留模型适配代码。
它们涉及各自的张量布局，不能仅凭相同专家数复用。通用缓存和分流入口不承诺任意
布局、任意批量都能命中优化路径。

## GLM 5.3 Flash

`ForwardGpuLayers` 与 `ForwardHybridLayers` 分别拥有完整的逐层循环。KDA / DSA、
Hyper-Connection、路由计算及共享专家块共用；各自的专家调度在独立源文件中维护。
混合部署中某层专家全部位于 CUDA 时，该层复用 GPU 专家块，其他层仍保留主机调度。

`glm5_next_tp_state.h` 只声明共享的 TP 运行时类型；TP 权重切分、线程、事件交接与
KDA Graph 的实现仍由原运行时维护。每个 TP rank 在最终设备配置确定后解析执行模式。
稠密 FFN 层不使用路由专家，自动模式忽略这些层上未使用的专家设备配置。

GLM 的缓存策略在各设备首个适用 MoE 层启动，维持原有时序；draft MoE 也使用通用
步骤生命周期。共享专家与主机专家的重叠、TP 结果归并、MTP/KDA 回滚和前缀快照保留。

当前 GLM 专用 GPU 专家缓存注册支持原有 NVFP4 block-16 和符合条件的 GGUF 格式。
FP8 源权重仍使用其既有主机/GPU 后端；选择 Hybrid 或设置缓存预算不会自动将 FP8
转换为 NVFP4/GGUF，也不代表该缓存已启用。

## Qwen4 模块边界

| 职责 | 实现 |
| --- | --- |
| GPU / Hybrid 普通前向与专家执行 | `qwen4_exp_gpu.cpp` / `qwen4_exp_hybrid.cpp` |
| 每层及草稿的专家执行计划、路由与公共共享专家数学 | `qwen4_exp_moe.cpp` |
| GPU 图编排 | `qwen4_exp_gpu_graph.cpp` |
| Hybrid 图段与段外专家编排 | `qwen4_exp_hybrid_graph.cpp` |
| 图存储、捕获/回放/销毁、共享数学与状态发布 | `qwen4_exp_graph.cpp` / `qwen4_exp_graph.h` |
| 混合专家协作、绑核与执行生命周期 | `qwen4_exp_hybrid_parallel.cpp` / `qwen4_exp_hybrid.h` |
| 分层 prefill 流水 | `qwen4_exp_pp.cpp` |
| TP 线程、通信与共享请求状态 | `qwen4_exp_tp.cpp` / `qwen4_exp_tp.h` |

全模型模式负责配置检查和 forward 选择；`MoEExecutionPlan` 另按实际设备为目标各层和草稿选择专家后端。
主模型专家在 NUMA、草稿专家在 GPU 时，两者分别调用对应后端。该计划记录后端、层号、前缀和原权重/偏置表；格式、形状和 Graph 能力仍由运行时检查，不是“权重在 GPU 即可进图”的承诺。

`HybridExecutionScope` 与 `HybridPrefillScope` 管理混合策略、CPU 亲和性及步骤收尾；公共 workspace 管理共享张量。
MTP 的接受、拒绝、前缀提交和 KV 回滚只有一份语义实现。共享专家、远端专家合并及 buffer 有效期通过专家接口和图边界协调。

维护时必须保持图地址有效、张量释放顺序、TP 集合通信顺序和 MTP 回滚语义。模型回归需覆盖固定输入 logits、CPU fallback、Graph 捕获/重放、TP、前缀恢复和投机回滚；精度测试与稳态测速分开。
