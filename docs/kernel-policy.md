# 算子注册与预热选择（第一版）

`--kernel-tune quick` 在实际算子形状首次预热时，对已注册且满足硬件、格式和布局约束的候选做数值校验与测速，然后固定该形状的执行方案。默认 `off` 使用各后端的默认分派，不参与测速。

当前接入 NVFP4 模型中的三类 GPU 热点：普通 FP8/BF16 投影、分组 FP8/BF16 投影，以及 NVFP4 的 gate/SwiGLU/down 专家融合计算。后者按整个计算块比较，允许一个候选包含多个 kernel。框架不依赖 CUDA；CPU 可以使用同一注册、校验、计时回调及选择接口。EXL3 普通及分组投影也已接入 `linear.exl3`；共享输入投影复用逐投影选择，全部选择压缩路径时仍合并启动。生产 CPU、GGUF 算子的迁移尚未包含在这一版中，其现有路径保持可用。

## 使用

```sh
ftllm ops
ftllm benchmark /path/to/model --auto --moe_device numa --max_batch 4 \
  --dspark 5 --speculative_dspark_confidence_threshold 0 \
  --batch 2 --warmup 3 --input_tokens 4096 --output_tokens 512 \
  --temperature 0.8 --top_p 0.95 --top_k 50 \
  --kernel-tune quick --kernel-profile /path/to/kernels.json
```

`ftllm ops` 输出当前构建中注册的候选、架构/行数容量、布局和参数。也可以使用 `python -m ftllm.kernel_policy`。

- `--kernel-tune off`：后端默认行为，不参与新选择器。
- `--kernel-tune quick`：优先复用匹配的缓存，未命中时在预热窗口内校验并测速。
- `--kernel-tune cache`：只复用匹配且重新通过数值校验的记录；缺失时使用后端的合法默认候选，不重新测速。
- `--kernel-profile FILE`：读取历史记录并原子保存本次报告。缓存必须匹配构建、设备型号/SM/SM 数量/显存、驱动与运行时、候选版本、形状、布局和精度要求。显式配置的 CUDA 算子环境选项及编译脚本也进入构建身份。
- `--kernel-policy FILE`：覆盖预算、精度和候选集合，例如：

```json
{
  "budget_ms": 30000,
  "repeats": 8,
  "samples": 3,
  "max_l2": 0.0032,
  "deny": ["cuda.projection.triton"],
  "force": {"moe.nvfp4.gate_swiglu_down": "cuda.moe.nvfp4.native"}
}
```

`force` 仍必须满足能力、准备和数值校验；不满足时明确报错。不能禁用参考/回退实现。旧的显式 Triton 禁用选项仍有效。启用此策略不要求设置全局 `FASTLLM_CUDA_TRITON=1`，也不会随之开启尚未注册的 Triton 算子。

## 预热、图与报告

计时前先完成编译、模块加载和数值检查。GPU 候选各自捕获一个临时 CUDA Graph，以相同回放方式计时，交替候选顺序并取多个样本的中位数；有噪声重叠时保留较早的实现。L2 是相对于原实现的归一化 L2，还会拒绝非有限输出。独立算子测试继续使用更高精度参考或既有 CUDA oracle。

每个签名第一次选定后保持不变。模型启动可能只覆盖单行，新请求形状可以在对应图的首次预热阶段补测；CUDA Graph 捕获和回放期间不会编译或测速。`ftllm benchmark` 用实际 `--batch` 预热，正式计时前全局冻结选择器。自定义驱动在完成代表性请求后调用 `model.kernel_policy.freeze()`；`save()` 可以随时保存报告。冻结后遇到新形状只走回退，报告说明原因。服务模式采用有总预算的首次形状预热，用户应预热预期并发形状后再接流量。

预算限制是否开始下一组测量；正在进行的一组校验/编译/测量允许完成，因此它不是编译服务的硬超时。编译时间不进入候选 kernel 的耗时，但计入本次准备预算。缓存复用也要准备模块并重新校验，不能将任意缓存 ID 直接放进图。

日志给出 family、实际行数、维度、分组/top-k、选中实现和来源。JSON 保存每个候选的排除原因、L2、各次计时、中位数、波动范围和选择结果。来源包括 `measured`、`forced`、`cache`、`only-supported`、`fallback`。第一版保存精确形状表，不从少量点插值出未经验证的性能阈值；相同行数、不同 N/K 的选择也可能不同。

## 注册新算子

核心接口位于 `include/utils/kernel_registry.h`，CUDA 计时/校验适配位于 `include/devices/cuda/fastllm-kernel-benchmark.cuh`。

1. 定义一个语义 family 和稳定的候选 ID，描述架构、格式、布局、图能力及参数；第零个候选是参考及默认回退实现。
2. 构造签名：必须包括所有影响合法性或执行方案的实际形状与布局字段，不能把服务的 `max_batch` 当成行数。
3. 提供无副作用的能力查询、执行、数值校验、计时及硬件身份回调。CPU 适配器需要把 ISA、线程数、NUMA/亲和性、布局等加入身份或签名，并等待线程池任务完成后计时。
4. 注册一次，在原分派入口调用选择器。参考执行使用 `Bypass` 避免递归分派，保留原回退。支持检查不能分配、编译或启动 kernel。
5. 增加独立数值测试、边界/不规则形状、缓存与图回放检查，再进行完整模型验证。

后端可以通过 `Call::fallback` 指定经过独立数值验证、无需额外准备的未测形状默认候选；选择器仍检查实际能力与 `deny` 策略，默认值为第零个实现。显式 `force` 必须经过校验与测量，或缓存重验证。

当前签名的前四项为 family ID、device、runtime arch 和实际行数。FP8 投影随后为输出宽度、输入宽度、groups、量化 block 参数；专家随后为 hidden、intermediate、top-k、有效 route 容量、planar 布局、权重类型和激活上限。内部保留固定长度签名是为了低成本查找，不是模型级规则。

## FP8/BF16 批量投影

普通投影和分组 WoA 保留原 `cuda.projection.triton` 小批候选，新增
`cuda.projection.triton.batch32` 和 `cuda.projection.triton.batch64`。
32/64 表示每个 CTA 处理的物理行 tile，内核遍历全部 tile，末尾不足部分使用 mask；
它们不是逻辑批次的最大行数。张量索引、CUDA 启动及显存容量约束仍适用。
这些 Triton 候选支持 SM120/121，其余架构继续使用原 CUDA 实现。

验证的 compact WoA 和草稿的普通 WoA 都参与同一 family 的预热选择，
签名区分两种输入语义和权重存储类型。普通 WoA 先完成原有激活量化，再比较投影候选，
不会因为切换内核跳过 FP8 舍入。不同 tile 与原实现按实际形状测速，未新增行数调优开关。
可以用 policy 的 `deny` 分别排除候选，或用 `force` 在通过能力、数值校验后指定候选；
冻结前应预热实际验证和草稿批次，正式计时查看报告中的 `selected` 与 `source`。

## EXL3 投影

EXL3 后端仅通过统一算子策略配置，导入内核不读取第三方工程的调优环境开关。需要比较实现时使用下面的 policy 候选；未编入导入内核的构建仍走独立 portable 路径。原有数值边界及硬件能力保护保留。

`linear.exl3` 有两个候选，保持已有 Maya/ExLlama 来源与许可说明：

- `cuda.exl3.tiles`：直接读取压缩权重，按后端物理计算 tile 分块；普通、共享输入和分组投影都不再用 32 行限制逻辑批次。其物理 tile 大小不会限制模型的并发或草稿行数。
- `cuda.exl3.expand_gemm`：调用已有 portable 路径，小批量使用直接乘积，大批量使用有界临时空间中的权重展开与 GEMM；作为 prefill 的测速候选保留。

两者支持 FP32、FP16、BF16 激活和输出。签名区分普通/分组布局、实际行数、K/N、分组/权重分片、EXL3 码率、输入/输出类型和 bias。数值校验恢复原输出；捕获期间只读取已选方案。共享输入目前依据各投影的独立测速选择，尚未对整组合并的收益单独建模。

建议使用 `--kernel-tune quick` 同时预热真实 prefill 和验证批次。未测速形状按实际内核的 tile 宽度和压缩码率估算权重流量，比较压缩权重重复读取与一次展开、写入/读取 FP16 矩阵的成本，选择合法的默认候选；这是回退估计，不是测速结论。已测形状仍以实测结果为准，报告同时记录 `default_candidate` 和 `selected`。禁用候选仍生效，force 不会跳过测量或校验。需要对照可通过 policy 的 `force` 指定上述候选，仍须在冻结前通过能力和数值校验。

## 本版边界

- Triton 的 SM120/121 和行数容量是现有实现的硬约束，不是测速推断的交叉点。较旧架构继续使用原 CUDA 路径。
- 原 `cuda.moe.nvfp4.triton` 的 2–8 行入口、8 路计算 tile 和缓存版本保持。新增 `cuda.moe.nvfp4.triton.batch` 使用独立 16 路 tile，支持 planar 布局及 `rows * topk <= 65535`（CUDA grid.y 容量）；它与原 kernel 一起参与预热选择。可在 `--kernel-policy` 的 `deny` 中禁用新候选进行对照，无需修改原路径。
- 原 CUDA 实现目前作为一个候选，内部现有 MMA/SIMT 参数选择尚未全部拆成独立候选；tile/warp 配置的搜索也未加入。
- NVFP4 为候选比较采用共享 planar 专家缓存布局；未联合搜索布局转换、PCIe 搬运、专家缓存容量和 CPU/GPU 动态分流策略。新接口没有替换这些调度器。
- 建图前专家预热使用真实驻留权重和私有输入/输出，构造重复与不重复专家混合的代表性路由。尚未按路由分布分别建模；实际负载仍应以端到端波形校验。
- 数值试跑保存并恢复共享输出，避免破坏部分专家调用已经产生的结果。候选执行拒绝时仍可退回原实现。
- 策略当前按进程配置；运行多个不同策略的模型应使用独立进程。不要在已有图运行时通过底层 C API 重新配置。
- 这里组织现有 FastLLM CUDA/Triton 算子，不引入外部算子源码；既有算子来源说明保持不变。
