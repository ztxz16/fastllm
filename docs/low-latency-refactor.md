# 低并发算子重构

第一阶段建立不依赖模型和请求调度的 CUDA 算子，随后迁移 B300 专用实现。
新分支从上游 master 开始，按功能重建提交，不合并整个历史优化分支。

## 分层与运行约束

- `include/devices/cuda/low_latency/` 暴露原始张量、容量、stride、stream 和准备好的执行计划；不依赖 `Data`、模型类或 TP/EP 状态。
- `src/devices/cuda/low_latency/` 只包含 CUDA 算子。计算行数与请求数分离；物理 tile 不限制逻辑 batch。
- `fastllm-cuda-norm-linear.h` 是 FastLLM 的薄适配层，校验 `Data` 类型、形状和所有权，不分配输出或复制权重。
- 能力查询不启动 kernel；资源准备发生在捕图前。执行时不编译、不调优、不捕图、不分配。输入地址变化可以重新绑定已有计划。
- 不支持的配置由调用方走原有算子组合。运行错误不能当作“不支持”吞掉，尤其不能在已启动 kernel 后重跑回退路径。
- 后续多候选选择接入现有 `kernel_registry`；这里的原始执行计划不建立第二套调优系统。尚未接入模型自动分派，也未替换现有 `RMSNormSmallLinear`。

## 首个算子：BF16 残差、RMSNorm 与投影

`PrepareNormLinear`、`CanRunNormLinear` 和 `RunNormLinear` 的头文件只依赖 CUDA runtime 与标准库。
权重保持原始行主序 BF16，gamma 为 FP32，输出为保留 BF16 舍入的 FP32。
可选 residual 只读，可选 mirror 写调用方提供的词表分片。无 GPU workspace，无权重或 KV 副本。

舍入契约为：

```
z = bf16(input + optional residual)
h = bf16(bf16(z * rsqrt(mean(z*z) + epsilon)) * gamma)
output = float(bf16(h @ weight.T))
```

归约和点积使用 FP32。这与只有一次归一化舍入的 `RMSNormSmallLinear` 语义不同，不能直接互换。
当前融合候选支持 BF16、K 为 256 的倍数且处于 256–8192，输出列数允许尾块；
行 tile 为 1–8，超过 8 行通过 grid.y 覆盖同一逻辑批次。
实际可用性还检查设备 SM、加载的 kernel image、线程数、共享内存及缓冲区容量。
该候选支持较大逻辑 batch 不代表它会比 GEMM 更快；模型自动选择要保留 GEMM 并实测交叉点。

计划不保存张量地址，允许同一计划在不同 stream 上使用独立 bindings。输入、输出不能重叠。
共享内存 opt-in 使用固定的 per-image 上限，避免准备较小形状破坏已有图。
mirror 的 peer access、生命周期和多生产者完成事件由调用方负责；kernel 负责发布自己的写入。

## 验证与最终验收

独立 CUDA 测试使用逐行 CPU 高精度参考，显式实现舍入边界，检查 L2 < 0.001。
覆盖尾块、padding、容量不足、别名、错误参数、图回放、重新绑定及并发；非法调用不写输出。
跨 SM 编译检查只证明能编译，不能替代相应硬件上的正确性或性能验证。

最终模型验收固定当前优化版、上游版、新版三套独立构建和二进制哈希。
目标为 TP8/EP1、TP8/EP8，DS7 无阈值，batch 1/2/4/8，并补无推测解码对照。
使用相同模型、精度、上下文、采样、KV 预算和图配置，多次交替运行。
**每个低 batch 场景的稳态轮时相对当前优化版回退不得超过 5%。**
同时记录验证、草稿、阶段外耗时、吞吐与波动；请求变化不能触发重新捕图。
只有算子测试通过时，不能宣称整个重构已达到这个端到端验收标准。

后续依次重构 router/top-k、dense SwiGLU/down、采样与交接、归约与归一化，
再迁移 SWA/DSA、MoE 与草稿整层的 B300 优化。每组实现、适配、回退及测试形成完整提交。

## 通用 Router 与 dense FFN 计算块

`router.cuh` 提供 BF16 输入、FP32 权重/输出的 sigmoid 投影；隐藏维度模板与模型无关。
单行/多行共享权重候选支持 K 为 256 的倍数，tiled 候选当前只支持 K=4096。
`dense.cuh` 提供可选 RMSNorm + gate/up + SwiGLU 和可选 residual 的 down 投影。
计算使用 SM80 起的 BF16 MMA，逻辑行数由 grid.y 切成 8 行 tile，支持尾行。
输入与权重为行主序且 16 字节对齐，输出由调用方预分配，不能与输入重叠。
TP 必须先归约局部 down，再加残差；公共计算块不拥有 epoch/mailbox。

这些模板是 device 计算接口。调用方在预热阶段检查 image、共享内存和线程配置；
不满足条件时保留原有实现。它们不新增运行时环境变量或 GPU 全局工作区。
独立测试覆盖不同隐藏维度、17/256 个 expert、64 行 Router、FFN 跨 tile 与尾行。
B300 CPU 参考的最大逐行 L2：gate/up 8.45e-7，down 4.84e-4；memcheck 0 errors。
已完成 SM80/90/100/103/120 编译检查；其他 GPU 的速度尚未实测。
