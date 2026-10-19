# 文档索引

- [快速开始](../README.md)
- [模型部署指南](../README.md#模型部署指南)
- [Benchmark](benchmark.md)：按模型保存最终结果及测量范围
- [开发与验证记录](development-validation.md)：文档分工、测试入口和 Benchmark 记录要求
- [Agent 部署速查](model-deployment-agent.md)：建议交给自动化 agent 先阅读
- [执行模式与模块边界](execution_modes.md)
- [混合推理](mixforward.md)与 [CUDA 专家缓存](cuda-expert-cache.md)
- [Benchmark 工具](../test/benchmark/README.md)
- [上下文扩展](context-length-extension.md)与 [KVMem](kvmem.md)
- [版本日志](version.md)

## 内容约定

部署文档说明当前参数、支持范围、回退行为和验证入口；性能结果集中在对应模型的 Benchmark 页面。
结果保留硬件、权重格式、被测版本（有记录时）、输入输出长度、并发、缓存/MTP 设置、计时口径和限制。没有源码版本的旧测量要明确注明，不能表述为当前 HEAD 的性能。

同一配置的结果更新时替换正文中的结论，不追加逐轮优化日记。未采用方案、临时调试命令、原始日志、profiler 波形、完整生成输出和跨框架性能对比不随文档提交；这些材料放在仓库外的实验归档中。
`test/` 保留可执行回归、必要夹具和运行说明。独立数值参考、第三方许可及来源说明保留，正式版本变更使用版本日志维护。
