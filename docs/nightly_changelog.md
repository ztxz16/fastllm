# Nightly 更新日志

本文档记录 `fastllm-nightly` 每次更新的详细变更内容。

---

## 0.0.0.3 (2026.3.7)

### 新模型支持

- 支持 Minimax-M2 的 GGUF 格式
- 初步支持 Qwen3.5 

### 新功能

- 初步支持 Anthropic API 兼容接口，包含工具调用（tool call）支持

### CUDA / 计算优化

- 预编译架构增加 SM80，支持A系列显卡，RTX 30系列显卡
- 修复一些显存泄漏问题

### Bug 修复

- 修复纯CPU计算的一些错误

---

## 0.0.0.2

- 初始的nightly版本
