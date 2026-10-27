# 开发与验证记录

这页整理可长期维护的开发结论和验证入口，方便维护者、部署人员和自动化 agent 快速判断一项改动是否已经有回归覆盖。它不是逐轮实验日志：临时命令、原始日志、profiler 波形和未采用的方案放在仓库外的实验归档中，最终性能数字写入对应的 Benchmark 页面。

## 文档分工

| 需要查什么 | 入口 | 内容边界 |
| --- | --- | --- |
| 自动部署和模型选择 | [Agent 部署速查](model-deployment-agent.md) | `--auto` 默认流程、资源限制、手动覆盖和验收步骤；建议先交给 agent 阅读 |
| 自动规划器的规则 | [混合推理自动规划](hybrid-auto.md) | 设备选择、MoE/PLE/Engram 放置、缓存预算和回退条件 |
| 单个模型的参数与限制 | 各模型部署指南 | 当前支持范围、手动命令、格式限制和已知风险 |
| 最终性能结论 | [Benchmark 索引](benchmark.md) 及 `docs/benchmarks/` | 硬件、权重格式、源码版本、输入输出长度、缓存设置、计时口径和限制 |
| 可执行正确性回归 | `test/`、CTest | 数值对齐、启动器、下载器、算子和 API 回归 |
| 版本变更 | [版本日志](version.md) | 面向发布的功能和兼容性变化 |

## 当前模型验证入口

| 模型簇 | 部署说明 | Benchmark / 回归 |
| --- | --- | --- |
| Qwen3.8 27B | [Qwen3 部署指南](qwen3.md) | [Qwen3 Benchmark](benchmarks/qwen3.md)、[DFlash2](benchmarks/qwen38_27b_dflash2.md) |
| Qwen3.8-Flash-Next GSQ / NVFP4 | [Flash-Next 用户手册](qwen3.8-flash-next/README.md) | [Qwen4-Exp Benchmark](benchmarks/qwen4_exp.md) |
| DeepSeek V4.1 Flash | [DeepSeek V4.1 说明](deepseek_v41.md) | [V4.1 Benchmark](benchmarks/deepseek_v41.md)、CPU fixture 和 DSpark 回归 |
| GLM-5.3-Flash | [GLM-5 部署指南](glm5.md) | [GLM Benchmark](benchmarks/glm5.md)、CUDA/NUMA/缓存回归 |

自动部署相关的可执行覆盖主要位于 `test/api/test_auto_config.py`、`test/api/test_launcher.py`、`test/api/test_launcher_webui_browser.py` 和 `test/api/test_modelscope_download.py`。模型算子和数值边界测试继续按模型指南列出的 CTest 或 Python fixture 执行。

## Benchmark 记录要求

新增或更新一组数字时，正文至少记录：

1. GPU 型号、数量、可见设备和主机内存；
2. 权重仓库、量化格式、分片路径和 FastLLM 源码版本；
3. 输入长度、输出长度、批大小、线程数和 warmup 方法；
4. `--auto` 的实际规划结果，或完整的手动启动命令；
5. `moe_cuda_cache`、`moe_cpu_cache`、prefix/history cache、MTP/DSpark、Graph 和 chunked prefill 设置；
6. TTFT、prefill、decode 的计时口径，以及未覆盖的场景。

同一配置重新测量时，更新原结论并保留源码版本和测量范围；不要在部署文档中追加逐轮优化日记。没有源码版本的旧结果要明确标记为历史参考，不能写成当前 HEAD 的性能。

## 提交前检查

部署器改动应在安装了 FastAPI、Uvicorn、requests 和 httpx 的测试环境中执行以下回归。浏览器测试还需要 Playwright 与 Chromium（`python3 -m playwright install chromium`）；未安装时该组会跳过，不能把跳过计为通过。

```bash
git diff --check
PYTHONPATH=test/api python3 -m unittest \
  test.api.test_auto_config \
  test.api.test_hybrid_auto \
  test.api.test_moe_cuda_cache_cli \
  test.api.test_launcher \
  test.api.test_launcher_webui_browser \
  test.api.test_modelscope_download
```

如果修改了 CUDA、模型算子或权重格式，还要运行对应的 `ctest --test-dir <build> ...` 回归，并在 Benchmark 或模型指南中记录实际构建配置。只修改文档时，`git diff --check` 和链接/命令检查即可；不要为了文档变更伪造性能数字。
