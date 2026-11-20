# Qwen1.5-MoE-A2.7B-Chat 批次不变模式性能对比

[返回性能对比汇总](README.md)

本报告持续汇总 Attention DP/EP 旧路径与批次不变模式的对比。当前记录对应运行 `20261120T081232Z_moe_batch_performance_3a89bda4`；后续场景按模型、硬件、并行配置和工作负载追加，并保留各次测量来源。

来源为用户提供的终端结果。本文件整理已有测量，没有重新运行测试；当前本地未取得服务器原始 JSON。

**模型**：Qwen1.5-MoE-A2.7B-Chat，FP16。**并行场景**：Attention DP=4 / EP=4，四张 V100-SXM2-32GB。并发 1 也使用四卡进程组，并非单 GPU 测试。

**输入/输出**：每请求 1024 / 128 Token；前缀缓存关闭。当前测试脚本设置为 warmup 1 次、每轮 1 次生成批次、正式重复 5 轮。

**加速比 = 旧路径 Median / 批次不变 Median**；大于 1 表示批次不变模式更快。耗时变化 = (新 / 旧 - 1) × 100%。

| 执行模式 | 请求并发 | 旧路径 Median(ms) | 批次不变 Median(ms) | 批次不变 P95(ms) | 批次不变吞吐(Token/s) | 加速比(旧/新) | 耗时变化(%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Eager | 1 | 10013.234 | 8351.875 | 8383.345 | 15.326 | 1.199 | -16.592 |
| Eager | 4 | 14803.898 | 10312.632 | 10499.799 | 49.648 | 不适用 | 不适用 |
| CUDA Graph | 1 | 9779.168 | 8297.472 | 8352.813 | 15.426 | 1.179 | -15.152 |
| CUDA Graph | 4 | 13901.804 | 9406.969 | 10461.047 | 54.428 | 不适用 | 不适用 |

并发 1：Eager 加速比 1.199，耗时降低 16.592%；CUDA Graph 加速比 1.179，耗时降低 15.152%。

并发 4：仅保留双方耗时及新路径吞吐，不计算加速比或耗时变化。原报告提示：

> outputs/cache/workload differ or a source case failed; speedup not comparable

该通用提示尚不能确定是哪一项不同，需用原始 JSON 复核。单个测量 case 的 PASS 不等于新旧模式具备加速比可比性。

计时覆盖一轮请求从提交到全部生成完成，包含 Prefill、Decode 与调度；模型加载、warmup 和结果检查不在计时范围。吞吐量是并发请求的总输出 Token 数除以该轮 Median 耗时，包含 Prefill 时间。

这些结果反映启用批次不变模式的整体变化，尚不能分离 GEMM 后端替换的收益与不变性约束本身的开销，也不表示完整功能矩阵或 HF 对齐已经通过。

额外建议记录的字段：

- **原始计时样本、双方 CV 和旧路径 P95**：判断波动和结果稳定性；当前不能由 Median/P95 推算原始样本或 CV。
- **本次实际后端及 worker 回执**：确认进入目标实现；配置名称不能替代实际执行证据。
- **本次 Git 提交、CUDA 版本、编译架构**：复现运行环境；当前工作区提交不能充当服务器测速提交。
- **并发 4 不可比的具体原因**：读取原始 JSON 区分输出、缓存或工作负载差异；当前只保留通用提示。
- **双方 TTFT、逐 Token 延迟、显存占用**：同时评估首 Token 等待、生成阶段延迟及内存代价。
- **Prefill/Decode 分段耗时或 profiling**：仅在需要解释加速来源时补测；端到端结果不能定位具体算子收益。

环境和证据说明：

- GPU 型号和模型配置来自此前用户提供的环境；CUDA 12.6 是部署配置，SM70 是目标架构，本次实际版本和编译回执待补。
- 当前预期候选后端为 CUTLASS SM70 MoE 专家 GEMM；本次 worker 后端回执未提供。
- 本次 Git 提交、CV、原始计时样本和显存指标未提供，JSON 使用 null；没有使用当前工作区信息替代远端测速信息。
- 所有数值保持用户粘贴的精度。Markdown、CSV 浮点展示保留 3 位小数；JSON 不补造更多有效数字。
- 原始运行目录：`/home/project/vllm_fastllm/test/distributed/results/20261120T081232Z_moe_batch_performance_3a89bda4`。

当前脚本依据（用于解释口径，尚未与远端测速提交核对）：

- [run_qwen_moe_batch_performance.py:20](/home/project/vllm_fastllm/test/distributed/run_qwen_moe_batch_performance.py:20)：可比性校验和旧/新加速比。
- [run_qwen_moe_full.py:554](/home/project/vllm_fastllm/test/distributed/run_qwen_moe_full.py:554)：Prompt、精度、EOS 与重复次数。
- [run_qwen_moe_full.py:587](/home/project/vllm_fastllm/test/distributed/run_qwen_moe_full.py:587)：四卡会话、warmup 与计时范围。
- [run_qwen_moe_full.py:605](/home/project/vllm_fastllm/test/distributed/run_qwen_moe_full.py:605)：目标后端验证。

用户提供的完整运行命令：

```bash
(
set -eo pipefail
source /workspace/fastllm-v100-setup/env.sh
cd /home/project/vllm_fastllm
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
"$ATTENTION_EP_PYTHON" test/distributed/run_qwen_moe_batch_performance.py \
    --build "$ATTENTION_EP_BUILD" \
    --model "$ATTENTION_EP_MODEL" \
    --prompt-lengths 1024 \
    --output-tokens 128 \
    --concurrencies 1 4 \
    --cache-states off \
    --output /home/project/vllm_fastllm/test/distributed/results
)
```

相关环境变量按此前部署记录为：

```text
ATTENTION_EP_PYTHON=/venv/fastllm-main-cu126/bin/python3
ATTENTION_EP_BUILD=/home/project/vllm_fastllm/build-attention-ep-cu126
ATTENTION_EP_MODEL=/home/models/Qwen1.5-MoE-A2.7B-Chat
```
