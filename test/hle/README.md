# HLE API Evaluation

本目录为 Humanity's Last Exam（HLE）提供 OpenAI-compatible API 评测，并支持
target-only 与 DFlash2 的成对精度、速度对比。

## 本次推荐口径

正式对比固定使用官方 `cais/hle` test split 中的 50 道题：

- 仅 text-only，避免把多模态预处理差异混入 speculative decode 对比；
- 仅 `multipleChoice`，答案字母可确定性判分，不依赖外部 LLM judge；
- 先过滤再以 `Random(42)` 打乱，取前 50 道；
- 单并发 `workers=1`；
- `temperature=0`；runner 不主动发送 `max_tokens`，当前 FastLLM 服务端使用
  默认的 32768-token completion 上限，同时仍受模型原生 262144-token 总上下文
  上限约束；
- Qwen thinking 显式开启；
- 正式计时前用同一题生成 64 token 预热；
- DFlash2 使用 checkpoint 原生 block=8（每轮 7 个 draft token）。

这是一套可重复的 **HLE text-only multiple-choice 50 题子集分数**，不是完整 HLE
官方分数。完整 HLE 还包含图片题和 `exactMatch` 题；后者需要官方的能力较强的
LLM judge。runner 虽可对 exactMatch 输出归一化字符串下界，但不会把它冒充官方分数。

## 数据准备

默认从 ModelScope 的公开 `cais/hle` 镜像下载：

```bash
bash test/hle/setup.sh
bash test/hle/download.sh
```

下载脚本固定 ModelScope revision
`1ec1f1f25ed4ad891e3a81d1cbc08f261f5e77c6`，并校验 parquet SHA-256
`6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a`。
该文件与 Hugging Face 官方固定 revision
`5a81a4c7271a2a2a312b9a690f0c2fde837e4c29` 的文件大小和 SHA-256 一致。
筛选结果输出到：

```text
test/hle/baseline/downloaded/hle_test_text_mc_seed42_50.jsonl
```

下载内容和运行结果均已 gitignore。`baseline/smoke.jsonl` 是合成链路测试，不是
HLE 真题。

若要直接从 Hugging Face 下载，先接受 gated dataset 条款并执行
`hf auth login`，然后运行 `bash test/hle/download.sh --source huggingface`。

## Target-only

从仓库根目录启动当前构建：

```bash
PYTHONPATH="$PWD/build-fastllm/tools" \
FASTLLM_CUDA_GRAPH=0 FASTLLM_CUDA_CUSTOM_ALLREDUCE=0 \
ftllm server /root/hfmodels/Qwen3.8-27B-FP8 \
  --model_name qwen38-hle-target \
  --dtype auto --tp 2 --cuda_embedding --max_batch 1 \
  --gpu_mem_ratio 0.98 \
  --host 127.0.0.1 --port 18126 --hide_input
```

另一个终端运行：

```bash
bash test/hle/run.sh \
  --base-url http://127.0.0.1:18126 \
  --model qwen38-hle-target \
  --data-file test/hle/baseline/downloaded/hle_test_text_mc_seed42_50.jsonl \
  --workers 1 --temperature 0 --max-tokens 0 --request-timeout 0 \
  --warmup-tokens 64 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":true}}' \
  --output-file test/hle/results/qwen38_hle50_target_only.jsonl \
  --overwrite
```

## DFlash2 最快推荐配置

```bash
PYTHONPATH="$PWD/build-fastllm/tools" \
FASTLLM_CUDA_GRAPH=0 FASTLLM_CUDA_CUSTOM_ALLREDUCE=0 \
FASTLLM_DFLASH_BLOCK_SIZE=8 \
ftllm server /root/hfmodels/Qwen3.8-27B-FP8 \
  --model_name qwen38-hle-dflash2-b8 \
  --dtype auto --tp 2 --cuda_embedding --max_batch 1 \
  --gpu_mem_ratio 0.98 \
  --host 127.0.0.1 --port 18126 --hide_input \
  --speculative_algorithm dflash \
  --speculative_draft_model_path /root/hfmodels/Qwen3.8-27B-DFlash2 \
  --speculative_num_draft_tokens 8
```

```bash
bash test/hle/run.sh \
  --base-url http://127.0.0.1:18126 \
  --model qwen38-hle-dflash2-b8 \
  --data-file test/hle/baseline/downloaded/hle_test_text_mc_seed42_50.jsonl \
  --workers 1 --temperature 0 --max-tokens 0 --request-timeout 0 \
  --warmup-tokens 64 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":true}}' \
  --output-file test/hle/results/qwen38_hle50_dflash2_b8.jsonl \
  --overwrite
```

## 成对比较

```bash
python3 test/hle/compare_results.py \
  test/hle/results/qwen38_hle50_target_only.jsonl \
  test/hle/results/qwen38_hle50_dflash2_b8.jsonl \
  --output test/hle/results/qwen38_hle50_comparison.json
```

输出包括两边 accuracy、总输出 token、请求总耗时、output tok/s、DFlash2
加速比、成对正确性变化，以及逐题输出完全一致率。单并发下吞吐按
`sum(completion_tokens) / sum(request_latency)` 计算，不包含模型加载和预热。

两边都固定 `FASTLLM_CUDA_CUSTOM_ALLREDUCE=0`，避免每次进程启动时的自动微基准
因小幅抖动选择不同规约实现，从而把规约数值路径差异混入 DFlash2 精度 A/B。
