# MMLU-Pro API Evaluation

## 测试内容与目的

本测试通过 OpenAI-compatible `/v1/chat/completions` API 跑 MMLU-Pro 风格的多选题评测，解析模型输出的答案字母，并生成 JSONL 明细和 summary。

它适合作为模型发布前的知识、考试型推理和格式遵循回归测试。`baseline/smoke.jsonl` 只是用于验证评测链路的极小样例，不代表官方 MMLU-Pro 分数。

## 目录结构

- `setup.sh`: 安装 Python 依赖。
- `download.sh`: 从 HuggingFace 下载 MMLU-Pro 数据到 `baseline/downloaded/`。
- `run.sh`: 调用 `api_eval.py` 执行评测。
- `baseline/smoke.jsonl`: 小型本地链路测试数据。
- `baseline/downloaded/`: 下载后的真实运行数据，默认 gitignored。
- `results/`: 运行输出目录，JSONL/summary 默认被忽略。

## 1. Setup

```bash
bash test/mmlu_pro/setup.sh
```

## 2. Download

下载完整 test split：

```bash
bash test/mmlu_pro/download.sh
```

下载每个 category 5 条，适合小规模验证：

```bash
bash test/mmlu_pro/download.sh \
  --sample-per-category 5 \
  --output test/mmlu_pro/baseline/downloaded/mmlu_pro_test_5_per_category.jsonl
```

为了完全复现实验，可以传入固定数据集 revision：

```bash
bash test/mmlu_pro/download.sh --dataset-revision <commit>
```

## 3. Run

默认使用 `baseline/smoke.jsonl` 跑链路冒烟：

```bash
bash test/mmlu_pro/run.sh \
  --base-url http://127.0.0.1:1616 \
  --model ds
```

使用下载后的数据运行：

```bash
bash test/mmlu_pro/run.sh \
  --base-url http://127.0.0.1:1616 \
  --model ds \
  --data-file test/mmlu_pro/baseline/downloaded/mmlu_pro_test.jsonl \
  --workers 8 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

CoT 模式：

```bash
bash test/mmlu_pro/run.sh \
  --base-url http://127.0.0.1:1616 \
  --model ds \
  --data-file test/mmlu_pro/baseline/downloaded/mmlu_pro_test.jsonl \
  --cot \
  --max-tokens 512
```

`run.sh` 至少需要 `--base-url` 和 `--model`，并会把 `--temperature`、`--top-p`、`--max-tokens`、`--cot`、`--extra-body`、`--sample-per-category`、`--output-file`、`--resume` 等参数继续传给底层 runner。

## 输出

默认输出：

```text
test/mmlu_pro/results/mmlu_pro_<model>_<split>_<timestamp>.jsonl
test/mmlu_pro/results/mmlu_pro_<model>_<split>_<timestamp>.summary.json
```

summary 包含总 accuracy、有效答案 accuracy、错误数、平均延迟和按 category 的准确率。
