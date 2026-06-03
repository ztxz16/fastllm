# GSM8K API Evaluation

## 测试内容与目的

本测试通过 OpenAI-compatible `/v1/chat/completions` API 跑 GSM8K 风格数学推理题，从模型输出中抽取最终数字答案，并生成 JSONL 明细和 summary。

它适合做算术、短链路数学推理和 CoT 输出解析的回归测试。`baseline/smoke.jsonl` 只是用于验证评测链路的小型样例，不代表官方 GSM8K 分数。

## 目录结构

- `setup.sh`: 安装 Python 依赖。
- `download.sh`: 从 HuggingFace 下载 GSM8K 数据到 `baseline/downloaded/`。
- `run.sh`: 调用 `gsm8k_api_eval.py` 执行评测。
- `baseline/smoke.jsonl`: 小型本地链路测试数据。
- `baseline/downloaded/`: 下载后的真实运行数据，默认 gitignored。
- `results/`: 运行输出目录，JSONL/summary 默认被忽略。

## 1. Setup

```bash
bash test/gsm8k/setup.sh
```

## 2. Download

下载完整 test split：

```bash
bash test/gsm8k/download.sh
```

只下载前 200 条：

```bash
bash test/gsm8k/download.sh \
  --limit 200 \
  --output test/gsm8k/baseline/downloaded/gsm8k_test_200.jsonl
```

为了完全复现实验，可以传入固定数据集 revision：

```bash
bash test/gsm8k/download.sh --dataset-revision <commit>
```

## 3. Run

默认使用 `baseline/smoke.jsonl` 跑链路冒烟：

```bash
bash test/gsm8k/run.sh \
  --base-url http://127.0.0.1:1616 \
  --model ds
```

使用下载后的数据运行 CoT：

```bash
bash test/gsm8k/run.sh \
  --base-url http://127.0.0.1:1616 \
  --model ds \
  --data-file test/gsm8k/baseline/downloaded/gsm8k_test.jsonl \
  --workers 8 \
  --cot \
  --max-tokens 512 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

`run.sh` 至少需要 `--base-url` 和 `--model`，并会把 `--temperature`、`--top-p`、`--max-tokens`、`--cot`、`--extra-body`、`--output-file`、`--resume` 等参数继续传给底层 runner。

## 输出

默认输出：

```text
test/gsm8k/results/gsm8k_<model>_<split>_<mode>_<timestamp>.jsonl
test/gsm8k/results/gsm8k_<model>_<split>_<mode>_<timestamp>.summary.json
```

summary 包含 accuracy、有效答案 accuracy、错误数、延迟分位数、token 数和吞吐。
