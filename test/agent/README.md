# Agent Tool-use API Evaluation

## 测试内容与目的

本测试通过 OpenAI-compatible `/v1/chat/completions` API 评估模型的轻量 Agent 工具使用能力。它不依赖服务端原生 function calling，而是要求模型按 JSON ReAct 协议选择本地内置工具、读取 observation，并给出最终答案。

主要覆盖：

- 工具选择与 JSON 参数生成
- 多步工具链
- policy 检索
- 计算器与日期工具使用
- 最终答案格式遵循

这是一套本地冒烟/回归测试，不是 WebArena、SWE-bench、tau-bench 这类公开重环境 benchmark。

## 目录结构

- `setup.sh`: 安装 Python 依赖。
- `download.sh`: 检查本地 baseline 数据；本测试没有外部数据下载。
- `run.sh`: 调用 `agent_tool_eval.py` 执行评测。
- `baseline/default_cases.jsonl`: 默认测试数据，当前 24 条。
- `results/`: 运行输出目录，JSONL/summary 默认被忽略，人工整理的 `.md` 报告可提交。

## 1. Setup

```bash
bash test/agent/setup.sh
```

## 2. Download

```bash
bash test/agent/download.sh
```

该脚本会确认 `baseline/default_cases.jsonl` 存在，并打印 case 数量。

## 3. Run

快速冒烟：

```bash
bash test/agent/run.sh \
  --base-url http://127.0.0.1:1616 \
  --model ds \
  --limit 8
```

传入模型专属参数：

```bash
bash test/agent/run.sh \
  --base-url http://127.0.0.1:1616 \
  --model ds \
  --workers 4 \
  --max-steps 5 \
  --max-tokens 384 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

`run.sh` 至少需要 `--base-url` 和 `--model`，并会把 `--temperature`、`--top-p`、`--max-tokens`、`--extra-body`、`--output-file`、`--resume` 等参数继续传给底层 runner。

## 输出

默认输出：

```text
test/agent/results/agent_tool_<model>_<timestamp>.jsonl
test/agent/results/agent_tool_<model>_<timestamp>.summary.json
```

summary 包含 accuracy、tool plan accuracy、invalid action case、tool error case、平均步数、延迟和 token 吞吐等基础结果。
