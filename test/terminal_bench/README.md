# Terminal-Bench 2.0 for Qwen3.6-27B

## 测试内容与目的

本目录整理 Qwen/Qwen3.6-27B 模型卡中 Terminal-Bench 2.0 的测试方式，并提供一个 Harbor/Terminus-2 的本地运行入口。

这不是本仓库自带的轻量回归测试。Terminal-Bench 2.0 的任务集由 Harbor 下载和调度，任务在隔离容器中执行，适合评估模型作为终端 agent 完成多步骤任务的能力。

## Qwen3.6-27B 模型卡中的测试方式

- Benchmark: Terminal-Bench 2.0
- Reported score: 59.3
- Harness / agent: Harbor / Terminus-2
- Timeout: 3h
- Sandbox resources: 32 CPU / 48 GB RAM
- Generation: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `max_tokens=80K`
- Context window: 256K
- Aggregation: 5 runs average

Terminal-Bench v2 公开说明为 89 个真实多步骤终端任务，覆盖编译、调试、系统管理、文件操作等场景；任务在隔离容器里执行，并用自动验证给出 0/1 分。

## 目录结构

- `setup.sh`: 安装 Harbor CLI。
- `run.sh`: 通过 Harbor 运行 Terminal-Bench 2.0。
- `qwen3_6_27b.env.example`: OpenAI-compatible endpoint 与 Qwen 测试参数模板。
- `qwen3_6_27b_methodology.json`: 结构化记录模型卡中的 Terminal-Bench 参数。
- `jobs/`: Harbor job 输出目录，默认 gitignored。

## 1. Setup

先确认 Docker 可用，再安装 Harbor：

```bash
bash test/terminal_bench/setup.sh
```

Harbor 当前版本通常需要较新的 Python；如果本机 `pip install harbor` 失败，优先使用 `uv tool install harbor`。

## 2. Run

以 FastLLM/OpenAI-compatible 服务为例，Harbor/LiteLLM 通常需要带 `/v1` 的 base URL：

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:1616/v1

bash test/terminal_bench/run.sh \
  --model openai/ds \
  --api-base http://127.0.0.1:1616/v1 \
  --dataset terminal-bench-sample@2.0
```

`terminal-bench-sample@2.0` 适合先做链路冒烟。完整运行改为默认的 `terminal-bench@2.0`。

使用模板参数：

```bash
source test/terminal_bench/qwen3_6_27b.env.example
bash test/terminal_bench/run.sh --dataset terminal-bench-sample@2.0
```

更接近模型卡设置的运行方式：

```bash
source test/terminal_bench/qwen3_6_27b.env.example
TB_N_ATTEMPTS=5 \
bash test/terminal_bench/run.sh
```

Qwen 模型卡报告使用 `32 CPU / 48 GB RAM`。当前安装的 `harbor==0.1.28` 不在 `harbor run` CLI 中暴露 CPU/内存覆盖参数；需要通过宿主机 Docker、云 sandbox 或调度环境保证资源。

## 关键参数

- `--model`: Harbor/LiteLLM 使用的模型名。OpenAI-compatible 本地服务常用 `openai/<served-model-name>`。
- `--api-base`: 自定义 OpenAI-compatible endpoint，例如 `http://127.0.0.1:1616/v1`。
- `--dataset`: 默认 `terminal-bench@2.0`；小规模链路测试可用 `terminal-bench-sample@2.0`。
- `--agent`: 默认 `terminus-2`。
- `--n-concurrent`: 并发 trial 数，默认 1。
- `--n-attempts`: 每个任务运行次数，默认 1；Qwen 模型卡报告为 5 runs average。
- `--llm-call-kwargs`: 透传给 Terminus-2 的 LLM 调用参数，模板里包含 `top_p`、`max_tokens`、`top_k`。

Qwen 模型卡还要求 3h timeout。当前安装的 Harbor 暴露 `--timeout-multiplier`，可按任务默认 timeout 乘数做调整。

## 输出

默认输出到：

```text
test/terminal_bench/jobs/
```

Harbor job 通常包含 `config.json`、`result.json`、每个 trial 的 `agent/` 日志、`verifier/` 验证日志和 reward。可用：

```bash
harbor view test/terminal_bench/jobs
```

查看结果和轨迹。

## 资料来源

- ModelScope model card: https://modelscope.cn/models/Qwen/Qwen3.6-27B
- Hugging Face mirror: https://huggingface.co/Qwen/Qwen3.6-27B
- Harbor Terminal-Bench docs: https://www.harborframework.com/docs/tutorials/running-terminal-bench
- Terminal-Bench 2.0 repo: https://github.com/harbor-framework/terminal-bench-2
- EvalScope Terminal-Bench 2.0 notes: https://evalscope.readthedocs.io/en/v1.5.2/benchmarks/terminal_bench_v2.html
