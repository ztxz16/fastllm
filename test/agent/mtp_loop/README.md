# MTP 思考循环复现

通过已有的 OpenAI-compatible `/v1/chat/completions` 服务重放 Agent 场景，保存思考和正文，并检测连续精确重复。测试不导入 FastLLM、不调用 CUDA，也不依赖采样实现的新接口，可以独立于 MTP 修复提交和运行。

## 场景

`cases.json` 保留六类构造场景：缓存一致性与字数限制、矛盾测试、工具权限不足、任务已完成的长工具历史、取消/提交竞态、网络分区下冲突的订单要求。工具历史均为固定夹具，`tool_choice=none`，测试不会执行模型生成的工具调用。提示词没有要求模型重复。

## 运行

需要 Python 3.10 或更新版本。默认只用标准库，按 Unicode 字符分析；如需按模型 token 分析，另安装 `tokenizers` 并提供本地 tokenizer 文件：

```bash
python3 -m pip install tokenizers
```

在任意已启动的待测版本上运行最明确的循环场景：

```bash
python3 test/agent/mtp_loop/probe.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3.8-27B-FP8 \
  --cases circular_requirements_zh \
  --temperature 0.6 --max-tokens 16384 --repeats 3 \
  --out test/agent/mtp_loop/results/before
```

`--model` 必须与服务提供的模型 ID 一致。服务若需要认证，设置 `OPENAI_API_KEY`；密钥不会写入测试元数据。可加 `--tokenizer /path/to/model/tokenizer.json` 或模型目录；工具不会自动下载模型或 tokenizer。

依次启动旧版、新版及 `--mtp 0` 的服务，用相同参数和不同输出目录运行。脚本仅请求 API，不启动或停止服务、不选择 GPU，也不切换安装库。完整六场景筛查使用 `--temperature 0.6 1.0`，省略 `--cases`。

FastLLM 的思考强度对照可加 `--reasoning-effort medium`；省略时保持服务/模型默认值。`top_k=20`、`top_p=0.95` 默认显式传入。跨服务比较时需确认这些参数和思考开关被服务支持。

每条请求保留 `request.json`、`events.jsonl`、`reasoning.txt`、`content.txt`、`tool_calls.json`、`result.json`。思考/正文和 SSE 在接收过程中写入；请求失败时保留已收到的内容。运行目录保存 `results.json`、`summary.json` 和 API 地址等元数据。输出目录必须不存在，生成结果默认被 Git 忽略。

## 判定与退出状态

- `finish=length` 只表示触及长度上限，不直接判为循环。
- 16/64 单位片段的重复次数用于筛查，代码引用和正常修改也可能重复。
- `cycle` 表示至少 3 轮、覆盖至少 256 单位的**连续精确周期**；`unit` 明确区分字符与 token。起止位置从 0 开始，`end` 不包含在区间内。
- 默认候选周期上限为 16384 字符或 4096 tokens。候选搜索有界，报告的周期经过精确比较，但未检出不能证明没有循环；近义反复和很长周期仍需人工审阅。

默认退出 0 表示请求正常完成，即使检测到循环；传 `--fail-on-cycle` 时检测到循环退出 2。HTTP/SSE 错误或不完整流退出 1，优先于循环状态。随机生成不保证每次复现，少量构造场景不能估计线上发生率。

离线分析已有思考文本，无需连接服务：

```bash
python3 test/agent/mtp_loop/analyze.py \
  test/agent/mtp_loop/fixtures/legacy_reasoning.txt
```

同样可传 `--tokenizer`、`--output analysis.json` 或 `--fail-on-cycle`。不能将字符周期长度直接当作 token 周期长度。

## 已知复现夹具

`fixtures/legacy_reasoning.txt` 是 2026-09-11 在 Qwen3.8-27B-FP8、TP2/MTP3、temperature=0.6 下由最初旧采样库生成的真实思考输出。循环发生在中文订单需求评审中：先得出结论，准备结束，再重新比较相同方案，然后回到相同结论。

该输出的 token 区间 `[4046, 16384)` 包含周期为 2312 tokens 的内容：完整重复 5 轮，再重复 778 tokens 后被上限截断，最终答案为空。字符分析对应区间 `[17638, 69811)`、周期 9750 字符。`legacy_cycle.json` 保存参数、库和原文 SHA256 及预期周期。

同场景 16K 上限复测中，旧库三次有一次出现上述长精确循环，其余两次退出；当前修正版三次均最终退出。正确采样和关闭 MTP 时仍可能长时间思考或触及更低的输出上限。因此本夹具用于验证检测和复现流程，不表示所有思考不收尾都由同一个 MTP 问题引起。夹具中的模型方案不是实现建议。

## 本地回归

以下测试仅需 Python 标准库：真实旧输出的周期检测、整数 token 周期、非循环引用、截断响应、UTF-8 SSE 拆包、错误和原始请求保留。

```bash
python3 -m unittest discover -s test/agent/mtp_loop -p 'test_*.py' -v
```

此目录只包含用例、API 探测器、离线分析和小规模夹具。采样内核单元测试及 CMake 注册仍由对应实现改动维护；历史 GPU 编排脚本、旧共享库和完整批量输出不属于此测试包。
