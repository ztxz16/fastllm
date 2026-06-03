# 大模型精度与 Agent 能力评测规划

本目录用于沉淀新模型发布前后的评测清单，方便后续逐步生成真正可执行的测试脚本。

当前文件是规划文档，不是可直接运行的 benchmark。配套的 `benchmark_catalog.json` 是机器可读清单，可作为后续生成 runner、CI case 或结果报表的输入。

整理日期：2026-06-02。

## 背景

近期新模型（如 Qwen 系列、Step 3.7、MiniMax M3、DeepSeek V4）发布时，已经不只依赖 MMLU 这类传统考试型评测。公开报告里常见的组合大致包括：

- 传统知识与推理：MMLU、MMLU-Pro、MMLU-Redux、C-Eval、CMMLU、GPQA、HLE、BBH、DROP。
- 数学与代码：GSM8K、MATH、AIME、HMMT、LiveCodeBench、BigCodeBench、HumanEval、SWE-bench。
- 长上下文：RULER、LongBench v2、MRCR、CorpusQA、LOCA-bench。
- 多模态：MMMU、MMBench、MathVista、MathVision、OmniDocBench、Video-MME。
- Agent 与工具使用：BFCL、Toolathlon、tau-bench / tau2-bench、AgentBench、GAIA、BrowseComp、SWE-bench、Terminal-Bench、WebArena / VisualWebArena、WorkArena、TheAgentCompany、MLE-bench、OSWorld、AndroidWorld、Windows Agent Arena。

传统 benchmark 仍然适合作为回归门禁；真正区分 frontier 模型的，通常是软件工程、终端、浏览器、GUI、工具调用、长上下文和多模态 Agent 能力。

## 推荐分层

### 1. 快速冒烟套件

目标：1-2 天内判断一个模型或量化版本是否明显退化。

建议覆盖：

- MMLU-Pro 或 MMLU-Redux
- GPQA Diamond
- SimpleQA 或 Chinese SimpleQA
- IFEval
- GSM8K / MATH / AIME 子集
- LiveCodeBench 子集
- Local Agent Tool-use Smoke
- BFCL
- RULER 32K / 128K

预估工作量：接入 1-2 人天；单模型运行 6-24 小时，取决于并发和模型速度。

### 2. 发布级公开能力套件

目标：形成较完整的 release scorecard。

建议覆盖：

- MMLU-Pro、GPQA、HLE
- CMMLU / C-Eval
- Full LiveCodeBench、BigCodeBench
- LongBench v2 或 MRCR
- MMMU / MathVista
- BFCL、tau-bench / tau2-bench
- MLE-bench 子集
- WorkArena 子集
- Terminal-Bench

预估工作量：接入 5-10 人天；单模型运行 2-5 天。Agent 类建议至少 3 次重复运行。

### 3. Frontier / Agent 宣发级套件

目标：评估“模型 + scaffold + 工具 + sandbox”的系统级能力。

建议覆盖：

- SWE-bench Verified / Pro / Multilingual
- Terminal-Bench 2.x
- AgentBench
- GAIA / BrowseComp / DeepResearch 类任务
- WebArena / VisualWebArena
- WorkArena / TheAgentCompany
- MLE-bench full
- OSWorld / AndroidWorld / Android Daily
- Windows Agent Arena
- Video-MME / VideoMMMU
- 私有业务 holdout 集

预估工作量：接入 2-6 周；单模型运行 1-3 周。GUI、浏览器、桌面、企业环境和视频类环境维护成本最高。

## 知名 Agent Benchmark 补充

这些是目前公开报告、论文和 leaderboard 里更常见的 Agent 评测。它们测的是“模型 + scaffold + 工具 + sandbox + 环境”的系统能力，不应直接当成裸模型能力。

| Benchmark | 类型 | 主要测什么 | 推荐用途 | 单模型接入与运行预估 |
|---|---|---|---|---|
| Local Agent Tool-use Smoke | 本地轻量工具调用 | JSON ReAct、工具选择、参数生成、多步工具链、policy 检索、计算器 | 内部冒烟；先挡 API/scaffold 退化 | 已实现；运行 1-10 分钟 |
| BFCL | 函数调用 | 函数选择、参数 AST/JSON、并行/多轮工具、no-call | 发布前必跑的工具调用基础项 | 接入 0.5-1 人天；运行 2-8 小时 |
| tau-bench / tau2-bench | 业务对话 Agent | 航司/零售等客服对话、API 操作、policy 遵循、Pass^k 稳定性 | 发布级业务 Agent 能力 | 接入 1-3 人天；运行 4-24 小时/轮 |
| AgentBench | 综合 Agent | OS、数据库、知识图谱、购物/浏览、游戏等多环境 | 广谱 Agent scaffold sanity check | 接入 2-5 人天；运行 12-72 小时 |
| GAIA | 通用助手 Agent | 搜索/浏览、多步推理、工具使用、多模态信息 | General assistant / research agent | 接入 1-3 人天；运行 6-48 小时 |
| BrowseComp | 深度浏览 Agent | 难找事实、多网站检索、搜索策略、短答案验证 | Deep research / browsing 宣发项 | 接入 1-2 人天；运行 12-72 小时 |
| SWE-bench Verified / Pro / Live / Multilingual | 软件工程 Agent | 真实 repo issue 修复、代码编辑、测试执行 | Coding agent 核心项 | 接入 2-7 人天；运行 1-4 天 |
| Terminal-Bench 2.x | 终端 Agent | CLI、文件、依赖安装、调试、端到端 sandbox 任务 | DevOps / terminal autonomy | 接入 1-3 人天；运行 6-48 小时 |
| WebArena / VisualWebArena | Web UI Agent | 自建网站长链路操作；VisualWebArena 加视觉 grounding | 浏览器操作核心项 | 接入 3-7 人天；运行 1-5 天 |
| WorkArena | 企业 Web Agent | ServiceNow/知识工作流、BrowserGym/AgentLab、企业任务 | 企业 web workflow | 接入 3-7 人天；运行 1-4 天 |
| TheAgentCompany | 数字员工 Agent | 模拟软件公司，浏览、代码、程序执行、文件、同事通信 | 企业数字员工宣发级 | 接入 1-2 周；运行 2-7 天 |
| MLE-bench | ML 工程 Agent | Kaggle 风格 ML 工程、训练、验证、提交 | ML/data-science agent | 接入 2-5 人天；运行 2-10 天 |
| OSWorld / OSWorld-Verified | 桌面 GUI Agent | Ubuntu/Windows/macOS 风格真实桌面任务、截图 grounding | Computer-use / GUI 核心项 | 接入 1-2 周；运行 1-4 天 |
| AndroidWorld / Android Daily | 手机 GUI Agent | Android emulator、真实 app 操作、动态任务 | Mobile agent | 接入 1-2 周；运行 1-4 天 |
| Windows Agent Arena | Windows GUI Agent | Windows 环境多模态 OS 操作、常见应用任务 | Windows-specific computer-use | 接入 1-2 周；运行 1-4 天 |
| Mind2Web / WebShop / MiniWoB++ | 经典 Web Agent | 早期/轻量浏览器导航、购物和网页控件操作 | 回归和 scaffold 调试 | 接入 1-3 人天；运行 4-24 小时 |

落地优先级建议：

1. 已实现的 `test/agent` 本地 smoke + BFCL：先验证 API、JSON、工具调用和参数生成。
2. tau-bench / tau2-bench + Terminal-Bench：对工具链、业务 policy、终端自治更敏感，适合 release scorecard。
3. SWE-bench / WebArena / WorkArena / GAIA：适合发布级或对外宣发，但需要固定 scaffold 和重复运行。
4. OSWorld / AndroidWorld / TheAgentCompany / MLE-bench：环境成本高，适合 frontier agent 系统评估。

注意：SWE-bench Verified 仍然很有名，但需要和 SWE-bench Live、Pro、Multilingual、SWE-rebench 或私有 holdout 互补，避免只看单一老 benchmark。

参考入口：

- BFCL: https://sky.cs.berkeley.edu/project/berkeley-function-calling-leaderboard/
- tau2-bench: https://github.com/sierra-research/tau2-bench
- AgentBench: https://github.com/THUDM/AgentBench
- GAIA: https://huggingface.co/gaia-benchmark
- BrowseComp: https://openai.com/index/browsecomp/
- SWE-bench: https://www.swebench.com/
- Terminal-Bench: https://www.tbench.ai/
- WebArena: https://github.com/web-arena-x/webarena
- VisualWebArena: https://github.com/web-arena-x/visualwebarena
- WorkArena: https://github.com/ServiceNow/WorkArena
- TheAgentCompany: https://github.com/TheAgentCompany/TheAgentCompany
- MLE-bench: https://github.com/openai/mle-bench
- OSWorld: https://os-world.github.io/
- AndroidWorld: https://github.com/google-research/android_world
- Windows Agent Arena: https://microsoft.github.io/WindowsAgentArena/

## Benchmark 清单

| 类别 | 常用测试 | 主要能力 | 单模型接入与运行预估 |
|---|---|---|---|
| 通用知识 | MMLU、MMLU-Pro、MMLU-Redux | 学科知识、考试题、选择题 | 接入 0.5-1 人天；运行 1-8 小时 |
| 中文/多语 | C-Eval、CMMLU、AGIEval、MMMLU、MultiLoKo | 中文考试、多语知识、综合考试 | 接入 0.5-1 人天；运行 1-6 小时 |
| 高难知识 | GPQA Diamond、SuperGPQA、HLE | 研究生/专家级科学和综合知识 | 接入 0.5-2 人天；运行 2-12 小时 |
| 事实性 | SimpleQA、Chinese SimpleQA、FACTS、TruthfulQA | 幻觉率、事实问答准确性 | 接入 0.5-1 人天；运行 1-4 小时，常需抽样复核 |
| 推理/NLP | BBH、DROP、ARC、HellaSwag、WinoGrande | 逻辑、阅读、常识 | 接入 0.5 人天；运行 1-6 小时 |
| 指令遵循 | IFEval、IFBench、MultiChallenge | 格式、约束、多条件指令 | 接入 0.5 人天；运行 1-3 小时 |
| 数学 | GSM8K、MATH、AIME、HMMT、IMOAnswerBench、MathArena | 算术、竞赛数学、证明类推理 | 接入 0.5-2 人天；运行 1-8 小时，证明类更慢 |
| 传统代码 | HumanEval、MBPP、BigCodeBench | 函数级代码生成和单测通过 | 接入 0.5-1 人天；运行 1-8 小时 |
| 新代码竞赛 | LiveCodeBench、OJBench、Codeforces | 新题、竞赛编程、抗污染 | 接入 1-2 人天；运行 4-24 小时 |
| 本地 Agent 冒烟 | Local Agent Tool-use Smoke | JSON ReAct、工具选择、参数、多步工具链 | 已实现；运行 1-10 分钟 |
| 软件工程 Agent | SWE-bench Verified、SWE-bench Pro、SWE-bench Live、SWE Multilingual、SWE-rebench | 真实 GitHub issue 修复 | 接入 2-7 人天；运行 1-4 天，重复运行线性增加 |
| 终端 Agent | Terminal-Bench 2.x | shell/sandbox 中完成真实任务 | 接入 1-3 人天；运行 6-48 小时 |
| 工具调用 | BFCL、ToolACE、API-Bank | 函数选择、参数、并行工具、多轮工具调用 | 接入 0.5-1 人天；运行 2-8 小时 |
| 业务型 Agent | tau-bench、tau2-bench | 航司/零售客服，多轮对话、API 操作、policy 遵循 | 接入 1-3 人天；运行 4-24 小时/轮 |
| 综合 Agent | AgentBench | 多环境通用 Agent 能力 | 接入 2-5 人天；运行 12-72 小时 |
| Web/研究 Agent | GAIA、BrowseComp、DeepResearch 类 | 搜索、浏览、多步推理、工具使用 | 接入 1-3 人天；运行 6-72 小时 |
| Web UI Agent | WebArena、VisualWebArena | 自建网站环境中的长链路操作 | 接入 3-7 人天；运行 1-5 天 |
| 企业 Agent | WorkArena、TheAgentCompany | 企业 web/数字员工任务、文件、通信、代码和程序执行 | 接入 1-2 周；运行 1-7 天 |
| ML 工程 Agent | MLE-bench | 机器学习工程、Kaggle 风格竞赛、训练和提交 | 接入 2-5 人天；运行 2-10 天 |
| GUI/电脑/手机 Agent | OSWorld、OSWorld-Verified、AndroidWorld、Android Daily、Windows Agent Arena | 控制桌面/手机完成任务 | 接入 1-2 周；运行 1-4 天 |
| 经典 Web Agent | Mind2Web、WebShop、MiniWoB++ | 浏览器导航、购物、网页控件操作 | 接入 1-3 人天；运行 4-24 小时 |
| 长上下文 | Needle/RULER、LongBench v2、MRCR、CorpusQA、LOCA-bench | 128K-1M token 检索、归纳、跨段推理 | 接入 1-3 人天；运行 8-48 小时 |
| 多模态图像 | MMMU/MMMU-Pro、MMBench、MMStar、MathVista、MathVision | 图文理解、图表、视觉数学 | 接入 1-2 人天；运行 2-12 小时 |
| 文档/图表 | DocVQA、ChartQA、OmniDocBench | OCR、表格、PDF/文档理解 | 接入 1-3 人天；运行 4-24 小时 |
| 视频 | Video-MME、VideoMMMU | 长视频理解、时序推理、字幕/帧输入 | 接入 2-5 人天；运行 1-3 天 |
| 安全/合规 | HarmBench、StrongREJECT、JailbreakBench、XSTest | 拒答、越狱、危险能力边界 | 接入 1-3 人天；运行 2-12 小时，通常要人工复核 |

## 后续落地建议

1. 先从 `quick_smoke` 套件做起：MMLU-Pro、CMMLU、GPQA、IFEval、LiveCodeBench 子集、Local Agent Tool-use Smoke、BFCL、RULER。
2. 每个 benchmark runner 统一输出 JSONL：至少包含 `benchmark_id`、`case_id`、`model`、`prompt`、`raw_output`、`parsed_answer`、`score`、`latency_ms`、`input_tokens`、`output_tokens`。
3. 对 Agent 测试固定 scaffold、系统提示、工具 schema、sandbox 配置、timeout 和最大 token budget，否则模型间结果不可比。
4. 对随机性较强的 Agent 测试至少跑 3 次，报告平均值、标准差和失败类型。
5. 保留私有业务 holdout 集。公开 benchmark 高分不能替代实际场景验收。
