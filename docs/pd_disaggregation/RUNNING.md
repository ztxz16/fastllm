# 单机 PD：当前代码与服务器验证

八阶段的首版代码已接入工作区，**尚未通过服务器编译或 GPU 验收，不能视为计划已交付**。
设计范围保持不变；本页记录目前能接入的代码路径和验证方法，不替代实现清单。

2026-11-26：PD 核心决策已迁移到 C++ 扩展，GPU 搬运由 C++ 直接调用 CUDA Runtime。
Python 保留 CLI、HTTP/Tokenizer、数据对象和原生接口适配。详细模块边界与验证状态见
[核心逻辑原生迁移](NATIVE_MIGRATION.md)。本次原生迁移尚未编译或执行功能回归，
迁移前的协议、Router、IPC PASS 记录不能直接作为迁移后验收结果。

## 当前接入范围

| 部分 | 当前代码状态 |
| --- | --- |
| 请求角色 | 未量化 Qwen3 与 Qwen2 MoE；`full`、`p`、`d`；P 不采样，D 重算 Prompt 末 Token 后生成全部输出 |
| 并行 | 普通 DP×TP；Qwen2 MoE 支持普通 TP 与 Attention DP/EP 两条入口；EP 统一协调联合轮次；普通 TP 沿用单请求串行计算 |
| 交接 | JSON 身份/布局握手、逐请求源/目标描述、逻辑 Token/head 映射；CUDA IPC 和异步 GPU 拷贝 |
| 存储 | 普通 TP 从原生分页池导出；Attention DP/EP 使用 GPU 交接缓冲适配连续 KV，计算与 Graph 继续使用原有固定地址 |
| 推送与拉取 | 两条控制路径；长 Prompt 提前登记位置，按源层就绪提交，完整完成后进入 D 计算 |
| Router | 八种策略、P/D 分开配置、完整服务模式、健康状态、注册/注销、有条件请求级重试、统一结果流 |
| 容量 | 基于原生空闲/可淘汰页检查增长预算；副本排队；D 页不足时本地重算，保留输出进度与独立 RNG |
| 多轮 | P 优先恢复本地前缀，只拉取远端额外命中的部分；480 秒保留、过期重算、在途租约保护 |
| CPU 协议脚本 | 41 个协议/映射/路由/生命周期/模式选择 case，尚未执行 |
| Router 脚本 | 20 个真实 HTTP 故障注入 case，后端为控制面桩，不冒充模型执行 |
| GPU 传输脚本 | 24 个实际跨进程拷贝、错误交接拒绝、取消及迟到通知 case，尚未执行 |
| 正式入口脚本 | 每部署组合 15 个生成场景，含 exact、并发、多轮；6 个性能 case，每项 5 轮；均未执行 |
| 生命周期脚本 | 每种模式 7 个真实服务旧消息拒绝、取消、释放与客户端断连场景；尚未执行 |

**仍缺验证：** 原生编译、四种模式组合的生成正确性、满载/抢占压力回归、rank 异常退出与
失联租约恢复、完整服务原路径回归，以及真实 GPU 计算/传输重叠时序。新增的源代码和脚本
不能替代上述结果；既有 Attention DP/EP 数值/前缀缓存问题保持原状态。

当前原生执行器沿用 FP16、总上下文最多 2048 Token 等既有限制；不接受量化模型、PP、
跨机或 CPU 数据兜底。连续 KV 交接有独立的 512 MiB 缓冲预算，额外占用显存；
可选的 Decode 结束回传快照若无法分配，则不提供该轮缓存票据，下一轮走本地命中或重算。
持续运行的请求不会因 480 秒保留期而释放 KV，该期限从生成结束后计算。

## 服务器构建

仅在用户的 GPU 服务器执行。当前工作区不执行编译。
复用工程现有 CMake 配置；协调器和 worker 必须一起更新，进程协议已新增 PD 消息。

```bash
cmake -S . -B build-process-tp -DUSE_CUDA=ON -DBUILD_PROCESS_TP=ON -DCUDA_ARCH=native
cmake --build build-process-tp --target fastllm_pd_core fastllm_tp_generate fastllm_moe_generate fastllm_tp_worker -j 4
```

构建还需要 CMake ≥ 3.18、与运行服务相同 Python 的开发头文件，以及 pybind11 头文件；
优先使用已有子模块，也可使用所选 Python 环境中安装的 pybind11。
`fastllm_pd_core` 生成 `build-process-tp/_pd_core.so`，源码入口会自动查找该产物。
其他构建目录通过 `FT_PD_CORE_LIBRARY=/绝对路径/_pd_core.so` 指定。
原生模块缺失时明确报错，不会运行旧 Python 核心实现。

要求 Linux、同一主机上的 NVIDIA GPU、CUDA IPC，以及所选 GPU 对的 P2P 能力。
服务进程需要能看到传输涉及的源与目标 GPU；通过 `--tp` 指定本服务的计算卡，
不要使用互不相交的 `CUDA_VISIBLE_DEVICES` 隐藏对端卡。
不支持 P2P 时明确失败，当前不通过 CPU 中转。

## 启动示例

以下是四卡、未量化 Qwen3 的验证部署示例，**不是已验证成功的部署结果**。
先设置 `FT_PD_TOKEN` 环境变量；每个服务和 Router 使用同一个内部令牌。
`MODEL_PATH` 指向本地模型目录。默认模型指纹会读取配置和权重；`--pd-model-id`
可指定可信的部署版本标识，使用该选项时部署者必须保证所有服务权重一致。

```bash
export MODEL_PATH=/models/Qwen3
export PYTHONPATH="$PWD/tools${PYTHONPATH:+:$PYTHONPATH}"
export PD_RUN_DIR="$PWD/test/distributed/results/$(date -u +%Y%m%dT%H%M%SZ)_pd_deploy_$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
mkdir -p "$PD_RUN_DIR"

nohup python3 -m fastllm_pytools.cli serve "$MODEL_PATH" \
  --tp_backend process --tp 0 --role p --instance p0 --port 8100 \
  --tp-binary "$PWD/build-process-tp/fastllm_tp_generate" \
  --tp-worker "$PWD/build-process-tp/fastllm_tp_worker" \
  --tp-output "$PD_RUN_DIR/p0" --prefix-cache true \
  >"$PD_RUN_DIR/p0.stdout.log" 2>"$PD_RUN_DIR/p0.stderr.log" </dev/null &
echo "$!" >"$PD_RUN_DIR/p0.pid"

nohup python3 -m fastllm_pytools.cli serve "$MODEL_PATH" \
  --tp_backend process --tp 1,2 --role d --instance d0 --port 8200 \
  --tp-binary "$PWD/build-process-tp/fastllm_tp_generate" \
  --tp-worker "$PWD/build-process-tp/fastllm_tp_worker" \
  --tp-output "$PD_RUN_DIR/d0" --prefix-cache true \
  >"$PD_RUN_DIR/d0.stdout.log" 2>"$PD_RUN_DIR/d0.stderr.log" </dev/null &
echo "$!" >"$PD_RUN_DIR/d0.pid"

nohup python3 -m fastllm_pytools.cli router "$MODEL_PATH" \
  --prefill http://127.0.0.1:8100 --decode http://127.0.0.1:8200 \
  --kv-mode pull --port 8000 \
  >"$PD_RUN_DIR/router.stdout.log" 2>"$PD_RUN_DIR/router.stderr.log" </dev/null &
echo "$!" >"$PD_RUN_DIR/router.pid"
```

推送验证使用独立 Router 端口和 `--kv-mode push`。P/D 传输模式随交接身份指定。
多轮回传要求 Router、P、D 都设置 `--pd-return-kv`；D 可用 `--pd-retention` 调整 480 秒保留期。
`--pd-long-threshold` 默认 512，`--pd-growth-budget` 默认 512，两者互相独立。
两端 `--cuda-graph`、前缀缓存状态、模型精度及采样设置应与对照组一致。
采样对照使用与 D 相同的并行模式：D 沿用该模式原有的 RNG，普通 TP 与 Attention DP/EP
的既有 RNG 不同，不能仅凭相同 seed 要求两种模式的随机采样序列一致。

完整推理对照服务可用第四张卡启动：沿用上述服务命令，用
`env CUDA_VISIBLE_DEVICES=3,0,1,2 python3 ...` 保留四卡可见并把第四张卡排在首位，参数改为
`--role full --router-managed --tp 0 --instance full0 --port 8300`，
再启动 `router "$MODEL_PATH" --worker http://127.0.0.1:8300 --port 8001`。
不带 `--router-managed` 的原有完整 `serve` 入口保持原路径。

## 并行模式与部署矩阵

Qwen2 MoE 普通 DP×TP 显式使用 `--moe-parallel-mode tp`，协调器为
`fastllm_tp_generate`。Attention DP/EP 使用 `--moe-parallel-mode ep --attention-dp N --ep N --tp 1`，
协调器为 `fastllm_moe_generate`。`auto` 保留原有 Qwen2 MoE 自动选择 EP 的行为。
普通 DP 的 `--dp >1` 时，`--tp` 表示每个副本的卡数；Attention DP/EP 中 N 支持 1、2、4。

例如同一台四卡机器的 Qwen2 MoE 混合验证配置：P 使用原可见顺序的
`--moe-parallel-mode tp --tp 0,1`；D 使用
`CUDA_VISIBLE_DEVICES=2,3,0,1` 与 `--moe-parallel-mode ep --attention-dp 2 --ep 2 --tp 1`。
D 的 `--tp-binary` 改为 `fastllm_moe_generate`，其余角色、令牌、实例和日志参数与上例相同。
两个服务都保留对方 GPU 可见；不能给 D 只暴露 2、3 两张卡。

以下每行分别验证 pull/push、Eager/Graph、Prefix off/on，并使用同一 Qwen2 MoE 权重。
每组执行独立脚本和结果目录，表格是待运行矩阵，不是通过记录。

| P | D | 重点 |
| --- | --- | --- |
| DP1×TP1 | DP1×TP1 | 基本首 Token、长短阈值 |
| DP1×TP2 | DP1×TP2 | 同拓扑 |
| DP1×TP1 | DP1×TP2 | KV 切分 |
| DP1×TP2 | DP1×TP1 | KV 汇集 |
| DP1×TP2 | DP1×TP4 | 更多目标分片 |
| DP1×TP4 | DP1×TP2 | 更多来源分片 |
| DP2×TP1 | DP1×TP2 | DP 副本选择与 TP 映射分离 |
| 普通 DP1×TP2 | Attention DP2/EP2 | 分页→连续 |
| Attention DP2/EP2 | 普通 DP1×TP2 | 连续→分页 |
| Attention DP2/EP2 | Attention DP2/EP2 | 独立 EP 组、不同请求联合轮次 |

TP4 组合要求足够的独立 GPU；`check_ipc.py` 的逻辑分片测试不能替代模型 TP4 部署。

## 验证

CPU 协议脚本与 GPU 正式入口脚本分开运行；下面的 launcher 使用独立进程会话，
保存 supervisor/test PID、完整命令、stdout、stderr 和最终退出码，不依赖 SSH 保持连接。

```bash
python3 test/distributed/pd/launch.py protocol
python3 test/distributed/pd/launch.py router
python3 test/distributed/pd/launch.py ipc -- --devices 0,1
python3 test/distributed/pd/launch.py generation -- \
  --router http://127.0.0.1:8000 --reference http://127.0.0.1:8001 \
  --profile pull_p_tp1_d_tp2_graph0_cache1 \
  --prompt-token-ids /path/to/fixed_prompt_token_ids.json
python3 test/distributed/pd/launch.py lifecycle -- \
  --prefill http://127.0.0.1:8100 --decode http://127.0.0.1:8200 \
  --router http://127.0.0.1:8000 --mode pull \
  --prompt-token-ids /path/to/fixed_prompt_token_ids.json
```

固定输入文件是本模型合法 Token ID 的 JSON 数组；测试按指定长度构造相同输入传给两边，
覆盖 1、128、511、512、513 Token、流式/非流式、首 Token 和固定种子采样。
每个部署组合单独运行。部署日志必须与本次测试结果一起保留。
生命周期脚本要求隔离测试部署，无其他客户端；`--mode` 必须与 Router 的传输模式一致。

读取本次结果目录的 `exit_code` 与 `summary.json`，不能把“后台已启动”当作通过。
报告同时输出 Markdown、CSV、JSON、XLSX；首次分歧和原始输出保存为单独 evidence JSON。
这些脚本使用真实输出 `exact`，失败返回非零；公开接口没有导出 logits，因此不会伪造
`top5`、`top5_near_tie` 或 numeric 通过结果。numeric 告警规则不能覆盖 exact 的 FAIL。

性能脚本在计时外读取 P、D、完整对照服务的实际配置和执行计数：

```bash
python3 test/distributed/pd/launch.py performance -- \
  --router http://127.0.0.1:8000 --reference http://127.0.0.1:8001 \
  --profile pull_p_tp1_d_tp2_graph0_cache1 --cache-state warm \
  --service http://127.0.0.1:8100 --service http://127.0.0.1:8200 \
  --service http://127.0.0.1:8300 --prompt-token-ids /path/to/fixed_prompt_token_ids.json
```

`--cache-state off` 要求两边关闭前缀缓存；`warm` 要求两边开启，且预热不计时。
记录 Median/P95、CV、TTFT、Token 间隔、吞吐、原始样本和相对完整服务的耗时比。
健康/证据接口提供实际完成的 GPU 拷贝批次、字节数、CUDA event 耗时总和；并行 stream
的耗时总和不能当作请求墙钟时间。真实重叠比例必须另外采集服务器 GPU 时间线。

完整异常矩阵还需要服务器故障演练：在隔离测试部署中终止某个 rank，观察相关组停止、
其他独立组继续；取消/断连发生在等待空间、逐层传输、Decode 和回传各阶段；同时记录
请求、GPU 页池、在途访问与释放证据。不得把控制面桩测试当作这些场景已通过。

## 控制面与退出

内部 HTTP 使用 Bearer 令牌，仅传请求、meta 和结果；KV 不经过 Router 或 CPU 消息体。
Router 的 `/internal/workers/register` 接受 `role`、`url`，注销接口接受 `url`；
注销会停止继续探测，暂时不健康则继续周期探测。

关闭服务时先拒绝新请求，保留控制接口用于完成通知及读取租约归还，再释放原生进程。
若无法确认在途访问结束，服务保持隔离并记录错误，不因为超时就复用 KV。
跨进程异常退出、GPU 故障及失联租约的完整恢复还没有验收，不能据此宣称无泄漏。
