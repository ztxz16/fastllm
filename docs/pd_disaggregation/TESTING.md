# PD 分离测试命令

在仓库根目录执行。启动器会自动后台运行测试。

## 测试步骤编号

沿用最初确认的 13 步编号。实际执行顺序可以调整，但提前执行某项时，仍使用它在原表
中的编号，并说明当前子项、部署配置和通过标准；不能根据本次执行先后重新编号。
`PD-Vxx` 是 TODO 中独立验收对象的标识，不等于这里的步骤号。当前状态统一见
[项目 TODO 主表](../../project/TODO.md#pd-分离)。

| 原步骤 | 测试项目 | 入口或范围 |
| --- | --- | --- |
| 1 | 编译与加载 | 原生扩展、模型执行器与实际加载产物 |
| 2 | 协议与逻辑映射 | `protocol`，41 个 case |
| 3 | Router 重试边界 | `router`，20 个 case |
| 4 | 真实 GPU 传输 | `ipc`，24 个 case |
| 5 | 非 PD 对照基线 | 完整推理服务；保存输入、输出和配置 |
| 6 | PD 生成正确性 | `generation`，每部署组合 15 个 case |
| 7 | 并行模式组合 | 更换 TP/DP/Attention DP/EP 与 KV 布局组合，重复生成验证 |
| 8 | 取消与资源释放 | `lifecycle`，每种传输模式 7 个 case |
| 9 | 真实服务管理 | 策略分发、注册/注销、健康恢复和会话绑定；需补集成用例 |
| 10 | 回传与多轮复用 | D→P、实际复用量、480 秒过期与在途保护；需补相应断言 |
| 11 | 满载、背压与恢复 | 容量不足、排队、抢占、重算；需补压力用例 |
| 12 | 真实故障隔离 | rank 退出、服务重启、迟到消息和传输异常；需补故障注入 |
| 13 | 性能与重叠 | `performance`，6 个 case、每项 5 轮；重叠另采 GPU 时间线 |

## 四卡机器的第 7 步：先录制基线，再执行 PD

2026-11-27 用户提供的服务器拓扑仅支持 GPU 0↔1、2↔3 的 P2P；两组之间不支持。
当前 PD 后端没有 CPU 中转。第 7 步先尝试普通 `DP1×TP2 → DP1×TP2`：

| 阶段 | 部署 | 检查范围 |
| --- | --- | --- |
| 准备基线 | 完整服务使用 GPU 1、3；普通 TP2，Graph/Prefix 关闭 | 完整服务的跨 NUMA TP 通信和实际生成；录制 15 个用例的输出 |
| PD pull | P 的 rank 0/1 使用 GPU 0、2；D 的 rank 0/1 使用 GPU 1、3 | 对应分片分别走 0→1、2→3；实际生成与 TP2 离线基线 exact 对照 |
| 后续组合 | 当前组合通过后再单列 push、其他执行配置及并行模式 | 不由某个小规模组合通过推断全部组合通过 |

上述 GPU 编号均为物理 GPU；服务内逻辑编号取决于 `CUDA_VISIBLE_DEVICES` 顺序。
基线阶段先停止旧 TP1 模型服务，保留独立 Router。基线录制成功后再停止完整服务，
把四张卡交给 P、D；不能同时保留一个额外模型实例，也不能拿已有 TP1 基线代替 TP2。
同一 TP 组内的 GPU 跨 NUMA，必须通过实际完整推理确认其通信路径，不能仅根据
两条 P→D 的 P2P 连通关系宣称 TP 通信可用。此部署尚未验收。

`generation` 增加两个互斥的对照来源，原有在线 `--reference` 行为保留：

- 录制：`--record-reference --reference <完整服务 Router>`，不指定 `--router`。
  仍运行 15 个原有 case，共保存 21 次实际请求与响应（并发、三轮对话分别展开记录）。
  仅在完整录制通过后生成 `reference.json`；失败时的 `reference.incomplete.json` 不可用于对照。
  录制 PASS 只表示输出长度、执行完整性和服务状态符合要求，不报告 PD exact PASS。
- 离线比较：`--router <PD Router> --reference-record <录制目录/reference.json>`。
  保持相同 15 个 case；只省去在线完整服务调用，PD 仍通过正式 Router 和原生执行器运行。
  校验模型标识、服务/副本数量、TP、存储布局、分页大小、Graph、Prefix 和自定义 AllReduce
  配置；逐请求校验 Prompt ID、生成长度、采样及 EOS 参数。只有每次运行新建的会话编号
  不要求与录制时相同，多轮历史与本轮会话连续性仍由原用例保留。
- 记录本次部署证据和基线文件摘要，拒绝不完整或报告不匹配的基线；运行中服务重启或
  配置变化判 FAIL。离线基线模式只比较正确性，不能把两个时间段的延迟直接作为性能对比。

以下是测试入口模板，必须先完成对应部署；启动器仍自动后台运行并记录退出码：

```bash
# 阶段一：完整 TP2 服务已就绪，保存基线。
python3 test/distributed/pd/launch.py generation -- \
  --reference http://127.0.0.1:8001 --record-reference \
  --profile full_tp2_graph0_cache0 \
  --prompt-token-ids "$PD_STEP6_DIR/prompt_token_ids.json"

# 阶段二：基线通过并停止完整服务、启动 TP2 P/D 后执行。
# PD_TP2_REFERENCE 指向上一条命令实际输出目录中的 reference.json。
python3 test/distributed/pd/launch.py generation -- \
  --router http://127.0.0.1:8000 \
  --reference-record "$PD_TP2_REFERENCE" \
  --profile pull_p_tp2_d_tp2_graph0_cache0 \
  --prompt-token-ids "$PD_STEP6_DIR/prompt_token_ids.json"
```

本次仅补充 Python 测试组织与证据保存，不修改 C++/CUDA 的路由、KV 传输或推理逻辑。
实现位置为 `check_generation.py` 和 `generation_reference.py`；本地不编译，服务器基线
录制及 PD exact 对照分别验收，当前状态见 TODO 的 PD-V16、PD-V17。

## 基础测试命令

```bash
python3 test/distributed/pd/launch.py protocol
python3 test/distributed/pd/launch.py router
python3 test/distributed/pd/launch.py ipc -- --devices 0,1
```

## protocol 测什么

**`protocol` 是一组 41 个 CPU 回归用例，检查 PD 协议、KV 映射、路由及资源生命周期。**
当前代码已迁移：Python 组织用例、构造输入和断言，通过原生接口调用正式 C++ 核心；
MoE 并行配置选择仍调用 Python 配置入口。传输后端使用模拟完成事件，不加载模型，不执行真实 GPU 拷贝。
以下范围按 2026-11-27 的 `9849a51f` 核对，入口见
[check_protocol.py:258](../../test/distributed/pd/check_protocol.py#L258)。

| 检查项 | 用例数 | 实际检查内容 | 用例位置 |
| --- | ---: | --- | --- |
| TP 分片映射 | 12 | TP 1→1、2→1、1→4、4→2、2→4、8→1，各测 push / pull；目标的每个 rank、层、K/V、Token、Head 都有对应来源，不能漏写或重复写 | [check_protocol.py:58](../../test/distributed/pd/check_protocol.py#L58)、[check_protocol.py:263](../../test/distributed/pd/check_protocol.py#L263) |
| 八种 Router 策略 | 8 | 每种策略能从两个健康 P 实例中选出目标；重复释放同一租约不能重复扣减在途计数 | [check_protocol.py:128](../../test/distributed/pd/check_protocol.py#L128)、[check_protocol.py:266](../../test/distributed/pd/check_protocol.py#L266) |
| MoE 并行配置 | 6 | 默认配置、显式 TP1 / TP2 / DP2 / EP2 选择预期执行模式；冲突配置拒绝 | [check_protocol.py:247](../../test/distributed/pd/check_protocol.py#L247)、[check_protocol.py:267](../../test/distributed/pd/check_protocol.py#L267) |
| 传输完成与取消 | 2 | 缺一层时不能报告 READY；取消后等待在途任务完成再释放资源，重复取消只能释放一次 | [check_protocol.py:91](../../test/distributed/pd/check_protocol.py#L91)、[check_protocol.py:111](../../test/distributed/pd/check_protocol.py#L111) |
| 历史 KV 与容量 | 3 | 480 秒保留过期后拒绝新读取，但保护已有读者；关闭时等读者退出；容量准入计入增长预算，占用未解除时禁止回收 | [check_protocol.py:153](../../test/distributed/pd/check_protocol.py#L153)、[check_protocol.py:177](../../test/distributed/pd/check_protocol.py#L177)、[check_protocol.py:234](../../test/distributed/pd/check_protocol.py#L234) |
| 健康、缓存与重试 | 4 | 连续失败 3 次摘除、成功 2 次恢复；旧实例探测不能恢复新实例；相同前缀保持亲和，恰好 30% 不算超过 0.3；503 允许重试，新尝试换交接 ID，响应客户端后禁止重试 | [check_protocol.py:139](../../test/distributed/pd/check_protocol.py#L139)、[check_protocol.py:168](../../test/distributed/pd/check_protocol.py#L168)、[check_protocol.py:187](../../test/distributed/pd/check_protocol.py#L187)、[check_protocol.py:204](../../test/distributed/pd/check_protocol.py#L204) |
| 元数据与非法输入 | 6 | JSON 往返一致；拒绝目标范围缺失、重复覆盖、请求 ID 不同、模型不同、交接 ID 不同；交接 ID 错误必须在提交传输前拒绝 | [check_protocol.py:213](../../test/distributed/pd/check_protocol.py#L213)、[check_protocol.py:280](../../test/distributed/pd/check_protocol.py#L280) |
| **合计** | **41** | **控制逻辑与映射回归** | [check_protocol.py:285](../../test/distributed/pd/check_protocol.py#L285) |

### 哪些是真检查，哪些是模拟

- **映射规划是真代码，GPU 数据是模拟描述。** 输入使用 2 层、4 个 KV Head、4 个 Token 的小布局，
  源端分页、目标连续；检查逻辑坐标覆盖，不分配显存或比较拷贝后的 KV 数值。TP8 也是逻辑布局，
  不需要八张卡。见 [check_protocol.py:43](../../test/distributed/pd/check_protocol.py#L43)、
  [mapping.py:23](../../tools/fastllm_pytools/pd/mapping.py#L23)。
- **取消与完成判断调用正式管理逻辑，底层完成事件是模拟的。** 测试主动设置事件，检查资源是否提前释放、
  重复释放；不会调用 CUDA IPC。见 [check_protocol.py:74](../../test/distributed/pd/check_protocol.py#L74)。
- **480 秒通过假时钟推进到 481 秒验证**，不用真的等待。重试项检查是否允许重试及交接 ID，
  不发送真实 HTTP 请求。见 [check_protocol.py:153](../../test/distributed/pd/check_protocol.py#L153)、
  [check_protocol.py:168](../../test/distributed/pd/check_protocol.py#L168)。
- **MoE 六项只测配置选路**，属于共用配置的回归检查，不启动 TP/EP 执行器。
  八种 Router 策略的八项只覆盖基本选取与租约释放，不代表策略全部语义、分配公平性或负载性能已验证。
  见 [check_protocol.py:247](../../test/distributed/pd/check_protocol.py#L247)、
  [check_protocol.py:128](../../test/distributed/pd/check_protocol.py#L128)。

因此，41 项通过只能说明**这些输入下的协议、映射和状态处理符合断言**。
真实 CUDA IPC、GPU 同步与数值、模型生成、性能仍需分别验收。

### 运行前提与结果

当前版本需要服务器上已构建的 `_pd_core.so`。缺少原生扩展会直接报错，不会回退到 Python 核心，
也不会由此命令自动编译。加载规则见 [_core.py:20](../../tools/fastllm_pytools/pd/_core.py#L20)，
服务器构建说明见 [原生迁移记录](NATIVE_MIGRATION.md#构建与打包)。

启动器自动后台运行，在 `test/distributed/results/` 下生成独立目录，保存命令、PID、
`stdout.log`、`stderr.log` 和 `exit_code`。启动成功不等于测试通过：完成后先查退出码和错误日志，
再看 `summary.md` / `summary.json` / `summary.csv` / `summary.xlsx`；任一用例失败返回非零。
见 [launch.py:16](../../test/distributed/pd/launch.py#L16)、
[launch.py:32](../../test/distributed/pd/launch.py#L32)、
[report.py:18](../../test/distributed/pd/report.py#L18)。

2026-11-26 回传的 41/41 PASS 属于原生迁移前的版本，不能替代当前 C++ 版本复验。
当前状态见 [PD 原生迁移待办](../../project/TODO.md#pd-01)。
