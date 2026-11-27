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
