# PD 测试跨会话交接

交接日期：2026-11-28。供同一项目的新 chat 接续服务器测试，不需要复制历史对话。
**当前验收状态只维护在 [TODO 主表](TODO.md#状态总览)**；本文是操作上下文和有日期的交接快照，不是第二份验收清单。后续接手时先读根目录 AGENTS.md，再查 TODO 的 PD-07、PD-V61/V62/V63 及各项详情。

## 接手后的第一件事

**最新接续（2026-11-28，逐级接受规则更新后）**：最小复验、普通 TP 完整服务对照及
原生 Logits 诊断均已结束，不重复提交。用户已将生成验收改为 exact → top5 →
top5_near_tie，任一通过即接受，未通过单项 WARNING，不再执行 numeric。
两条原生贪心对照已按 Top5 接受；当前待补齐原采样用例候选证据与独立运行核查，
具体状态及离线重判证据见 TODO 的 PD-V62、PD-E02、QA-02。

最后一次服务器诊断目录为
`test/distributed/results/20261128T124641Z_pd_v62_logits_16089634`。
D/full 已在该轮停止后恢复，后续身份核查使用其中 `deployment.restored.json`、
`restore_d.command.json`、`restore_full.command.json` 及对应 PID；旧 D/full 启动身份
已失效，不能重跑下文旧启动或对照命令。新任务前仍须核对服务器当前任务与服务状态。
后续直接给完整可执行命令，不新增或改写 `.sh`；下面脚本仅归档此前已执行命令。

**历史接续（最小复验结束时；下文旧验收标准和下一步建议均为历史记录）**：目录
`test/distributed/results/20261128T115530Z_pd_v62_replay_191dee94`，退出码 1，
stderr 为空。6 次场景全部 exact FAIL，共 12 次请求中 6 次 exact FAIL、
6 次 PASS；12 次输出均与各自原始 PD actual 完全一致。不要再次执行最小复验
启动命令，也不要把该结果理解为需要等待的后台任务。

- 采样两轮均首差 output 3，分别在 D 副本 1、0，均复现原输出。
- request3 单独执行两轮均首差 output 6，分别在 D 副本 1、0，均复现原输出。
- 并发 4 两轮均只有 request3 失败，首差 output 6，目标副本均为 0；
  request0/1/2 均 exact PASS，路由为 [1,1,0,0]。
- 本次复现无需并发，且两个 D 副本均出现相同结果；不支持“仅并发触发”或
  “仅某一 D 副本异常”的解释。仍未定位普通 TP/Attention 计算、KV 搬运或
  离线基线影响，不能把稳定复现判为数值误差已证实。
- 下一步建议：先核对 GPU1 / 8001 的原普通 TP 完整服务身份、配置与空闲，
  使用原采样和原 request3 的完整请求参数各两轮作诊断对照（4 次请求），
  分别与旧 Attention reference、原 PD actual 比较。该对照不替代正式
  Attention 基线，不修改原 FAIL；新对照命令尚未准备，也未提交请求。
  如需当下 Attention 非 PD 对照或 KV 内容核查，再据结果明确下一步。

**此前只读快照（2026-11-28 用户回传）**：目录
`test/distributed/results/20261128T114741Z_pd_v62_status_text_68a8a9d9` 的输出，
主机名为 `557aa3be03a7`，采集 `errors=[]`。原生成及总流程仍为退出码 1，
当时没有新复验结果。P=20006、D=20007、Router=20008 及对应监护进程
20003/20004/20005 均在进程快照中；原测试/监护 PID 19964/19962、
生成测试/监护 PID 20306/20305 未出现在匹配列表中。结合原退出码，测试已结束；
同一个运行脚本的服务监护进程仍在不代表测试仍在运行。快照只证明进程存在，
未检查服务健康、空闲或启动身份，不能直接据此放行新测试。

两条失败响应已提取：采样为“中国北部”与“中国东北部”，含义不完全相同；
并发 request3 的后半段为 `北京，简称“京` 与 `北京，简称京，`，
可见差异主要是引号/逗号。双方各生成 8 Token，字符长度及
句尾不同不能另外判漏字，也不能据短片段评估完整回答质量。
用户随后询问下一步，曾建议按现行 exact 规则进行三种场景各两轮最小复验，
先做服务身份/健康/空闲核查，不重跑旧切换命令、不进入 PD-V63。
已准备 [最小复验后台启动命令](pd_test_commands/PD-V62_launch_minimal.sh)：
使用独立启动锁、已有复验目录检查及公共 PD 测试锁防止重复提交，复制已存在的
复验脚本到新目录，后台记录 PID、完整命令、日志和退出码。脚本没有停止或启动
模型服务的操作；复验报告保留 Markdown/CSV/JSON/XLSX，原生成元信息另行复制。
命令可直接粘贴到服务器执行，不需要先同步新增的启动包装文件；依赖的复验
Python 脚本已在此前同步的 master 中。该命令现已由用户执行并回传上述结果；
本次代理未连接 GPU 服务器、未代跑测试。

2026-11-28 用户已经回传 **PD-V62 失败结果**：13/15 case PASS；固定种子采样首差索引 3、并发 4 汇总首差索引 6（索引从 0 开始），生成和总退出码均为 1。运行核查在生成门槛处中止，未完成后续验收。前一项 PD-V61 的成功结论保留。

1. 已完成已有证据提取：用户诊断目录 `test/distributed/results/20261128T111230Z_pd_v62_exact_diag_e29698a5`。并发 4 仅 request3 失败；第 7 个输出 reference=2073、actual=46553，D 副本 1；同副本 request0 通过，request1/2 在副本 0 通过。四条记录均 D completed/released、P released、error 为空；pull 的 P completed=false 不单独判错。
2. 采样第 4 个输出 reference=106758、actual=104756。非流式响应未提供请求 ID，不能猜副本。并发失败使用 temperature=0，所以不能把两处都归因为随机采样。
3. 验收口径讨论：用户询问差异是否很大，并提出“语义一样是否还需要 exact”。已收到上述响应文本；并发片段主要是标点差异，采样存在方位描述变化，不能把两处都归为语义相同。按位置比较，采样为 8 个输出中后 5 个不同，并发 request3 为后 2 个不同；不将这些计数解释为独立错误数量或质量分数。没有变更验收口径；最新下一步建议和待执行命令见上文。
4. [最小复验脚本](pd_test_commands/PD-V62_replay_minimal.py) 为三种场景各两轮（6 次场景/12 次请求），现已完成，结果见最上方最新接续。保留现行 exact 结果与原失败证据，不把“语义可能相同”写成验收成功。

本次失败目录：`test/distributed/results/20261128T110052Z_pd_tp1_ep2_pull_5ac24694`；生成目录：`test/distributed/results/20261128T110401Z_pd_generation_d98fff95`。原脚本在生成门槛前保存了 `services.after_generation.json`，本次没有通过后的 `deployment.after.json`，不要误用 PD-V61 的快照代替。

版本核对：原基线报告 Git 为 `6ac3c9101ab45dc8fc8e5ff8c18c6e56dfe01bf3`，失败报告为 `67bcb885b3dc2c2e57a44192fcb78f2d0653cf46`，测试进程加载的 `_pd_core.so` 摘要不同。已 fetch 并逐文件比较这两个提交：功能源码只将证据接口的 usage 查询前移到 cache 之前，其余是测试/文档；模型计算与采样源码未变化。因此版本标签不同不是已定位根因；报告中的扩展摘要也不等于服务器所有模型进程产物的完整版本证明。

新 chat 不应假设自己能直接访问用户 GPU 服务器。此前由代理本地准备/静态检查命令，用户在服务器执行并回传报告。服务器路径即使与本地相同，也不证明是同一台机器。

## 本次交接位置与范围

- 原计划 **第 7 步 = 并行模式组合**；PD-V61 等是独立验收编号，不是“第 61 步”。
- PD-V61：Attention DP2/EP2 → 普通 TP1，push，Eager / Prefix 关闭。用户回传 15/15 actual Token ID exact PASS，生成与总退出码均 0，独立 `ep2_tp1_push_runtime` PASS。完整证据与当前结论见 [PD-V61](TODO.md#pd-v61)。
- 已约定本轮四卡基础方向收尾为 [PD-V62 pull](TODO.md#pd-v62) 和 [PD-V63 push](TODO.md#pd-v63)：普通 TP1 → Attention DP2/EP2，各 15 个生成 case、1 项运行核查。15 个 case 内展开 21 次请求，不要把两种计数混用。
- 这两组之后不自动追加全部 Graph/Prefix 交叉矩阵；扩展配置、规模、策略仍在 TODO 独立待验。本轮收尾不等于第 7 步全量范围或整体交付通过。
- 第 8 步 TP1、Graph/Prefix 关闭的 pull/push 生命周期各 7 个 case 已有用户验收记录。后续按原表范围继续，不能重新编号或将未覆盖范围记为通过。

PD-V61 服务器目录（路径和数据来自用户回传，不是本地执行结果）：

```text
/home/project/vllm_fastllm/test/distributed/results/20261128T105105Z_pd_ep2_tp1_push_be393099
/home/project/vllm_fastllm/test/distributed/results/20261128T105111Z_pd_generation_757c2054
```

该轮 P 两个副本请求数 [10,11]，D/full 各 21 次请求；D Eager Decode +161，full +140；P 推送 504 批、1290338304 字节。服务不变、配对、释放及连续→分页 KV 核查通过。

## 服务器环境与两个部署快照

用户服务器：C.62948395；项目 `/home/project/vllm_fastllm`；Python `/venv/main/bin/python3`；CUDA Toolkit 12.6，GCC 13.3，Python 3.12.16。模型 `/home/models/Qwen1.5-MoE-A2.7B-Chat`，模型类型 qwen2_moe。

2026-11-28 用户回传新机器 **4 × V100-SXM2-32GB，两两 NV2，P2P 读写全部 OK**。旧机器仅 0↔1、2↔3 可达的限制不适用于本次机器；重新安装或换机后仍应以现场为准。

| 物理卡 | 新机器 UUID |
| --- | --- |
| 0 | GPU-0faee1cf-7bfa-5d12-997d-eb1a0f02f325 |
| 1 | GPU-5451d6d1-e24d-7cc3-d557-7f5cc0e73625 |
| 2 | GPU-c1fa1353-8a10-b12f-5937-206aacfc941d |
| 3 | GPU-9fcde866-591c-050f-6330-82250114d9b0 |

**PD-V61 结束时的已确认部署**，不是对当前在线状态的承诺：

| 服务 | 端口 | 物理 GPU | 配置 |
| --- | --- | --- | --- |
| P | 8100 | 0、2 | Attention DP2/EP2，Attention TP1，load，连续 KV |
| D | 8200 | 3 | 普通 TP1，分页 KV，page 128 / tokens 2048 |
| full | 8300 | 1 | 普通 TP1，同 D 配置的在线对照 |
| PD Router | 8000 | 无模型 | push，登记 P 8100、D 8200 |
| reference Router | 8001 | 无模型 | 登记 full 8300 |

上述模型服务 FP16、Eager，Prefix/回传关闭。P 使用原生 MoE 执行器，D/full 使用普通 TP 执行器。

**PD-V62 首次运行已完成切换并进入生成**：P GPU3 普通 TP1；D GPU0、2 Attention DP2/EP2、Attention TP1、load；8000 为 pull。FP16、Eager、Prefix/回传关闭。失败发生在输出 exact 对照；后续现场是否仍存活需查询，不能以历史快照替代实时状态。

切换前检查旧命令、启动身份、空闲与 UUID，并通过正式参数/设备规划；然后停止旧 P、D、8000 Router，等待对应 worker 和端口释放后重启。full GPU1 和 8001 保留，但本轮不向 full 发送生成请求。

## 对照基线与输入

PD-V62/V63 使用已验收的 **Attention DP2/EP2 Eager / Prefix 关闭离线基线（PD-V44）**：

```text
/home/project/vllm_fastllm/test/distributed/results/20261128T033907Z_pd_generation_536f636f/reference.json
```

必须同时保留同目录的 `summary.json`，核对摘要、完整 21 次请求、模型标识和 D 配置；不能用普通 TP1 的在线 full 替代 Attention 对照。归档固定输入为 **6 Token**，应从归档恢复，不能沿用上一混合方向的 25 Token。边界 Prompt 仍覆盖 1、128、511、512、513，另有首 Token、固定种子、并发 2/4 和三轮历史输入。

模型标识：`5be1d21e7fb5013084e38154dc0d6a0ec7a9146cc1dc78ef0c2658e73ee22027`。

## 已给出命令的持久化归档

[PD-V62 启动命令](pd_test_commands/PD-V62_tp1_ep2_pull.sh) 保存首次运行的完整 Shell/Python 内容，用于追溯。**用户已执行并切换服务，不要再次运行此旧启动命令**。[已有证据提取命令](pd_test_commands/PD-V62_collect_exact.sh) 对应已完成的证据提取；状态与响应文本只读提取也已完成，当前接续见本文最上方。命令已做静态检查，没有代理本地编译/GPU 运行结果。

该命令仍依赖服务器历史归档；保存命令不等于已备份服务器结果。依赖包括：

- PD-V61 总目录：退出码、运行核查、`deployment.after.json`、五个服务的 `*.command.json`、`helpers.source.py`。
- PD-V44 生成目录：上述 `reference.json` 和配套报告。
- `test/distributed/results/20261128T074703Z_pd_ep2_prefix_pull_cce92abb/` 下的 `source.helpers.source.py` 与 `source.run.py`，供已有助手抽取查询/停止/启动函数。
- 现有原生产物 `_pd_core.so`、`fastllm_tp_generate`、`fastllm_moe_generate`、`fastllm_tp_worker`，位于 `build-process-tp/`。

依赖丢失时先恢复归档或重新核实现场，不通过删除校验继续执行。不执行旧结果目录的 `command.sh` 来“看看”，它可能再次启动测试或切换服务。

正式生成使用 [launch.py:16](../test/distributed/pd/launch.py#L16) 后台入口；参数定义见 [check_generation.py:89](../test/distributed/pd/check_generation.py#L89)，部署/离线基线核查见 [generation_reference.py:9](../test/distributed/pd/generation_reference.py#L9)。后续引用前重新核对行号。

## 只读恢复进度

重新 SSH 后临时环境变量可能不存在，按目录发现结果。下面命令只查看报告，不发生成请求，不改服务；在用户服务器执行：

```bash
cd /home/project/vllm_fastllm
python3 - <<'PY'
import json
from pathlib import Path

root = Path('test/distributed/results').resolve()
runs = sorted(p for p in root.glob('*_pd_tp1_ep2_pull_*')
              if (p / 'command.sh').is_file())
if not runs:
    raise SystemExit('未找到 PD-V62 运行目录；先核对是否已启动，不要直接重启服务')
run = runs[-1]
print('本次目录：', run)
for name in ('stdout.log', 'stderr.log', 'p.stderr.log', 'd.stderr.log', 'router.stderr.log'):
    path = run / name
    if path.is_file():
        print(f'\n{name}：\n' + '\n'.join(path.read_text(errors='replace').splitlines()[-40:]))
launch = run / 'generation.launch.json'
if launch.is_file():
    try:
        generation = Path(json.loads(launch.read_text())['directory'])
    except (ValueError, KeyError) as error:
        print('生成启动记录尚不完整：', error)
    else:
        print('\n生成目录：', generation)
        for name in ('exit_code', 'summary.md', 'stderr.log'):
            path = generation / name
            if path.is_file():
                print(f'\n生成 {name}：\n{path.read_text(errors="replace")}')
for name in ('exit_code', 'summary.md', 'runtime_check.json'):
    path = run / name
    if path.is_file():
        print(f'\n总流程 {name}：\n{path.read_text(errors="replace")}')
if not (run / 'exit_code').is_file():
    print('\n总流程尚未记录结束，结合 PID 与日志判断状态。')
PY
```

## PD-V63 准备原则与已遇到的坑

PD-V62 验收后，V63 沿用同一 P/D、Attention 离线基线和输入，只将 8000 从 pull 改为 push，不重新加载模型。具体 V63 命令在本次交接时尚未准备，不将 V62 简单字符串替换后直接交付。

- 两层验收：生成 15/15 exact、生成退出码 0，以及独立运行核查 PASS、总退出码 0。`started` 或健康就绪不能代替验收。
- 每轮只统计新增请求和计数增量；双方各 21 次请求，D 两个 Attention 副本均实际执行。保留匹配的请求/交接 ID、配对、副本、UUID、head 范围、分页→连续布局和释放证据。
- pull 实际传输看 **D**，push 看 **P**。pull 不强制源 P 的 `completed`；push 的 D 不要求 `prefill_done`，应核对 `completed`、`released`、无错误。完整服务也不以 KV 交接 `completed` 判定输出成功。
- 已释放的旧记录会自然过期，不能用整个 evidence 字典相等判“服务不变”；比较配置、boot、worker 身份，核查本轮请求完成释放。
- `--tp 3` 表示卡数，不表示物理 GPU3。用 UUID 顺序选择物理卡，再传 `--tp 1`。P 预期可见顺序 [3,0,1,2]，D 为 [0,2,1,3]；保留四卡可见供 IPC 查询，分别校验实际 UUID。
- Attention 组有两个 rank，但每个 Attention 副本视图只有一个 worker，Attention TP=1；不要把 EP 组大小当 TP，也不要要求两个视图各有两个 worker。
- 服务退出和端口释放可能不同步；按监听端口找进程，核对启动时间/命令，等待父进程、worker、端口释放。该 Python 环境没有 `os.pidfd_open`。不要宽泛匹配后批量 kill。
- 无 PyTorch/TensorFlow/Flax 的提示曾是可选框架警告，不能单凭它判失败；但缺 `transformers` 会阻止 Tokenizer/Router 启动。

## 后续维护与交付边界

收到用户结果后，先更新 TODO 主表与原编号详情，注明用户服务器证据来源；保留首次 FAIL 和修正后的结果。再更新本文的接手位置、部署快照、下一步以及命令归档。不要因为换 chat 重跑已验收矩阵。

默认本地只做静态检查，服务器测试后台运行，独立目录、PID/命令/日志/退出码齐全；表格报告需 Markdown/CSV/JSON/XLSX。严格遵守 AGENTS.md，不修改第三方代码，不擅自扩大已约定范围。

Pull 长短 Prompt 接收流程统一是 [PD-33](TODO.md#pd-33) 待讨论的新方案，本次接续不实施它；Attention 前缀缓存全链路一致性也不能由这些 PD exact 结果替代。验收细项通过与整个功能范围通过分开记录。

交接文件与命令归档不代表执行了服务器命令、编译或停止服务。相同工作区的新 chat 可直接读取这些文件；若换成另一份仓库，必须先同步包含这些文件的提交。服务器历史结果通常不会随 Git 同步，需单独保留在服务器。
