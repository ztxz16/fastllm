# 单机多副本部署：ftllm deploy

`ftllm deploy` 启动多个独立模型服务和一个流式 HTTP 路由。每个模型服务仍使用
原有 `ftllm.server`，TP/EP、KVMem、模型加载和生成行为由原有服务处理。
`ftllm serve` 的入口、默认参数和单实例行为不变。首版进程管理支持 Linux。

安装带部署依赖的包：`pip install 'ftllm[deploy]'`。源码环境可使用
`pip install -r requirements-deploy.txt`。`render` 不需要 aiohttp，不加载模型或 CUDA；
`start` 会在启动模型前检查路由依赖。

## 启动配置

例如，创建 `deployment.json`，把模型路径和参数替换为自己的设置：

```json
{
  "schema_version": 1,
  "name": "my-model",
  "state_dir": "./deploy-runs/my-model",
  "serve_args": ["/models/my-model", "--tp", "1", "--max_batch", "8"],
  "env": {},
  "replicas": [
    {"name": "replica0", "gpus": [0], "port": 8081},
    {"name": "replica1", "gpus": [1], "port": 8082}
  ],
  "router": {
    "host": "127.0.0.1",
    "port": 8080,
    "queue_limit_per_replica": 8,
    "queue_timeout_seconds": 300,
    "request_timeout_seconds": 3600,
    "affinity_ttl_seconds": 3600,
    "max_sessions": 10000
  },
  "startup_timeout_seconds": 900,
  "stop_timeout_seconds": 30,
  "sequential_start": true
}
```

以上数字是示例，不是模型部署预设。`serve_args` 是原生参数列表，不是 shell 字符串，
可原样使用现有的 `--tp`、`--ep`、`--kvmem`、`--tokens` 等选项。参数类型、别名和
默认值复用 `util.make_normal_parser()` 与 `util.add_server_args()`；只把显式参数
传给子进程，保留 `--auto` 的显式覆盖语义。为便于准确校验，参数名不接受缩写。

`--kvmem` 的 JSON 在参数列表中写为一个字符串，例如：

```json
["/models/my-model", "--max_batch", "8", "--kvmem", "{\"max_tokens\":32768}"]
```

部署文件与旧的模型启动 JSON 不是同一种格式。旧格式在模型加载时会覆盖绑定端口，
因此不能把旧启动 JSON 作为 `serve_args` 的位置参数；应把模型路径和需要的参数写入
`serve_args`。模型内部的 `config.json` 不受影响。

- 模型参数均保持原有单实例语义。两个副本各自 `--max_batch 380 --tokens 50000000`
  得到总计 760 路、100M token 池；组件不会自动把数值除以或乘以副本数后传给模型。
  这些具体实验数值应放在外部启动配置，不作为部署组件默认值。
- `gpus` 是物理编号或 CUDA GPU/MIG 标识；子进程通过 `CUDA_VISIBLE_DEVICES` 看到自己
  的设备。`--tp` 中的设备编号仍按原有“可见设备”语义解释。空列表表示不暴露 GPU。
  显式重复的 GPU 标识和重复端口会被拒绝。
- 工作目录、相对路径以部署配置所在目录为基准；`serve_args` 内的路径由原服务按该
  工作目录解释。每个副本可设置自己的 `env`，覆盖公共 `env`。
- 模型服务统一绑定回环地址；API 入口由 `router.host/port` 控制。不要在 `serve_args`
  中再传 `--host`、`--port` 或 `--startup-progress`。
- 路由每副本在途上限默认使用显式的 `--max_batch`。自动 batch 时必须配置
  `router.max_inflight_per_replica`；也可用 `replicas[].max_inflight` 限制某个副本。
  显式上限不能超过模型的显式 `--max_batch`。
- 选填 `worker.python` 可指定 Python，默认使用当前解释器。选填 `worker.script`
  可使用外部适配脚本替代 `python -m ftllm.server`，参数仍使用同一套 serve 参数。
  例如 SWE 的 token-count 适配器可以留在测试仓库中；deploy 不依赖 SWE。

## 命令与进程归属

### GPU 拓扑与 NUMA 绑定

部署配置顶层 `"numa": "auto"` 为默认值。在多 NUMA 节点的 Linux 机器上，
`start` 通过 `nvidia-smi` 查询物理 GPU 编号、UUID、PCI 地址和 NUMA 拓扑；
拓扑命令不可用时读取 PCI sysfs 的 `numa_node`。每个副本的 GPU 应位于同一节点。
随后在 Python、CUDA 初始化及 KVMem 分配前，以 `numactl --physcpubind=... --membind=N`
启动该副本。CPU 范围取目标节点与启动环境允许 CPU 的交集，内存节点也必须在
容器允许范围内。路由和 supervisor 不绑定到某个副本的节点。

自动解析后的 `CUDA_VISIBLE_DEVICES` 使用完整 GPU/MIG UUID，防止 CUDA 枚举顺序与
物理编号不一致。可以直接在 `gpus` 中填写 UUID 或唯一 UUID 前缀；MIG 标识通过
`nvidia-smi -L` 查找父 GPU 的 NUMA 节点。该过程不导入模型、PyTorch 或初始化 CUDA。

一个副本跨多个节点、拓扑未知、目标节点无可用 CPU、缺少 `numactl` 或绑定权限不足时，
启动会明确失败，不退化为只绑 CPU 或 preferred 内存策略。可重新分组 GPU，或在
`replicas[]` 中显式设置 `"numa_node": 0` 等节点编号覆盖自动判断；该编号属于机器的
部署配置，组件不预设。显式覆盖会跳过该副本的 GPU 拓扑解析，保留原 GPU 标识。
跨节点副本显式绑定单个节点时，其他节点的 GPU 仍会访问远端内存。

若需要由外部程序管理绑定，可显式设置顶层 `"numa": "off"`；不能同时指定
`replicas[].numa_node`。单 NUMA 节点或 CPU 副本默认不额外绑定，仍可显式指定节点。
`auto` 会为副本设定新的内存策略，保留启动环境的 CPU/cpuset 限制；要完整继承外部
内存策略应使用 `off`。

普通 `render` 保持离线、跨平台行为。检查当前机器实际 GPU 映射和最终绑定命令：

```sh
ftllm deploy render --config deployment.json --resolve-topology
```

该命令只查询拓扑，不分配模型内存。`start` 还会对全部副本的绑定命令执行独立小进程
探测，检查实际 CPU affinity 和 `MPOL_BIND` 内存策略，全部成功后才开始加载模型。
`manifest.json`、`status` 的 `numa` 字段及 supervisor 日志记录目标节点、CPU、GPU
映射和探测结果；模型就绪后再记录线程 CPU 范围及 `/proc/PID/numa_maps` 的分节点
驻留内存。驻留统计包含共享文件页，不等同于 KV 缓存容量，也不要求所有页都在本节点。
这些是启动时快照，不是持续采样；不向推理热路径添加日志。

硬绑定只允许在目标节点分配新内存。部署时须为每个节点分别预留 KVMem、模型加载和
运行时内存，不能把其他节点的空闲内存算作本副本容量。这个检查不替代模型自身的
内存预算；绑定变更需要重新启动副本，不能迁移已经分配的锁页缓存。

### 生命周期

```sh
ftllm deploy render --config deployment.json
ftllm deploy start  --config deployment.json
ftllm deploy status --config deployment.json
ftllm deploy stop   --config deployment.json
```

也可使用 `python -m ftllm.deploy ...`。普通 `render` 只输出参数和命令，不创建运行目录或
启动进程；`--resolve-topology` 会调用系统工具查询拓扑，不启动模型服务。
认证参数在输出中隐藏。`start` 等所有副本及路由就绪后返回，后台
supervisor 持续拥有这些进程，终端退出不影响已启动服务。

默认按顺序加载副本以减少同时加载产生的内存峰值，`sequential_start=false` 可并行
加载。就绪检查使用原生 `/v1/models`，不发送生成请求。启动失败会清理已启动的本轮
进程；服务运行中任一子进程退出会停止其他本轮进程并标记失败，不自动循环重启。
HTTP 健康检查失败会在路由状态中体现，并阻止向不健康副本分配新请求。

启动前检查 Linux `RLIMIT_NOFILE`。每个在途代理请求需要客户端、后端各一个连接，
排队请求只占客户端连接；自动预算为 `2 × 总在途容量 + 总排队容量 + 256`，
最后一项给健康检查、日志及空闲连接留出余量。独立 supervisor 会按需提高软限制，
所有模型进程和路由继承该限制；已有更高限制保持不变，不修改硬限制或调用方进程。
硬限制不足时在加载模型前报错，应在启动环境或容器中提高限制后再启动。

需要为大量空闲连接等额外开销预留空间时，可在部署 JSON 顶层显式设置
`"nofile": 65536`；这是软限制的最低目标，必须不小于上述自动预算。
这项设置只属于 `deploy`，不改变 `ftllm serve` 的默认行为。
`status` 和每轮 `manifest.json` 中的 `resource_limits.nofile` 记录预算、调整前后的
软限制与硬限制。该预算不是连接总数限制，不替代访问入口的连接管理。

每次启动在 `state_dir/runs/<时间>_<随机ID>/` 保存生效配置、精确 argv、显式环境、
源码摘要、进程身份、各副本日志、路由日志和生命周期状态。目录权限为 0700，配置
快照可能含 API key，应按私有运行资料管理。`status` 不输出认证参数。

`stop` 校验 PID、进程启动时间、boot ID 和进程组，清理记录中的服务，禁止全局
`pkill`。在启动等待期间也可以执行 `stop`。即使模型路径失效或模型参数被编辑，
`stop/status` 仍根据 `name/state_dir` 找到已有运行记录。停止旧运行时保留原有
`state_dir`；需要时可将该轮的 `config.json` 传给 `stop/status`。

运行期间不要修改正在使用的 Python 包；配置修改只影响后续启动。多个部署应使用
不同的 `state_dir`、GPU 分组和端口。`status` 的进程存活与下面的 HTTP 健康状态是
不同信息，HTTP 超时不会被伪装成进程已退出。

## 路由行为

所有 `/v1/*` 请求透传至副本，保持请求内容、认证头、HTTP 状态、响应头和 SSE 数据；
客户端断开会关闭上游连接。生成请求不会自动重试，也不会在流式输出中途切换副本。

- 有 `X-Session-Id` 或 `X-SMG-Routing-Key`：新会话按副本容量均衡分配，后续请求固定
  使用同一副本。不健康的已绑定副本返回 503，不隐式迁移会话。
- 新会话及无会话头请求会避开队列已满的副本；已有会话仍保持副本绑定。
- 没有会话头：按当前在途/排队请求数与容量的比例选择健康副本，支持普通 API 客户端。
- 每副本独立限流和排队，队满返回 429，排队超时返回 504。默认不额外排队，可设置
  `queue_limit_per_replica`。
- 会话表与推理槽位分开计数。会话空闲超时或表满时回收空闲 LRU，正在排队/请求的
  会话不会被回收。回收后新请求可能分到另一副本，影响缓存命中但不改写消息。

管理接口：

| 接口 | 含义 |
|---|---|
| `GET /health` | 所有副本健康返回 200，否则 503 |
| `GET /deploy/status` | 每副本健康、在途、排队、会话数与聚合路由容量 |
| `POST /deploy/release` | 带会话头，主动释放空闲亲和映射；忙碌时返回 409 |

`--api_key` 同时保护路由的模型和管理接口，支持 Bearer 或 `x-api-key`；`/health`
只暴露健康计数，允许无认证探测。默认入口绑定回环地址，远程访问地址需显式配置。

## 回归验证

在仓库根目录，使用带 aiohttp 的 Python 执行：

```sh
python -m unittest discover -s test/api -p 'test_deploy*.py' -v
```

测试使用假 HTTP worker，验证参数复用、默认值保持、进程回滚/崩溃清理、停止启动中
的部署、会话亲和、无会话路由、SSE、取消、限流、超时、认证与健康状态；不加载模型。
