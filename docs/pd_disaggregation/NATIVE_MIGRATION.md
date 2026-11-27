# PD 核心逻辑原生迁移

依据 2026-11-26 确认的范围，核心决策使用 C++，GPU 数据搬运由 C++ 直接调用 CUDA Runtime。
迁移保留现有 PD 行为、公开 Python 调用签名与命令行配置；模型计算和算子继续复用。
原有非 PD、非 Router 托管的完整服务入口不在这次 PD 核心迁移范围内。

## 模块与实现边界

| 模块 | 实现语言/技术 | 核心职责 | 原生文件（`src/distributed/`） |
| --- | --- | --- | --- |
| 协议与映射 | C++ | 身份、布局、范围校验，TP 分片映射与覆盖检查 | `pd_metadata.cpp` |
| 服务内调度 | C++ | DP 选择、评分、队列、容量检查、撤回及恢复 | `pd_service.cpp`、`pd_execution.cpp`、`pd_resources.cpp` |
| Attention DP/EP 协调 | C++ | 每 rank 每轮至多一条命令，统一提交联合轮次、失败唤醒 | `pd_moe.cpp` |
| Router 策略 | C++ | 八种策略、前缀/缓存选择、负载计数和会话绑定 | `pd_routing.cpp` |
| Router 服务管理与流程 | C++ | 注册/注销、健康检查、P/D 配对、meta 转交、有条件请求级重试 | `pd_router.cpp` |
| KV 交接管理 | C++ | 层就绪、传输汇总、取消、超时、在途保护及引用解除 | `pd_tasks.cpp`、`pd_service_transfer.cpp` |
| 单机搬运 | C++、CUDA Runtime | IPC 映射、P2P 检查、异步拷贝、完成事件、映射释放 | `pd_cuda.cpp` |
| 历史 KV | C++ | 480 秒保留、读取租约、过期回收、来源失效与前缀恢复 | `pd_resources.cpp`、`pd_service_transfer.cpp` |
| 生成与输出控制 | C++ | Decode 推进、停止判断、位置与身份检查、结果去重/拒绝 | `pd_execution.cpp`、`pd_service_output.cpp`、`pd_router.cpp` |
| Python 接入与原生绑定 | Python、pybind11 | CLI、配置、HTTP/NDJSON/SSE、Tokenizer、数据对象、回调与迭代器适配 | `pd_bindings.cpp`、`pd_bindings.h`；Python `pd/` 包 |
| 验证与报告 | Python | 调用正式入口，复用协议、Router、GPU 及模型测试矩阵 | `test/distributed/pd/` |

Python 中的 dataclass/Enum 保留为接口数据对象，现有 `asdict`、`replace` 和测试注入方式可继续使用。
这些对象的校验、变更时机和决策在 C++ 中实现。为兼容现有接口，原生逻辑通过 pybind11
读写这些对象，复用 Python 的编码、时钟、随机数、锁、线程及 I/O 原语。
**这不是脱离 Python 解释器的独立 C++ 服务**，也不宣称已经解除 GIL 或获得性能提升。
后续如替换容器或线程原语，不得改变已确认的策略、同步与资源生命周期。

`_core.py` 只负责加载、调用及将原生回调转换成迭代器。每条输出须待调用方继续读取后才确认，
保持原 `yield` 的背压；HTTP 适配器记录“客户端已响应/断开”，C++ 决定是否还允许重试。
服务启动、JSON/HTTP 格式转换和原生进程消息的 I/O 适配继续使用原有 Python 接口。
`RoundTransport` 仍负责进程消息收发，组内命令合并与执行轮次由 C++ 控制。

GPU 测试的 `cuda_fixture.py` 仅在测试所有者进程分配、上传和读回显存；
测试中的 KV 传输、映射与完成判断全部调用正式 C++/CUDA 后端。
没有新增 Python 核心回退，也没有新增 CPU KV 中转。

## 构建与打包

新增独立目标 `fastllm_pd_core`，输出 `_pd_core.so`。它不链接模型/CPU 算子对象，
可在 `USE_CUDA=OFF` 时单独用于协议与 Router 回归；此类产物调用 GPU 传输会明确拒绝。
GPU 运行必须使用 `USE_CUDA=ON` 构建。Python 开发头文件必须与服务解释器匹配；
CMake 至少 3.18，pybind11 头文件可来自已有子模块或所选 Python 环境。

进程包和工具包都包含扩展；源码运行优先加载包内扩展，其次查找
`build-process-tp/_pd_core.so`、`build/_pd_core.so`，也可用 `FT_PD_CORE_LIBRARY` 指定路径。
缺少扩展或 ABI 不匹配时直接报错。

仅在服务器执行构建，当前本地工作区未获编译授权：

```bash
cd /home/project/vllm_fastllm
cmake -S . -B build-process-tp -DUSE_CUDA=ON -DBUILD_PROCESS_TP=ON -DCUDA_ARCH=70 -DMAKE_WHL_X86=OFF
cmake --build build-process-tp --target fastllm_pd_core fastllm_tp_generate fastllm_moe_generate fastllm_tp_worker -j 4
```

这里 `70` 对应当前验证服务器的 V100；其他服务器沿用各自 GPU 架构配置。
PD 核心迁移本身不解决第三方 CPU SIMD 编译问题；后续完整组件升级单独记录如下。

### 2026-11-27：GGUF 完整组件升级与 CPU 指令集适配

将 FastLLM 上游的整个 `third_party/gguf/` 兼容组件固定到提交
`e33b3cb795119e9d0b00d65b9c9991c4ca054483`：全部 7 个文件保持上游原样，
相对升级前有 4 个文件变化，3 个文件本来就一致；公共头文件未变。
此前仅替换两个文件的尝试已撤销，本次按完整子树重新导入。
此组件整合 llama.cpp 与 ik_llama.cpp，不表示这两个原始仓库已整体升级。

| 模块 | 实现语言/技术 | 本次变化 |
| --- | --- | --- |
| 上游 GGUF 组件 | C++、CPU SIMD | 完整版本升级，包含 AVX/IQK 修复及元数据读取、反量化和模型映射变化 |
| FT CPU 算子适配 | C++ 条件编译 | 位于 `src/devices/cpu/cpudevice.cpp`；两处 FP16 转换要求 F16C，BF16 SIMD 转换要求 AVX2；函数体、MSVC 约定与标量路径保留 |
| 版本管理与校验 | JSON、Python 标准库 | 位于 `tools/third_party/`，记录固定提交、完整文件清单及哈希；不参与推理，不触发编译 |

版本来源、升级流程、兼容边界和校验命令见[第三方版本记录](../../tools/third_party/README.md)。
已 Review 调用兼容性并直接对照上游全部文件与 Git tree；完整性静态检查通过。
未本地编译或执行模型/数值回归；GGUF 兼容问题仍待服务器验收，不能用文件一致性代替构建通过。
服务器 CPU 指令集与 `CUDA_ARCH=70` 指定的 V100 GPU 架构分别处理；
不强制添加 CPU 不支持的 AVX2/FMA，也不宣称已补齐所有 IQ 格式的无 AVX2 点积能力。

构建后逐阶段后台验收，每次检查运行目录的 `exit_code` 和汇总报告：

```bash
python3 test/distributed/pd/launch.py protocol
python3 test/distributed/pd/launch.py router
python3 test/distributed/pd/launch.py ipc -- --devices 0,1
```

上述三个脚本分别保留 41、20、24 个 case；不能代替模型生成、生命周期、抢占恢复和性能验收。
后续正式服务矩阵沿用 [运行说明](RUNNING.md)；要求 exact/Top-K 的 FAIL 和仅 numeric 的 WARNING
继续按项目统一规则处理。

## Review 与验证状态

迁移代码已接入，尚未通过编译和功能验收。已进行源代码 review、Python AST/定义导入检查、
C++ 语法树检查、绑定入口与参数核对、差异空白检查；这些检查不能证明 C++ 编译/链接成功。

Review 中修正了安装包缺少原生扩展、流式桥接的逐条确认、取消通知、
预提交失败后残留 IPC 映射清理，以及地址计算的整数边界等问题。
未执行本地 CMake、编译器、链接或可能触发编译的测试。

2026-11-27 服务器使用 pybind11 2.13.6 构建时，发现 `py::float_(0)` 的 float/double
构造函数重载歧义。已在 C++ 传输计时默认值与副本 KV 使用率初始化两处改用 `0.0`，
并静态复查其余浮点构造调用；数值仍为浮点零，待服务器重新构建验证。

待服务器完成：原生构建、协议/Router 回归、真实 CUDA IPC、四种模式组合的模型生成、
满载/撤回/恢复、多轮回传及取消生命周期验收。迁移前的 PASS 记录不改写，不能替代本次验证。
