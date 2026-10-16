# FastLLM 多模型上下文扩展

支持适用模型的静态 YaRN 扩展，包含公共配置解析、CPU/CUDA 旋转、分页缓存写入、启动容量检查和服务元数据。本文同时说明已验证范围及尚未接入的布局。下述真实模型验证基于 `9d948ec6` 之后的上下文扩展实现，日期为 2026-09-06。

## 使用方法

`--max_context_length`（别名 `--max-context-length`）设置单会话输入与输出合计上限。`--rope_scaling`（别名 `--rope-scaling`）设置位置编码，接受 `yarn` 或 JSON。二者均为公共模型加载参数，CLI、server 和 Launcher 共用同一套 C++ 解析规则；不修改 checkpoint 的 `config.json`，也不引入扩展专用环境变量。

Qwen3.8-27B-FP8 的 1,000,000 token 配置：

```bash
numactl -C 0-31 -m 0 ftllm server /path/to/Qwen3.8-27B-FP8 \
  --tp 2 --kv_cache_dtype fp4 \
  --max_context_length 1000000 --rope_scaling yarn
```

这里 `1M` 为十进制 1,000,000，包含输出空间。该命令表达目标配置，**不代表双 24GB 显卡能够容纳它**；显存检查发生在 warmup 校准后、服务就绪前。不要为了通过检查随意增加 `--tokens`，它同样需要实际显存支持。

`yarn` 简写仅在已知原始 RoPE 长度时可用。当前 Qwen3.5 适配器确认声明长度 262144、theta=10000000、rotary_dim=64、交错三轴 sections=`[11,11,10]` 的配方后，可以采用 original=262144，并根据目标向上取整得到 factor=4。其他 checkpoint 应显式提供原始长度，例如 Qwen3：

```bash
numactl -C 0-31 -m 0 ftllm server /path/to/Qwen3-0.6B \
  --device cuda --tokens 65536 --chunked_prefill_size 8192 \
  --max_context_length 65536 \
  --rope_scaling '{"rope_type":"yarn","factor":2,"original_max_position_embeddings":32768}'
```

Qwen3 的配置声明可能是 40960，而官方 YaRN 原始长度为 32768，二者不能混用。省略 factor 时根据目标和原始长度计算；配置里已有 factor 时保留它，目标超过覆盖范围则报错。Qwen3.5 的完整写法为：

```json
{"rope_type":"yarn","factor":4,"original_max_position_embeddings":262144}
```

Launcher 的高级参数“RoPE 扩展”可填写同一个字符串或 JSON；JSON 作为一个 argv 元素传递。工作室通过 `/v1/models` 获取实际上下文。

Python 在创建模型时传入扩展参数，随后 warmup 校准容量：

```python
from ftllm import llm

llm.set_device_map("cuda:0")
llm.set_max_tokens(65536)
model = llm.model(
    "/path/to/Qwen3-0.6B", dtype="float16",
    max_context_length=65536,
    rope_scaling={"rope_type": "yarn", "factor": 2,
                  "original_max_position_embeddings": 32768},
)
model.warmup()
print(model.context_config)
```

旧 `set_max_context_length` 保留只缩小的兼容语义；扩大窗口必须在构造时配置。模型启动后不支持热改 RoPE，已有 KV、前缀和 CUDA Graph 始终使用本实例的固定参数。

## 支持范围与行为

| 模型/布局 | 支持范围 | 验证范围 |
| --- | --- | --- |
| Qwen2/2.5 普通文本 | 公共配置、普通旋转和融合分页写入 | Qwen2-7B 真实模型；普通布局 CPU/CUDA 算子 |
| Qwen3 dense | 普通、独立 CUDA、TP、分页及非分页入口；YaRN 不分配全长 sin/cos 表 | Qwen3-0.6B 默认回归、YaRN、41,000 token 输入 |
| Qwen3 MoE | 复用公共旋转和 TP helper | Qwen3-30B-A3B-FP8 双卡真实推理及共享算子回归 |
| Qwen3.5 dense/MoE，以及使用该架构的 Qwen3.8 | partial rotary、交错三轴 M-RoPE、TP、内置 MTP、融合 decode | Qwen3.5-2B、Qwen3.8-27B-FP8；不据此声明所有 MoE checkpoint 均完成实测 |
| 其他模型 | 无新参数时沿用原路径；不开放新的 RoPE 扩展 | 需要模型适配和相应路径验证 |

新 RoPE 扩展入口接入 **HF 模型目录和 GGUF 文件**，仍受上表的模型布局限制。GGUF 显式传入 `--rope_scaling` 时，在 `InitParams` 前应用公共配置；读取 GGUF 的窗口、theta、旋转维度、scaling 元数据和 Qwen3.5 交错 sections，原生加载路径保持不变。GGUF、FLM、自定义 GraphLLM 的 CLI 仅设置 `--max_context_length` 时保留旧的限长路径：可以缩小，超过模型窗口时仍取原上限。FLM 和自定义 GraphLLM 显式设置 `--rope_scaling` 仍会报不支持。Python 的新构造参数支持 HF 目录和 GGUF；其他格式继续使用旧长度 setter。上下文扩展不会额外增加某模型对 FP4、MTP 或其他后端的支持。结合主机 KV 备份的长上下文用法见 [KVMem](kvmem.md)。

当前支持 default、linear 与静态 YaRN。linear 扩大声明窗口同样需要明确原始长度。已有 dynamic NTK 在没有 RoPE 覆盖且没有扩大声明窗口时保留原路径，不开放新的动态扩展。外部 DSpark 与 YaRN 同用仍拒绝启动。Qwen3.5 的 DFlash2 可保留独立的原生 default RoPE 与目标 YaRN 同用，但草稿必须显式声明所有层均为滑窗注意力，窗口不超过其原生长度，且不使用缩放、partial rotary 或 M-RoPE。HF/GGUF 两个加载入口执行同一项检查。该组合不承诺短上下文相同的草稿接受率；主模型与草稿分别使用自己的旋转配置。

| HF 新入口的输入 | 行为 |
| --- | --- |
| 无新参数，模型为 default RoPE | 保持旧执行路径 |
| 模型配置已经声明 YaRN | 使用归一后的 YaRN，配置窗口不重复乘 factor |
| 显式目标小于或等于支持窗口 | 设置目标，并检查实际容量 |
| 目标超过声明窗口，缺少有效扩展配置 | 初始化报错，不再静默压回原上限 |
| 缺少 YaRN original、非法 factor、未知字段/算法或冲突别名 | 给出配置错误 |
| 位置目标超过 16777216 | 拒绝，当前 position IDs 为 FP32 |
| 校准后的 KV 容量小于显式目标 | warmup 失败，报告目标及校准容量 |
| 未显式设置目标 | 保留按物理 KV 容量限制实际窗口的行为 |

## 实现结构与模型接入

公共实现位于 [`include/contextconfig.h`](../include/contextconfig.h) 和 [`src/contextconfig.cpp`](../src/contextconfig.cpp)。

- `ContextOptions`：用户目标长度和 RoPE 覆盖。
- `ModelContextSpec`：由模型类声明是否支持、配置作用域和交错 M-RoPE 布局。
- `RopeConfig`：归一后的 theta、factor、原始长度、旋转维度、YaRN beta 区间、幅度和三轴 sections。
- `ContextPlan`：模型声明长度、用户目标、最终有效窗口及 RoPE 参数。模型实例持有，多个模型不共享可变配置。

HF loader 和显式配置上下文的 GGUF loader 在 `InitParams` **之前**调用 `ConfigureContext`，避免先按旧窗口分配资源再重建模型。模型的 `InitContextParams` 将计划应用到自身旋转参数；C API 保留旧入口，增加带 context options 的入口和已解析配置查询，错误不会跨 ctypes 直接抛出。

HF/GGUF loader 通过 RAII 管理初始化中的模型，HF loader 还管理 LoRA 资源；CLI/服务公共工厂在 warmup 或容量检查失败时释放已加载模型，避免重试启动遗留权重。直接使用 Python `llm.model` 的调用方仍负责在失败或使用结束后调用 `release_memory()`。

解析器区分顶层与 `text_config`，归一 `rope_parameters`、旧 `rope_scaling` 及 `type` 别名；不修改视觉编码器配置。新旧字段冲突必须显式覆盖消解。跨算法切换仅保留 theta 和布局等兼容字段，不继承旧算法的专属参数。当前接受裸单组配置及 `{"full_attention": {...}}` 的单组覆盖；**尚未实现多个 RoPE 组同时持有不同计划**，不能把此语法当成已支持 Laguna 的 full/sliding 多组布局。

新增标准布局模型时：

1. 在模型类提供 `GetContextSpec()`，选择配置作用域，核实原始长度来源、head_dim 和旋转布局。
2. 在初始化连接 `InitContextParams`，在全部 Q/K 旋转入口使用同一份配置。普通和融合路径都必须覆盖，包括 TP、图及投机分支；缺少融合实现时可以走语义一致的普通路径。
3. 添加该 checkpoint 的默认/扩展回归和容量测试。普通文本与 Qwen3.5 的实现提供两种已接入布局，无需再改 CLI、Launcher 或 API 长度字段。

特殊布局仍需要相应工作：MLA、不同幅度/score 规则、多组 RoPE、绝对位置表及新动态算法不能仅注册模型名称。首批模型均包含随 token 增长的全注意力，容量检查复用已有 `AutoWarmup` 校准后的 token 容量。纯滑窗、纯线性及压缩缓存模型接入时，需要扩展容量策略，不能直接套用此比较。当前错误报告目标与校准 token 容量，**尚未提供各卡“所需/可用字节”详细清单**。

## 旋转、KV 与执行路径

复用现有 `YarnRopeEncoding` 和 CUDA `FastllmYarnInvFreq`，完整应用频率插值、外推、beta 区间及 attention factor。原始位置长度独立保存，不会被服务窗口替换。

Qwen3.5 的 head_dim=256、partial rotary factor=0.25，实际仅旋转 64 维。CPU/CUDA 的 YaRN 增加与原 M-RoPE 一致的交错三轴选择。幅度只施加在旋转部分；K 旋转后再写入 FP16/BF16/FP8/FP4 KV，V 和 gate 保持原值。

Qwen3.5 专用融合 decode 接收同一份 YaRN 参数；普通 prefill、TP 与内置 MTP 的对应入口也连接该计划。专用融合 prefill 尚未增加 YaRN，启用扩展时使用现有拆分操作完成相同计算。这可能影响 prefill 性能。

算子按实际位置计算频率，不使用额外的设备端 `inv_freq` 缓存。Qwen3 的 YaRN 路径跳过全长 sin/cos 表及依赖该表的融合 attention。

## 服务长度和显存

三个长度分别展示：`model_context_window` 是 checkpoint 原声明，`configured_context_window_limit` 是用户目标，`context_window` 是实际接受窗口。C++ 计划在 warmup 后更新，Python 和服务元数据读取该结果；不再把新计划重新压回 checkpoint 声明长度。请求端仍按模板及多模态展开后的完整输入加输出限制检查长度。

`--tokens` 表示共享 KV 池容量，不是每条会话的单独预留。显式窗口必须能够放进校准后的池；并发仍受共享页和状态池限制。chunked prefill 限制中间工作区，不能消除最终 KV 存储需求。

Qwen3.8-27B-FP8 有 16 层全注意力、4 个 KV head、head_dim=256。按当前格式计算，其 1,000,000 token 全注意力 KV 为：

| 类型 | 每 token 字节 | 两卡合计 | TP=2 每卡 |
| --- | ---: | ---: | ---: |
| FP16/BF16 | 65536 | 61.04 GiB | 30.52 GiB |
| FP8 | 32768 | 30.52 GiB | 15.26 GiB |
| FP4，包含 block16 scale | 18432 | 17.17 GiB | 8.58 GiB |

还需要权重、线性注意力状态、MTP、图及工作区。本机两张 24GB 显卡的既有 8K FP4/TP2 配置常驻合计约 37GiB，换成 1M KV 的估算超过 54GiB，不能承诺完整 1M。已测自动预算为 **462080 token**（TP2、FP4、pageLen=128、MTP/图关闭、GPU 内存比例 0.9）；请求 1M 时明确失败。容量会随启动参数和空闲显存变化。

## 验证与复现

所有构建和测试限定在 NUMA0：`numactl -C 0-31 -m 0`。构建产物位于独立 build 目录，没有覆盖本机已安装的 Python 包。在仓库根目录可用 `PYTHONPATH=build-fastllm/tools numactl -C 0-31 -m 0 python3 -m ftllm.cli server /path/to/model` 运行构建版本，再追加上文的上下文参数。

```bash
numactl -C 0-31 -m 0 cmake --build build-fastllm \
  --target fastllm_tools contextExtensionRegression cuda_fp4_kv_test -j 16
numactl -C 0-31 -m 0 ctest --test-dir build-fastllm \
  -R 'context_extension|cuda_fp4_kv' --output-on-failure
numactl -C 0-31 -m 0 compute-sanitizer --tool memcheck --error-exitcode 99 \
  build-fastllm/contextExtensionRegression --cuda
```

测试目标需要 `-DUNIT_TEST=ON`；CPU-only 构建也可以运行 `context_extension_cpu`。`test/api/context_extension_model_probe.py` 是显式运行的真实模型探针，不下载模型；记录首 token logits、生成 token、耗时和解析后的窗口。MTP 探针使用普通 token 输出，避免 `output_logits` 自动禁用 MTP；`--no-logits` 可用于对应的非 MTP 对照。例如复现实际超过原声明的 Qwen3.5 输入：

```bash
numactl -C 0-31 -m 0 python3 test/api/context_extension_model_probe.py \
  --python-root build-fastllm/tools --model /path/to/Qwen3.5-2B \
  --output /tmp/context-qwen35 --context 300000 --tokens 300032 \
  --input-tokens 270000 --kv fp4 --rope yarn
```

## 验证范围

回归覆盖配置解析和冲突检查、独立标量 RoPE/M-RoPE 参考、融合分页写入、FP4 KV、CUDA Graph、启动容量不足时的释放和重试，以及 CPU-only 构建。
高位置算子输入覆盖 32768、262144、999999、1000000；这不等于完整 1M 请求验证。

真实模型检查覆盖 Qwen2-7B、Qwen3-0.6B、Qwen3.5-2B 和 Qwen3-30B-A3B-FP8 的短输入；Qwen3-0.6B 的 41000-token 输入、Qwen3.5-2B 的 270000-token FP4 输入均完成开头口令检索。
Qwen3.8-27B-FP8 的 TP2、FP4、MTP1 和 Graph 在短请求中通过；该双卡配置申请 1M 上下文时校准容量为 462080，按预期拒绝启动。Qwen3.5-2B 的图片请求完成，logits 有限。
这些是有限请求与回归检查，不是长上下文质量基准；更复杂检索、并发、长前缀复用和完整 1M 请求未据此验收。

## 模型配置参考

- [Qwen3 的原始长度和 YaRN 配方](https://huggingface.co/Qwen/Qwen3-8B#processing-long-texts)
- [Qwen3.8-27B-FP8 的 1M 配置](https://huggingface.co/Qwen/Qwen3.8-27B-FP8#best-practices)

声明长度、RoPE 配置和物理 KV 容量必须同时满足；放宽声明长度不能替代位置编码或显存支持。
