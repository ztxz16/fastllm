# Qwen3.5/Qwen3.8 多模态持久前缀缓存

为 dense Qwen3.5/Qwen3.8 增加可跨服务进程重启的 SSD 前缀缓存，支持 DFlash2，
并修复 Merkyor 等 `qwen3_5_text` / `Qwen3_5ForCausalLM` 纯文本检查点的工厂识别与权重名称映射。
纯文本模型本身不因此获得视觉能力。

## 启用

在现有 `ftllm server` 命令前配置：

```bash
export FASTLLM_PREFIX_CACHE=1
export FASTLLM_MULTIMODAL_PREFIX_CACHE=1
export FASTLLM_PREFIX_CACHE_DIR=/path/to/ssd-cache
export FASTLLM_PREFIX_CACHE_DISK_BYTES=274877906944
export FASTLLM_PREFIX_CACHE_RESTORE_POLICY=always
```

也可使用 `--prefix_cache_dir`、`--prefix_cache_disk_gb`、`--prefix_cache_restore_policy`；
CLI 优先于环境变量。目录未设置时不启用 SSD。容量默认 256 GiB，策略默认 `auto`。
`always` 适合正确性验证；`auto` 在缺少有效读取成本标定时跳过恢复，不能保证自动加速；
`never` 不从 SSD 恢复。完整性和身份检查始终保留。

构建依赖 OpenSSL、SQLite3；文件后端复用原样的 LMCache C++ 组件，来源/许可证在
`third_party/lmcache_fs/`。当前持久化后端要求 Linux/POSIX；启用 SSD 但运行库缺少 v2 能力时明确报错。

## 行为与限制

- 依据实际 token、媒体内容及位置、模型/运行库内容、执行布局建立前缀身份；共同前缀允许共享，分叉后的状态隔离。
- 保存普通 KV、GDN、DFlash2 和多模态位置相关身份。图片 embedding 通过独立身份持久化。
- 读取完整对象并校验后发布恢复状态；损坏记录可回退到较短检查点。不可变对象、提交文件与 SQLite 索引共同管理引用及 FIFO 容量。
- 后台写入，在已有合法 prefill 边界保存检查点，不改变用户的 `chunked_prefill_size`。
- 普通 KV 保持实际存储精度。FP16→FP8 复用现有转换算子并记录来源；FP8→FP16 不反向恢复。
  转换后的计算历史不等同于从头使用 FP8 的冷计算，不应把两者输出一致当作合同。
- 启动时主模型/视觉/草稿共用最多 8 个张量哈希线程，最多 16 个在途任务和 64 MiB 权重读取缓冲；
  仍计算原始完整 SHA-256。缓冲上限不是进程总内存上限。
- 正常停服等待后台提交最多 30 秒；只应将 `committed` 的记录视为可重启恢复。
- 当前不支持 MoE、MTP/DSpark、LoRA、自定义权重等持久恢复组合；视频等缺少支持身份的入口不复用该前缀。

验证真正跨进程命中：等待 `[Prefix SSD] committed:`，停止服务并保留目录，
重启后发送相同完整前缀，检查 `[Prefix SSD] restored:` 和响应中的 cached token 计数。
普通内存 `prefix cache hit` 或图片内存命中不能单独证明 SSD 恢复。

## 验证范围

以下真实模型结果来自提交者独立部署版本 `8cd030532`，不是对更新后上游 PR 分支的重复 GPU 验收：

- GPU 0–3、TP4、FP8 KV、2048 prefill、DFlash2、多模态配置下，文本/图片重启、A/B/A 分支、换图和追加图通过 HTTP 对照。
- 同精度含图 8K、32K、128K 输入，在独立进程间恢复后检查 4 步全词表 logits，最大绝对差为 0。
- 损坏第 4 个 rank 的后部对象后拒绝长检查点、回退短检查点，重算修复并再次重启恢复，输出一致。
- 8 张量哈希与旧版完整 manifest 一致；一次本机对照 API 就绪时间 145.65→134.31 秒，旧 SSD 恢复 8192 token。
  未清空系统文件缓存，不将结果视为物理冷盘吞吐或跨硬件性能承诺。
- Merkyor 纯文本兼容具有工厂/名称映射、小型 FP8 loader 与检查点配置/索引验证；
  不将多模态模型实测冒称 Merkyor 纯文本模型的完整 GPU 验收。

尚未完成：FP16→FP8 同源内存转换的严格数值对照、全部并发/故障矩阵、auto 成本策略验证。
未执行机器断电测试。当前实现仍需按这些范围审查，不能宣称全部生产场景已经验证。

CPU/集成测试在 `test/api/test_persistent_prefix.py`、`test/basic/test_disk_prefix_cache.cpp`、
`test/basic/test_qwen35_text_weights.cpp` 及多模态前缀相关测试中。
`test/multimodal/` 的 SSD 与启动验收脚本默认使用提交者的本地模型和证据路径，
其他机器运行前需要调整路径，不属于无需模型的通用 CI。
