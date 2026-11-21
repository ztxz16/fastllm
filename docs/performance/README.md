# 性能对比汇总

本目录长期汇总不同模型、硬件、并行方式和工作负载下的性能对比，作为查看与追加结果的固定入口。

## 已有对比

- [Attention DP/EP：旧路径与批次不变模式](attention_dp_ep_batch_invariant.md)
  - 模型：Qwen1.5-MoE-A2.7B-Chat，FP16。
  - 硬件与并行：4 × V100-SXM2-32GB，Attention DP=4 / EP=4。
  - 当前覆盖：每请求输入 1024 Token、输出 128 Token，请求并发 1/4，Eager/CUDA Graph，前缀缓存关闭。
  - 可比结果：并发 1 的 Eager 加速比为 1.199，CUDA Graph 为 1.179；方向为旧路径耗时 / 批次不变模式耗时。
  - 并发 4 的两项保留耗时和不可比原因，不给出加速比。
  - 补充数据：[JSON](attention_dp_ep_batch_invariant.json)、[CSV](attention_dp_ep_batch_invariant.csv)。

## 后续记录方式

分批后台验收生成的当前版本性能报告保存在 `runs/<独立运行目录名>/summary.md/json/csv`。
对应原始证据、XLSX 与完整执行记录仍在 `test/distributed/results/`，汇总包含直接证据路径。
每次运行独立保存，不覆盖此前记录；缓存一致性问题未关闭时，报告明确标记最终交付未完成。

同一对比主题的新场景追加到对应报告；新的对比主题新建报告，并在本页登记。不同硬件或版本的重复测量保留各自记录，不覆盖历史结果。

每条结果至少记录以下内容：

- 模型名称、精度、GPU 型号与数量、并行配置、请求并发数。
- 每请求输入和输出 Token 数、Eager/Graph、缓存状态、采样与 EOS 设置。
- 对比双方的模式或后端、计时范围、Median、P95、吞吐、加速比及其计算方向。
- 是否满足可比条件；不满足时保留原因，加速比留空。
- Warmup、迭代及重复次数、CV、原始样本、运行日期、Git 提交、CUDA 和编译配置。
- 完整运行命令和原始结果目录，便于追溯。

尚未取得的字段标明“未提供”，区分用户粘贴的结果、配置推断与原始日志证据。需要分析性能变化来源时，再补充首 Token 延迟、逐 Token 延迟、显存占用及 Prefill/Decode 分段耗时。

原始测试日志与诊断证据仍保存在 `test/distributed/results/`；本目录保存可持续维护的对比汇总。按用户本次要求，移除刚生成的 Excel，本次汇总保留 Markdown、JSON 和 CSV。
