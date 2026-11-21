# PD 分离并行模式

PD 分离拟支持以下并行组合：

| P 端 | D 端 |
|---|---|
| `DP×TP` | `DP×TP` |
| `DP×TP` | `Attention DP/EP` |
| `Attention DP/EP` | `DP×TP` |
| `Attention DP/EP` | `Attention DP/EP` |

DP 指普通副本 DP。DP、TP 可以为 1，P/D 两端规模可以不同。

**不支持 PP。具体实现方案后续补充。**

## KV 重分片

PD 分离拟支持 P/D 两端 KV 分布不同时的重分片，包括不同 TP 规模之间的
切分、合并和必要的复制，将 KV 按 D 端的分片方式放到对应 GPU，保持 KV 内容与逻辑位置不变。

仅改变普通 DP 副本数量时，按请求将 KV 交给目标副本，不要求按头重新切分。
具体实现方案后续补充。
