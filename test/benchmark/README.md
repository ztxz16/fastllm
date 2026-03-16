# Benchmark Scripts

这个目录目前有两类 benchmark：

- `prefill.py`：测长上下文 prefill 性能。
- `decode.py`：测多 batch 并发请求下的 decode 性能。

本文主要说明 `decode.py` 和 `decode_config.example.json` 的用法。

## 1. `decode.py` 是做什么的

`decode.py` 会在本地拉起一个 FastLLM OpenAI 兼容服务，然后并发发起多条流式 `chat/completions` 请求，统计多 batch 场景下的 decode 吞吐。

它不是单个 HTTP 请求里塞多个 prompt，而是：

1. 启动服务端模型；
2. 构造统一的 prompt；
3. 同时发起 `batch_size` 条流式请求；
4. 等所有请求结束后，统计整体 decode 吞吐和单请求 decode 速度。

这种测法更接近实际服务端的多并发、多 batch 推理场景。

## 2. 依赖

先安装依赖：

```bash
pip install -r test/benchmark/requirements.txt
```

## 3. 最简单的单次测试

在仓库根目录执行：

```bash
python test/benchmark/decode.py /path/to/model \
  --batch-size 16 \
  --max_batch 16 \
  --prefill-length 512 \
  --max-tokens 256
```

参数含义：

- `model` 或 `--path`：模型路径，支持 fastllm 模型文件或 HF 模型目录。
- `--batch-size`：同时发起多少条请求。
- `--max_batch`：服务端允许同时推理的最大 batch。要测“真实多 batch decode”，通常应满足 `max_batch >= batch_size`。
- `--prefill-length`：按字符数构造 prompt。想把测试重点放在 decode 时，这个值不要太大。
- `--max-tokens`：每条请求最多生成多少 token。

例如测 `Qwen3-8B` 的 64 batch decode：

```bash
python test/benchmark/decode.py ~/hfmodels/Qwen/Qwen3-8B \
  --batch-size 64 \
  --max_batch 64 \
  --prefill-length 512 \
  --max-tokens 256
```

## 4. 用配置文件批量跑

如果要连续跑多组 batch，可以用 `decode_config.example.json`：

```bash
python test/benchmark/decode.py --config test/benchmark/decode_config.example.json
```

这个文件结构分两部分：

- `defaults`：所有 case 共用的默认参数；
- `cases`：每组 benchmark 的覆盖项。

示例：

```json
{
  "defaults": {
    "path": "/path/to/model",
    "prefill_length": 512,
    "max_tokens": 256
  },
  "cases": [
    {
      "name": "batch1-out256",
      "batch_size": 1,
      "max_batch": 1
    },
    {
      "name": "batch4-out256",
      "batch_size": 4,
      "max_batch": 4
    },
    {
      "name": "batch16-out256",
      "batch_size": 16,
      "max_batch": 16
    }
  ]
}
```

建议把 `path` 改成你自己的模型路径，再按需要增加 case。

## 5. 关键参数说明

- `--batch-size`
  同时发起的请求数，也是你想测的并发 batch。

- `--max_batch`
  FastLLM 服务端实际允许并行处理的上限。
  如果这个值小于 `batch_size`，结果会包含排队时间，不再是纯粹的多 batch decode 吞吐。

- `--prefill-length`
  直接控制 prompt 字符数，优先于 `--prompt-repeat`。
  decode benchmark 推荐用较短 prompt，比如 `256`、`512`、`1024`。

- `--prompt-repeat` / `--prompt-unit` / `--question`
  如果不想直接指定字符数，可以用这三个参数拼 prompt。

- `--request-stagger-ms`
  请求启动间隔，默认 `0` 表示尽量同时发起。

- `--skip-warmup`
  跳过预热。默认会先发一个很短的预热请求。

## 6. 输出怎么看

脚本会打印单次结果和 JSON 结果，重点看这些字段：

- `batch_decode_speed`
  所有请求合并后的 decode 吞吐，单位 `tokens/s`。这是最核心指标。

- `end_to_end_speed`
  以整批请求总耗时计算的整体吞吐，包含 TTFT 影响。

- `ttft_min / ttft_avg / ttft_max`
  首 token 延迟统计。

- `per_request_decode_speed_avg`
  单请求平均 decode 速度。

- `per_request_decode_speed_min / max`
  单请求 decode 速度分布。

- `finish_reason_counts`
  每个请求的结束原因统计，正常通常是 `stop`。

## 7. 实际使用建议

- 只想看 decode，不想让 prefill 干扰结果：
  用较小的 `prefill_length`，例如 `512`。

- 想比较不同 batch 的扩展性：
  固定 `prefill_length` 和 `max_tokens`，只改 `batch_size` 与 `max_batch`。

- 想看服务端是否已经到瓶颈：
  观察 `batch_decode_speed` 是否继续增长，以及 `per_request_decode_speed_avg` 是否开始明显下降。

- 想避免把排队时间算进去：
  保证 `max_batch >= batch_size`。

## 8. 和 `prefill.py` 的区别

- `prefill.py` 重点看长上下文首 token 延迟和 prefill 速度；
- `decode.py` 重点看多请求并发下的 decode 吞吐。

如果你在做服务压测，通常两个都值得跑：

- 用 `prefill.py` 看长上下文是否慢；
- 用 `decode.py` 看多 batch 下吞吐能到多少。
