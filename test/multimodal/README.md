# Multimodal Tests

这里放通用的多模态测试脚本。

当前入口：

- `fastllm_multimodal_check.py`：通用 fastllm 图文 smoke test

示例：

```bash
python test/multimodal/fastllm_multimodal_check.py \
  --model-path /home/use/gemma-4-31B-it \
  --text-only-first

如果显存紧张，也可以切到 CPU：

```bash
python test/multimodal/fastllm_multimodal_check.py \
  --model-path /home/use/gemma-4-31B-it \
  --device cpu \
  --cpu-threads 32
```
```

自定义 messages：

```bash
python test/multimodal/fastllm_multimodal_check.py \
  --model-path /path/to/model \
  --messages-file /path/to/messages.json
```

支持的 profile：

- `auto`：按模型 architecture 自动推断
- `generic`：通用图文配置
- `gemma4`：Gemma4 默认配置
- `cogvlm`：CogVLM 默认配置

如果后续要支持新的多模态模型，优先在 `common.py` 里补一个新 profile，而不是再新写一份模型专用脚本。
