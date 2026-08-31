# Kimi-K3 Deployment

[中文](kimi_k3.md) · [Back to README](../README_EN.md) · [Benchmark](benchmarks/kimi_k3_en.md)

FastLLM supports Kimi-K3 KDA/MLA, chunked prefill, CUDA/NUMA execution, CPU/GPU and disk experts, external DSpark, XTML serving, and tool calling.

## API server quick start

~~~bash
ftllm server /data/models/kimi-k3 \
  --model_name kimi-k3 \
  --host 0.0.0.0 --port 8080
~~~

## GPU + NUMA hybrid MoE

~~~bash
ftllm server /data/models/kimi-k3 \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

Use `--moe_device cpu` on a single-socket host.

## Disk experts

~~~bash
ftllm server /data/models/kimi-k3 \
  --device cuda \
  --moe_device "{'cuda':1,'numa':8,'disk':1}" \
  --chunked_prefill_size 8192
~~~

Start with a small disk fraction because decode can become random-I/O bound.

## External DSpark

The draft directory must contain a matching `DSparkDraftModel` checkpoint:

~~~bash
ftllm server /data/models/kimi-k3 \
  --speculative_algorithm dspark \
  --speculative_draft_model_path /data/models/kimi-k3-dspark \
  --speculative_dspark_confidence_threshold 0.5
~~~

The block size defaults to the draft configuration and is validated at startup.

## Long context, thinking, and tools

~~~bash
ftllm server /data/models/kimi-k3 \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --enable_thinking true \
  --tool_call_parser auto
~~~

Kimi-K3 uses its XTML conversation protocol; keep the checkpoint's tokenizer and template.

## Benchmark status

No publishable Kimi-K3 throughput report is currently stored in the repository. See the [Kimi-K3 benchmark page](benchmarks/kimi_k3_en.md) for suggested device commands and the current data status.
