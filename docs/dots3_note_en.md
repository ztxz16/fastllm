# Dots3-Note Deployment

[中文](dots3_note.md) · [Back to README](../README_EN.md) · [Benchmark](benchmarks/dots3_note_en.md)

FastLLM supports Dots3-Note DSA indexing, sparse attention, long-context caching, chunked prefill, reasoning content, and tool calling.

Dots3-Note is not currently supported by the runtime `--auto` planner. Use the manual commands on this page. See the [`--auto` guide](auto_en.md) for supported models.

## API server quick start (manual configuration)

~~~bash
ftllm server /data/models/dots3-note \
  --model_name dots3-note \
  --host 0.0.0.0 --port 8080
~~~

## GPU + NUMA hybrid MoE

~~~bash
ftllm server /data/models/dots3-note \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

## GPU + CPU

~~~bash
ftllm server /data/models/dots3-note \
  --device cuda --moe_device cpu \
  --chunked_prefill_size 8192
~~~

For FP8 checkpoints, FastLLM selects model- and device-specific MoE activation and CPU fast paths. Start with `--dtype auto`.

## Long context and caching

~~~bash
ftllm server /data/models/dots3-note \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --gpu_mem_ratio 0.9
~~~

## Thinking and tool calling

~~~bash
ftllm server /data/models/dots3-note \
  --enable_thinking true \
  --tool_call_parser auto
~~~

The API supports streaming and non-streaming `reasoning_content`. Tool parsing defaults to automatic selection.

## Benchmark status

Operator, long-cache, and API regressions exist, but the repository has no complete Dots3-Note device-throughput table. See the [Dots3-Note benchmark page](benchmarks/dots3_note_en.md) for suggested device commands and the current data status.
