# Qwen3.5 / Qwen3.6 / Qwen3.8 Deployment

[中文](qwen3.md) · [Back to README](../README_EN.md) · [Qwen4-Exp](qwen4_exp.md) · [Benchmarks](benchmarks/qwen3_en.md)

This guide covers current Qwen3.5, Qwen3.6, and Qwen3.8 text models. Qwen4-Exp / Qwen3.8-Flash-Next uses a separate architecture and is documented in the [Qwen4-Exp guide](qwen4_exp.md).

Checkpoint layouts may be dense, MoE, FP8, NVFP4, or another quantized format. Keep `--dtype auto` for an initial launch, then tune from measured memory use and throughput.

## API server quick start

~~~bash
ftllm server /data/models/qwen \
  --model_name qwen \
  --host 0.0.0.0 --port 8080
~~~

## Commands by device

### Single NVIDIA GPU

~~~bash
ftllm server /data/models/qwen \
  --device cuda \
  --max_batch 8 \
  --gpu_mem_ratio 0.9
~~~

### Multi-GPU tensor parallelism

~~~bash
ftllm server /data/models/qwen \
  --device cuda --tp 0,1 \
  --cuda_embedding \
  --max_batch 16 \
  --gpu_mem_ratio 0.9
~~~

`--tp 0,1` selects two explicit GPUs; `--tp 2` selects the first two visible GPUs.

### GPU + NUMA hybrid MoE

~~~bash
ftllm server /data/models/qwen-moe \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

Use `--moe_device cpu` on a single-socket host.

### CPU / NUMA

~~~bash
ftllm server /data/models/qwen --device cpu -t 32
ftllm server /data/models/qwen --device numa -t 64
~~~

Treat the thread counts as examples and tune them for physical cores and memory bandwidth.

## Long context and caching

~~~bash
ftllm server /data/models/qwen \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --gpu_mem_ratio 0.9
~~~

Extending the model-declared window requires a valid `--rope_scaling` configuration; insufficient calibrated KV capacity fails startup. For Qwen3, specify the original RoPE length explicitly (commonly 32768). See [context extension](context-length-extension-design.md) for examples and validation limits.

## MTP

For a model with compatible MTP weights:

~~~bash
ftllm server /data/models/qwen-with-mtp \
  --device cuda --mtp 4
~~~

`--mtp` configures draft tokens per step; `0` disables it and the current maximum is 8. MTP is not exclusive to one Qwen version.

Qwen3.5 GGUF files use an embedded NextN/MTP layer directly when present. A GGUF without that layer can attach a standalone `mtp.safetensors` extracted from a structurally matching model:

~~~bash
ftllm server /data/models/qwen3.5-target.gguf \
  --device cuda --cuda_embedding \
  --draft /data/models/qwen3.5-fp8/mtp.safetensors \
  --draft_tokens 5
~~~

A matching `config.json` must be adjacent to `mtp.safetensors`; `--draft` may also point to the directory containing both files. Loading validates the Qwen3.5 architecture, layer and hidden dimensions, head counts, vocabulary, and every required MTP tensor shape. Standalone MTP currently attaches only to GGUF targets. MTP accesses the vocabulary projection frequently, so keep `--cuda_embedding` when memory permits or throughput drops substantially.

## Qwen3.8 DFlash2

~~~bash
ftllm server /data/models/qwen3.8-27b-fp8 \
  --model_name qwen3.8 \
  --tp 2 --cuda_embedding \
  --max_batch 1 \
  --speculative_algorithm dflash \
  --speculative_draft_model_path /data/models/qwen3.8-27b-dflash2 \
  --speculative_num_draft_tokens 8
~~~

DFlash2 requires a matching standalone draft checkpoint. See the [Qwen3.8 DFlash2 benchmark](benchmarks/qwen38_27b_dflash2.md).

## Thinking and tool calling

~~~bash
ftllm server /data/models/qwen \
  --enable_thinking false \
  --tool_call_parser auto
~~~

Thinking can also be controlled per API request through `chat_template_kwargs.enable_thinking`.

## Benchmarks

- [Qwen3.6-27B-FP8 on dual RTX 5090 TP2](benchmarks/qwen36_27b_fp8.md)
- [Qwen3.8-27B DFlash2 / MTP on dual RTX 5090](benchmarks/qwen38_27b_dflash2.md)
- [Benchmark index](benchmark_en.md)

Each result applies only to the recorded checkpoint, precision, hardware, batch, and context.
