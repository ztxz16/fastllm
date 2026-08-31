# Laguna Benchmark

[中文](laguna.md) · [Benchmark index](../benchmark_en.md) · [Deployment guide](../laguna_en.md)

The repository currently contains functional validation for Laguna multi-GPU, CUDA Graph, hybrid MoE, and quantized paths, but no formal speed table on a consistent hardware platform. The configurations below are suggested measurement starting points.

| Device layout | Suggested launch command | Speed |
| --- | --- | --- |
| Four-GPU TP4 | `ftllm server /data/models/laguna --tp 4 --max_batch 8 --gpu_mem_ratio 0.9` | Not measured |
| Eight-GPU TP8 | `ftllm server /data/models/laguna --tp 0,1,2,3,4,5,6,7 --max_batch 16 --gpu_mem_ratio 0.9` | Not measured |
| GPU with NUMA MoE | `ftllm server /data/models/laguna --device cuda --moe_device numa --chunked_prefill_size 8192` | Not measured |

Published results should include the GPU interconnect, checkpoint precision, CUDA Graph state, context length, and batch size. NVFP4 and INT4_GROUP32 results must be reported separately. See the [benchmark tools](../../test/benchmark/README.md).
