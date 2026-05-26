## 推理速度

可以使用benchmark程序进行测速，根据不同配置、不同输入，推理速度也会有一些差别

### ftllm benchmark

Python 包安装后可以直接使用 `ftllm benchmark`，也可以简写为 `ftllm bench`：

``` sh
ftllm bench Qwen/Qwen3-0.6B --device cuda --input_tokens 512 --output_tokens 128 --batch 4
```

常用参数：

- `--input_tokens`：构造 benchmark 输入的 token 长度。
- `--output_tokens`：每个请求最多生成的 token 数。
- `--batch`：同时启动的 benchmark 请求数量；如果未显式设置 `--max_batch`，会自动使用该值。
- `--temperature`：生成温度，设置为 `0` 或负数时使用 greedy decoding。

输出中会统计平均 TTFT、TPOP，以及 batch 总吞吐、首 token 后 decode 吞吐、单请求平均 token/s 等信息。

### C++ benchmark

例如:

``` sh
./benchmark -p ~/chatglm-6b-int4.flm -f ../example/benchmark/prompts/beijing.txt -b 1
./benchmark -p ~/chatglm-6b-int8.flm -f ../example/benchmark/prompts/beijing.txt -b 1
./benchmark -p ~/chatglm-6b-fp16.flm -f ../example/benchmark/prompts/hello.txt -b 512 -l 18
```

|              模型 | Data精度 | 平台               | Batch    | 最大推理速度(token / s) |
|-----------------:|---------|--------------------|-----------|---------------------:|
| ChatGLM-6b-int4  | float32 |  RTX 4090          |         1 |                  176 |
| ChatGLM-6b-int8  | float32 |  RTX 4090          |         1 |                  121 |
| ChatGLM-6b-fp16  | float32 |  RTX 4090          |        64 |                 2919 |
| ChatGLM-6b-fp16  | float32 |  RTX 4090          |       256 |                 7871 |
| ChatGLM-6b-fp16  | float32 |  RTX 4090          |       512 |                10209 |
| ChatGLM-6b-int4  | float32 |  Xiaomi 10 Pro - 4 Threads | 1 |                4 ~ 5 |
