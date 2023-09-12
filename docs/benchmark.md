## 推理速度

可以使用benchmark程序进行测速，根据不同配置、不同输入，推理速度也会有一些差别

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
