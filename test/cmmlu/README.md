CMMLU是一个综合性的中文评估基准，专门用于评估语言模型在中文语境下的知识和推理能力。
项目官网网址为: https://github.com/haonan-li/CMMLU

本目录下的chatglm.py程序会调用fastllm框架进行测试

测试步骤如下：

- 1. 克隆CMMLU仓库

``` sh
git clone https://github.com/haonan-li/CMMLU
```

- 2. 测试

```
python3 chatglm.py --model_name_or_path 此处填写模型路径 --save_dir 此处填写结果保存路径 --dtype float16
```

这里model_name_or_path可以使用ChatGLM2-6b官方的原始模型、int4模型，dtype支持float16, int8, int4

测试数据较多，过程比较漫长（我本地测试时，4090下float16模型约耗时38分钟），测试中途可以通过以下命令查看已完成的测试成绩

```
python3 eval.py 此处填写结果保存路径
```

- 3. 参考结果

|              模型  | Data精度 |  CMMLU分数 |
|-----------------: |-------- |------------|
| ChatGLM2-6b-fp16  | float32 |  50.16     |
| ChatGLM2-6b-int8  | float32 |  50.14     |
| ChatGLM2-6b-int4  | float32 |  49.63     |

