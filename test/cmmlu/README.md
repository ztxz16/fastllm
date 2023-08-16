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
# chatglm测试脚本
# 这里model_name_or_path可以使用ChatGLM2-6b官方的原始模型、int4模型，dtype支持float16, int8, int4
python3 chatglm.py --model_name_or_path 此处填写模型路径 --save_dir 此处填写结果保存路径 --dtype float16

# baichuan13b测试脚本
# 这里model_name_or_path可以使用Baichuan13B-Base或Baichuan13B-Chat官方的原始模型，dtype支持float16, int8, int4
python3 baichuan.py --model_name_or_path 此处填写模型路径 --save_dir 此处填写结果保存路径 --dtype float16
```

测试数据较多，过程比较漫长，测试中途可以通过以下命令查看已完成的测试成绩

```
python3 eval.py 此处填写结果保存路径
```

- 3. 参考结果

|              模型        | Data精度 | Shot     |  CMMLU分数 |
|-----------------------: |-------- |----------|-----------|
| ChatGLM2-6b-fp16        | float32 |0         |  50.16    |
| ChatGLM2-6b-int8        | float32 |0         |  50.14    |
| ChatGLM2-6b-int4        | float32 |0         |  49.63    |
| QWen-7b-Base-fp16       | float32 |0         |  57.43    |
| QWen-7b-Chat-fp16       | float32 |0         |  54.82    |
| Baichuan-13b-Base-int8  | float32 |5         |  55.12    |
| Baichuan-13b-Base-int4  | float32 |5         |  52.22    |
