## 模型获取

### 模型库

可以在以下链接中下载已经转换好的模型

[huggingface](https://huggingface.co/huangyuyang) 

### 模型导出

#### ChatGLM模型导出 (默认脚本导出ChatGLM2-6b模型)

``` sh
# 需要先安装ChatGLM-6B环境
# 如果使用自己finetune的模型需要修改chatglm_export.py文件中创建tokenizer, model的代码
cd build
python3 tools/chatglm_export.py chatglm2-6b-fp16.flm float16 #导出float16模型
python3 tools/chatglm_export.py chatglm2-6b-int8.flm int8 #导出int8模型
python3 tools/chatglm_export.py chatglm2-6b-int4.flm int4 #导出int4模型
```

#### baichuan模型导出 (默认脚本导出baichuan-13b-chat模型)

``` sh
# 需要先安装baichuan环境
# 如果使用自己finetune的模型需要修改baichuan2flm.py文件中创建tokenizer, model的代码
# 根据所需的精度，导出相应的模型
cd build
python3 tools/baichuan2flm.py baichuan-13b-fp16.flm float16 #导出float16模型
python3 tools/baichuan2flm.py baichuan-13b-int8.flm int8 #导出int8模型
python3 tools/baichuan2flm.py baichuan-13b-int4.flm int4 #导出int4模型
```

#### baichuan2模型导出 (默认脚本导出baichuan2-7b-chat模型)

``` sh
# 需要先安装baichuan2环境
# 如果使用自己finetune的模型需要修改baichuan2_2flm.py文件中创建tokenizer, model的代码
# 根据所需的精度，导出相应的模型
cd build
python3 tools/baichuan2_2flm.py baichuan2-7b-fp16.flm float16 #导出float16模型
python3 tools/baichuan2_2flm.py baichuan2-7b-int8.flm int8 #导出int8模型
python3 tools/baichuan2_2flm.py baichuan2-7b-int4.flm int4 #导出int4模型
```

#### MOSS模型导出

``` sh
# 需要先安装MOSS环境
# 如果使用自己finetune的模型需要修改moss_export.py文件中创建tokenizer, model的代码
# 根据所需的精度，导出相应的模型
cd build
python3 tools/moss_export.py moss-fp16.flm float16 #导出float16模型
python3 tools/moss_export.py moss-int8.flm int8 #导出int8模型
python3 tools/moss_export.py moss-int4.flm int4 #导出int4模型
```

#### LLAMA系列模型导出
``` sh
# 修改build/tools/alpaca2flm.py程序进行导出
# 不同llama模型使用的指令相差很大，需要参照torch2flm.py中的参数进行配置
```
一些模型的转换可以[参考这里的例子](docs/llama_cookbook.md)

#### QWEN模型导出
* **Qwen**
```sh
# 需要先安装QWen环境
# 如果使用自己finetune的模型需要修改qwen2flm.py文件中创建tokenizer, model的代码
# 根据所需的精度，导出相应的模型
cd build
python3 tools/qwen2flm.py qwen-7b-fp16.flm float16 #导出float16模型
python3 tools/qwen2flm.py qwen-7b-int8.flm int8 #导出int8模型
python3 tools/qwen2flm.py qwen-7b-int4.flm int4 #导出int4模型
```

* **Qwen1.5**

```sh
# 需要先安装QWen2环境（transformers >= 4.37.0）
# 根据所需的精度，导出相应的模型
cd build
python3 tools/llamalike2flm.py qwen1.5-7b-fp16.flm float16 "qwen/Qwen1.5-4B-Chat" #导出wen1.5-4B-Chat float16模型
python3 tools/llamalike2flm.py qwen1.5-7b-int8.flm int8 "qwen/Qwen1.5-7B-Chat" #导出Qwen1.5-7B-Chat int8模型
python3 tools/llamalike2flm.py qwen1.5-7b-int4.flm int4 "qwen/Qwen1.5-14B-Chat" #导出Qwen1.5-14B-Chat int4模型
# 最后一个参数可替换为模型路径
```

#### MINICPM模型导出
```sh
# 需要先安装MiniCPM环境（transformers >= 4.36.0） 
# 默认脚本导出iniCPM-2B-dpo-fp16模型
cd build 
python tools/minicpm2flm.py minicpm-2b-float16.flm #导出dpo-float16模型
./main -p minicpm-2b-float16.flm # 执行模型
```