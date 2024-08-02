# 支持的模型 Supported Models

## 说明

目前Fastllm加载模型有以下几种方式。

* **加载后转换（两行加速模式）** (convert on-the-fly)  
    将原始模型加载为HuggingFace模型，再通过`from_hf()`方法，转换并加速，这种方法内存占用大且速度慢，目前不再推荐。

* **离线转换** (convert offline)  
    将原始模型转换为.flm格式的模型，一些[模型](#flm模型库)已经转换好。

* **直接读取** (load from Huggingface .safetensors)  
    直接读取HuggingFace上发布的模型，仅支持.safetensors格式的模型。


## 支持模型一览 Model List


* ✔ 表示支持该方式，并测试通过；  
    ✔ means supports this mode and passes the test.

* ❌ 表示本应该支持该方式，但实际测试后发现本功能并不受支持，可能在后续版本修复；  
    ❌ means this method is supposed to be supported, but failed after actual testing.

* √ 表示支持，但是还没有测试过。  
    √ means supported, but not tested.

### GLM系列

|              模型  | 加载后转换 |  离线转换  |  直接读取  |
|-----------------: |------------|------------|------------|
| THUDM/ChatGLM-6b | [✔](#chatglm系列) | [✔](#chatglm模型导出-默认脚本导出chatglm2-6b模型) |  |
| THUDM/ChatGLM-6b-int8 | [✔](#Cchatglm系列) | ❌ |  |
| THUDM/ChatGLM-6b-int4 | [✔](#chatglm系列) | ❌ |  |
| THUDM/ChatGLM2-6b | [✔](#chatglm系列) | [✔](#chatglm模型导出-默认脚本导出chatglm2-6b模型) |  |
| THUDM/glm-large-chinese |  | [✔](tools\scripts/glm_export.py) | |
| THUDM/ChatGLM2-6b-int8 | [✔](#chatglm系列) | ❌ |  |
| THUDM/ChatGLM2-6b-int4 | [✔](#chatglm系列) | ❌ |  |
| THUDM/ChatGLM2-6b-32k | [✔](#chatglm系列) | [✔](#chatglm模型导出-默认脚本导出chatglm2-6b模型) |  |
| THUDM/ChatGLM3-6b | [✔](#chatglm系列) | [✔](#chatglm模型导出-默认脚本导出chatglm2-6b模型) |  |
| THUDM/ChatGLM3-6b-32k | [✔](#chatglm系列) | [✔](#chatglm模型导出-默认脚本导出chatglm2-6b模型) |  |
| THUDM/ChatGLM3-6b-128k | ❌ | ❌ |  |
| THUDM/glm-4-9b-chat | [✔](#chatglm系列) | [✔](#chatglm模型导出-默认脚本导出chatglm2-6b模型) | ✔ |
| THUDM/codegeex4-all-9b | [✔](#chatglm系列)<sup>2</sup> | [✔](#chatglm模型导出-默认脚本导出chatglm2-6b模型)<sup>2</sup> | ✔ |

> 注2：需要手动设置 pre_prompt

### Qwen系列

|              模型  | 加载后转换 |  离线转换  |  直接读取  |
|-------------------: |------------|------------|------------|
| Qwen/Qwen-7B-Chat   | [✔](#其它模型) | [✔](#qwen模型导出) |  |
| Qwen/Qwen-14B-Chat  | [✔](#其它模型) | [✔](#qwen模型导出) |  |
| Qwen/Qwen-72B-Chat  | [✔](#其它模型) | [✔](#qwen模型导出) |  |
| Qwen/Qwen-1_8B-Chat | [✔](#其它模型) | [✔](#qwen模型导出) |  |
| Qwen/Qwen1.5-0.5B-Chat | [✔](#其它模型) | [✔](#qwen模型导出) | ✔<sup>3</sup> |
| Qwen/Qwen1.5-1.8B-Chat | [✔](#其它模型) | [✔](#qwen模型导出) | ✔<sup>3</sup> |
| Qwen/Qwen1.5-4B-Chat   | [✔](#其它模型) | [✔](#qwen模型导出) | ✔<sup>3</sup> |
| Qwen/Qwen1.5-7B-Chat   | [✔](#其它模型) | [✔](#qwen模型导出) | ✔<sup>3</sup> |
| Qwen/Qwen1.5-14B-Chat  | [✔](#其它模型) | [✔](#qwen模型导出) | ✔<sup>3</sup> |
| Qwen/Qwen1.5-72B-Chat  | [✔](#其它模型) | [✔](#qwen模型导出) | ✔<sup>3</sup> |
| Qwen/Qwen1.5-32B-Chat  | [✔](#其它模型) | [✔](#qwen模型导出) | ✔<sup>3</sup> |
| Qwen/Qwen1.5-110B-Chat | [√](#其它模型) | [√](#qwen模型导出) | √<sup>3</sup> |
| Qwen/CodeQwen1.5-7B-Chat | [✔](#其它模型) | [✔](#qwen模型导出) | ✔ |
| Qwen/Qwen2-0.5B-Instruct | [✔](#其它模型) | [✔](#qwen模型导出) | ✔ |
| Qwen/Qwen2-1.5B-Instruct | [✔](#其它模型) | [✔](#qwen模型导出) | ✔ |
| Qwen/Qwen2-7B-Instruct   | [✔](#其它模型) | [✔](#qwen模型导出) | ✔ |
| Qwen/Qwen2-72B-Instruct  |  | [✔](#qwen模型导出) | ✔ |

> 注3： 需要更新，检查 `tokenizer_config.json` 是否为最新版本

### DeepSeek系列

|                                       模型  | 加载后转换 |  离线转换  |  直接读取  |
|-------------------------------------------: |------------|------------|------------|
| deepseek-ai/Deepseek-Coder-1.3B-Instruct    | [✔](llama_cookbook.md#deepseek-coder) | [✔](llama_cookbook.md#deepseek-coder) | ✔ |
| deepseek-ai/Deepseek-Coder-6.7B-Instruct    | [✔](llama_cookbook.md#deepseek-coder) | [✔](llama_cookbook.md#deepseek-coder) | ✔ |
| deepseek-ai/Deepseek-Coder-7B-Instruct v1.5 | [✔](llama_cookbook.md#deepseek-coder) | [✔](llama_cookbook.md#deepseek-coder) | ✔ |
| deepseek-ai/deepseek-coder-33b-instruct     | [√](llama_cookbook.md#deepseek-coder) | [√](llama_cookbook.md#deepseek-coder) | ✔ |
| deepseek-ai/DeepSeek-V2-Chat                | √ | ✔ | √ |
| deepseek-ai/DeepSeek-V2-Lite-Chat           | √ | ✔ | ✔ |
| deepseek-ai/DeepSeek-Coder-V2-Instruct      | √ | ✔ | √ |
| deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | √ | ✔ | ✔ |

> 注4： Python ftllm用AutoTokenizer而不使用Fastllm Tokenizer可以实现加载，但是C++程序尚不支持加载该模型的Tokenizer。


### LLaMA类模型

|              模型  | 加载后转换 |  离线转换  |  直接读取  |
|-----------------: |------------|------------|------------|
| meta-llama/Llama-2-7b-chat-hf | [✔](llama_cookbook.md#llama2-chat) | [✔](llama_cookbook.md#llama2-chat) |  |
| meta-llama/Llama-2-13b-chat-hf | [✔](llama_cookbook.md#llama2-chat) | [✔](llama_cookbook.md#llama2-chat) |  |
| codellama/CodeLlama-7b-Instruct-hf | [✔](llama_cookbook.md#llama2-chat) | [✔](llama_cookbook.md#llama2-chat) |  |
| codellama/CodeLlama-13b-Instruct-hf | [✔](llama_cookbook.md#llama2-chat) | [✔](llama_cookbook.md#llama2-chat) |  |
| xverse/XVERSE-13B-Chat | [✔](llama_cookbook.md#xverse) | [✔](llama_cookbook.md#xverse) |  |
| xverse/XVERSE-7B-Chat | [✔](llama_cookbook.md#xverse) | [✔](llama_cookbook.md#xverse) |  |
|  |  |  |  |
| internlm/internlm-chat-7b | [✔](llama_cookbook.md#internlm书生) | [✔](llama_cookbook.md#internlm书生) |  |
| internlm/internlm-chat-20b | [✔](llama_cookbook.md#internlm书生) | [✔](llama_cookbook.md#internlm书生) |  |
| internlm/internlm2-chat-1_8b | [✔](llama_cookbook.md#internlm书生) | [✔](llama_cookbook.md#internlm书生) | ❌<sup>4</sup> |
| internlm/internlm2-chat-7b | [✔](llama_cookbook.md#internlm书生) | [✔](llama_cookbook.md#internlm书生) | ❌<sup>4</sup> |
| internlm/internlm2-chat-20b | [✔](llama_cookbook.md#internlm书生) | [✔](llama_cookbook.md#internlm书生) | ❌<sup>4</sup> |
|  |  |  |  |
| 01-ai/Yi-6B-Chat | [✔](llama_cookbook.md#yi) | [✔](llama_cookbook.md#yi) | ❌<sup>4</sup> |
| 01-ai/Yi-34B-Chat | [✔](llama_cookbook.md#yi) | [✔](llama_cookbook.md#yi) | ❌<sup>4</sup> |
| SUSTech/SUS-Chat-34B | [✔](llama_cookbook.md#llama2-chat) | [✔](llama_cookbook.md#llama2-chat) |  |
|  |  |  |  |
| meta-llama/Meta-Llama-3-8B-Instruct |  | [✔](tools/scripts/llama3_to_flm.py) | ✔ |
| meta-llama/Meta-Llama-3-70B-Instruct |  | [✔](tools/scripts/llama3_to_flm.py) | ✔ |

### 其它模型

|              模型  | 加载后转换 |  离线转换  |  直接读取  |
|-----------------: |------------|------------|------------|
| fnlp/moss-moon-003-sft | [✔]() | [✔](#moss模型导出) |  |
| fnlp/moss-moon-003-sft-plugin | [✔]() | [✔](#moss模型导出) |  |
|  |  |  |  |
| baichuan-inc/baichuan-13b-chat | [✔](#其它模型) | [✔](#baichuan模型导出-默认脚本导出baichuan-13b-chat模型) |  |
| baichuan-inc/Baichuan2-7B-Chat | [✔](#其它模型) | [✔](#baichuan2模型导出-默认脚本导出baichuan2-7b-chat模型) |  |
| baichuan-inc/baichuan2-13b-chat | [✔](#其它模型) | [✔](#baichuan2模型导出-默认脚本导出baichuan2-7b-chat模型) |  |
|  |  |  |  |
| openbmb/MiniCPM-2B-sft-fp16 | [✔](#其它模型) | [✔](#minicpm模型导出) |  |
| openbmb/MiniCPM-2B-dpo-fp16 | [✔](#其它模型) | [✔](#minicpm模型导出) |  |


### 加载后转换（两行加速模式）(convert on-the-fly)

#### ChatGLM系列

``` python
# 这是原来的程序，通过huggingface接口创建模型
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code = True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code = True)

# 加入下面这两行，将huggingface模型转换成fastllm模型
# 目前from_hf接口只能接受原始模型，或者ChatGLM的int4, int8量化模型，暂时不能转换其它量化模型
from ftllm import llm
model = llm.from_hf(model, tokenizer, dtype = "float16") # dtype支持 "float16", "int8", "int4"
model = model.eval()
```

model支持了ChatGLM的API函数`chat()`, `stream_chat()`，因此ChatGLM的demo程序无需改动其他代码即可运行

#### 其它模型

``` python
# 通过huggingface接口创建模型，参考每个模型readme.md中的加载方式
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code = True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code = True)

# 加入下面这两行，将huggingface模型转换成fastllm模型
# 目前from_hf接口只能接受原始模型，或者ChatGLM的int4, int8量化模型，暂时不能转换其它量化模型
from ftllm import llm
model = llm.from_hf(model, tokenizer, dtype = "float16") # dtype支持 "float16", "int8", "int4"
```
ftllm实现了兼容Transformers的`generate()`方法。


转好的模型也可以导出到本地文件，之后可以直接读取，也可以使用fastllm cpp接口读取

``` python
model.save("model.flm"); # 导出fastllm模型
new_model = llm.model("model.flm"); # 导入flm模型
```

### flm模型库

可以在以下链接中找到一部分已经转换好的模型

[huggingface](https://huggingface.co/huangyuyang) [modelscope](https://modelscope.cn/profile/huangyuyang)

### 模型导出(convert offline)

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
一些模型的转换可以[参考这里的例子](llama_cookbook.md)

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