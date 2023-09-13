# LLaMA 类模型转换参考

这个文档提供了了转换LLaMA同结构模型的方法。

Llama类模型有着基本相同的结构，但权重和prompt构造有差异。在fastllm中，通过转转模型时修改部分配置，实现对这些变体模型的支持、

## 声明

以下配置方案收集于网友整理，本项目不保证模型推理结果与原版完全一致。

## 修改方案

这里以支持推理各类Llama结构的基座模型为例，介绍如何应用本教程。

* 方案一：修改转换脚本

以alpaca2flm.py为模板修改。在创建model之后添加：

```python
    model = LlamaForCausalLM.from_pretrained(model_name).float()
    # config.json中定义了自己的model_type的需要添加
    conf = model.config.__dict__
    conf["model_type"] = "llama"
    # 接下来的部分各个Chat模型有差别，Base模型有的需要添加pre_prompt。
    torch2flm.tofile(exportPath, model, tokenizer, pre_prompt = "", 
                     user_role = "", bot_role = "", history_sep = "", 
                     dtype = dtype)
```

* 方案二：修改config.json
在下载的模型目录下，修改配置文件`config.json`中，修改"model_type"为`llama`，并增加下面的键-值对：

```json
    "pre_prompt": "",
    "user_role": "",
    "bot_role": "",
    "history_sep":  "",
```

如需添加Token ID而非字符串（类似baichuan-chat模型），可以使用“<<FLM_FIX_TOKEN_{ID}>”的格式添加。

## Base Model

见上方“[修改方案](#修改方案)”。

## Chat Model

对Chat Model，同样是修改转换脚本，或修改模型的config.json，以下是目前常见的chat model的配置：

### InternLM（书生）

* internlm/[internlm-chat-20b](https://huggingface.co/internlm/internlm-chat-20b)

```python
    conf = model.config.__dict__
    conf["model_type"] = "llama"
    torch2flm.tofile(exportPath, model, tokenizer, pre_prompt = "<s><s>", 
                     user_role = "<|User|>:", bot_role = "<eoh>\n<|Bot|>:", 
                     history_sep = "<eoa>\n<s>", dtype = dtype)
```

### XVERSE-13B

* xverse/[XVERSE-13B-Chat](https://huggingface.co/xverse/XVERSE-13B-Chat)
* xverse/[XVERSE-13B-Chat](https://huggingface.co/xverse/XVERSE-7B-Chat)
```python
    conf = model.config.__dict__
    conf["model_type"] = "llama"
    torch2flm.tofile(exportPath, model, tokenizer, pre_prompt = "", 
                     user_role = "Human: ", bot_role = "\n\nAssistant: ", 
                     history_sep = "<FLM_FIX_TOKEN_3>", dtype = dtype)
```

### llama2-chat

* meta-llama/Llama-2-chat

|Model|Llama2-chat|Llama2-chat-hf|
|---|---|---|
|7B| [meta-llama/Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
|13B| [meta-llama/Llama-2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat) | [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |

官方示例代码中，可以不用系统提示语：

```python
    torch2flm.tofile(exportPath, model, tokenizer, pre_prompt = "<FLM_FIX_TOKEN_1>", 
                     user_role = "[INST] ", bot_role = " [/INST]", 
                     history_sep = " <FLM_FIX_TOKEN_2><FLM_FIX_TOKEN_1>", dtype = dtype)
```

**Llama-2系列支持系统提示语需要修改代码**，单轮可以使用以下带有系统提示语的版本：

```python
    torch2flm.tofile(exportPath, model, tokenizer, 
                     pre_prompt = "<FLM_FIX_TOKEN_1>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, " \
        "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
        "Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, " \
        "or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, " \
        "please don't share false information.\n<</SYS>>\n\n", 
                     user_role = " ", bot_role = " [/INST]", 
                     history_sep = " <FLM_FIX_TOKEN_2><FLM_FIX_TOKEN_1>", dtype = dtype)
```

* ymcui/Chinese-Alpaca-2

|Model|Chinese-Alpaca-2|Chinese-Alpaca-2-16K|
|---|---|---|
|7B | [ziqingyang/chinese-alpaca-2-7b](https://huggingface.co/ziqingyang/chinese-alpaca-2-7b) | [ziqingyang/chinese-alpaca-2-7b-16k](https://huggingface.co/ziqingyang/chinese-alpaca-2-7b-16k) |
|13B | [ziqingyang/chinese-alpaca-2-13b](https://huggingface.co/ziqingyang/chinese-alpaca-2-13b) | [ziqingyang/chinese-alpaca-2-13b-16k](https://huggingface.co/ziqingyang/chinese-alpaca-2-13b-16k) |

```python
    torch2flm.tofile(exportPath, model, tokenizer, 
                     pre_prompt = "<FLM_FIX_TOKEN_1>[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n"
                     user_role = " ", bot_role = " [/INST]", 
                     history_sep = " <FLM_FIX_TOKEN_2><FLM_FIX_TOKEN_1>", dtype = dtype)
```