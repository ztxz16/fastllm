# pyfastllm 

by [wildkid1024](https://github.com/wildkid1024) 

pyfastllm是基于fastllm的python api接口实现，通过pyfastllm可以更加灵活地编码实现pythonic场景，满足更复杂更个性化的业务需求。

- 对接fastapi、flask等web框架，向外提供数据接口
- 利用python yield生成器语言特性，流式问答响应
- 对接Lora、Ptuning等微调方法，下游任务可微调(开发中...)
- 无缝对接加速HugingFace模型库，无痛加速迁移原有业务代码(开发中...)
- 其他更多...

## 版本更新

### v0.1.5 2023-09-12

- 修复wheel编译安装部分
- 文件合并，修复导入

### v0.1.4 2023-09-12

- 修复了一些后端接口变动的bug
- 增加了新的ops, 支持低级op操作


### v0.1.3 2023-07-08

- 增加使用和API接口文档
- 增加fastllm-convert模型转换接口及命令行转化工具
- 修复部分因为cpp新接口导致的bug


## 编译安装

本地编译安装fastllm的python接口，以两种方式编译运行：
1. 动态库方式：编译为动态库，需放在python运行加载目录下
2. wheel包方式：编译为wheel包，安装在python的site-packages下，但暂不支持cuda

### 动态库方式

> 动态库安装方式暂不支持模型转换

首先下载pybind11 c++依赖:

```sh
git submodule init 
git submodule update  # 下载pybind11依赖
```

Cpp手动编译：
```sh
mkdir build-py
cd build-py
cmake .. -DUSE_CUDA=ON -DPY_API=ON
make -j4
python cli.py -p chatglm-6b-int8.bin -t 8  # 与cpp编译的运行结果保持一致
```

Python脚本编译：

```sh
cd pyfastllm
python build_libs --cuda
python cli.py -p chatglm-6b-int8.bin -t 8 
```

### wheel包方式

> 注意wheel包安装方式暂不支持cuda 

首先下载pybind11：

```bash
pip install pybind11
```

```sh
cd pyfastllm
python setup.py build
python setup.py install 
python cli.py -p chatglm-6b-int8.bin -t 8 
```

## 使用

### python 调用
在demo文件夹中存放了几种常见的代码示例：

demo/cli.py: 以回调函数方式输出回答示例
demo/cli_thread.py: 多线程调用api接口示例(推荐)
demo/cli_low_api.py: 底层API调用示例
demo/convert_model.py: 模型转换示例
demo/web_api.py, demo/web_api_client.py: fastapi webapi调用
demo/test_ops: 部分op的使用样例及测试

### 命令行工具

使用命令行工具对模型进行转换，使用方法与convert_model.py类似：

```sh
$ fastllm-convert --help
$ fastllm-convert -m chatglm6B -p hf_model_path -o output_flm_path  
```

### 动态batch使用示例
```sh
mkdir build-py
cd build-py && cmake .. -DPY_API=ON -DUSE_CUDA=ON && make -j && cd -
cd pyfastllm/demo
python web_api.py -m 0 -p path_for_chatglm --max_batch_size 32
```
可以使用locust进行压测。A100 40G，chatglm fp16 压测部分结果如下：
|    并发数 | 平均调用时间(s) | TP95(s) | TP99(s) |
|----------:|------|------|------|
| 1         | 3.07 | 4.2  | 4.8 |
| 10        | 6.11 | 11.0 | 12.0 |
| 16        | 6.82 | 15.0 | 16.0 |
| 32        | 10.74 | 16.0 | 20.0 |
## API编程接口

### fastllm数据结构

> fattllm.Tensor数据类型
- fastllm.float32  
- fastllm.bfloat16
- fastllm.int16
- fastllm.int8
- fastllm.int4
- fastllm.int2
- fastllm.float16

> fastllm.Tensor: fastllm基础张量结构
- fastllm.Tensor()
- fastllm.Tensor(Datatype)
- fastllm.Tensor(Datatype, Dims:list[int])
- fastllm.Tensor(Datatype, Dims:list[int], Data:list[float])
- fastllm.Tensor(Data:fastllm.Tensor)
- fastllm.Tensor.to_list() # 将Tensor转化list并返回
- fastllm.Tensor.to() # 将Tensor转移到对应设备上
- fastllm.Tensor.zeros(Dims:list[int]) # 按照Dims生成全零矩阵
- fastllm.Tensor.cat(Data:list[fastllm.Tensor], axis:int) # 将Tensor按照axis(默认为0)方向上拼接

### fastllm函数

> fastllm.get_llm_type(model_path:str)->str # 获取当前model的类型
> fastllm.set_threads(thread:int) -> None # 设置当前运行线程数，默认为4
> fastllm.get_threads()->int  # 获取当前运行线程数
> fastllm.set_low_memory(flag:bool) # 低内存模式下运行，默认为False
> fastllm.get_low_memory() # 查看当前是否为低内存运行模式
> fastllm.create_llm(model_path: str)-> fastllm.model  # 从本地权重文件生成对应的模型实例，基于规则匹配

### fastllm模块

> fastllm.Tokenizer: 分词及编解码工具
> Tips: 该类不可直接实例化，只可通过model.weight.tokenizer访问具体实例
- fastllm.Tokenizer.encode(prompt:str) # 将prompt分词并进行编码
- fastllm.Tokenizer.decode(output_ids:fastllm.Tensor) # 将fastllm.Tensor解码为对应字符串
- fastllm.Tokenizer.decode(output_ids: list[int]) # 将list[int]解码为对应的字符串
- fastllm.Tokenizer.decode_byte(output_ids: fastllm.Tensor) # 将Tensor解码对应字节流

> fastllm.WeightMap: 模型的权重词典
> Tips: 该类不可直接实例化，只可通过model.weight访问具体实例
- fastllm.WeightMap.tokenizer： 访问权重中的tokenizer实例
- fastllm.WeightMap.save_lowbit(output_path:str, bit:int)：量化并保存低bit的权重
- fastllm.WeightMap.set_kv(key:str, value:str)：设置模型的weight字典
- fastllm.WeightMap.set_weight(key:str, )：为weight添加具体Tensor
- .fastllm.WeightMap\['key'\]: 根据key的名称得到对应的Tensor

### fastllm模型

> fastllm.ChatGLMModel: 具体模型实例，其中chatglm可以更换为llama、alpaca、Moss等模型
- fastllm.ChatGLMModel.model_type: 模型类型属性，区分不同的模型
- fastllm.ChatGLMModel.weight：对应的weightmap
- fastllm.ChatGLMModel.block_cnt：模型中block的数量
- fastllm.ChatGLMModel() # 初始化模型实例
- __call__(input_ids:fastllm.Tensor, attention_mask:fastllm.Tensor, position_ids:fastllm.Tensor, penalty_factor:fastllm.Tensor, pastKeyValues:memory_view) # 以类call function的方式调用模型进行推理 
- fastllm.ChatGLMModel.load_weights(model_path:str) # 从文件路径中加载模型权重
- fastllm.ChatGLMMode.make_history(history:str, round:int, input:str, output:str) # 基于历史对话和当前输入输出构造送入模型的历史对话
- fastllm.ChatGLMMode.make_input(history:str, round:int, input:str) # 基于历史对话和当前输入构造送入模型的对话输入
- fastllm.ChatGLMModel.response(inputs:str, callback:function) # 发送字符串到模型中并使用callback函数接受处理返回的答案
- fastllm.ChatGLMModel.response_batch(inputs:list[str], callback:function) -> outputs:list[str] # 发送列表字符串到模型中并使用callback函数接受处理返回的答案
- fastllm.ChatGLMModel.warmup()  # GPU热身，填充GPU，防止冷启动 
- fastllm.ChatGLMModel.launch_response(inputs:str)->handle_id:int  # 多线程下使用，填充第一个token，并返回多线程的线程id
- fastllm.ChatGLMModel.fetch_response(handle_id:int) # 根据线程ID从消息队列中取出对应的消息并返回
- fastllm.ChatGLMModel.save_lowbit_model(model_path:str, q_bit:int) # 量化保持低bit的权重并保存模型


支持的模型列表：

| 模型名称 | 对应类 | 备注 
| -- | -- | -- 
| ChatGLM-6B | fastllm.ChatGLMModel |
| ChatGLM2-6B | fastllm.ChatGLMModel | 在权重中标注版本
| Moss | fastllm.MossModel |
| Alpaca | fastllm.llamaModel | 


## 开发计划(TODO)

- [x]  修改response_batch的output_str函数，以返回值的形式返回答案
- [x]  编解码部分优化，合并不同的返回类型
- [ ]  对接numpy等矩阵库
- [ ]  Tensor的深复制和浅复制，以及基础运算符重载
- [ ]  fix low_api下pastKV复制的bug
- [x]  模型运行参数对象类，封装模型运行时参数，包含模型路径、运行线程数、是否为低内存模型、惩罚因子、温度等
- [ ]  增加更多的op
- [ ]  增加module

