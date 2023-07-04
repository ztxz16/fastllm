# pyfastllm 

by [wildkid1024](https://github.com/wildkid1024) 

pyfastllm是基于fastllm的python api接口实现，通过pyfastllm可以更加灵活地编码实现pythonic场景，满足更复杂更个性化的业务需求。

- 对接fastapi、flask等web框架，向外提供数据接口
- 利用python yield生成器语言特性，流式问答响应
- 对接Lora、Ptuning等微调方法，下游任务可微调(开发中...)
- 无缝对接加速HugingFace模型库，无痛加速迁移原有业务代码(开发中...)
- 其他更多...

## 编译安装

本地编译安装fastllm的python接口，以两种方式编译运行：
1. 动态库方式：编译为动态库，需放在python运行加载目录下
2. wheel包方式：编译为wheel包，安装在python的site-packages下，但暂不支持cuda

### 动态库方式

Cpp手动编译：
```
mkdir build-py
cd build-py
cmake .. -DUSE_CUDA=ON -DPY_API=ON
make -j4
python cli.py -p chatglm-6b-int8.bin -t 8  # 与cpp编译的运行结果保持一致
```

Python脚本编译：

```
cd pyfastllm
python build_libs --cuda
python cli.py -p chatglm-6b-int8.bin -t 8 
```

### wheel包方式

```
cd pyfastllm
python setup.py build
python setup.py install 
python cli.py -p chatglm-6b-int8.bin -t 8 
```

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
> fastllm.create_llm(model_path: str)-> fastllm.model  # 从本地权重文件生成对应的模型实例，基于规则匹配
> fastllm.set_low_memory() # 低内存模式下运行，默认为False
> fastllm.get_low_memory() # 查看当前是否为低内存运行模式

### fastllm模块

> fastllm.Tokenizer: 分词及编解码工具
> Tips: 该类不可直接实例化，只可通过model.weight.tokenizer访问具体实例
- fastllm.Tokenizer.encode(prompt:str) # 将prompt分词并进行编码
- fastllm.Tokenizer.decode(output_ids:fastllm.Tensor) # 将fastllm.Tensor解码为对应字符串
- fastllm.Tokenizer.decode(output_ids: list[int]) # 将list[int]解码为对应的字符串
- fastllm.Tokenizer.decode_byte(output_ids: fastllm.Tensor) # 将Tensor解码对应字节流

### fastllm模型

> fastllm.ChatGLMModel: 具体模型实例，其中chatglm可以更换为llama、alpaca、Moss等模型
- fastllm.ChatGLMModel() # 初始化模型实例
- __call__(input_ids:fastllm.Tensor, attention_mask:fastllm.Tensor, position_ids:fastllm.Tensor, penalty_factor:fastllm.Tensor, pastKeyValues:memory_view) # 以类call function的方式调用模型进行推理 
- fastllm.ChatGLMModel.load_weights(model_path:str) # 从文件路径中加载模型权重
- fastllm.ChatGLMModel.response(inputs:str, callback:function) # 发送字符串到模型中并使用callback函数接受处理返回的答案
- fastllm.ChatGLMModel.response_batch(inputs:list[str], callback:function) -> outputs:list[str] # 发送列表字符串到模型中并使用callback函数接受处理返回的答案
- fastllm.ChatGLMModel.warmup()  # GPU热身，填充GPU，防止冷启动 
- fastllm.ChatGLMModel.launch_response(inputs:str)->handle_id:int  # 多线程下使用，填充第一个token，并返回多线程的线程id
- fastllm.ChatGLMModel.fetch_response(handle_id:int) # 根据线程ID从消息队列中取出对应的消息并返回
- fastllm.ChatGLMModel.save_lowbit_model(model_path:str, q_bit:int) # 量化保持低bit的权重并保存模型


## 开发计划

- [ ]  模型运行参数对象类，封装模型运行时参数，包含模型路径、运行线程数、是否为低内存模型、惩罚因子、温度等
- [ ]  Tensor的深复制和浅复制，以及基础运算符重载
- [ ]  编解码部分优化，合并不同的返回类型
- [ ]  暴露更多的底层api接口，按照module的方式定义模型的点，拼接model实现自定义model
- [ ]  修改response_batch的output_str函数，以返回值的形式返回响应