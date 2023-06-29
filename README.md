# fastllm

## 介绍

fastllm是纯c++实现，无第三方依赖的高性能大模型推理库

6~7B级模型在安卓端上也可以流畅运行

部署交流QQ群： 831641348

| [快速开始](#快速开始) | [模型获取](#模型获取) | 

## 功能概述

- 🚀 纯c++实现，便于跨平台移植，可以在安卓上直接编译
- 🚀 ARM平台支持NEON指令集加速，X86平台支持AVX指令集加速，NVIDIA平台支持CUDA加速，各个平台速度都很快就是了
- 🚀 支持浮点模型（FP32), 半精度模型(FP16), 量化模型(INT8, INT4) 加速
- 🚀 支持Batch速度优化
- 🚀 支持流式输出，很方便实现打字机效果
- 🚀 支持并发计算时动态拼Batch
- 🚀 支持python调用
- 🚀 前后端分离设计，便于支持新的计算设备
- 🚀 目前支持ChatGLM模型，各种LLAMA模型(ALPACA, VICUNA等)，BAICHUAN模型，MOSS模型

## 推理速度

6B级int4模型单4090延迟最低约5.5ms

6B级fp16模型单4090最大吞吐量超过10000 token / s

6B级int4模型在骁龙865上速度大约为4~5 token / s

[详细测试数据点这里](docs/benchmark.md)

## 快速开始

### 编译

建议使用cmake编译，需要提前安装c++编译器，make, cmake

gcc版本建议9.4以上，cmake版本建议3.23以上

GPU编译需要提前安装好CUDA编译环境，建议使用尽可能新的CUDA版本

使用如下命令编译

``` sh
cd fastllm
mkdir build
cd build
cmake .. -DUSE_CUDA=ON # 如果不使用GPU编译，那么使用 cmake .. -DUSE_CUDA=OFF
make -j
```

编译完成后，可以使用如下命令安装简易python工具包

``` sh
cd fastllm/build/tools
python setup.py install
```

### 运行demo程序

我们假设已经获取了名为`model.flm`的模型（参照 [模型获取](#模型获取)，初次使用可以先下载转换好的模型)

编译完成之后在build目录下可以使用下列demo:
``` sh
# 命令行聊天程序, 支持打字机效果
./main -p model.flm 

# 简易webui, 使用流式输出 + 动态batch，可多路并发访问
./webui -p model.flm --port 1234 

# python版本的命令行聊天程序，使用了模型创建以及流式对话效果
python tools/cli_demo.py -p model.flm 

# python版本的简易webui，需要先安装streamlit-chat
streamlit run tools/web_demo.py model.flm 

```

### 简易python调用

编译后如果安装了简易python工具包，那么可以使用python来调用一些基本的API （如果没有安装，也可以在直接import编译生成的tools/fastllm_pytools来使用)

``` python
# 模型创建
from fastllm_pytools import llm
model = llm.model("model.flm")

# 生成回复
print(model.response("你好"))

# 流式生成回复
for response in model.stream_response(query):
    print(response, flush = True, end = "")

```


### PC 使用python api

```
mkdir build-py
cd build-py
cmake .. -DPY_API=ON -DUSE_CUDA=ON （只使用CPU则使用 cmake .. -DPY_API=ON 即可）
make -j
cd -
python cli.py  -m chatglm -p chatglm-6b-int8.bin 或  
python web_api.py  -m chatglm -p chatglm-6b-int8.bin  
```
上述web api可使用python web_api_client.py进行测试

### Android

```
# Android上需要下载NDK工具编译
mkdir build-android
cd build-android
export NDK=<your_ndk_directory>
# 如果手机不支持，那么去掉 "-DCMAKE_CXX_FLAGS=-march=armv8.2a+dotprod" （比较新的手机都是支持的）
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_CXX_FLAGS=-march=armv8.2a+dotprod ..
make -j4
```


编译后会在build目录下生成：

1. main: 示例程序

2. quant: 量化程序

## 运行示例程序

./main -h 可以查看具体参数信息，以下是一些简单示例：

### 运行ChatGLM-6B模型

```
./main -m chatglm -p chatglm-6b-int8.bin
```

### 运行baichuan模型

```
./main -m baichuan -p baichuan-int8.bin
```

### 运行MOSS模型

```
./main -m moss -p moss-int8.bin
```

### 在Android上运行

可以在Android设备上安装termux软件，并在其中执行termux-setup-storage获得读取手机文件的权限。然后将NDK编译出的main文件和模型存入手机，然后在termux中运行main文件（需要把main文件拷贝到termux的根目录下，否则无权限运行）

### 运行webui

webui 由 [Jacques CHEN](http://whchen.net/index.php/About.html) 提供

编译出webui后，需要在运行目录中放入example/webui/web文件夹以及模型文件（默认为chatglm-6b-v1.1-int4.bin文件)，然后运行既可部署网页端服务

## 模型获取

### 原始模型

如果使用原生的ChatGLM-6B模型或者MOSS模型，可以在百度网盘中直接获得量化的模型：

[原始模型](https://pan.baidu.com/s/1DyGOWqKFbpBSSi93PJe6Ug) 提取码：pk7q

如果需要导出自己的模型，可参照如下步骤

### ChatGLM模型导出

```
# 需要先安装ChatGLM-6B环境
# 如果使用自己finetune的模型需要修改chatglm_export.py文件中创建tokenizer, model的代码
# 如果使用量化模型，需要先编译好quant文件，这里假设已经存在build/quant文件
cd tools
python3 chatglm_export.py ../chatglm-6b.bin # 导出浮点模型
cd ../build
./quant -m chatglm -p ../chatglm-6b.bin -o ../chatglm-6b-fp16.bin -b 16 #导出float16模型
./quant -m chatglm -p ../chatglm-6b.bin -o ../chatglm-6b-int8.bin -b 8 #导出int8模型
./quant -m chatglm -p ../chatglm-6b.bin -o ../chatglm-6b-int4.bin -b 4 #导出int4模型
```

### baichuan模型导出

```
# 需要先安装baichuan环境
# 默认使用的是经过sft训练的对话模型，如果使用其余模型需要修改导出文件
# 如果使用量化模型，需要先编译好quant文件，这里假设已经存在build/quant文件
cd tools
python3 baichuan_peft2flm.py ../baichuan.bin # 导出浮点模型
cd ../build
./quant -m baichuan -p ../baichuan.bin -o ../baichuan-fp16.bin -b 16 #导出float16模型
./quant -m baichuan -p ../baichuan.bin -o ../baichuan-int8.bin -b 8 #导出int8模型
./quant -m baichuan -p ../baichuan.bin -o ../baichuan-int4.bin -b 4 #导出int4模型
```

### MOSS模型导出

```
# 需要先安装MOSS环境
# 如果使用自己finetune的模型需要修改moss_export.py文件中创建tokenizer, model的代码
# 如果使用量化模型，需要先编译好quant文件，这里假设已经存在build/quant文件
cd tools
python3 moss_export.py ../moss.bin # 导出浮点模型
cd ../build
./quant -m moss -p ../moss.bin -o ../moss-fp16.bin -b 16 #导出float16模型
./quant -m moss -p ../moss.bin -o ../moss-int8.bin -b 8 #导出int8模型
./quant -m moss -p ../moss.bin -o ../moss-int4.bin -b 4 #导出int4模型
```

## TODO

1、opencl支持

2、完善Sample功能
