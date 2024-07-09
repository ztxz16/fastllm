# fastllm

[English Document](README_EN.md)

## 介绍

fastllm是纯c++实现，无第三方依赖的多平台高性能大模型推理库

部署交流QQ群： 831641348

| [快速开始](#快速开始) | [模型获取](#模型获取) |

## 功能概述

- 🚀 纯c++实现，便于跨平台移植，可以在安卓上直接编译
- 🚀 支持读取Hugging face原始模型并直接量化
- 🚀 支持部署Openai api server
- 🚀 支持多卡部署，支持GPU + CPU混合部署
- 🚀 支持动态Batch，流式输出
- 🚀 前后端分离设计，便于支持新的计算设备
- 🚀 目前支持ChatGLM系列模型，Qwen2系列模型，各种LLAMA模型(ALPACA, VICUNA等)，BAICHUAN模型，MOSS模型，MINICPM模型等

## 快速开始

### 编译

建议使用cmake编译，需要提前安装gcc，g++ (建议9.4以上), make, cmake (建议3.23以上)

GPU编译需要提前安装好CUDA编译环境，建议使用尽可能新的CUDA版本

使用如下命令编译

``` sh
bash install.sh -DUSE_CUDA=ON # 编译GPU版本
# bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=89 # 可以指定CUDA架构，如4090使用89架构
# bash install.sh # 仅编译CPU版本
```

其他不同平台的编译可参考文档
[TFACC平台](docs/tfacc.md)

### 运行demo程序 (python)

假设我们的模型位于"~/Qwen2-7B-Instruct/"目录

编译完成后可以使用下列demo:

``` sh
# 使用float16精度的模型对话
python3 -m ftllm.chat -t 16 -p ~/Qwen2-7B-Instruct/ 

# 在线量化为int8模型对话
python3 -m ftllm.chat -t 16 -p ~/Qwen2-7B-Instruct/ --dtype int8

# openai api server (目前处于测试调优阶段)
# 需要安装依赖: pip install -r requirements-server.txt
# 这里在8080端口打开了一个模型名为qwen的server
python3 -m ftllm.server -t 16 -p ~/Qwen2-7B-Instruct/ --port 8080 --model_name qwen

# webui
# 需要安装依赖: pip install streamlit-chat
python3 -m ftllm.webui -t 16 -p ~/Qwen2-7B-Instruct/ --port 8080
```

以上demo均可使用参数 --help 查看详细参数

目前模型的支持情况见: [模型列表](docs/models.md)

有一些架构暂时无法直接读取Hugging face模型，可以参考 [模型转换文档](docs/convert_model.md) 转换fastllm格式的模型

### 运行demo程序 (c++)

```
# 进入fastllm/build-fastllm目录

# 命令行聊天程序, 支持打字机效果 (只支持Linux）
./main -p ~/Qwen2-7B-Instruct/ 

# 简易webui, 使用流式输出 + 动态batch，可多路并发访问
./webui -p ~/Qwen2-7B-Instruct/ --port 1234 
```

Windows下的编译推荐使用Cmake GUI + Visual Studio，在图形化界面中完成。

如编译中存在问题，尤其是Windows下的编译，可参考[FAQ](docs/faq.md)

### python API

``` python
# 模型创建
from ftllm import llm
model = llm.model("~/Qwen2-7B-Instruct/")

# 生成回复
print(model.response("你好"))

# 流式生成回复
for response in model.stream_response("你好"):
    print(response, flush = True, end = "")
```

另外还可以设置cpu线程数等内容，详细API说明见 [ftllm](docs/ftllm.md)

这个包不包含low level api，如果需要使用更深入的功能请参考 [Python绑定API](#Python绑定API)

## 多卡部署

### python命令行调用中使用多卡部署

``` sh
# 使用参数--device来设置多卡调用
#--device cuda:1 # 设置单一设备
#--device "['cuda:0', 'cuda:1']" # 将模型平均部署在多个设备上
#--device "{'cuda:0': 10, 'cuda:1': 5, 'cpu': 1} # 将模型按不同比例部署在多个设备上
```
### ftllm中使用多卡部署

``` python
from ftllm import llm
# 支持下列三种方式，需要在模型创建之前调用
llm.set_device_map("cuda:0") # 将模型部署在单一设备上
llm.set_device_map(["cuda:0", "cuda:1"]) # 将模型平均部署在多个设备上
llm.set_device_map({"cuda:0" : 10, "cuda:1" : 5, "cpu": 1}) # 将模型按不同比例部署在多个设备上
```

### Python绑定API中使用多卡部署

``` python
import pyfastllm as llm
# 支持以下方式，需要在模型创建之前调用
llm.set_device_map({"cuda:0" : 10, "cuda:1" : 5, "cpu": 1}) # 将模型按不同比例部署在多个设备上
```

### c++中使用多卡部署

``` cpp
// 支持以下方式，需要在模型创建之前调用
fastllm::SetDeviceMap({{"cuda:0", 10}, {"cuda:1", 5}, {"cpu", 1}}); // 将模型按不同比例部署在多个设备上
```

## Docker 编译运行
docker 运行需要本地安装好 NVIDIA Runtime,且修改默认 runtime 为 nvidia

1. 安装 nvidia-container-runtime
```
sudo apt-get install nvidia-container-runtime
```

2. 修改 docker 默认 runtime 为 nvidia

/etc/docker/daemon.json
```
{
  "registry-mirrors": [
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ],
  "runtimes": {
      "nvidia": {
          "path": "/usr/bin/nvidia-container-runtime",
          "runtimeArgs": []
      }
   },
   "default-runtime": "nvidia" // 有这一行即可
}

```

3. 下载已经转好的模型到 models 目录下
```
models
  chatglm2-6b-fp16.flm
  chatglm2-6b-int8.flm
```

4. 编译并启动 webui
```
DOCKER_BUILDKIT=0 docker compose up -d --build
```

## Android上使用

### 编译
``` sh
# 在PC上编译需要下载NDK工具
# 还可以尝试使用手机端编译，在termux中可以使用cmake和gcc（不需要使用NDK）
mkdir build-android
cd build-android
export NDK=<your_ndk_directory>
# 如果手机不支持，那么去掉 "-DCMAKE_CXX_FLAGS=-march=armv8.2a+dotprod" （比较新的手机都是支持的）
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_CXX_FLAGS=-march=armv8.2a+dotprod ..
make -j
```

### 运行

1. 在Android设备上安装termux软件
2. 在termux中执行termux-setup-storage获得读取手机文件的权限。
3. 将NDK编译出的main文件，以及模型文件存入手机，并拷贝到termux的根目录
4. 使用命令```chmod 777 main```赋权
5. 然后可以运行main文件，参数格式参见```./main --help```

