## TFACC介绍

TFACC是ThinkForce公司7000系列处理器的AI算力平台，可用于TF 7000系列处理器的大模型推理加速。

## 快速开始

### 加载驱动

``` sh
cd fastllm/third_party/tfacc/driver/tfacc2
./build_driver.sh
modprobe tfacc2
```

### 打开TFACC计算服务

``` sh
cd fastllm/third_party/tfacc
python3 ./launch.py 4 & #这里的参数是numa节点数量，需要根据7000服务器具体的型号设定
```

### 编译

建议使用cmake编译，需要提前安装c++编译器，make, cmake

gcc版本建议9.4以上，cmake版本建议3.23以上

使用如下命令编译

``` sh
cd fastllm
mkdir build
cd build
cmake .. -DUSE_TFACC=ON
make -j
```

编译完成后，可以使用如下命令安装简易python工具包。

``` sh
cd tools # 这时在fastllm/build/tools目录下
python setup.py install
```

### 运行demo程序

我们假设已经获取了名为`model.flm`的模型（参照 [模型获取](#模型获取)，初次使用可以先下载转换好的模型)

编译完成之后在build目录下可以使用下列demo:

``` sh
# 这时在fastllm/build目录下

# 命令行聊天程序, 支持打字机效果 (只支持Linux）
./main -p model.flm 

# 简易webui, 使用流式输出 + 动态batch，可多路并发访问
./webui -p model.flm --port 1234 

# python版本的命令行聊天程序，使用了模型创建以及流式对话效果
python tools/cli_demo.py -p model.flm 

# python版本的简易webui，需要先安装streamlit-chat
streamlit run tools/web_demo.py model.flm 

```

更多功能及接口请参照[详细文档](../README.md)