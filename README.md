# fastllm

## 介绍

fastllm是纯c++实现，无第三方依赖的大模型库，目前支持国产大模型ChatGLM-6B，MOSS;

可以在安卓设备上流畅运行ChatGLM-6B

可以在支持CUDA的设备上加速计算

## 编译

fastllm使用c++编写，建议使用cmake编译，需要提前安装c++编译器，make, cmake

如果需要使用GPU（目前仅int8模型支持GPU加速），需要提前安装好CUDA编译环境

### PC (CPU)

```
mkdir build
cd build
cmake ..
make -j4
```

### PC (CPU + GPU)

```
mkdir build-cuda
cd build-cuda
cmake .. -DUSE_CUDA=ON
make -j4
```

### Android

```
# Android上需要下载NDK工具编译
mkdir build-android
cd build-android
export NDK=<your_ndk_directory>
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 ..
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

### 运行MOSS模型

```
./main -m moss -p moss-int8.bin
```

### 在Android上运行

可以在Android设备上安装termux软件，并在其中执行termux-setup-storage获得读取手机文件的权限。然后将NDK编译出的main文件和模型存入手机，然后在termux中运行main文件（需要把main文件拷贝到termux的根目录下，否则无权限运行）

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
./quant -m chatglm -p ../chatglm-6b.bin -o ../chatglm-6b-int8.bin -b 8 #导出int8模型
./quant -m chatglm -p ../chatglm-6b.bin -o ../chatglm-6b-int4.bin -b 4 #导出int4模型
```

### MOSS模型导出

```
# 需要先安装MOSS环境
# 如果使用自己finetune的模型需要修改moss_export.py文件中创建tokenizer, model的代码
# 如果使用量化模型，需要先编译好quant文件，这里假设已经存在build/quant文件
cd tools
python3 moss_export.py ../moss.bin # 导出浮点模型
cd ../build
./quant -m moss -p ../moss.bin -o ../moss-int8.bin -b 8 #导出int8模型
./quant -m moss -p ../moss.bin -o ../moss-int4.bin -b 4 #导出int4模型
```

## TODO

1、各种算子NEON优化, AVX指令集优化

2、CUDA优化

3、CUDA模式下，当显存不够时使用显存+内存混合计算模式

4、支持更多模型
