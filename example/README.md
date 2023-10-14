# example 示例项目

## Benchmark

测速示例程序，方便大家测试不同软硬件下的推理性能。作者测试的速度可以参考[这里](doc/benchmark.md)。

由于实际使用时很难满足batch的条件，也并非贪婪解码，该速度与真实使用时的速度有一定差异。

### 使用方法：

CPU：

   `./benchmark -p chatglm-6b-int4.flm -f prompts.txt -t [线程数] --batch [Batch大小]`

GPU：

   `./benchmark -p chatglm-6b-int4.flm -f prompts.txt --batch [Batch大小]`



## Web UI 

由 Jacques CHEN 提供，鸣谢！

## Win32Demo (Windows平台)

Win32Demo，是windows平台上运行FastLLM程序的一个Visual Studio工程。
由于Windows控制台默认编码为ANSI（中文是GBK编码，code page 936），而FastLLM默认输入输出编码为UTF-8，故与`main`存在一些差异，特提供专门的版本。为防止部分token是半个字符（如BPE编码），目前连续的中文字符是一并输出的。

生成的exe位置为：`Win32Demo\bin\Win32Demo.exe`

请尽量编译Release版本，速度快！

除此之外提供了fastllm的.vcproj文件，带GPU支持，本项目最低可在Visual Studio 2015 Update 3 下编译通过。
（但是**编译pyfastllm至少需要 MSVC 2017**）

### 编译

fastllm工程目前分为CPU版本和GPU版本，为简单上手，在没有cmake时，本项目可以使用Visual Studio工程文件并配置预处理器定义开关功能项。默认使用CPU版本。

签出代码后，**修改 include/fastllm.h**，Visual Studio中点击”文件“ -> "高级保存选项"，在编码中选择”Unicode (UTF-8 **带签名**) -代码页 65001“，或在其他文本编辑器中转为”UTF-8 BOM“编码。（由于linux下gcc不识别BOM头，该修改只能手动处理。）

* **CPU版本**：
  * 如果本机没有安装CUDA，在Win32Demo项目“属性”中找到"链接器" -> "输入" -> "附加依赖项"，点击'从父级或项目设置继承'。

* **GPU版本**：
  - 需要正确安装CUDA及其中的Visual Studio Integration；
  - 正确配置CUDA_PATH环境变量，指向要编译的CUDA版本；
  - 在解决方案资源管理器中移除fastllm.vcproj，引入fastllm-gpu.vcproj，
  - 对fastllm-gpu项目，在”生成依赖项“ -> "生成自定义" 中手动添加已安装的CUDA的自定义项文件；
  - 对fastllm-gpu项目，在”属性“中找到"CUDA C/C++" -> "Device" -> "Code Generation" 中配置编译后支持的[GPU计算能力](https://developer.nvidia.com/cuda-gpus#compute)；
  - 在Win32Demo项目上选择”添加“ -> "引用“，勾选fastllm-gpu项目；
  - 在Win32Demo项目上配置预处理器定义”USE_CUDA“。

### 使用方法：

1. 打开命令提示符cmd；

2. `cd example\Win32Demo\bin` ；

3. 运行时参数与`main`基本一致。但多一个参数 `-w` ，表示启动webui,不加为控制台运行。如：

   `Win32Demo.exe -p c:\chatglm-6b-v1.1-int4.flm -w`

## Android (android平台)
Android，使用Android studio工具建立的一個Android平台上运行LLM程序的例子。

### 使用方法：

1.在Android Studio直接打开工程运行。

2.直接下载release目录里里面的apk体验。

3.可以通过CMake工具链编译main文件(具体步骤见主页的readme)，通过adb shell运行，

1. `adb push main /data/local/tmp` 将main文件放到手机的tmp文件夹，
2. `adb shell` ,
3. `cd /data/local/tmp` 
4. `./main` 运行。

注意：demo apk 会将模型文件复制到应用 data 目录以方便 native 读取，因此设备需准备至少两倍模型大小的空余空间。