# example

## 介绍

### Win32Demo (Windows平台)
Win32Demo，使用Vitual Studio工具建立的一个windows平台上运行LLM程序的一个例子。
生成的exe位置为：
Win32Demo\bin\Win32Demo.exe
请尽量编译Release版本，速度快！
使用方法：
1.win+r 打开cmd控制台

2.cd 到Win32Demo\bin，在当前路径中运行Win32Demo.exe

3.运行时需要添加参数：Win32Demo.exe -p c:\chatglm-6b-v1.1-int4.bin -w
	-p 后面接bin模型，-w表示启动webui,不加为控制台运行。
	
4.webui 由 Jacques CHEN 提供，鸣谢！

### Android (android平台)
Android，使用Android studio工具建立的一個Android平台上运行LLM程序的例子。

使用方法：

1.直接AS打开运行。

2.直接下载release目录里里面的apk体验。

2.可以通过CMake工具链编译main文件(具体步骤见主页的readme)，通过adb shell运行，1. adb push main /data/local/tmp 将main文件放到手机的tmp文件夹，2. adb shell ,3.cd /data/local/tmp 4. ./main 运行。

注意：本demo中是写死了一个model文件(chatglm-6b-int4.bin),其路径为SD卡的根目录(/sdcard/chatglm-6b-int4.bin)。要想愉快的运行请先下载模型文件放到手机的根目录下！！！！