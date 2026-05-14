# 昇腾 NPU 编译

## 认识昇腾 NPU产品

初次接触华为昇腾产品线时，厘清混乱的版本和编号规则十分重要，

下表总结了目前最常见的昇腾产品、与芯片型号的对应关系：

| 系列                      | 产品                                         | 芯片系列 | 内存        | 内存带宽       |
| ------------------------- | -------------------------------------------- | -------- | ----------- | -------------- |
| Atlas 200/500 推理产品    | Atlas 200加速模块   Atlas 500智能小站        | _昇腾?_  | 8G          | 51.2GB/s       |
| Atlas 200/500 A2 推理产品 | Atlas 200I A2加速模块   Atlas 500 A2智能小站 | _昇腾?_  | 12G         | 51.2GB/s       |
| Atlas 300 推理产品        | Atlas 300I                                   | 昇腾310  | 32G（4*8G） | 204.8GB/s      |
| Atlas 训练系列产品        | Atlas 800T 训练服务器/ Atlas 300T            | 昇腾910A | 32G         | HBM, > 640GB/s |
| Atlas A2训练系列产品      | Atlas 800T A2 训练服务器                     | 昇腾910B | 32G / 64G   | HBM, > 640GB/s |
| Atlas 推理系列产品        | Atlas 300I Pro                               | 昇腾310P | 24G         | 204.8 GB/s     |
|                           | Atlas 300I Duo                               | 昇腾310P | 48G / 96G   | 408GB/s        |

其中，第一代昇腾310 / 910A系列产品按架构分为不同版本，3000 / 9000 适配 ARM64架构， 3010 / 9010 适配x86架构，彼此不可混用。如Atlas 800 推理服务器（型号：3000）表示ARM架构推理服务器，插 Atlas 300I 卡时只能插型号为3000的。

执行`npu-smi info`命令时，可以看到设备的SoC信息，SoC与对芯片类型的对应关系如下：

| 芯片系列 | SOC 型号(Name)                                |
| -------- | --------------------------------------------- |
| 昇腾310  | 310B1                                         |
| 昇腾910A | 910A **910B** 910ProA **910ProB** 910PremiumA |
| 昇腾310P | 310P1 310P3                                   |
| 昇腾910B | 910B1 910B2 910B2C 910B3 910B4                |

其中，910B、910ProB仍然是910A；910B3、910B4为910B屏蔽部分计算单元的版本。

## 环境搭建

### 安装驱动和CANN

* 安装NPU驱动：

  参考官方文档：[310P](https://www.hiascend.com/document/detail/zh/quick-installation/24.0.RC1/quickinstg/800I_A2/quickinstg_800I_A2_0007.html) [910A](https://www.hiascend.com/document/detail/zh/quick-installation/23.0.RC2/quickinstg/800_9000/quickinstg_800_9000_0007.html) [910B](https://www.hiascend.com/document/detail/zh/quick-installation/24.0.RC1/quickinstg_train/800_9000A2/quickinstg_800_9000A2_0007.html)

* 安装CANN：

  需要先安装对应版本的Python. (CANN < 7.0 安装 Python3.7， CANN > 7.0 安装 Python3.9/Python3.11)

  参考官方文档：[310P](https://www.hiascend.com/document/detail/zh/quick-installation/24.0.RC1/quickinstg/800_3000/quickinstg_800_3000_0021.html) [910A](https://www.hiascend.com/document/detail/zh/quick-installation/23.0.RC2/quickinstg/800_9000/quickinstg_800_9000_0018.html) [910B](https://www.hiascend.com/document/detail/zh/quick-installation/24.0.RC1/quickinstg_train/800_9000A2/quickinstg_800_9000A2_0020.html)

此外，6.0 版本以上需要按对应的NPU安装预编译算子包(kernels)：

```shell
chmod a+x Ascend-cann-kernels-910*_*_linux.run
Ascend-cann-kernels-910*_*_linux.run --install --quiet
```

* 配置环境

  算子编译需要依赖python。执行`pip3 check`，根据提示安装需要的包；（其中tensorflow不需要安装）  

  如果使用虚拟环境，如vituralenv / conda，启动前需要把虚拟环境的库目录加到`PYTHONPATH`环境变量中。

### 使用Docker镜像

昇腾910A 参考paddleCustomDevice镜像，如

* `paddlepaddle/paddle:latest-dev-cann5.0.2.alpha005-gcc82-aarch64`
* `paddlepaddle/paddle:latest-dev-cann5.0.2.alpha005-gcc82-x86_64` 
* `registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-aarch64-gcc82`
* `registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-x86_64-gcc82`
* `registry.baidubce.com/device/paddle-npu:cann701-ubuntu20-aarch64-gcc84-py39`
* `registry.baidubce.com/device/paddle-npu:cann701-ubuntu20-x86_64-gcc84-py39`

昇腾910B 参考paddleCustomDevice镜像，如

* registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-aarch64-gcc84-py39

现在可以基于其他公开可得的镜像，如`quay.io/ascend/vllm-ascend`系列。

git clone本仓库后，使用类似如下的命令创建一个容器：

```shell
docker run -it -v ${PWD}:/workspace --network host -u root --name fastllm_builder --device=/dev/davinci0 -v /etc/localtime:/etc/localtime --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu/:/usr/slog -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro -v /dev/shm/:/dev/shm/ ${IMAGE_NAME} /bin/bash
```
其中`IMAGE_NAME`为使用的镜像名称。

## 编译

进入环境后，执行如下脚本设置环境变量：
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

然后即可执行编译流程：
```shell
mkdir build-ascend/ && cd build-ascend/
cmake .. -DUSE_ASCEND_NPU=ON  # 使用NPU
make -j
```

## 运行demo程序

如果在Docker容器中执行，需要验证当前环境下NPU是否可用。执行：
```shell
npu-smi info
```
类似下面的结果表示NPU可正常使用：
```
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.rc1                 Version: 24.1.rc1                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 81.1        37                0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          2773 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
+===========================+===============+====================================================+
```
类似下面的结果表示NPU被占用，不能使用NPU设备：
```
DrvMngGetConsoleLogLevel failed. (g_conLogLevel=3)
dcmi model initialized failed, because the device is used. ret is -8020
```

我们假设已经获取了名为`model.flm`的模型（参照 [模型获取](#模型获取)，初次使用可以先下载转换好的模型)

编译完成之后在build目录下可以使用下列demo:

```shell
# 这时在fastllm/build目录下

# 命令行聊天程序, 支持打字机效果（只支持Linux）
./main -p model.flm 

# 性能测试
./benchmark -p model.flm -b 1 -t 16
 
# 简易webui, 使用流式输出 + 动态batch，可多路并发访问
./webui -p model.flm --port 1234

# python版本的命令行聊天程序，使用了模型创建以及流式对话效果
python tools/cli_demo.py -p model.flm
```

> ~~\# python版本的简易webui，需要先安装streamlit-chat~~
> ~~streamlit run tools/web_demo.py model.flm~~ 
> 

更多功能及接口请参照[详细文档](../README.md)