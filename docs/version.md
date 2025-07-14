## V0.1.3.4

- 修复Kimi-K2模型分词出现的一些错误

## V0.1.3.3

- 支持Kimi-K2模型
- 修复8numa时的一些bug
- 加速CPU上的AWQ计算

## V0.1.3.2

- 修复上个版本一些模型无法读取的bug

## V0.1.3.1

- 支持Hunyuan模型 （混元）
- 支持Ernie_4.5模型 （文心）
- 支持PanguPro模型 （盘古）
- 支持Minimax 01, Minimax M1模型
- 优化旧GPU上的一些计算性能
- 在dev_mode下支持主动终止请求 [查看详情](https://github.com/ztxz16/fastllm/pull/535)
- 修复numa模式下无法计算float16的问题
- 修复numa个数过多时可能出现的一些错误

## V0.1.3.0

- DeepSeek模型中，默认改用cuda执行共享专家，可通过参数`--cuda_se false`来关闭
- 支持使用类似`--device "{'cuda:0':1,'cuda:1':1}"`的命令来串行执行，参考[如何设定运行设备](../README.md#3-如何设定运行设备)
- 增加动态量化功能，参考[动态量化指南](dtype_config.md)
- 修复一些超长prompt可能引起的出错
- 多卡张量并行时将平分显存，目前测试阶段，仅在llama系模型（如Qwen2, Qwen2.5, QwQ等）生效
- 所有moe模型支持`--moe_expert`参数 （之前的版本中，qwen3-moe此参数不会生效）


## V0.1.2.0

- 规范版本号 a.b.c.d
- a为保留位，目前为0
- b为大版本号
- c为小版本号
- d为bug修复版本的编号

## V0.0.1.2

- 优化了numa加速
- 略微提升了prefill和decode速度
- 支持了moe的混合张量并行，参考[混合推理指南](mixforward.md)
- 修复了multicuda的一些bug，支持了所有精度的混合张量并行
- 修复了C++下Jinja模板的一些bug，支持Qwen3, DS等一系列模型的内置分词器

## V0.0.1.1

- 支持了 `FP8_E4M3` 精度（新老硬件均可）
- MOE模型支持用`--moe_dtype`来设置混合精度
- 可以在`ROCM`环境下使用`pip`安装了
- 修复了C++下Jinja模板的一些bug
- api server的默认输出token数由8K提升到32K

## V0.0.1.0

- 支持了千问3模型 [部署指南](qwen3.md)
- 优化了DeepSeek模型的显存使用
- 增加参数`--cache_fast`来指定是否使用显存缓存

## V0.0.0.9

- 优化了使用DeepSeek模型时的多轮对话缓存
- 略微提升了DeepSeek模型的多并发速度
- 减少了DeepSeek模型Prefill时的显存消耗，可以支持更长的上下文
- 支持了DeepSeek模型的INT8量化 （使用原始模型时`--dtype int8`，或者导出时`--dtype int8`）
- 隐藏了 "None of PyTorch, TensorFlow >= 2.0 ..." 的警告信息
- 增加了`--cache_dir`参数来指定缓存目录
- server增加了`--hide_input`参数来隐藏日志中的请求信息
- webui增加了`--max_token`参数来指定最大输出，--think参数来强制思考

## V0.0.0.8

- api server增加api_key参数，来设定api_key
- api server支持了一些复合输入
- 提升了moe模型prefill的速度
- 增加了--version参数查看版本号

## V0.0.0.7

- 增加config选项，可通过config.json文件来启动模型
- 提升moe模型的速度

## V0.0.0.6

- 降低GLIBC版本，PIP安装包兼容更多系统
- PIP安装包支持更多架构（目前最低支持到SM_52）

## V0.0.0.5

- 修改文档，增加了一些pip安装后无法使用的情况说明
- 聊天模式下自动读取模型的生成配置文件
- 修复一些情况下kv_cache_limit计算错误的问题

## V0.0.0.4

- 增加ftllm run, chat, webui, server接口