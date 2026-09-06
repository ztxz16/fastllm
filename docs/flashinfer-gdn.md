# 可选的 SM90 FlashInfer GDN prefill

H800 的推荐命令、参数取舍和实测速度，见 [H 系列显卡调优指南](hopper-tuning.md)。

此实验路径复用 FlashInfer 的四个 SM90 CuTe CP kernel，通过现有 Python 编译服务
导出包含 cubin 和 TMA 描述符构造代码的共享库，由 C++ 调用。GDN kernel 启动不经过 Python。

默认关闭，包括 SM90。显式启用方式：

```bash
FASTLLM_CUDA_TRITON_FLASHINFER_GDN=1 ftllm server MODEL --triton
```

这是本路径唯一新增的环境变量。`--triton` 启用现有编译服务；单独使用它仍走原来的
CUDA/Triton GDN。无需修改现有 max-chunks 参数来启用 FlashInfer。

当前入口只接受 Linux CUDA、SM90、batch=1、FP16、K/V 维度均为 128 的 Qwen3.5 GDN
prefill；每段 token 数为 1024–131072，V heads 为 32–128 且可被 K heads 整除。
CUDA Graph capture、其他架构、数据类型和不兼容布局均回退到原路径。
缓存保留原 FP16 `[K,V]` 状态布局，FlashInfer 的 FP32 `[V,K]` 状态在内部转换。

首次编译需要可用的 FlashInfer、CuTe DSL、CUDA Python bindings、CUDA toolkit 和 g++。
没有增加 fastllm 的默认安装依赖。编译解释器可通过已有的
`FASTLLM_CUDA_TRITON_PYTHON=/path/to/python` 指定；但 CLI 的 `--triton` 会用当前
ftllm 解释器覆盖该值，因此通过 CLI 启动时应直接使用具备可选依赖的 Python 环境。
共享库仍依赖编译环境的 CUDA/CuTe 运行时动态库；不能将 cubin 单独复制后直接运行。
已验证环境为 FlashInfer 0.6.16.post3、nvidia-cutlass-dsl 4.6.2、CUDA toolkit 12.8、
CUDA 13 系列 CuTe 运行时和 H800。其他依赖版本尚未验证。

编译缓存沿用 `FASTLLM_CUDA_TRITON_CACHE_DIR`。每种 head 配置保存一个索引，以及
共享库和用于检查的独立 cubin；不保留头文件、目标文件和编译 IR。
依赖缺失、编译或加载失败会回退，同一进程不会每层重试。修复环境或更换编译依赖后，
删除对应 `flashinfer_gdn_v1_*.json` 索引并重启服务即可重新解析编译缓存。

算子验证：

```bash
FASTLLM_CUDA_TRITON_PYTHON=/path/to/python \
FASTLLM_CUDA_TRITON_SERVER_SCRIPT="$PWD/tools/fastllm_triton_server.py" \
/path/to/python test/ops/check_flashinfer_gdn.py
```

测试涵盖空状态、非空状态、尾段、连续调用、独立 FP32 recurrence、禁用开关、非法输入、
CUDA Graph 回退、部分 launch 失败的状态保护及损坏缓存恢复。
模型级验证范围是 H800 上的 Qwen3.8-27B-FP8（内部架构 Qwen3.5，K/V heads=16/48）；
暂未覆盖其他 GPU、head 配置或完整质量评测。
