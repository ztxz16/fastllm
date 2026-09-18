# 可选 CUDA 低比特 prefill 路径

原有 decode 权重格式与 W8A16/W4A16 路径保留。本功能增加 W8A8/W4A4 prefill；激活量化会引入数值误差，仅在 SM120 的已验证单卡形状上默认开启。FastLLM 中间张量为 FP16。

## 默认策略与显式开关

以下三个开关未设置时使用自动白名单，显式 `0` 禁用，显式 `1`（或 `true`）允许其他受支持 SM/形状尝试，但仍须通过所有 CanRun 检查。其他值按禁用处理。

| 开关 | 自动开启范围（仅 SM120，即 compute capability 12.0） |
|---|---|
| `FASTLLM_CUDA_NATIVE_FP8_PREFILL` | M=32..4096；(N,K)=(16384,5120)、(14336,5120)、(34816,5120)、(5120,6144)、(5120,17408) |
| `FASTLLM_CUDA_NATIVE_NVFP4_PREFILL` | M=9..4096；(N,K)=(34816,5120)、(5120,17408)，后续布局、对齐及分块限制仍适用 |
| `FASTLLM_CUDA_GDN_PREPARE_WY` | [batch,heads,chunks,chunkSize,headDim]=[1,48,1..64,64,128] |

这些是 Qwen3.8-27B 单卡已验证形状；不读取模型名字。其他模型只有匹配形状且通过数据类型、布局等检查才会进入。TP 切分形状、其他 SM 默认关闭，保留手动开启。`FASTLLM_CUDA_NVFP4_NATIVE_LAYOUT` 独立保留 opt-in，不随这三个开关自动打开。

## 手动启用

```bash
export FASTLLM_CUDA_NATIVE_FP8_PREFILL=1
export FASTLLM_CUDA_NATIVE_NVFP4_PREFILL=1
export FASTLLM_CUDA_GDN_PREPARE_WY=1
ftllm server /path/to/Qwen3.8-27B-NVFP4 --max_batch 1 --mtp 0 \
  --kv_cache_dtype fp8_e4m3 --tokens 16384 --prefix_cache false \
  --chunked_prefill_size 2048
```

不保留第二份权重布局：每次调用在复用的固定 scratch 内重排当前层权重，调用结束后不缓存。额外 5.6 GB 布局缓存及其环境开关已移除。

## 条件及回退

| 路径 | 条件 | 回退 |
|---|---|---|
| FP8 cuBLASLt | CUDA >=12.8，SM89+，M=32..4096，N/K 为16倍数；row FP8 权重、FP16输入输出 | 原 W8A16 路径 |
| NVFP4 cuBLASLt | CUDA >=12.8，SM10x/12x，M=128..4096且为128倍数，N为128倍数、K为64倍数 | 原 Marlin 路径 |
| NVFP4 TMA融合 | SM120；M=256..4096，支持尾块；gate/up+SwiGLU N=34816,K=5120；down+residual N=5120,K=17408；无 bias | 上述通用 cuBLASLt，再回退原权重布局对应的 Linear 路径 |
| GDN prepare | SM80+，FP16 dense，chunk64、K/V head128；非 ragged、未启用已有 Triton GDN 路径 | 原 KKT、mask、逆三角、WY/WU 完整路径 |
| gated RMSNorm | 原 FP16 head128 路径，每 CTA 4 行；行内归约顺序不变 | 小行数使用每 CTA 1 行 |

所有路径还检查指针、维度、设备、布局、对齐、别名和可用编译镜像。cuBLASLt 路径需要可用算法和足够的固定 scratch。CUDA graph capture 中的新低比特/GDN prepare 路径返回 false，保留原计算。普通 decode 不进入新低比特 GEMM。

Block 的 CanRun 检查结构条件，Run 内仍允许在写出结果前因库算法或描述符不可用而回退；kernel 启动错误抛出，不能在写过 residual 后静默重复计算。

`FASTLLM_CUDA_NATIVE_NVFP4_TMA=0` 禁用 TMA 特化；只有父开关允许（自动白名单或显式 `1`）时 TMA 才可能运行。
`FASTLLM_CUDA_GDN_NORM_MULTIROW=0` 恢复原单行 gated norm。

每 GPU 的 cuBLASLt 状态使用互斥锁和 CUDA event 协调跨 stream scratch；预热时预留 32 MiB workspace 和 384 MiB scratch，进入服务后不增长。形状算法在首次使用时选择，测速必须预热。

## 实现与来源

FP8 逐 token E4M3 激活量化；NVFP4 使用 block16 E2M1 数据、E4M3 block scale 和逐 token global scale。通用 GEMM 使用 FP32 输出中间缓冲，避免在恢复缩放之前 FP16 溢出，随后融合缩放、SwiGLU 或 residual。

TMA 特化使用自包含的手写 CUDA 内核，将缩放及 SwiGLU/residual 融合到矩阵乘尾部，支持 FP16 输出；不依赖外部推理引擎的算子源码。不同 FP32 累加顺序可能在 FP16 舍入边界产生少量差异。

GDN prepare 融合 KKT、cumsum/decay/mask、分块三角逆和 WY/WU。三角逆使用 TF32×3 乘积以控制误差；WMMA accumulator 经共享内存转换，不依赖不同 fragment 类型的内部元素排列。

## 验证

```bash
CUDA_HOME=/path/to/cuda-13.1 CUDA_VISIBLE_DEVICES=0 \
  python3 test/cuda/native_prefill/run.py --arch 120f --tma
```

测试包含 FP8/NVFP4 量化参考、SwiGLU/residual、TMA/cuBLASLt 对照、decode/graph/不支持尺寸回退、GDN prefix/decay 和 compact scratch 别名。GPU 测试需串行在空闲设备运行。

本轮使用 CUDA 13.1.80，在 SM120 RTX 5090 上完成运行验证。通用源文件另通过 SM75/80/86/89/90/100/120/121 编译检查；不代表已在这些硬件上实测。TMA 单独使用 compute_120f/sm_120f 编译。

整模型 2K/8K、每组33个 teacher-forced token 的 logits 抽样用于发现明显数值问题，不替代完整困惑度/下游质量评估。因此自动开启仅限上述 SM120 形状，其他 SM 仍保持 opt-in。

## 仅关闭 Marlin 的回退对照

进程启动前设置 `FASTLLM_CUDA_NVFP4_MARLIN=0` 禁止 NVFP4 权重转换成 Marlin 布局；默认值为1。已经转换的权重仍必须由理解其布局的路径处理，因此需要重启服务切换，不能在线改变布局。

这是原始格式 GEMV / 解量化后 FP16 GEMM 的回退对照，当前不等价于原生 NVFP4 TMA prefill。设置 `FASTLLM_CUDA_FP8_MARLIN=0` 可同时禁用 FP8 Marlin（SM120上默认已关闭）。不引入任何权重副本缓存。

## SM120 共用 NVFP4 权重布局

在上述 prefill 开关外，设置以下选项并重启服务：

```bash
export FASTLLM_CUDA_NVFP4_NATIVE_LAYOUT=1
export FASTLLM_CUDA_NVFP4_MARLIN=0
export FASTLLM_CUDA_FP8_MARLIN=0
```

这条路径使 NVFP4 prefill 和 decode 共用一份权重。预热时在原权重 allocation 内转换为 E2M1 codes、cuBLASLt tiled E4M3 scales 和 global scale。prefill 直接读取，不再执行 Marlin→NVFP4 重排；decode 使用读取同一布局的 W4A16 GEMV，并融合 SwiGLU/residual。

没有第二份常驻权重，也没有 5.6 GB 缓存开关。转换期间仅有当前层临时缓冲（最大约 100 MB），校验和安装后立即释放。低比特 prefill 仍使用前文的 416 MiB 固定共享工作区；大 M 回退最多请求 32 MiB 解量化 scratch。原权重 allocation 不缩小。

初始转换仅启用于 SM12x、可用 SM120+ 编译镜像、单设备完整权重，且限于 N=34816/K=5120 和 N=5120/K=17408 两种尺寸。需要 FP16 输入输出、NVFP4 block16、预热同步阶段；不能在 graph capture 中转换。E4M3 scales 必须通过还原检查，失败则不改写原权重。开关默认关闭。其他 SM、TP shard 尺寸或不满足条件的权重保留原有调度；当前新布局仅在 SM120 单卡实测。

转换之后：

- M=1..8：原生 W4A16 GEMV，FP32 累加、FP16 输出，支持 CUDA graph；M=1 对上述 K 特化。
- M=9..4096：SM120 上符合手写融合条件的 M=256..4096 直接使用 TMA（支持尾块）；其他情况将 M 补齐到 128 倍数后使用 cuBLASLt。输入只读真实行、输出只写真实行。
- 库算法、scratch、形状或 capture 条件不适合低比特 prefill 时，使用理解新布局的 GEMV 或分块解量化 GEMM。融合路径返回 false 后，Linear 仍能正确读取新布局，再执行原后处理。

`Data` 只增加布局标记，不增加权重缓存指针。复制保留布局信息；迁回 CPU、reshape/resize 和 expansion 前恢复原始格式；释放清除标记。不能直接对 tiled 权重调用 `FakeFrom` 创建原始字节切片，必须先恢复源权重，接口会显式拒绝错误用法。

新增测试覆盖 pack/restore、M=1/2/3/8 GEMV、graph replay、两个实际 K 特化、M=9/17/63/127/129/255 的补齐和输出边界，以及新布局 prefill 与 cuBLASLt 对照。`native-layout-lifecycle.cpp` 另链接构建好的 `libfastllm_tools.so`，检查复制、CPU 迁移和 reshape。测试在空闲 GPU 上串行执行。


## SM120 手写融合 prefill

开启原 NVFP4 native prefill 后，符合条件时默认使用自包含的手写 kernel。`FASTLLM_CUDA_NATIVE_NVFP4_FRESH=0` 或 `FASTLLM_CUDA_NATIVE_NVFP4_TMA=0` 可禁用此路径并回退 cuBLASLt；没有保留外部引擎的旧 TMA 算子。

支持 SM120、M=256..4096（包括非整倍数尾块）、gate/up N34816 K5120 的 SwiGLU 融合，以及 down N5120 K17408 的 residual 融合。保留无 bias、指针对齐、非 CUDA Graph capture 约束。普通 Linear mode0、其他尺寸、SM 或资源不满足时回退 cuBLASLt/原 Linear 路径；M<256 沿用旧通用路径。

复用原有权重与固定 scratch，没有新增常驻权重副本。activation scale 按 256 行 tile 预留容量，量化只写实际行；尾块 kernel 不读取无效行。

`test/cuda/native_prefill/run.py --arch 120f --tma` 使用 cuBLASLt 作为独立数值参考，并包含 `native-fresh-test`，覆盖量化、native weight layout、原地 residual、379/487/775 等尾块及小 M/非融合模式回退。测试不要求逐位相等。

真实模型首层 down 回放中，379/775 行分别有5/2个 FP16 输出与 cuBLASLt 不同，最大绝对差3.05e-5/6.10e-5，相对RMS误差约2e-7；双精度参考表明差异来自FP32累加顺序及FP16舍入。多层低比特量化可能放大差异，因此不承诺任意长度的最终logits或生成文本逐位一致。
