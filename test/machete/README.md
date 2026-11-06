# Native / Machete对照

## 范围与边界

新增后端为Hopper SM90a上的整数W4A16/W8A16，不包括NVFP4、NF4、FP8、W8A8。
保留原生分发（包括其Marlin/cuBLAS选择），默认仍为native。
显式machete严格拒绝不支持的目标权重；auto只在支持检查拒绝时回退，执行错误不回退。
浮点层和其他量化族不参与本次选择。

正式适配器接受FastLLM的INT8、INT4_NOZERO、INT4_GROUP存储：

- 激活FP16/BF16；输出相同类型；可选FP32 bias。
- N为128的倍数，K为64的倍数；分组须整除K，且为64的倍数或等于K。
- INT8当前正式权重格式是per-channel；INT4_GROUP使用现有groupCnt。
- INT4_GROUP32、INT4_GROUP128的特殊打包不能当作普通INT4字节数组直接输入，明确拒绝。
- 不新增GPTQ checkpoint解析器，也不将已有GPTQ/AWQ外部编码直接视为FastLLM内部布局。
- BF16的这些普通INT8/INT4_GROUP组合在原生分发中缺失，报告标记UNSUPPORTED。
  BF16算子用独立FP64累加参考验证；模型缺少同配置参考时标记UNPAIRED，汇总INCOMPLETE，不能冒充正确性PASS。

底层对称量化使用上游biased整数类型与scale；其他情况使用unsigned权重与加法offset。
原FP32 scale转换为激活类型，offset=-scale*zero或min同样转换。
日志记录最大scale/offset舍入误差，算子报告记录输出误差；不重新量化整数权重。
这不是bitwise等价，也不能默认认为模型精度已经通过。

预打包在每个权重/设备/激活类型/工作线程首次调用时创建，之后复用；源权重保留。
额外显存包括预打包副本、scale/offset与workspace。并发线程隔离资源。
Data深拷贝、FreeSpace、设备迁移和析构退休缓存。
CUDA Graph必须先在相同工作线程预热目标形状；捕获期禁止首次预打包和workspace扩容。
多卡使用原有张量并行切分后的Data进入同一正式Linear入口；切分后不满足对齐条件时严格拒绝。
上述GPU行为需要在服务器实际验证，静态单元测试不能替代GPU验证。

## 编译（在GPU服务器执行）

沿用已有构建目录和配置，确保USE_CUDA、UNIT_TEST启用，CUDA_ARCH包含90a，
FASTLLM_ENABLE_MACHETE_BUILD=ON，CUDA工具链>=12.0且CUTLASS依赖已安装。
构建日志应出现`Machete W4A16/W8A16: ON`。
新增测试目标为`macheteLinearTest`；Machete核心使用独立sm_90a对象目标。
本次代码变更包含C++/CUDA，必须重新编译安装，不能仅更新Python脚本。

## 正式推理选择

CLI新增`--linear-backend native|machete|auto`和`--linear-backend-trace`。
Python在加载模型前调用`llm.set_linear_backend("machete")`。
非CLI调用可在进程启动前设置`FASTLLM_LINEAR_BACKEND=machete`。
`FASTLLM_LINEAR_BACKEND_TRACE=1`开启首次路径/舍入日志，默认关闭。
模型存活期间不要切换后端；对比测试使用独立模型进程。

## 算子正确性、性能及Nsight

```bash
python3 test/machete/operator_compare.py \
  --executable ./build-fastllm/macheteLinearTest \
  --result-dir /tmp/machete-operators-$(date -u +%Y%m%dT%H%M%SZ) \
  --batch-sizes 1,7,8,9,16,17,32,64 \
  --n 256 --k 256 --repeats 5 --iterations 100
```

矩阵覆盖两种位宽、两种激活类型、有无bias、分组、旧实现分支边界、Eager/Graph，
并加入非对称zero、min偏移、非法N和不支持分组的测试。
Machete case还执行Data深拷贝和设备往返迁移。
形状可替换为实际模型N/K；CPU参考会随矩阵大小增加耗时，但完全在计时外。
默认误差阈值atol=0.02、rtol=0.05，只作为算子回归门槛，并非模型精度保证。
`--nsight`额外采集相同计时区间的kernel/API统计；大报告与SQLite放临时目录，
统计CSV保留后临时文件自动清理。

## 模型Forward、整体性能及Nsight

```bash
python3 test/machete/model_compare.py \
  --model /absolute/path/to/model \
  --result-dir /tmp/machete-model-$(date -u +%Y%m%dT%H%M%SZ) \
  --dtypes int4g128,int8 --atypes float16,bfloat16 \
  --stages forward,performance,nsight \
  --batch-sizes 1,2,4,8,16,32 \
  --input-tokens 512 --output-tokens 64 --warmup 1 --repeats 5
```

浮点源模型由原有加载器统一生成目标权重；预量化模型必须使用加载器实际支持的格式，
不能用更改dtype掩盖不支持的checkpoint。不同backend使用相同模型、dtype与Prompt Token文件。
Forward为正式生成路径首token、公共TopK logprob对比，不是完整logits逐元素验证。
整体性能复用正式HTTP接口，Eager/Best、Cold/Cache-hit分别比较，不能跨配置计算加速比。
Nsight复用现有稳定Decode采集、统计和大文件清理流程。
正式测试需Python requests/transformers及正确安装的FastLLM；Nsight需nsys命令。

输出MD/CSV/JSON/XLSX/PNG；JSON保留原始样本，其他表格浮点展示三位小数。
加速比明确为native耗时/Machete耗时，大于1表示Machete更快。
没有native参考的组合不计算加速比；FAIL和INCOMPLETE均返回非零退出状态。

## 不编译的检查

```bash
python3 -m unittest discover -s test/machete -p 'test_*.py' -v
```

该检查仅覆盖脚本/报告/配置契约，不会执行GPU，也不能证明CUDA kernel编译或数值正确。
