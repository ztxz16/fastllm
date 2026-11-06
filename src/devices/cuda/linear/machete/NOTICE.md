# Machete来源及本地适配

上游：vLLM，提交 `643c125fab66d5ed5ec3143b7e764a77e7ae8ac7`。
源目录：`csrc/libtorch_stable/quantization/machete` 及其CUTLASS辅助头文件。
上游许可证为Apache-2.0，完整文本保存在本目录LICENSE；原始注释保留。
Machete主循环基于NVIDIA CUTLASS，使用项目现有CUTLASS依赖及其许可证。

本地修改：移除PyTorch张量/分配接口，使用裸指针、显式stream及RAII资源；
仅实例化A16、整数W4/W8、分组scale和加法offset组合。
保留核心预打包布局、主循环及CUTLASS调度，不以普通GEMM冒充Machete。
固定偏移的整数编码通过等价的unsigned+offset表达。

本地调度选择是FastLLM适配策略，并非声称已复现vLLM所有形状的最优调度。
