# ROCm 编译

## 0. 支持平台

ROCm 编译目前仅支持Linux平台。

目前支持的GPU型号如下：

- AMD Radeon Instinct MI系列，如MI50, MI100，MI210等
- AMD Radeon RDNA RX 7000 游戏卡和工作站卡系列，W7800，W7900等
- 海光系列GPU，如K100等（未验证，理论可行）

## 1. 安装 ROCm，获取 ROCm Arch

请参考 [ROCm 官方文档](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) 安装 ROCm。

可以在 [架构列表](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)的LLVM target列中找到GPU的 ROCm Arch。

常见GPU对应的架构：
- MI50: gfx906
- MI100: gfx908
- MI200系列: gfx90a
- RX 7000系列和W7000系列: gfx1100

把需要编译的GPU架构用`;`分隔，填入`-DROCM_ARCH`参数中。默认为`gfx908;gfx90a;gfx1100`。

注意，部分GPU（比如RX6000系列、MI50不支持矩阵乘法加速`rocwmma`，只要列表中有一个GPU不支持`rocwmma`，则编译时不会使用`rocwmma`。

## 2. 编译

``` sh
bash install.sh -DUSE_ROCM=ON -DROCM_ARCH="gfx908;gfx90a;gfx1100"
```

## TODO

- [ ] 海光系列GPU的验证
- [ ] 支持`rocwmma`，能使用矩阵乘法加速



