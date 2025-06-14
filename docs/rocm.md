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
| 架构代号 | 架构系列 | 代表产品示例                                | 推荐 ROCm 版本 |
|----------|-----------|---------------------------------------------|----------------|
| gfx900   | GCN5.0    | Radeon Instinct MI25                        | ❌不支持         |
| gfx906   | GCN5.1    | Radeon VII, Instinct MI50                   | [6.3.3](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.3/install/quick-start.html) |
| gfx908   | CDNA      | Radeon Instinct MI100                       | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |
| gfx90a   | CDNA2     | Radeon Instinct MI210/MI250/MI250X         | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |
| gfx942   | CDNA3     | Instinct MI300A/MI300X/MI325X              | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |
| gfx1030  | RDNA2     | Radeon PRO W6800/V620                       | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |
| gfx1100  | RDNA3     | Radeon PRO W7800/W7900, RX 7900 XT/XTX/GRE  | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |
| gfx1101  | RDNA3     | Radeon PRO V710                         | [6.4.0](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/install/quick-start.html) |



把需要编译的GPU架构用`;`分隔，填入`-DROCM_ARCH`参数中。如果不填这个参数，会自动检测。

注意，部分GPU（比如RX6000系列、MI50不支持矩阵乘法加速`rocwmma`，只要列表中有一个GPU不支持`rocwmma`，则编译时不会使用`rocwmma`。

## 2. 编译

如果使用自动检测，直接运行以下命令。适用于只在本机运行，不拷贝到有其他GPU的机器上运行的情况。

``` sh
bash install.sh -DUSE_ROCM=ON
```

如果需要编译成支持多个GPU的版本，或者在其他机器上运行，需要手动指定`ROCM_ARCH`参数。

``` sh
bash install.sh -DUSE_ROCM=ON -DROCM_ARCH="gfx908;gfx90a;gfx1100"
```

## TODO

- [ ] 海光系列GPU的验证
- [ ] 支持`rocwmma`，能使用矩阵乘法加速

## 鸣谢

[leavelet](https://github.com/leavelet) 提供ROCM支持
