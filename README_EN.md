# fastllm

[中文文档](README.md)

| [Quick Start](#quick-start) | [DeepSeek Deployment Guide](docs/deepseek.md) | [Qwen3 Deployment Guide](docs/qwen3.md) | [Changelog](docs/version.md) |

## Introduction

fastllm is a high-performance LLMs inference library implemented in C++ with no backend dependencies (e.g. PyTorch).

It enables hybrid inference of MOE models, achieving 20+ tps on consumer-grade single GPUs (e.g., 4090) for DeepSeek R1 671B INT4 model inference.


Deployment discussion QQ group: 831641348  

WeChat group: ![QR Code](docs/wechat_group0.jpg)

## Key Features

- 🚀 DeepSeek hybrid inference - deploy with multi-concurrency on consumer-grade single GPUs
- 🚀 Multi-NUMA node acceleration support
- 🚀 Dynamic batch and streaming output
- 🚀 Multi-GPU deployment and GPU+CPU hybrid deployment
- 🚀 Frontend-backend separation design for easy support of new computing devices
- 🚀 Support ROCm, so it's possible to inference with AMD GPU.
- 🚀 Pure C++ backend for easy cross-platform porting (can be directly compiled on Android)
- 🚀 Support [customize model structures](docs/english_custom.md) in Python

## Quick Start

### Installation

- PIP install (currently Nvidia GPU only)

Linux systems can try direct pip installation:

```
pip install ftllm -U
```

(Note: Due to PyPI size limitations, the package doesn't include CUDA dependencies - manual installation of CUDA 12+ is recommended)
```
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run
```

#### Compile From Source

If pip installation fails or you have special requirements, you can build from source:

Thie project is built with cmake. Requires pre-installed gcc, g++ (7.5+ tested, 9.4+ recommended), make, cmake (3.23+ recommended)

GPU compilation requires CUDA environment (9.2+ istested). Use the newest CUDA version possible.

Compilation commands:

``` sh
bash install.sh -DUSE_CUDA=ON -D CMAKE_CUDA_COMPILER=$(which nvcc) # GPU version
# bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=89 -D CMAKE_CUDA_COMPILER=$(which nvcc) # Specify CUDA arch (e.g. 89 for RTX 4090)
# bash install.sh # CPU-only version
```

##### Compilation on Different Platforms

For compilation instructions on other platforms, please refer to the documentation:  

[TFACC Platform](docs/tfacc.md)
[ROCm Platform](docs/rocm.md)

If you meet problem during compilation, see [FAQ](docs/faq.md) doc.

### Running Demos

Taking the Qwen/Qwen3-0.6B model as an example:

#### Command-line Chat:

```
ftllm run Qwen/Qwen3-0.6B
```

#### WebUI:

```
ftllm webui Qwen/Qwen3-0.6B
```

#### API Server (OpenAI-style):

```
ftllm server Qwen/Qwen3-0.6B
```

#### Local Models

You can launch a locally downloaded Hugging Face model. Assuming the local model path is `/mnt/Qwen/Qwen2-0.5B-Instruct/`, use the following command (similar for `webui` and `server`):

```
ftllm run /mnt/Qwen/Qwen3-0.6B/
```

#### Fuzzy Launch

If you can't remember the exact model name, you can input an approximate name (matching is not guaranteed).  
For example:
```
ftllm run qwen2-7b-awq
```

```
ftllm run deepseek-v3-0324-int4
```

#### Setting Cache Directory

If you don't want to use the default cache directory, you can set it via parameter `--cache_dir`, for example:

```
ftllm run deepseek-v3-0324-int4 --cache_dir /mnt/
```

Or you can set it via the environment variable `FASTLLM_CACHEDIR`. For example, on Linux:  

```
export FASTLLM_CACHEDIR=/mnt/
```

## Parameters

The following are common parameters when running the `ftllm` module:

### General Parameters

- `-t` or `--threads`:  
  - **Description**: Sets the number of CPU threads to use.  
  - **Example**: `-t 27`  

- `--dtype`:  
  - **Description**: Specifies the data type of the model.  
  - **Options**: `int4` or other supported data types.  
  - **Example**: `--dtype int4`  

- `--device`:  
  - **Description**: Specifies the computing device for the model.  
  - **Common Values**: `cpu`, `cuda`, or `numa`.  
  - **Example**: `--device cpu` or `--device cuda`  

- `--moe_device`:  
  - **Description**: Specifies the computing device for the MOE (Mixture of Experts) layer.  
  - **Common Values**: `cpu`, `cuda`, or `numa`.  
  - **Example**: `--moe_device cpu`  

- `--moe_experts`:  
  - **Description**: Specifies the number of experts to use in the MOE layer. If not set, it follows the model's configuration. Reducing the number of experts may speed up inference but could lower accuracy.  
  - **Example**: `--moe_experts 6`  

- `--port`:  
  - **Description**: Specifies the port number for the service.  
  - **Example**: `--port 8080`  


### Parameters for differnet Modules
  
Please read [Arguments for Demos](docs/english_demo_arguments.md) for further information.
## Obtain Model

### Model Download

Use the following command to download a model locally:  

```
ftllm download deepseek-ai/DeepSeek-R1
```


### Model Export

If using quantized model loading (e.g., `--dtype int4`), the model will be quantized online each time it is loaded, which can be slow.  

`ftllm.export` is a tool for exporting and converting model weights. It supports converting model weights to different data types. Below are detailed instructions on how to use `ftllm.export`.  

#### Command Format  

``` sh
ftllm export <model_path> -o <output_path> --dtype <data_type> -t <threads>  
```

#### Example Command

``` sh
ftllm export /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-INT4 --dtype int4 -t 16
```

#### Mixed Precisions

You can specify `--moe_dtype` for mixed precision of a MoE model, for example:

``` sh
ftllm export /mnt/DeepSeek-V3 -o /mnt/DeepSeek-V3-FP16INT4 --dtype float16 --moe_dtype int4 -t 16
```

#### Loading the Exported Model

The exported model can be used similarly to the original model. The `--dtype` parameter will be ignored when using the exported model.

For example:

``` sh
ftllm run /mnt/DeepSeek-V3-INT4/
```

### Supported Models

Fastllm supports original, AWQ and FASTLLM models. Please refer [Supported Models](docs/models.md) for older models.
