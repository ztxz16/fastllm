# fastllm

## Introduction

fastllm is a high-performance large model inference library implemented purely in C++ with no third-party dependencies, supporting multiple platforms.

Deployment and communication QQ group: 831641348

| [Quick Start](#quick-start) | [Model Acquisition](#model-acquisition) |

## Features Overview

- 🚀 Pure C++ implementation, facilitating cross-platform移植, directly compilable on Android
- 🚀 Supports reading Hugging Face raw models and direct quantization
- 🚀 Supports deploying OpenAI API server
- 🚀 Supports multi-card deployment, supports GPU + CPU hybrid deployment
- 🚀 Supports dynamic batching, streaming output
- 🚀 Front-end and back-end separation design, easy to support new computing devices
- 🚀 Currently supports ChatGLM series models, Qwen2 series models, various LLAMA models (ALPACA, VICUNA, etc.), BAICHUAN models, MOSS models, MINICPM models, etc.

## Quick Start

### Compilation

It is recommended to use cmake for compilation, requiring pre-installed gcc, g++ (recommended 9.4 or above), make, cmake (recommended 3.23 or above).

GPU compilation requires a pre-installed CUDA compilation environment, using the latest CUDA version is recommended.

Compile using the following commands:

``` sh
bash install.sh -DUSE_CUDA=ON -D CMAKE_CUDA_COMPILER=$(which nvcc) # Compile GPU version
# bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=89 -D CMAKE_CUDA_COMPILER=$(which nvcc) # Specify CUDA architecture, e.g., 4090 uses architecture 89
# bash install.sh # Compile CPU version only
```

For compilation on other platforms, refer to the documentation:
[TFACC Platform](docs/tfacc.md)

### Running the demo program (python)

Assuming our model is located in the "~/Qwen2-7B-Instruct/" directory:

After compilation, you can use the following demos:

``` sh
# OpenAI API server (currently in testing and tuning phase)
# Requires dependencies: pip install -r requirements-server.txt
# Opens a server named 'qwen' on port 8080
python3 -m ftllm.server -t 16 -p ~/Qwen2-7B-Instruct/ --port 8080 --model_name qwen

# Use a model with float16 precision for conversation
python3 -m ftllm.chat -t 16 -p ~/Qwen2-7B-Instruct/ 

# Online quantization to int8 model for conversation
python3 -m ftllm.chat -t 16 -p ~/Qwen2-7B-Instruct/ --dtype int8

# webui
# Requires dependencies: pip install streamlit-chat
python3 -m ftllm.webui -t 16 -p ~/Qwen2-7B-Instruct/ --port 8080
```

Detailed parameters can be viewed using the --help argument for all demos.

For detailed parameter explanations, please refer to [Parameter Documentation](docs/english_demo_arguments.md).

Current model support can be found at: [Model List](docs/models.md)

For architectures that cannot directly read Hugging Face models, refer to [Model Conversion Documentation](docs/convert_model.md) to convert models to fastllm format.

If you need to customize the model structure, you can refer to the detailed instructions [Custom Model](docs/english_custom.md)

### Running the demo program (c++)

```
# Enter the fastllm/build-fastllm directory

# Command line chat program, supports typewriter effect (Linux only)
./main -p ~/Qwen2-7B-Instruct/

# Simple webui, uses streaming output + dynamic batch, supports concurrent access
./webui -p ~/Qwen2-7B-Instruct/ --port 1234 
```

Compilation on Windows is recommended using Cmake GUI + Visual Studio, completed in the graphical interface.

For compilation issues, especially on Windows, refer to [FAQ](docs/faq.md).

### Python API

``` python
# Model creation
from ftllm import llm
model = llm.model("~/Qwen2-7B-Instruct/")

# Generate response
print(model.response("你好"))

# Stream generate response
for response in model.stream_response("你好"):
    print(response, flush = True, end = "")
```

Additional settings such as CPU thread count can be found in the detailed API documentation: [ftllm](docs/ftllm.md)

This package does not include low-level APIs. For deeper functionalities, refer to [Python Binding API](#Python-binding-API).

## Multi-Card Deployment

### Using Multi-Card Deployment in Python Command Line Calls

``` sh
# Use the --device parameter to set multi-card calls
#--device cuda:1 # Set single device
#--device "['cuda:0', 'cuda:1']" # Deploy model evenly across multiple devices
#--device "{'cuda:0': 10, 'cuda:1': 5, 'cpu': 1} # Deploy model proportionally across multiple devices
```

### Using Multi-Card Deployment in ftllm

``` python
from ftllm import llm
# Supports the following three methods, must be called before model creation
llm.set_device_map("cuda:0") # Deploy model on a single device
llm.set_device_map(["cuda:0", "cuda:1"]) # Deploy model evenly across multiple devices
llm.set_device_map({"cuda:0" : 10, "cuda:1" : 5, "cpu": 1}) # Deploy model proportionally across multiple devices
```

### Using Multi-Card Deployment in Python Binding API

``` python
import pyfastllm as llm
# Supports the following method, must be called before model creation
llm.set_device_map({"cuda:0" : 10, "cuda:1" : 5, "cpu": 1}) # Deploy model proportionally across multiple devices
```

### Using Multi-Card Deployment in c++

``` cpp
// Supports the following method, must be called before model creation
fastllm::SetDeviceMap({{"cuda:0", 10}, {"cuda:1", 5}, {"cpu", 1}}); // Deploy model proportionally across multiple devices
```

## Docker Compilation and Running
Running docker requires the local installation of NVIDIA Runtime and modification of the default runtime to nvidia.

1. Install nvidia-container-runtime
```
sudo apt-get install nvidia-container-runtime
```

2. Modify docker default runtime to nvidia

/etc/docker/daemon.json
```
{
  "registry-mirrors": [
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ],
  "runtimes": {
      "nvidia": {
          "path": "/usr/bin/nvidia-container-runtime",
          "runtimeArgs": []
      }
   },
   "default-runtime": "nvidia" // This line is required
}

```

3. Download the converted models to the models directory
```
models
  chatglm2-6b-fp16.flm
  chatglm2-6b-int8.flm
```

4. Compile and start webui
```
DOCKER_BUILDKIT=0 docker compose up -d --build
```

## Usage on Android

### Compilation
``` sh
# Compilation on PC requires downloading NDK tools
# You can also try compiling on the phone, using cmake and gcc in termux (no need for NDK)
mkdir build-android
cd build-android
export NDK=<your_ndk_directory>
# If the phone does not support, remove "-DCMAKE_CXX_FLAGS=-march=armv8.2a+dotprod" (most new phones support this)
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_CXX_FLAGS=-march=armv8.2a+dotprod ..
make -j
```

### Running

1. Install the termux app on the Android device.
2. Execute termux-setup-storage in termux to gain permission to read phone files.
3. Copy the main file and model file compiled with NDK into the phone and into the termux root directory.
4. Use the command ```chmod 777 main``` to grant permissions.
5. Run the main file, refer to ```./main --help``` for parameter format.