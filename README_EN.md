# fastllm

## Introduction

fastllm is a high-performance large model inference library implemented in pure C++ with no third-party dependencies.

6~7 billion parameter models can run smoothly on Android devices.

Deployment and communication QQ group: 831641348

| [Quick Start](#quick-start) | [Model Acquisition](#model-acquisition) | [Development Plan](#development-plan) |

## Overview of Features

- 🚀 Pure C++ implementation, easy to port across platforms, can be compiled directly on Android.
- 🚀 ARM platform supports NEON instruction set acceleration, X86 platform supports AVX instruction set acceleration, NVIDIA platform supports CUDA acceleration, and all platforms are very fast.
- 🚀 Supports floating-point models (FP32), half-precision models (FP16), and quantized models (INT8, INT4) for acceleration.
- 🚀 Supports multi-card deployment, supports GPU + CPU hybrid deployment.
- 🚀 Supports batch speed optimization.
- 🚀 Supports dynamic batch stitching during concurrent computation.
- 🚀 Supports streaming output, convenient for implementing typewriter effects.
- 🚀 Supports Python invocation.
- 🚀 Front-end and back-end separation design, easy to support new computing devices.
- 🚀 Currently supports ChatGLM series models, various LLAMA models (ALPACA, VICUNA, etc.), BAICHUAN models, QWEN models, MOSS models, MINICPM models, etc.

## Two lines of code to accelerate (under testing, currently only supports ChatGLM series)

Use the following command to install the fastllm_pytools package:

```sh
cd fastllm
mkdir build
cd build
cmake .. -DUSE_CUDA=ON # If not compiling with GPU, then use cmake .. -DUSE_CUDA=OFF
make -j
cd tools && python setup.py install
```

To utilize fastllm for acceleration in your original inference program, you simply need to add two lines of code.

```python
# This is the original program, creating the model through the huggingface interface
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

# Add the following two lines to convert the huggingface model to the fastllm model
# Currently, the from_hf interface can only accept original models or ChatGLM's int4, int8 quantized models, and cannot convert other quantized models temporarily
from fastllm_pytools import llm
model = llm.from_hf(model, tokenizer, dtype="float16")  # dtype supports "float16", "int8", "int4"

# Comment out this line model.eval()
#model = model.eval()
```

The model now supports the ChatGLM API functions chat and stream_chat, so the ChatGLM demo program can run without needing any other code modifications.

The model also supports the following APIs for generating replies.
```python
# Generate a response
print(model.response("Hello"))

# Streaming response generation
for response in model.stream_response("Hello"):
    print(response, flush=True, end="")
```

```python
model.save("model.flm")  # Export the fastllm model
new_model = llm.model("model.flm")  # Import the fastllm model
```

Note: This feature is in the testing phase, and currently, only ChatGLM and ChatGLM2 models have been verified to be accelerated with two lines of code.

## PEFT Support (In Testing, Currently Only Supports ChatGLM + LoRA)

Using 🤗PEFT, you can easily run fine-tuned large models. You can use the following method to accelerate your PEFT model with fastllm:

```python
import sys
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
sys.path.append('..')
model = AutoModel.from_pretrained("THUDM/chatglm-6b", device_map='cpu', trust_remote_code=True)
model = PeftModel.from_pretrained(model, "path/to/your/own/adapter")  # Use your own PEFT adapter here
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

# If there is an active_adapter in the model, it will also be enabled by default in the fastllm model
from fastllm_pytools import llm
model = llm.from_hf(model, tokenizer, dtype="float16")  # dtype supports "float16", "int8", "int4"
```

Next, you can use the model just like a regular model (e.g., by calling the chat and stream_chat functions).

You can also change the adapter used by the PEFT model:

```python
model.set_adapter('your adapter name')
```

Or disable PEFT to use the original pre-trained model:

```python
model.disable_adapter()
```

## Inference Speed

6B-level int4 model has a minimum latency of about 5.5ms on a single 4090.

6B-level fp16 model achieves a maximum throughput of over 10000 tokens/s on a single 4090.

The speed of the 6B-level int4 model on Snapdragon 865 is approximately 4~5 tokens/s.

[For detailed benchmark data points, click here.](docs/benchmark.md)

## CMMLU Accuracy Test

|              Model  | Data Accuracy |  CMMLU Score |
|-----------------: |-------- |------------|
| ChatGLM2-6b-fp16  | float32 |  50.16     |
| ChatGLM2-6b-int8  | float32 |  50.14     |
| ChatGLM2-6b-int4  | float32 |  49.63     |

Currently, ChatGLM2 model has been tested. For specific testing steps, please click [here](test/cmmlu/README.md).

## Quick Start

### Compile


It is recommended to compile using cmake, and you'll need to have a C++ compiler, make, and cmake installed beforehand.

For gcc, version 9.4 or higher is recommended, and for cmake, version 3.23 or higher is recommended.

For GPU compilation, ensure that you have the CUDA compilation environment installed, and it's recommended to use the latest possible CUDA version.

To compile, use the following command:

```sh
cd fastllm
mkdir build
cd build
cmake .. -DUSE_CUDA=ON # If not compiling with GPU, use cmake .. -DUSE_CUDA=OFF
make -j
```

After compiling, you can install the simple Python toolkit with the following command.

```sh
cd tools # Now you are in the fastllm/build/tools directory
python setup.py install

```

The compilation for different platforms can be referenced in the documentation. 
[TFACC Platform](docs/tfacc.md)

### Run Demo Program.

Assuming you have obtained the model named `model.flm` (refer to [Model Acquisition](#model-acquisition))

After compilation, you can use the following demo in the build directory:

```sh
# Now you are in the fastllm/build directory

# Command-line chat program, supports typewriter effect (Linux only)
./main -p model.flm 

# Simple web UI, using streaming output + dynamic batch, can handle multiple concurrent accesses
./webui -p model.flm --port 1234 

# Python version of the command-line chat program, using model creation and streaming conversation effects
python tools/cli_demo.py -p model.flm 

# Python version of the simple web UI, you need to install streamlit-chat first
streamlit run tools/web_demo.py model.flm 
```

For compiling on Windows, it's recommended to use CMake GUI + Visual Studio. You can complete the process in a graphical interface.

If you encounter any issues during compilation, especially on Windows, you can refer to the documentation for troubleshooting. [FAQ](docs/faq.md)

### Simple Python Commands

If you have installed the simple Python toolkit after compilation, you can use Python to call some basic APIs. (If you haven't installed it, you can still use 'import' to directly use the compiled 'tools/fastllm_pytools')

```python
# Model creation
from fastllm_pytools import llm
model = llm.model("model.flm")

# Generate response
print(model.response("Hello"))

# Stream response generation
for response in model.stream_response("Hello"):
    print(response, flush=True, end="")
```

Additionally, you can set the number of CPU threads and other parameters. For detailed API documentation, see [fastllm_pytools](docs/fastllm_pytools.md).

This package does not include low-level APIs. If you need more advanced features, please refer to [Python Binding API](#Python-Binding-API).

## Python Binding API

```
cd pyfastllm
export USE_CUDA=OFF    # Use CPU only; remove this line to use GPU
python3 setup.py build
python3 setup.py install 
cd examples/
python cli_simple.py -m chatglm -p chatglm-6b-int8.flm
# or  
python web_api.py -m chatglm -p chatglm-6b-int8.flm
```
You can test the above web API using web_api_client.py. For more usage details, see the documentation [API Documents](pyfastllm/README.md).

## Multi-GPU deployment

### Using multi-GPU deployment in fastllm_pytools

``` python
from fastllm_pytools import llm

# Support the following three methods, need to be called before model creation
llm.set_device_map("cuda:0")  # Deploy the model on a single device
llm.set_device_map(["cuda:0", "cuda:1"])  # Deploy the model evenly across multiple devices
llm.set_device_map({"cuda:0": 10, "cuda:1": 5, "cpu": 1})  # Deploy the model on multiple devices with different ratios
```

### Using multi-GPU deployment in the Python Binding API.
``` python
import pyfastllm as llm

# Support the following method, needs to be called before model creation
llm.set_device_map({"cuda:0": 10, "cuda:1": 5, "cpu": 1})  # Deploy the model on multiple devices with different ratios
```
### Using multi-GPU deployment in C++.

``` cpp
// Support the following method, needs to be called before model creation
fastllm::SetDeviceMap({{"cuda:0", 10}, {"cuda:1", 5}, {"cpu", 1}}); // Deploy the model on multiple devices with different ratios
```

## Compiling and running with Docker
Docker runtime requires NVIDIA runtime to be installed locally and default runtime needs to be changed to nvidia.

1. Installing nvidia-container-runtime
```
sudo apt-get install nvidia-container-runtime
```

2. Change the Docker default runtime to Nvidia.

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
   "default-runtime": "nvidia" // Only this line is needed
}
```

3. Download the pre-trained model to the models directory.
```
models
  chatglm2-6b-fp16.flm
  chatglm2-6b-int8.flm
```

4. Compile and start the web UI.
```
DOCKER_BUILDKIT=0 docker compose up -d --build
```

## Android上使用

### Compile
```sh
# Compiling on a PC requires downloading the NDK tool.
# Alternatively, you can try compiling on a mobile device. In Termux, you can use cmake and gcc (no need for NDK).
mkdir build-android
cd build-android
export NDK=<your_ndk_directory>
# If the mobile device does not support it, remove "-DCMAKE_CXX_FLAGS=-march=armv8.2a+dotprod" (most newer phones support it).
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_CXX_FLAGS=-march=armv8.2a+dotprod ..
make -j
```

### Run

1. Install the Termux app on your Android device.
2. Run 'termux-setup-storage' in Termux to grant permission to access phone files.
3. Transfer the main file compiled with NDK, as well as the model file, to your phone and copy them to the root directory of Termux.
4. Use the command chmod 777 main to grant permissions.
5. You can then run the main file. For parameter format, see ./main --help.

## Model Acquisition

### Model Repository

You can download pre-converted models from the following link.

[huggingface](https://huggingface.co/huangyuyang)

### Model export

#### Exporting the ChatGLM model (default script exports the ChatGLM2-6b model).

``` sh
# ChatGLM-6B environment needs to be installed first.
# If using a fine-tuned model, modify the code in chatglm_export.py file to create tokenizer and model.
cd build
python3 tools/chatglm_export.py chatglm2-6b-fp16.flm float16 # Export float16 model
python3 tools/chatglm_export.py chatglm2-6b-int8.flm int8 # Export int8 model
python3 tools/chatglm_export.py chatglm2-6b-int4.flm int4 # Export int4 model
```

#### Exporting the Baichuan model (default script exports the baichuan-13b-chat model).

``` sh
# Baichuan environment needs to be installed first.
# If using a fine-tuned model, modify the code in baichuan2flm.py file to create tokenizer and model.
# Export the corresponding model based on the required precision.
cd build
python3 tools/baichuan2flm.py baichuan-13b-fp16.flm float16 # Export float16 model
python3 tools/baichuan2flm.py baichuan-13b-int8.flm int8 # Export int8 model
python3 tools/baichuan2flm.py baichuan-13b-int4.flm int4 # Export int4 model
```

#### Exporting the Baichuan2 model (default script exports the baichuan2-7b-chat model).

``` sh
# Baichuan2 environment needs to be installed first.
# If using a fine-tuned model, modify the code in baichuan2_2flm.py file to create tokenizer and model.
# Export the corresponding model based on the required precision.
cd build
python3 tools/baichuan2_2flm.py baichuan2-7b-fp16.flm float16 # Export float16 model
python3 tools/baichuan2_2flm.py baichuan2-7b-int8.flm int8 # Export int8 model
python3 tools/baichuan2_2flm.py baichuan2-7b-int4.flm int4 # Export int4 model
```

#### Exporting the MOSS model

``` sh
# MOSS environment needs to be installed first.
# If using a fine-tuned model, modify the code in moss_export.py file to create tokenizer and model.
# Export the corresponding model based on the required precision.
cd build
python3 tools/moss_export.py moss-fp16.flm float16 # Export float16 model
python3 tools/moss_export.py moss-int8.flm int8 # Export int8 model
python3 tools/moss_export.py moss-int4.flm int4 # Export int4 model
```

#### Exporting LLAMA series models
``` sh
# Modify the build/tools/alpaca2flm.py program for exporting.
# The commands used for different LLAMA models vary greatly, so you need to configure them according to the parameters in torch2flm.py.
```
Some models' conversions can be referenced in the examples [here](docs/llama_cookbook.md).

#### Exporting the QWEN model
* **Qwen**
``` sh
# QWEN environment needs to be installed first.
# If using a fine-tuned model, modify the code in qwen2flm.py file to create tokenizer and model.
# Export the corresponding model based on the required precision.
cd build
python3 tools/qwen2flm.py qwen-7b-fp16.flm float16 # Export float16 model
python3 tools/qwen2flm.py qwen-7b-int8.flm int8 # Export int8 model
python3 tools/qwen2flm.py qwen-7b-int4.flm int4 # Export int4 model
```

* **Qwen1.5**
``` sh
# QWen2 environment needs to be installed first (transformers >= 4.37.0).
# Export the corresponding model based on the required precision.
cd build
python3 tools/llamalike2flm.py qwen1.5-7b-fp16.flm float16 "qwen/Qwen1.5-4B-Chat" # Export qwen1.5-4B-Chat float16 model
python3 tools/llamalike2flm.py qwen1.5-7b-int8.flm int8 "qwen/Qwen1.5-7B-Chat" # Export Qwen1.5-7B-Chat int8 model
python3 tools/llamalike2flm.py qwen1.5-7b-int4.flm int4 "qwen/Qwen1.5-14B-Chat" # Export Qwen1.5-14B-Chat int4 model
# The last parameter can be replaced with the model path.
```

#### Exporting the MINICPM model

```sh
# MINICPM environment needs to be installed first (transformers >= 4.36.0).
# The default script exports the iniCPM-2B-dpo-fp16 model.
cd build 
python tools/minicpm2flm.py minicpm-2b-float16.flm # Export dpo-float16 model
./main -p minicpm-2b-float16.flm # Execute the model
```

## Development Plan

If you have any features you need, feel free to bring them up in the discussion area.

### Short-term plan

- Add MMLU, CMMLU, and other test programs.
- Support direct conversion of pre-quantized Hugging Face models.
- Implement extrapolation to 8K length.

### Mid-term Plan

- Support more backends, such as OpenCL, Vulkan, and some NPU acceleration devices.
- Support and validate more models, improve the model repository.
- Optimize the tokenizer (since currently the original model's tokenizer can be directly used for tokenization in Python, this task is not urgent for now).

### Long-term Plan

- Support ONNX model import and inference.
- Support model fine-tuning.
