# 常见问题

## CMAKE

### CMAKE_CUDA_ARCHITECTURES must be non-empty if set.

**现象：**

> CMake Error at cmake/Modules/CMakeDetermineCUDACompiler.cmake:277 (message):  
>   CMAKE_CUDA_ARCHITECTURES must be non-empty if set.  
> Call Stack (most recent call first):  
>   CMakeLists.txt:39 (enable_language)

**解决办法：**

部分版本cmake存在该问题，需手动指定`CMAKE_CUDA_ARCHITECTURES`。执行：

```shell
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
```

### Unsupported gpu architecture 'compute_native'

**现象：**

> nvcc fatal : Unsupported gpu architecture 'compute_native'

**解决办法：**

手动修改 CMakeLists.txt，根据GPU型号手动指定GPU的[Compute Capability](https://developer.nvidia.com/cuda-gpus)。如：

``` diff
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -52,7 +52,7 @@
     #message(${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
     set(FASTLLM_CUDA_SOURCES src/devices/cuda/cudadevice.cpp src/devices/cuda/cudadevicebatch.cpp src/devices/cuda/fastllm-cuda.cu)
     set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} cublas)
-    set(CMAKE_CUDA_ARCHITECTURES "native")
+    set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)
 endif()
 
 if (PY_API)
```

### identifier "__hdiv" is undefined

**现象：**

> src/devices/cuda/fastllm-cuda.cu(247): error: identifier "hexp" is undefined  
> src/devices/cuda/fastllm-cuda.cu(247): error: identifier "__hdiv" is undefined  
> ...

**原因：** [计算能力（Compute Capability）](https://developer.nvidia.com/cuda-gpus) <= 5.3 的GPU不支持半精度计算。

**解决办法：** 如需要支持这些GPU，执行cmake时使用编译选项`CUDA_NO_TENSOR_CORE`：

```shell
cmake .. -DUSE_CUDA=ON -DCUDA_NO_TENSOR_CORE=ON
```

## Windows

### fastllm.h error

**现象：**

> include\fastllm.h(50): error : identifier "top_k" is undefined  
> include\fastllm.h(172): error : expected a "}"  
> include\fastllm.h(234): error : identifier "DataDevice" is undefined  
> ....

**解决办法：** 参考 [example\README.md](/example/README.md)。签出代码后，**修改 include/fastllm.h**，Visual Studio中点击”文件“ -> "高级保存选项"，在编码中选择”Unicode (UTF-8 **带签名**) -代码页 65001“，或在其他文本编辑器中转为”UTF-8 BOM“编码。（由于linux下gcc不识别BOM头，MSVC依赖BOM判断文件编码，该修改只能手动处理。）

### main.exe 无法识别中文输入

**原因：** Windows下cmd不支持UTF-8编码，

**解决办法：** 编译[Win32Demo](/example/README.md#win32demo-windows平台) 或使用 [WebUI](/example/README.md#web-ui)

### Windows（MSVC）编译下，int4出现乱码

**原因：** MSVC编译器优化选项 "`/Ob2`"、"`/Ob3`"与的现有代码冲突，

**解决办法：** 编译时，在”属性“中找到"C/C++" -> "优化" -> "内联函数扩展" 中选择“只适用于 __inline (/Ob1)”。

### 导入提示 FileNotFoundError

**现象：**

> File "...Python\lib\ctypes\_\_init\_\_.py", line 374, in \_\_init\_\_  
>     self._handle = _dlopen(self._name, mode)  
> FileNotFoundError: Could not find module 'tools\fastllm_pytools\fastllm_tools.dll' (or one of its dependencies). Try using the full path with constructor syntax.

**解决办法：** 非CPU编译时，部分版本的python存在这一问题。

GPU编译时，根据使用的CUDA版本，将cudart cublas的相关dll文件复制到fastllm_tools同一目录下，例如：

* CUDA 9.2
  * %CUDA_PATH%\bin\cublas64_92.dll
  * %CUDA_PATH%\bin\cudart64_92.dll
* CUDA 11.x 
  * %CUDA_PATH%\bin\cudart64_110.dll
  * %CUDA_PATH%\bin\cublas64_11.dll
  * %CUDA_PATH%\bin\cublasLt64_11.dll
* CUDA 12.x 
  * %CUDA_PATH%\bin\cudart64_12.dll
  * %CUDA_PATH%\bin\cublas64_12.dll
  * %CUDA_PATH%\bin\cublasLt64_12.dll

## fastllm_pytools

### 释放内存报错： CUDA error when release memory

**现象：**
退出时报错：
> Error: CUDA error when release memory!  
> CUDA error = 4, cudaErrorCudartUnloading at fastllm/src/devices/cuda/fastllm-cuda.cu:1493  
> 'driver shutting down'

**原因：** python解释器在终止时常常会优先终止自己的进程，而没有现先析构调用的第三方库，因此在退出python时CUDA Runtime已关闭，释放显存操作失败。由于大多数时候显存已释放，并不会引起问题。

**解决办法：** python程序退出时，先显式调用 `llm.release_memory()`方法。
