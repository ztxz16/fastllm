# Hipify_torch

hipify_torch is a python utility to convert CUDA C/C++ code into HIP C/C++ code.
It is NOT a parser; it does a smart but basic search-and-replace based on CUDA-to-HIP mappings which are specified in the hipify_torch module.
It can also "hipify" the header include statements in your source code to ensure that it's the hipified header files that are included.

<!-- toc -->

- [Interface](#interface)
  - [Through cmake](#through-cmake)
  - [Through python](#through-python)
- [Utilities](#utilities)
  - [CMake utility function](#cmake-utility-function)
- [Custom hipify mapping](#custom-hipify-mapping)
- [Intended users](#intended-users)

<!-- tocstop -->

# Interface

## Through cmake

From the parent `CMakeLists.txt` file include the `Hipify.cmake` file from `./cmake/Hipify.cmake`

### API function -- ***hipify()***

This function executes the hipify conversion logic on an input directory recursively.

```
function(hipify CUDA_SOURCE_DIR HIP_CONFIG_DIR CONFIG_FILE)
```
- Takes 3 optional arguments, either CUDA_SOURCE_DIR or CONFIG_FILE argument is required
- `CUDA_SOURCE_DIR` - Full path of input cuda source directory which needs to be hipified.
- `HIP_SOURCE_DIR` - Full path of output directory where the hipified files will be placed.
                     If not provided, it is set to CUDA_SOURCE_DIR.
- `CONFIG_FILE` - JSON format file, which provides additional arguments for hipify_cli.py file.
                  When set, it is having higher precendence over CUDA_SOURCE_DIR/HIP_SOURCE_DIR.

#### Usage examples

```
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/hipify_torch/cmake")
include(Hipify)
# Example invocation - Provides cuda source dir and output hip source dir
hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR} HIP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/hip")

# Example invocation - Only cuda source dir provide and output hip files into same dir.
hipify(CUDA_SOURCE_DIR "/home/usr/project_sources/")

# Example invocation - Through config file
hipify(CONFIG_FILE "project_hipify_config_file.json")
```
Note: Update the CMAKE_MODULE_PATH list accordingly, if the hipify_torch repo is cloned into a different directory.

Above lines trigger the hipify script for all sources & header files under the `CUDA_SOURCE_DIR`

## Through python
### hipify_cli.py 
hipify_cli.py takes arguments through command line or a json file(hipify_config.json)
```
python hipify_cli.py --config-json hipify_config.json
```
hipify_config.json
```json
{
    "project_directory": <absolute path of project_directory to config_json>,
    "output_directory" : <absolute path of output_directory to config_json>,
    "header_include_dirs": [<absolute path to proj_dir of directories where headers are present>],
    "includes": [<absolute path to proj_dir of directories to be hipified>],
    "ignores": [<absolute path to proj_dir of directories to explicitly not hipify>],
    "extra_files": [<absolute or relative paths of extra files to be hipified>],
    "hipify_extra_files_only": <true or false, whether to only hipify extra files>
}
```
### hipify.hipify_python
```
from <path to hipify_torch>.hipify.hipify_python import hipify
```
Note: We are in the process of making hipify_torch as an installable python package, then `<path to hipify_torch>` isn't required.

# Utilities

## CMake utility function

### API function -- ***get_hipified_list()***

This utility function can be used to get a list of hipified files from a list of cuda files.

```
function(get_hipified_list INPUT_LIST OUTPUT_LIST)
```
- `INPUT_LIST` - CMake list containing a list of cuda file names
- `OUTPUT_LIST` - Cmake list containing a list of hipified files names. If the cuda file name is not changed after hipify, then it is NOT replaced in the list.

#### Usage example

```
get_hipified_list("${TP_CUDA_SRCS}" TP_CUDA_SRCS)
```

Here the `TP_CUDA_SRCS` in the input list containing cuda source files and doing a inplace update with  output list `TP_CUDA_SRCS`
For the file suffix unique string, list variable name itself is passed as a string.

# Custom hipify mapping

Users can define their own custom mapping by adding a custom_hipify_mapping.json from project_directory from where the hipify() function is being called.
To use a JSON file from a different directory, users can pass in the JSON file path via ```custom_map``` argument in the hipify method.
The custom hipify mappings will be applied *before* any other default hipify mappings. 
The below is the sample JSON file:

```
{
    "custom_map" : {
        "src mapping 1" : "dst mapping 1",
        "src mapping 2" : "dst mapping 2",
        ...
    }
}
```

# Intended users

This module can be used to
- Build [PyTorch](https://github.com/pytorch/pytorch)
- PyTorch CUDA extensions such as [torchvision](https://github.com/pytorch/vision), [detectron2](https://github.com/facebookresearch/detectron2) etc.
- PyTorch submodules CMake-based such as [tensorpipe](https://github.com/pytorch/tensorpipe), etc.
- And any other repo having CUDA files requiring to hipify to build on ROCm.
