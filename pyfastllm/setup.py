import glob
import os.path
from setuptools import setup, Extension
from setuptools import find_packages

import sys
import argparse
parser = argparse.ArgumentParser(description='build pyfastllm wheel')
parser.add_argument('--cuda', dest='cuda', action='store_true', default=False,
                    help='build with cuda support')
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

__VERSION__ = "'0.1.4'"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def config_ext():
    ext_modules = []
    try:
        from pybind11.setup_helpers import Pybind11Extension
        source_files = glob.glob(os.path.join(BASE_DIR, "src/**/*.cpp"), recursive=True)
        for file in source_files:
            if file.endswith("cudadevice.cpp"): 
                source_files.remove(file)

        extra_compile_args = ["-w", "-DPY_API"]
        # If any libraries are used, e.g. libabc.so
        include_dirs = [os.path.join(BASE_DIR, "include/"), os.path.join(BASE_DIR, "include/devices/cpu/"), os.path.join(BASE_DIR, "include/models"), os.path.join(BASE_DIR, "include/utils")]
        library_dirs = []
        
        # (optional) if the library is not in the dir like `/usr/lib/`
        # either to add its dir to `runtime_library_dirs` or to the env variable "LD_LIBRARY_PATH"
        # MUST be absolute path
        runtime_library_dirs = []
        libraries = []

        if args.cuda:
            assert False, "Not Implement Yet!"
            extra_compile_args.append("-DUSE_CUDA -Wl,-rpath,$ORIGIN/")

            source_files.append(os.path.join(BASE_DIR, "src/devices/cuda/cudadevice.cpp"))
            include_dirs.append(os.path.join(BASE_DIR, "include/devices/cuda/"))

            library_dirs.append("/usr/local/cuda/lib64/")
            library_dirs.append(os.path.join(BASE_DIR, "pyfastllm/"))

            libraries.append("fastllm_cuda")

        ext_modules = [
            Pybind11Extension(
                "pyfastllm", 
                source_files,
                define_macros=[('VERSION_INFO', __VERSION__)],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                runtime_library_dirs=runtime_library_dirs,
                libraries=libraries,
                extra_compile_args=extra_compile_args,
                cxx_std=17,
                language='c++'
            ),
        ]
    except Exception as e:
        print(f"some errors happened: ")
        print(e)
        sys.exit(1)
    
    return ext_modules

cmdclass = {}
dyn_libs = glob.glob("*.so", root_dir="./fastllm")
dyn_libs += glob.glob("*.dll", root_dir="./fastllm")
# print(dyn_libs)
setup(
    name='fastllm',  
    version=eval(__VERSION__),
    description='python api for fastllm',
    author='wildkid1024',
    author_email='wildkid1024@outlook.com',
    maintainer='',
    maintainer_email='',
    url='',
    long_description='',
    # ext_modules=ext_modules,
    # packages = ['fastllm', 'fastllm.utils'],
    packages = find_packages(), 
    package_data={
        'fastllm': dyn_libs,
    }, 
    cmdclass=cmdclass,
    setup_requires=[""],
    install_requires=[""],
    python_requires='>=3.6',
    # data_files = [('', dyn_libs)],
    include_package_data=True,
    entry_points={
        'console_scripts':[
            'fastllm-convert = fastllm.convert:main'
        ]
    },
    zip_safe=False,
    classifiers=[
        'AI::ChatGPT'
        'AI::InfereEngine',
        'LLM::ChatGLM',
        'LLM::Moss',
        'LLM::LLama'
    ]
)