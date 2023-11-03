from setuptools import setup, find_packages

setup (
    name = "fastllm_pytools",
    version = "0.0.1",
    author = "huangyuyang",
    author_email = "ztxz16@foxmail.com",
    description = "Fastllm pytools",
    url = "https://github.com/ztxz16/fastllm",
    packages = ['fastllm_pytools'],

    package_data = {
        '': ['*.dll', '*.so', '*.dylib']
    }
)
