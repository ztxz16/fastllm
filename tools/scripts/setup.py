from setuptools import setup, find_packages

setup (
    name = "ftllm",
    version = "0.0.0.1",
    author = "huangyuyang",
    author_email = "ztxz16@foxmail.com",
    description = "Fastllm",
    url = "https://github.com/ztxz16/fastllm",
    packages = ['ftllm', 'ftllm/openai_server', 'ftllm/openai_server/protocal'],

    package_data = {
        '': ['*.dll', '*.so', '*.dylib']
    }
)
