from setuptools import setup, find_packages

server_require = ['fastapi', 'pydantic', 'openai', 'shortuuid', 'uvicorn']
webui_require = ['streamlit-chat']
all_require = server_require + webui_require

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
    },
    install_requires=[
        'transformers',
        'jinja2>=3.1.0'
    ],
    extras_require={
        'all': all_require,
        'server': server_require,
        'webui': webui_require
    },
)
