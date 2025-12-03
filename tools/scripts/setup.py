from setuptools import setup, find_packages

server_require = ['fastapi', 'pydantic', 'openai', 'shortuuid', 'uvicorn']
webui_require = ['streamlit-chat']
download_require = ['aria2']
tokenizer_require = ['tiktoken', 'blobfile']
all_require = server_require + webui_require + download_require + tokenizer_require

setup (
    name = "ftllm",
    version = "0.1.5.1",
    author = "huangyuyang",
    author_email = "ztxz16@foxmail.com",
    description = "Fastllm",
    url = "https://github.com/ztxz16/fastllm",
    entry_points = {
        'console_scripts' : [
            'ftllm=ftllm.cli:main'
        ]
    },
    packages = ['ftllm', 'ftllm/openai_server', 'ftllm/openai_server/protocal', 'ftllm/openai_server/tool_parsers'],
    package_data = {
        '': ['*.dll', '*.so', '*.dylib', '*.so.*']
    },
    install_requires=[
        'pyreadline3',
        'transformers',
        'jinja2>=3.1.0'
    ] + all_require,
    extras_require={
        'all': all_require,
        'server': server_require,
        'webui': webui_require
    },
)
