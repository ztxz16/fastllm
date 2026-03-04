import os
from setuptools import setup, find_packages

server_require = ['fastapi', 'pydantic', 'openai', 'shortuuid', 'uvicorn']
webui_require = ['streamlit-chat']
download_require = ['aria2']
tokenizer_require = ['tiktoken', 'blobfile', 'partial_json_parser']
all_require = server_require + webui_require + download_require + tokenizer_require

PACKAGE_INFO = {
    "release": {"name": "ftllm", "version": "0.1.5.1"},
    "nightly": {"name": "ftllm-nightly", "version": "0.0.0.2"},
}
variant = "nightly" if os.environ.get("FASTLLM_NIGHTLY", "0") == "1" else "release"
package_name = PACKAGE_INFO[variant]["name"]
package_version = PACKAGE_INFO[variant]["version"]

setup (
    name = package_name,
    version = package_version,
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
        '': ['*.dll', '*.so', '*.dylib', '*.so.*', 'build_info.json']
    },
    install_requires=[
        'pyreadline3',
        'transformers',
        'jinja2>=3.1.0',
        'nvidia-cuda-runtime-cu12',
        'nvidia-cublas-cu12',
        'nvidia-nccl-cu12'
    ] + all_require,
    extras_require={
        'all': all_require,
        'server': server_require,
        'webui': webui_require
    },
)
