import email
import os
from setuptools import setup, find_packages

server_require = ['fastapi', 'pydantic', 'openai', 'shortuuid', 'uvicorn']
webui_require = ['streamlit-chat']
download_require = ['aria2']
tokenizer_require = ['tiktoken', 'blobfile', 'partial_json_parser']
all_require = server_require + webui_require + download_require + tokenizer_require

PACKAGE_INFO = {
    "release": {"name": "ftllm", "version": "0.1.6.0"},
    "nightly": {"name": "ftllm-nightly", "version": "0.0.0.3"},
}


def load_package_info_from_metadata():
    metadata_paths = [
        "PKG-INFO",
        "ftllm.egg-info/PKG-INFO",
        "ftllm_nightly.egg-info/PKG-INFO",
    ]
    for metadata_path in metadata_paths:
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r", encoding = "utf-8") as metadata_file:
            metadata = email.message_from_file(metadata_file)
        name = metadata.get("Name")
        version = metadata.get("Version")
        if name and version:
            return {
                "name": name,
                "version": version,
            }
    return None


def resolve_package_info():
    if os.environ.get("FASTLLM_NIGHTLY", "0") == "1":
        return PACKAGE_INFO["nightly"]
    metadata_package_info = load_package_info_from_metadata()
    if metadata_package_info is not None:
        return metadata_package_info
    return PACKAGE_INFO["release"]


package_info = resolve_package_info()
package_name = package_info["name"]
package_version = package_info["version"]

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
