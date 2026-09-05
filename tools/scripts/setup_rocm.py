from setuptools import setup

server_require = ['fastapi', 'pydantic', 'openai', 'shortuuid', 'uvicorn']
pptx_require = ['python-pptx>=1.0.0']
document_require = ['pypdf>=4.0.0']
data_require = ['pandas>=2.0.0', 'openpyxl>=3.1.0', 'XlsxWriter>=3.1.0']
webui_require = ['fastapi', 'uvicorn'] + pptx_require + document_require + data_require
download_require = ['aria2', 'modelscope>=1.34.0,<2']
video_require = ['imageio', 'imageio-ffmpeg']
all_require = server_require + download_require + video_require + pptx_require + document_require + data_require

setup (
    name = "ftllm_rocm",
    version = "0.1.8.1",
    author = "huangyuyang",
    author_email = "ztxz16@foxmail.com",
    description = "High-performance C++ inference engine for dense and MoE language models",
    url = "https://github.com/ztxz16/fastllm",
    entry_points = {
        'console_scripts' : [
            'ftllm=ftllm.cli:main'
        ]
    },
    packages = ['ftllm', 'ftllm/openai_server', 'ftllm/openai_server/protocal'],
    package_data = {
        '': ['*.dll', '*.so', '*.dylib', '*.so.*', '*.html', '*.js', '*.svg',
             'launcher_assets/*', 'launcher_assets/locales/*.json', 'webui_assets/*']
    },
    install_requires=[
        'pyreadline3',
        'transformers',
        'jinja2>=3.1.0'
    ] + all_require,
    extras_require={
        'all': all_require,
        'server': server_require,
        'webui': webui_require,
        'pptx': pptx_require,
        'document': document_require,
        'data': data_require,
        'video': video_require
    },
)
