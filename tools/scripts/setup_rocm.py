"""Build the ROCm distribution without modifying the NVIDIA package sources."""
import json
from pathlib import Path

from setuptools import Distribution, find_namespace_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py

root = Path(__file__).resolve().parent
info = json.loads((root / "ftllm/rocm_build_info.json").read_text())
sdk_version = info["sdk_version"]


class RocmBuildPy(build_py):
    def run(self):
        super().run()
        package = Path(self.build_lib) / "ftllm"
        self.copy_file(str(root / "rocm_runtime.py"), str(package / "_rocm_init.py"))
        init = package / "__init__.py"
        init.write_text((root / 'ftllm/__init__.py').read_text() + "\n\nfrom ._rocm_init import initialize as _initialize_rocm\n"
                        "_initialize_rocm()\ndel _initialize_rocm\n")
        # Adapt only the staged ROCm wheel, not the shared NVIDIA sources.
        llm = package / "llm.py"
        source = (root / 'ftllm/llm.py').read_text()
        old = "preload_diagnostics = _preload_cuda_runtime_dependencies()"
        if source.count(old) != 1:
            raise RuntimeError("FastLLM loader changed; review the ROCm wheel adapter")
        source = source.replace(old, "preload_diagnostics = {}  # ROCm initialized by the package")
        source = source.replace('print("If CUDA/NCCL runtime is missing, try:")',
                                'print("Check the ROCm wheel installation and AMD driver:")')
        source = source.replace('print("pip install -U nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-nccl-cu12")',
                                'print("See the installation instructions shipped with ftllm-rocm.")')
        source = source.replace('print("Load fastllm failed. (Try update glibc)")\n        exit(0)',
                                'raise RuntimeError("Could not load FastLLM ROCm: " + "; ".join(load_errors))')
        llm.write_text(source)


class RocmDistribution(Distribution):
    def has_ext_modules(self):
        # The prebuilt ctypes library belongs in platlib, not purelib.
        return True


class RocmWheel(bdist_wheel):
    def get_tag(self):
        # ctypes does not depend on a CPython extension ABI.
        return "py3", "none", super().get_tag()[2]


server_require = ['fastapi', 'pydantic', 'openai', 'shortuuid', 'uvicorn']
tokenizer_require = ['tiktoken', 'blobfile', 'partial_json_parser', 'sentencepiece']
pptx_require = ['python-pptx>=1.0.0']
document_require = ['pypdf>=4.0.0']
data_require = ['pandas>=2.0.0', 'openpyxl>=3.1.0', 'XlsxWriter>=3.1.0']
webui_require = ['fastapi', 'uvicorn'] + pptx_require + document_require + data_require
download_require = ['aria2', 'modelscope>=1.34.0,<2']
video_require = ['imageio', 'imageio-ffmpeg']
all_require = (server_require + tokenizer_require + download_require + video_require
               + pptx_require + document_require + data_require)
extras = {
    'all': all_require, 'server': server_require + tokenizer_require,
    'webui': webui_require, 'pptx': pptx_require,
    'document': document_require, 'data': data_require, 'video': video_require,
}
for arch in info['architectures']:
    extras[arch] = [f'rocm-sdk-device-{arch}=={sdk_version}']
extras['all-gpus'] = [f'rocm-sdk-device-{arch}=={sdk_version}'
                      for arch in info['architectures']]

setup(
    name="ftllm-rocm",
    version="0.1.8.1.post1",
    python_requires=">=3.10",
    author="huangyuyang",
    author_email="ztxz16@foxmail.com",
    description="FastLLM inference engine for AMD ROCm",
    url="https://github.com/ztxz16/fastllm/tree/rocm",
    entry_points={'console_scripts': ['ftllm=ftllm.cli:main']},
    packages=find_namespace_packages(include=['ftllm', 'ftllm.openai_server', 'ftllm.openai_server.*']),
    package_data={'ftllm': ['*.so', '*.so.*', '*.json', '*.html', '*.js', '*.svg',
                            'licenses/*', 'launcher_assets/*', 'launcher_assets/locales/*.json']},
    install_requires=['numpy', 'pillow', 'requests', 'transformers', 'jinja2>=3.1.0',
                      f'rocm[libraries]=={sdk_version}'] + all_require,
    extras_require=extras,
    cmdclass={'build_py': RocmBuildPy, 'bdist_wheel': RocmWheel},
    distclass=RocmDistribution,
)
