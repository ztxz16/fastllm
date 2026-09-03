import platform
import sys

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class LinuxX64Wheel(bdist_wheel):
    """The bundled Pi executable makes this a Linux x86-64 wheel."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", "linux_x86_64"


if any(command in sys.argv for command in ("bdist_wheel", "build")):
    machine = platform.machine().lower()
    if not sys.platform.startswith("linux") or machine not in {"x86_64", "amd64"}:
        raise RuntimeError(
            "ftllm-agent-runtime currently builds only on Linux x86-64"
        )


setup(cmdclass={"bdist_wheel": LinuxX64Wheel})
