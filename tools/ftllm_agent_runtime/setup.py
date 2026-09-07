import platform
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    """Install the bundled executables into platlib, including on split-lib systems."""

    def has_ext_modules(self):
        return True


class LinuxX64Wheel(bdist_wheel):
    """The bundled Pi executable makes this a Linux x86-64 wheel."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", "linux_x86_64"


class CompleteRuntimeBuild(build_py):
    """Fail before building a wheel that cannot run its advertised tools."""

    def run(self):
        root = Path(self.get_package_dir("ftllm_agent_runtime"))
        required = [
            "bin/pi", "bin/rg", "bin/fd", "bin/package.json", "bin/photon_rs_bg.wasm",
            "bin/theme/dark.json", "bin/theme/light.json", "bin/theme/theme-schema.json",
            "extensions/project_tools.ts", "licenses/PI_LICENSE",
            "licenses/agent-tools/manifest.json", "licenses/agent-tools/rg/COPYING",
            "licenses/agent-tools/rg/LICENSE-MIT", "licenses/agent-tools/rg/UNLICENSE",
            "licenses/agent-tools/fd/LICENSE-APACHE", "licenses/agent-tools/fd/LICENSE-MIT",
        ]
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise RuntimeError(
                "Incomplete Pi runtime: " + ", ".join(missing)
                + ". Run python scripts/fetch_pi.py and python scripts/fetch_tools.py before building."
            )
        super().run()


if any(command in sys.argv for command in ("bdist_wheel", "build")):
    machine = platform.machine().lower()
    if not sys.platform.startswith("linux") or machine not in {"x86_64", "amd64"}:
        raise RuntimeError(
            "ftllm-agent-runtime currently builds only on Linux x86-64"
        )


setup(
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": LinuxX64Wheel, "build_py": CompleteRuntimeBuild},
)
