import base64
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
COMMON = Path(__file__).resolve().parents[1] / "common.ps1"


@unittest.skipUnless(POWERSHELL, "PowerShell is required")
class PowerShellHelperTests(unittest.TestCase):
    def read_architectures(self, source):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "make_whl.sh").write_text(source, encoding="utf-8")
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            script = ("$ErrorActionPreference='Stop';$RepoRoot=" + quote(root) +
                      ";. " + quote(COMMON) + ";Get-FastllmCudaArch")
            encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
            return subprocess.run(
                [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-EncodedCommand", encoded],
                capture_output=True, text=True,
            )

    def test_current_bash_default_and_legacy_assignment(self):
        for source in [
            "CUDA_ARCH_LIST=${CUDA_ARCH_LIST:-'75-real;86;120-virtual'}\n",
            'CUDA_ARCH_LIST="75-real;86;120-virtual"\n',
        ]:
            with self.subTest(source=source):
                result = self.read_architectures(source)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "75-real;86;120-virtual")

    def test_missing_architecture_default_fails(self):
        result = self.read_architectures("#!/bin/bash\n")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
