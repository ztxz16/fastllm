"""Exercise the public EXEs with an isolated PATH and relocated bundle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(
        ("PYTHON", "CUDA", "CONDA", "VIRTUAL_ENV", "PIP_", "FTLLM_", "FASTLLM_"))}
    env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    cli = bundle / "ftllm.exe"
    with tempfile.TemporaryDirectory(prefix="ftllm-entry-") as temp:
        def run(arguments, **kwargs):
            result = subprocess.run([str(a) for a in arguments], cwd=temp, env=env,
                                    capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=60, **kwargs)
            result.check_returncode()
            return result
        for executable in (cli, bundle / "ftllm-launch-webui.exe"):
            run([executable, "--help"])
        config = Path(temp) / "配置 space & 'quoted'.json"
        run([cli, "config", config])
        assert isinstance(json.loads(config.read_text(encoding="utf-8")), dict)
        bad = subprocess.run([str(cli), "--not-a-real-option"], cwd=temp, env=env,
                             capture_output=True, timeout=60)
        assert bad.returncode == 2, bad.returncode
        # Explorer launches a console EXE in a new console. Supplying stdin
        # lets the test close the otherwise interactive PowerShell cleanly.
        process = subprocess.Popen([str(cli)], cwd=temp, env=env,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            output, _ = process.communicate("ftllm --version\nexit\n", timeout=60)
            assert process.returncode == 0, output
            assert "FastLLM portable environment ready" in output, output
            assert "ftllm version:" in output, output
        finally:
            if process.poll() is None:
                subprocess.run([str(Path(env["PATH"]) / "taskkill.exe"), "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=20)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"passed": True, "isolated_path": env["PATH"],
        "cli_help": True, "browser_help": True, "unicode_arguments": True,
        "exit_code_forwarding": True, "double_click_terminal": True}, indent=2), encoding="utf-8")
    print("[OK] Public entrypoints: isolated PATH, Unicode arguments, exit codes and double-click terminal")


if __name__ == "__main__":
    main()
