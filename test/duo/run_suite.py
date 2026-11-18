"""Run native regression against pre-generated reference_fixture.py fixtures."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--fixtures", type=Path, required=True)
parser.add_argument(
    "--production-head", type=Path, help="128-wide FP8 reference fixture"
)
args = parser.parse_args()
binary = args.build / "src/models/duo"
results = {}


def run(argv):
    result = subprocess.run(
        list(map(str, argv)), text=True, stdout=subprocess.PIPE, check=False
    )
    if result.returncode:
        sys.stderr.write(result.stdout)
        result.check_returncode()
    return result.stdout


results["layout"] = run([binary / "duo_layout_test"]).strip()
cases = [
    (name, args.fixtures / name)
    for name in ("tiny", "tiny-lora", "tiny-fp8", "tiny-causal")
]
if args.production_head:
    cases.append(("tiny-fp8-head128", args.production_head))
for name, root in cases:
    output = args.fixtures / (name + "-native")
    command = [binary / "duo_generate", root, root / "request.json", output]
    if not name.startswith("tiny-fp8"):
        command.append(root / "adapter")
    (args.fixtures / (name + "-native.log")).write_text(run(command))
    results[name] = json.loads(
        run([sys.executable, Path(__file__).with_name("compare.py"), root, output])
    )
results["http"] = json.loads(
    run(
        [
            sys.executable,
            Path(__file__).with_name("test_server.py"),
            "--binary",
            binary / "duo_server",
            "--fixture",
            args.fixtures / "tiny",
        ]
    )
)
results["binaries"] = {
    name: hashlib.sha256((binary / name).read_bytes()).hexdigest()
    for name in ("duo_generate", "duo_server")
}
(args.fixtures / "verification.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
