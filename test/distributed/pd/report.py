"""PD 验证报告：保留每个 case、执行异常和原始精度，不触发编译。"""

import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/nvfp4"))
from xlsx_report import write_xlsx


def run(cases, output, *, backend, metadata_extra=None, scope="控制与映射；不代表 CUDA IPC 或模型端到端验收"):
    if not 1 <= len(cases) <= 50 or len({name for name, _ in cases}) != len(cases):
        raise ValueError("each script requires 1..50 uniquely named cases")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("refusing to overwrite a historical test run")
    from fastllm_pytools.pd import _core
    native = _core.load()
    if native.implementation != "cpp" or native.abi_version != 1:
        raise RuntimeError("PD verification requires the production C++ core")
    native_path = Path(native.__file__).resolve()
    native_metadata = dict(path=str(native_path), sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        implementation=native.implementation, abi=native.abi_version, entries=len(native.entries()),
        compiler=native.compiler, cpp_standard=native.cpp_standard, host_arch=native.host_arch,
        cuda_enabled=native.cuda_enabled, cuda_runtime_headers=native.cuda_runtime_headers)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), pid=os.getpid(), backend=backend,
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        python=sys.version, gpu="not used by protocol checks",
        cuda="runtime headers " + str(native.cuda_runtime_headers) if native.cuda_enabled else "disabled",
        architecture=native.host_arch + "; CUDA Runtime copies only; no new device kernel",
        native_core=native_metadata,
        seed=20261126, warmup=0, iterations=1, repeats=1)
    metadata.update(metadata_extra or {})
    rows = []
    try:
        for name, case in cases:
            try:
                evidence = case() or {}
                row = dict(case=name, status="PASS", reason="", **evidence)
            except Exception as error:
                row = dict(case=name, status="FAIL", reason=f"{type(error).__name__}: {error}", trace=traceback.format_exc())
            rows.append(row)
            (output / f"{name}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False))
            print(name, row["status"], flush=True)
    finally:
        status = "PASS" if len(rows) == len(cases) and all(r["status"] == "PASS" for r in rows) else "FAIL"
        (output / "summary.json").write_text(json.dumps(dict(status=status, metadata=metadata, cases=rows),
            ensure_ascii=False, indent=2, allow_nan=False))
        columns = ["case", "status", "reason"]
        for row in rows:
            for key, value in row.items():
                if key not in columns and key != "trace" and isinstance(value, (str, int, float, bool)):
                    columns.append(key)
        data = [[f"{row[key]:.3f}" if isinstance(row.get(key), float) else row.get(key, "")
                 for key in columns] for row in rows]
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(columns); writer.writerows(data)
        (output / "summary.md").write_text("# PD 验证：" + status + "\n\n| " + " | ".join(columns) + " |\n| " +
            " | ".join("---" for _ in columns) + " |\n" +
            "\n".join("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in line) + " |" for line in data) +
            "\n\n范围：" + scope + "。\n")
        write_xlsx(output / "summary.xlsx", [("Cases", columns, data),
            ("Metadata", ["key", "value"], [[k, str(v)] for k, v in metadata.items()])])
    return 0 if status == "PASS" else 1
