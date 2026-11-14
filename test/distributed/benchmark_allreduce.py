#!/usr/bin/env python3
"""AllReduce 算子分阶段测试入口；不会自动安装依赖或执行编译。"""

import argparse
import json
from pathlib import Path

from allreduce_bench import plan
from allreduce_bench.reporting import metadata, write_report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["plan"])
    p.add_argument("--output", required=True, help="新目录；已存在时拒绝覆盖")
    p.add_argument("--devices", default="0,1,2,3")
    return p


def main():
    args = parser().parse_args()
    plan.devices(args.devices)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    meta, rows = metadata(args), []
    status = "FAIL"
    try:
        spec = plan.manifest()
        (output / "manifest.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = plan.plan_rows()
        meta["manifest"] = spec
        status = "PLANNED"
    except Exception as exc:
        rows.append(dict(case=args.stage, stage=args.stage, status="FAIL", reason=str(exc)))
    write_report(output, rows, meta, status)
    print(f"{status}: {output}")
    return 0 if status in ("PASS", "PLANNED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
