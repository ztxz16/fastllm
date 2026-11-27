#!/usr/bin/env python3
# GGUF 完整快照的离线校验工具；只读取文件，不下载、不编译、不改写依赖。

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description="校验 GGUF 文件清单、内容与固定版本的 Git tree")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="仓库根目录，默认根据脚本位置确定")
    args = parser.parse_args()
    root = args.root.resolve()
    lock = json.loads((root / "tools/third_party/gguf.lock.json").read_text(encoding="utf-8"))

    if lock["schema_version"] != 1 or lock["destination"] != "third_party/gguf":
        raise ValueError("不支持的版本清单或目标目录")
    source = lock["source"]
    if source["path"] != lock["destination"]:
        raise ValueError("组件必须保持上游完整目录路径")
    for field in ("revision", "git_tree"):
        if not re.fullmatch(r"[0-9a-f]{40}", source[field]):
            raise ValueError(f"无效的 {field}")
    files = lock["files"]
    if not files:
        raise ValueError("版本清单不能为空")
    for name, info in files.items():
        if name in (".", "..") or not name or "/" in name or "\\" in name:
            raise ValueError("当前版本只支持完整的单层文件目录")
        if info["mode"] != "100644":
            raise ValueError(f"不支持的文件模式：{name}")
        for field, length in (("git_blob", 40), ("sha256", 64)):
            if not re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", info[field]):
                raise ValueError(f"无效的 {field}：{name}")

    component = root / lock["destination"]
    if (root / "third_party").is_symlink() or component.is_symlink():
        raise ValueError("组件目录不能是符号链接")
    actual_names = {path.name for path in component.iterdir()}
    errors = []
    for name in sorted(set(files) - actual_names):
        errors.append(f"缺少文件：{name}")
    for name in sorted(actual_names - set(files)):
        errors.append(f"多余文件：{name}")

    tree_entries = []
    for name in sorted(files, key=lambda value: value.encode("utf-8")):
        info = files[name]
        tree_entries.append(b"100644 " + name.encode("utf-8") + b"\0" + bytes.fromhex(info["git_blob"]))
        path = component / name
        if path.is_symlink() or not path.is_file():
            errors.append(f"不是普通文件：{name}")
            continue
        if path.stat().st_mode & 0o111:
            errors.append(f"执行权限与上游不一致：{name}")
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
        if blob != info["git_blob"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
            errors.append(f"内容与固定版本不一致：{name}")

    tree_data = b"".join(tree_entries)
    tree = hashlib.sha1(b"tree " + str(len(tree_data)).encode("ascii") + b"\0" + tree_data).hexdigest()
    if tree != source["git_tree"]:
        errors.append("文件清单与固定版本的 Git tree 不一致")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"GGUF 完整性校验通过：{len(files)} 个文件，版本 {source['revision']}，tree {tree}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"GGUF 完整性校验失败：{error}", file=sys.stderr)
        sys.exit(1)
