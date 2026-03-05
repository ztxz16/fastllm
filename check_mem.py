#!/usr/bin/env python3
"""比较两个 CUDA Memory Debug Report 文件，输出差异部分（忽略两个文件中都存在的未释放指针）。"""

import sys
import re
from collections import OrderedDict


def parse_summary(lines):
    """解析文件头部的 Summary 信息。"""
    info = {}
    for line in lines:
        line = line.strip()
        if line.startswith("Time:"):
            info["time"] = line[len("Time:"):].strip()
        elif line.startswith("GPU Memory:"):
            info["gpu_memory"] = line
        elif line.startswith("Tracked allocations:"):
            info["tracked"] = line
        elif line.startswith("Big buffer pool:"):
            info["big_pool"] = line
        elif line.startswith("Small buffer pool:"):
            info["small_pool"] = line
    return info


def parse_unreleased_blocks(filepath):
    """解析文件中 '--- Unreleased Blocks Detail ---' 部分，返回 {ptr: (size_str, bytes, [callstack_lines])}。"""
    blocks = OrderedDict()
    summary_lines = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    in_unreleased = False
    current_ptr = None
    current_size_str = None
    current_bytes = 0
    current_callstack = []
    in_callstack = False

    for line in lines:
        raw = line.rstrip("\n")

        if not in_unreleased:
            summary_lines.append(raw)
            if re.match(r"^---\s*Unreleased Blocks Detail", raw):
                in_unreleased = True
            continue

        if raw.strip() == "========== End of Report ==========":
            break

        m = re.match(r"^ptr=(0x[0-9a-fA-F]+),\s*size=(.+)\((\d+)\s*bytes\)", raw)
        if m:
            if current_ptr is not None:
                blocks[current_ptr] = (current_size_str, current_bytes, current_callstack)
            current_ptr = m.group(1)
            current_size_str = m.group(2).strip()
            current_bytes = int(m.group(3))
            current_callstack = []
            in_callstack = False
            continue

        stripped = raw.strip()
        if stripped == "callstack:":
            in_callstack = True
            continue

        if in_callstack and stripped:
            current_callstack.append(stripped)
            continue

        if stripped == "" and in_callstack:
            in_callstack = False

    if current_ptr is not None:
        blocks[current_ptr] = (current_size_str, current_bytes, current_callstack)

    summary = parse_summary(summary_lines)
    return summary, blocks


def format_size(nbytes):
    if nbytes >= 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.2f} MB"
    elif nbytes >= 1024:
        return f"{nbytes / 1024:.2f} KB"
    else:
        return f"{nbytes} B"


def print_blocks(blocks, label):
    if not blocks:
        print(f"  （无）")
        return

    total_bytes = sum(b[1] for b in blocks.values())
    print(f"  共 {len(blocks)} 个块, 总计 {format_size(total_bytes)}")
    print()

    size_groups = {}
    for ptr, (size_str, nbytes, callstack) in blocks.items():
        cs_key = tuple(callstack)
        key = (nbytes, cs_key)
        if key not in size_groups:
            size_groups[key] = []
        size_groups[key].append(ptr)

    for (nbytes, cs_key), ptrs in sorted(size_groups.items(), key=lambda x: -x[0][0]):
        print(f"  [{format_size(nbytes)}] x {len(ptrs)} 块:")
        for ptr in ptrs:
            print(f"    ptr={ptr}")
        if cs_key:
            print(f"    callstack:")
            for frame in cs_key:
                print(f"      {frame}")
        print()


def main():
    if len(sys.argv) != 3:
        print(f"用法: {sys.argv[0]} <file1> <file2>")
        print(f"  比较两个 CUDA Memory Debug Report，输出差异（忽略两个文件中共有的未释放指针）。")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    summary1, blocks1 = parse_unreleased_blocks(file1)
    summary2, blocks2 = parse_unreleased_blocks(file2)

    common_ptrs = set(blocks1.keys()) & set(blocks2.keys())
    only_in_1 = OrderedDict((p, blocks1[p]) for p in blocks1 if p not in common_ptrs)
    only_in_2 = OrderedDict((p, blocks2[p]) for p in blocks2 if p not in common_ptrs)

    print("=" * 60)
    print("CUDA Memory Debug Report 差异比较")
    print("=" * 60)
    print()

    print(f"文件 A: {file1}")
    print(f"  时间: {summary1.get('time', 'N/A')}")
    print(f"  未释放块总数: {len(blocks1)}")
    print()
    print(f"文件 B: {file2}")
    print(f"  时间: {summary2.get('time', 'N/A')}")
    print(f"  未释放块总数: {len(blocks2)}")
    print()

    print("-" * 60)
    print(f"共同未释放指针: {len(common_ptrs)} 个（已忽略）")
    total_common = sum(blocks1[p][1] for p in common_ptrs)
    print(f"共同未释放内存: {format_size(total_common)}")
    print("-" * 60)
    print()

    print(f">>> 仅在文件 A 中未释放的块（文件 B 中已释放或不存在）:")
    print_blocks(only_in_1, "A")

    print(f">>> 仅在文件 B 中未释放的块（文件 A 中已释放或不存在）:")
    print_blocks(only_in_2, "B")

    if not only_in_1 and not only_in_2:
        print("两个文件的未释放指针完全相同，没有差异。")
    else:
        delta_bytes = sum(b[1] for b in only_in_2.values()) - sum(b[1] for b in only_in_1.values())
        print("-" * 60)
        print(f"净变化: 文件 B 相对于文件 A {'增加' if delta_bytes >= 0 else '减少'} {format_size(abs(delta_bytes))}")
        print(f"  文件 A 独有: {len(only_in_1)} 块, {format_size(sum(b[1] for b in only_in_1.values()))}")
        print(f"  文件 B 独有: {len(only_in_2)} 块, {format_size(sum(b[1] for b in only_in_2.values()))}")


if __name__ == "__main__":
    main()
