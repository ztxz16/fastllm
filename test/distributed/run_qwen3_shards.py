#!/usr/bin/env python3
"""验证已有 C++ rank-local 读取器；只生成小型固定输入，不编译、不加载推理库。"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import uuid

from run_process_tp import ROOT, capture, save_report


def encode(values, dtype):
    if dtype == "BF16":
        return b"".join(struct.pack("<f", value)[2:] for value in values)
    return struct.pack("<" + ("e" if dtype == "F16" else "f") * len(values), *values)


def decode(data, dtype):
    if dtype == "BF16":
        return [struct.unpack("<f", b"\0\0" + data[i:i + 2])[0] for i in range(0, len(data), 2)]
    width, code = (2, "e") if dtype == "F16" else (4, "f")
    return list(struct.unpack("<" + code * (len(data) // width), data))


def write_safetensors(path, tensors):
    header, chunks, offset = {}, [], 0
    for name, tensor in tensors.items():
        data = tensor["data"]
        header[name] = {"dtype": tensor["dtype"], "shape": tensor["shape"],
                        "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(chunks))


def fixture(directory, dtype="F16", tied=False, bias=False, indexed=False, **dimensions):
    config = dict(model_type="qwen3", num_hidden_layers=2, hidden_size=6,
                  num_attention_heads=8, num_key_value_heads=2, head_dim=2,
                  intermediate_size=11, vocab_size=13, tie_word_embeddings=tied)
    config.update(dimensions)
    tensors = {}

    def add(name, shape):
        # 精确可表示的正负浮点数，无随机依赖；不同来源张量具有不同排列。
        values = [((i * 13 + len(tensors) * 37) % 251 - 125) / 8 for i in range(math.prod(shape))]
        tensors[name] = {"shape": shape, "dtype": dtype, "data": encode(values, dtype)}

    h, d, m, v = (config[k] for k in ("hidden_size", "head_dim", "intermediate_size", "vocab_size"))
    q, kv = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
    add("model.embed_tokens.weight", [v, h])
    add("model.norm.weight", [h])
    if not tied:
        add("lm_head.weight", [v, h])
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            add(p + name, [h])
        for name in ("q_norm.weight", "k_norm.weight"):
            add(p + "self_attn." + name, [d])
        for name, width in (("q_proj", q), ("k_proj", kv), ("v_proj", kv)):
            add(p + "self_attn." + name + ".weight", [width, h])
            if bias:
                add(p + "self_attn." + name + ".bias", [width])
        add(p + "self_attn.o_proj.weight", [h, q])
        for name in ("gate_proj", "up_proj"):
            add(p + "mlp." + name + ".weight", [m, h])
        add(p + "mlp.down_proj.weight", [h, m])
        if bias:
            add(p + "self_attn.o_proj.bias", [h])
            add(p + "mlp.down_proj.bias", [h])
    directory.mkdir()
    save_fixture(directory, config, tensors, indexed)
    return config, tensors


def save_fixture(directory, config, tensors, indexed=False):
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if indexed:
        groups = [dict(list(tensors.items())[i::2]) for i in range(2)]
        mapping = {}
        for i, group in enumerate(groups):
            filename = f"model-{i + 1:05d}-of-00002.safetensors"
            write_safetensors(directory / filename, group)
            mapping.update({name: filename for name in group})
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping}), encoding="utf-8")
    else:
        write_safetensors(directory / "model.safetensors", tensors)


def partition(n, world):
    # 独立参考按元素列表分组；不读取 C++ 返回的分片边界。
    quotient, remainder = divmod(n, world)
    sizes = [quotient + (i < remainder) for i in range(world)]
    starts = [sum(sizes[:i]) for i in range(world)]
    return [(start, start + size) for start, size in zip(starts, sizes)]


def reference(config, tensors, rank, world):
    h, d = config["hidden_size"], config["head_dim"]
    qh, kh = config["num_attention_heads"], config["num_key_value_heads"]
    group = qh // kh
    if world <= kh:
        k0, k1 = partition(kh, world)[rank]
        q0, q1 = k0 * group, k1 * group
    else:
        rank_groups = partition(world, kh)
        for head, (start, stop) in enumerate(rank_groups):
            if start <= rank < stop:
                a, b = partition(group, stop - start)[rank - start]
                q0, q1, k0, k1 = head * group + a, head * group + b, head, head + 1
                break
    m0, m1 = partition(config["intermediate_size"], world)[rank]
    v0, v1 = partition(config["vocab_size"], world)[rank]
    targets = {}

    def add(name, shape, slices, zero=False):
        data = bytearray()
        dtype = tensors[slices[0][0]]["dtype"]
        item = 4 if dtype == "F32" else 2
        for source, r0, r1, c0, c1 in slices:
            tensor = tensors[source]
            columns = tensor["shape"][1] if len(tensor["shape"]) == 2 else 1
            for row in range(r0, r1):
                data.extend(tensor["data"][(row * columns + c0) * item:(row * columns + c1) * item])
        targets[name] = dict(shape=shape, dtype=dtype, slices=slices,
                             data=bytes(len(data)) if zero else bytes(data), zero=zero)

    def replicated(name):
        shape = tensors[name]["shape"]
        add(name, shape, [(name, 0, shape[0], 0, shape[1] if len(shape) == 2 else 1)])

    replicated("model.embed_tokens.weight")
    replicated("model.norm.weight")
    lm = "lm_head.weight" if "lm_head.weight" in tensors else "model.embed_tokens.weight"
    add("lm_head.weight", [v1 - v0, h], [(lm, v0, v1, 0, h)])
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        a, m = p + "self_attn.", p + "mlp."
        for name in (p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
                     a + "q_norm.weight", a + "k_norm.weight"):
            replicated(name)
        for suffix in ("weight", "bias"):
            if a + "q_proj." + suffix not in tensors:
                continue
            width = (q1 - q0 + 2 * (k1 - k0)) * d
            columns = h if suffix == "weight" else 1
            add(a + "mergeqkv." + suffix, [width, h] if suffix == "weight" else [width],
                [(a + name + "." + suffix, begin * d, end * d, 0, columns)
                 for name, begin, end in (("q_proj", q0, q1), ("k_proj", k0, k1), ("v_proj", k0, k1))])
        add(a + "o_proj.weight", [h, (q1 - q0) * d], [(a + "o_proj.weight", 0, h, q0 * d, q1 * d)])
        add(m + "gateup_proj.weight", [2 * (m1 - m0), h],
            [(m + name + ".weight", m0, m1, 0, h) for name in ("gate_proj", "up_proj")])
        add(m + "down_proj.weight", [h, m1 - m0], [(m + "down_proj.weight", 0, h, m0, m1)])
        for name in (a + "o_proj.bias", m + "down_proj.bias"):
            if name in tensors:
                add(name, [h], [(name, 0, h, 0, 1)], zero=rank != 0)
    metadata = dict(rank=rank, world_size=world,
                    query_head_begin=q0, query_head_count=q1 - q0, kv_head_begin=k0, kv_head_count=k1 - k0,
                    intermediate_begin=m0, intermediate_count=m1 - m0, vocab_begin=v0, vocab_count=v1 - v0)
    return targets, metadata


def invoke(binary, directory, rank, world, log_dir, output=None, extra=(), cwd=None):
    command = [str(binary), "--model", str(directory), "--rank", str(rank), "--world-size", str(world)]
    if output is not None:
        command += ["--output", str(output)]
    command += list(extra)
    log_dir.mkdir()
    (log_dir / "command.txt").write_text(shlex.join(command), encoding="utf-8")
    (log_dir / "invocation.json").write_text(json.dumps(
        dict(command=command, cwd=str(Path(cwd).resolve() if cwd else Path.cwd())), ensure_ascii=False), encoding="utf-8")
    # 直接落盘，超时、中断或非法输出也保留已经写出的原始证据。
    with (log_dir / "stdout.log").open("wb") as stdout, (log_dir / "stderr.log").open("wb") as stderr:
        try:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=30, cwd=cwd)
        except OSError as error:
            stderr.write(str(error).encode("utf-8"))
            raise
    return result.returncode, json.loads((log_dir / "stdout.log").read_text(encoding="utf-8"))


def verify_rank(report, output, expected, metadata, tensors, reconstructed, coverage, metrics):
    assert report["status"] == "PASS" and not report["plan_only"], report
    assert report["backend"] == "qwen3_rank_local_safetensors", report
    assert all(report[key] == value for key, value in metadata.items()), (report, metadata)
    entries = {item["name"]: item for item in report["tensors"]}
    assert len(entries) == len(report["tensors"]) and entries.keys() == expected.keys(), "target list mismatch"
    for name, target in expected.items():
        entry = entries[name]
        assert entry["shape"] == target["shape"] and entry["dtype"] == target["dtype"], name
        assert entry["zero_fill"] == target["zero"], name
        data = (output / entry["file"]).read_bytes()
        assert len(data) == len(target["data"]) == entry["bytes"], name
        assert entry["bytes_read"] == (0 if target["zero"] else len(data)), name
        expected_slices = [dict(source=s, row_begin=a, row_end=b, column_begin=c, column_end=d)
                           for s, a, b, c, d in target["slices"]]
        assert entry["slices"] == expected_slices, name
        item = 4 if target["dtype"] == "F32" else 2
        actual_values, desired_values = decode(data, target["dtype"]), decode(target["data"], target["dtype"])
        for i, (actual, desired) in enumerate(zip(actual_values, desired_values)):
            mismatch = data[i * item:(i + 1) * item] != target["data"][i * item:(i + 1) * item]
            error = abs(actual - desired) if math.isfinite(actual) else float("inf")
            metrics["max"] = max(metrics["max"], error)
            metrics["sum"] += error
            metrics["count"] += 1
            metrics["failed"] += int(mismatch)
        if data != target["data"]:
            raise AssertionError(f"{name}, shape={target['shape']}: output differs from reference")
        if target["zero"]:
            continue
        position = 0
        for source, r0, r1, c0, c1 in target["slices"]:
            shape = tensors[source]["shape"]
            columns = shape[1] if len(shape) == 2 else 1
            for row in range(r0, r1):
                begin, length = (row * columns + c0) * item, (c1 - c0) * item
                reconstructed[source][begin:begin + length] = data[position:position + length]
                coverage[source][begin:begin + length] = b"\1" * length
                position += length
    assert report["tensor_bytes"] == sum(len(t["data"]) for t in expected.values()), "allocation size mismatch"
    assert report["payload_bytes_read"] == sum(len(t["data"]) for t in expected.values() if not t["zero"])
    assert report["max_tensor_buffer_bytes"] == max(len(t["data"]) for t in expected.values())


def positive_case(binary, directory, spec, metrics):
    options = dict(spec)
    world = options.pop("world")
    implicit_head_dim = options.pop("implicit_head_dim", False)
    change_cwd = options.pop("change_cwd", False)
    json_strings = options.pop("json_strings", False)
    config, tensors = fixture(directory / "model", **options)
    if implicit_head_dim:
        assert config["hidden_size"] // config["num_attention_heads"] == config["head_dim"]
        (directory / "model/config.json").write_text(
            json.dumps({key: value for key, value in config.items() if key != "head_dim"}))
    if json_strings:
        config["notes"] = '中文 " quote \\ slash {[]} : comma, newline\n'
        config["nested"] = [{"same_key": "one"}, {"same_key": ["two", {"same_key": "three"}]}]
        save_fixture(directory / "model", config, tensors)
    model, cwd, extra = directory / "model", None, []
    if change_cwd:
        shadow = directory / "other/model"
        shadow.mkdir(parents=True)
        save_fixture(shadow, config, {name: dict(tensor, data=bytes(len(tensor["data"])))
                                     for name, tensor in tensors.items()})
        # 同名、同大小、内容不同的模型：旧的相对路径实现会悄悄读到 shadow。
        model, cwd = Path("model"), directory
        extra = ["--change-cwd", str(shadow.parent)]
    code, plan = invoke(binary, model, 0, world, directory / "plan-log", extra=extra, cwd=cwd)
    assert code == 0 and plan["status"] == "PASS" and plan["plan_only"], plan
    _, plan_metadata = reference(config, tensors, 0, world)
    assert all(plan[key] == value for key, value in plan_metadata.items()), (plan, plan_metadata)
    assert plan["payload_bytes_read"] == 0 and plan["max_tensor_buffer_bytes"] == 0, plan
    reconstructed = {name: bytearray(len(t["data"])) for name, t in tensors.items()}
    coverage = {name: bytearray(len(t["data"])) for name, t in tensors.items()}
    for rank in range(world):
        expected, metadata = reference(config, tensors, rank, world)
        code, report = invoke(binary, model, rank, world, directory / f"log-{rank}",
                              directory / f"rank-{rank}", extra=extra, cwd=cwd)
        assert code == 0, report
        verify_rank(report, directory / f"rank-{rank}", expected, metadata, tensors,
                    reconstructed, coverage, metrics)
    for name, tensor in tensors.items():
        assert all(coverage[name]), f"reconstruction left holes: {name}"
        assert reconstructed[name] == tensor["data"], f"reconstruction mismatch: {name}"


def rejection_case(binary, directory, kind):
    config, tensors = fixture(directory / "model", bias=True)
    rank, world, extra = 0, 2, []
    source = "model.layers.0.self_attn.q_proj.weight"
    reason = ""
    if kind == "rank":
        rank, reason = 2, "invalid rank/world size"
    elif kind == "empty_shard":
        world, reason = 9, "empty shard"
    elif kind == "shape":
        tensors[source]["shape"] = [8, 12]
        reason = "shape mismatch"
    elif kind == "missing":
        del tensors["model.layers.0.mlp.down_proj.weight"]
        reason = "missing tensor"
    elif kind == "partial_bias":
        del tensors["model.layers.0.self_attn.k_proj.bias"]
        reason = "incomplete QKV bias"
    elif kind == "unknown_tensor":
        tensors["unsupported.weight"] = dict(tensors["model.norm.weight"])
        reason = "unexpected tensors"
    elif kind == "quantization":
        config["quantization_config"] = {"quant_method": "fp8"}
        reason = "quantized checkpoints"
    elif kind == "dtype":
        tensors[source]["dtype"] = "I16"
        reason = "unsupported dtype"
    elif kind == "mixed_fused_dtype":
        tensors[source]["dtype"] = "BF16"
        reason = "mixed fused dtype"
    elif kind == "model_type":
        config["model_type"] = "qwen3_moe"
        reason = "only dense"
    elif kind == "head_ratio":
        config["num_key_value_heads"] = 3
        reason = "Q heads must be divisible"
    elif kind == "fractional_dimension":
        config["hidden_size"] = 6.5
        reason = "invalid integer"
    elif kind == "memory_limit":
        extra = ["--max-bytes", "1"]
        reason = "allocation limit exceeded"
    save_fixture(directory / "model", config, tensors)
    file = directory / "model/model.safetensors"
    if kind == "truncated":
        file.write_bytes(file.read_bytes()[:-1])
        reason = "tensor size/offset mismatch"
    elif kind == "header_limit":
        data = file.read_bytes()
        file.write_bytes(struct.pack("<Q", 2**63) + data[8:])
        reason = "invalid safetensors header size"
    elif kind in ("offset_overlap", "offset_fractional"):
        data = file.read_bytes()
        size = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8:8 + size])
        key = list(header)[1]
        if kind == "offset_overlap":
            width = header[key]["data_offsets"][1] - header[key]["data_offsets"][0]
            header[key]["data_offsets"] = [0, width]
            reason = "overlapping or noncontiguous"
        else:
            header[key]["data_offsets"][0] += 0.5
            reason = "invalid integer"
        text = json.dumps(header).encode()
        file.write_bytes(struct.pack("<Q", len(text)) + text + data[8 + size:])
    elif kind == "index_mismatch":
        mapping = {name: "model.safetensors" for name in tensors}
        mapping["wrong.weight"] = mapping.pop(source)
        (directory / "model/model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        reason = "weight_map/header mismatch"
    elif kind == "index_path":
        (directory / "model/model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {source: "../other.safetensors"}}))
        reason = "invalid weight_map file"
    elif kind.startswith("duplicate_"):
        reason = "duplicate JSON key"
        if kind in ("duplicate_config", "duplicate_escaped_key", "duplicate_nested_key"):
            path = directory / "model/config.json"
            text = path.read_text()
            if kind == "duplicate_nested_key":
                addition = ',"extra":{"same":1,"same":2}'
            else:
                key = 'hidd\\u0065n_size' if kind == "duplicate_escaped_key" else 'hidden_size'
                addition = ',"' + key + '":' + str(config["hidden_size"])
            path.write_text(text[:-1] + addition + '}')
        elif kind == "duplicate_index":
            mapping = {name: "model.safetensors" for name in tensors}
            text = json.dumps(mapping)[:-1] + ',' + json.dumps(source) + ':"model.safetensors"}'
            (directory / "model/model.safetensors.index.json").write_text('{"weight_map":' + text + '}')
        else:
            data = file.read_bytes()
            size = struct.unpack("<Q", data[:8])[0]
            header = json.loads(data[8:8 + size])
            text = json.dumps(header)
            if kind == "duplicate_tensor":
                text = text[:-1] + ',' + json.dumps(source) + ':' + json.dumps(header[source]) + '}'
            elif kind == "duplicate_tensor_dtype":
                text = text.replace('"dtype": "F16"', '"dtype": "F16", "dtype": "F16"', 1)
            else:
                raise ValueError(kind)
            encoded = text.encode()
            file.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + size:])
    assert reason, kind
    code, report = invoke(binary, directory / "model", rank, world, directory / "log", directory / "rank", extra)
    verify_rejection(code, report, reason)


def verify_rejection(code, report, reason):
    # 已打印 FAIL 后崩溃也不是正常拒绝；C++ 检查程序规定拒绝时退出码为 1。
    assert code == 1 and report["status"] == "FAIL", (code, report)
    assert report["backend"] == "qwen3_rank_local_safetensors", report
    assert reason in report.get("reason", ""), report


def main():
    if not __debug__:
        raise RuntimeError("Do not run this correctness checker with python -O/PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True, help="已有 fastllm_tp_shard_check 可执行文件")
    parser.add_argument("--output", type=Path, default=ROOT / "test/distributed/results")
    parser.add_argument("--build-cache", type=Path)
    args = parser.parse_args()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_shards_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(["git", "rev-parse", "HEAD"]),
                    git_status=capture(["git", "status", "--short"]), seed=0, warmup=0, iterations=1, repeats=1,
                    test_kind="CPU rank-local weight loader correctness; not model inference or a performance benchmark",
                    gpu="n/a (CPU file I/O only)", cuda_runtime_version="n/a (no CUDA calls)",
                    compile_architecture="host C++17; no device kernels", checker=str(args.check.resolve()),
                    memory_metric="max_tensor_buffer_bytes is the largest returned buffer, not measured RSS",
                    io_metric="payload_bytes_read is requested payload bytes, not measured physical disk I/O")
    metadata["report_title"] = "Qwen3 rank-local safetensors correctness"
    metadata["report_description"] = "检查逐元素误差、字节一致性及分片重组；无数值输出的拒绝检查为 n/a。原始输出和输入保存在各 case 目录。"
    cases = []
    for dtype in ("F16", "BF16", "F32"):
        for world in (1, 2, 3, 4, 8):
            cases.append((f"{dtype}_tp{world}", dict(dtype=dtype, world=world)))
    cases += [("tied_bias_indexed", dict(world=4, tied=True, bias=True, indexed=True)),
              ("mha_uneven", dict(world=3, num_key_value_heads=8)),
              ("gqa_uneven_kv", dict(world=2, num_attention_heads=6, num_key_value_heads=3)),
              ("gqa_uneven_replicas", dict(world=5, num_attention_heads=12, num_key_value_heads=3)),
              ("implicit_head_dim", dict(world=2, hidden_size=16, implicit_head_dim=True)),
              ("relative_path_after_chdir", dict(world=2, change_cwd=True)),
              ("json_escaped_values", dict(world=2, json_strings=True)),
              ("minimal", dict(world=1, num_hidden_layers=1, hidden_size=2, num_attention_heads=1,
                               num_key_value_heads=1, head_dim=2, intermediate_size=1, vocab_size=1))]
    negative = ("rank", "empty_shard", "shape", "missing", "partial_bias", "unknown_tensor", "quantization",
                "dtype", "mixed_fused_dtype", "model_type", "head_ratio", "fractional_dimension", "memory_limit",
                "truncated", "header_limit", "offset_overlap", "offset_fractional", "index_mismatch", "index_path",
                "duplicate_config", "duplicate_escaped_key", "duplicate_nested_key", "duplicate_index",
                "duplicate_tensor", "duplicate_tensor_dtype")
    cases += [("reject_" + name, name) for name in negative]
    metadata["expected_case_count"] = len(cases)
    rows = []
    try:
        if args.build_cache:
            metadata["build_cache"] = args.build_cache.read_text()
        for name, spec in cases:
            directory = output / name
            directory.mkdir()
            metrics = dict(max=0.0, sum=0.0, count=0, failed=0)
            row = dict(case=name, backend="qwen3_rank_local_safetensors", status="FAIL", reason="",
                       max_abs_error=None, mean_abs_error=None, failed_elements=None, input=spec)
            try:
                if isinstance(spec, dict):
                    positive_case(args.check.resolve(), directory, spec, metrics)
                else:
                    rejection_case(args.check.resolve(), directory, spec)
                row["status"] = "PASS"
            except Exception as error:
                row["reason"] = f"{type(error).__name__}: {error}"
            finally:
                if metrics["count"]:
                    maximum, mean = metrics["max"], metrics["sum"] / metrics["count"]
                    row.update(max_abs_error=maximum if math.isfinite(maximum) else "inf",
                               mean_abs_error=mean if math.isfinite(mean) else "inf",
                               failed_elements=metrics["failed"], compared_elements=metrics["count"])
                rows.append(row)
                save_report(output, rows, metadata)
    except BaseException as error:
        rows.append(dict(case="runner", backend="python", status="FAIL",
                         reason=f"{type(error).__name__}: {error}"))
        raise
    finally:
        status = save_report(output, rows, metadata)
    print(f"{status}: {output}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
