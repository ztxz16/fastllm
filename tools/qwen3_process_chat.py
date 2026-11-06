#!/usr/bin/env python3
"""Qwen3 多进程 TP 单请求文本入口；tokenizer 在 CPU，权重和前向只在独立 worker。"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import uuid

from tp_process_utils import isolated_process, wait_case

ROOT = Path(__file__).resolve().parents[1]
BACKEND = "qwen3_process_generate_cublas_nccl"


def validate_request(request, vocab):
    """拒绝越界输入；不截断 Prompt，不改变 EOS 或采样参数。"""
    for name in ("max_new_tokens", "top_k", "seed"):
        value = request[name]
        if type(value) is not int or not 0 <= value <= 4294967295:
            raise ValueError(f"{name} must be uint32")
    ids = request["token_ids"]
    if not 1 <= len(ids) <= 128:
        raise ValueError(f"Prompt contains {len(ids)} tokens including chat template; supported range is 1..128")
    if request["max_new_tokens"] < 1 or len(ids) + request["max_new_tokens"] - 1 > 2048:
        raise ValueError("max_new_tokens must be positive and prompt + max_new_tokens - 1 <= 2048")
    for token in ids + request["eos_token_ids"]:
        if type(token) is not int or not 0 <= token < vocab:
            raise ValueError(f"token ID outside model vocabulary: {token}")
    temperature, top_p = request["temperature"], request["top_p"]
    if not math.isfinite(temperature) or temperature < 0 or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("temperature must be finite >=0; top_p must be in (0,1]")


def prepare_request(model, tokenizer, prompt, *, raw=False, thinking=False, max_new_tokens=32,
                    temperature=0.0, top_k=0, top_p=1.0, seed=0, eos_ids=None, ignore_eos=False):
    """复用本地模型的聊天模板和 EOS 配置，保存实际送入模型的完整 token ID。"""
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError("only dense Qwen3 (model_type=qwen3) is supported")
    if raw:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        if not tokenizer.chat_template:
            template = model / "chat_template.jinja"
            if template.is_file():
                tokenizer.chat_template = template.read_text()
            else:
                raise ValueError("model has no chat template; use --raw only for raw completion")
        ids = tokenizer.apply_chat_template([dict(role="user", content=prompt)], tokenize=True,
            add_generation_prompt=True, enable_thinking=thinking)

    eos = []
    if not ignore_eos:
        if eos_ids is not None:
            eos = list(eos_ids)
        else:
            generation_path = model / "generation_config.json"
            generation = json.loads(generation_path.read_text()) if generation_path.is_file() else {}
            configured = generation.get("eos_token_id", config.get("eos_token_id"))
            if configured is not None:
                eos.extend(configured if isinstance(configured, list) else [configured])
            if tokenizer.eos_token_id is not None:
                eos.append(tokenizer.eos_token_id)
            if not eos:
                raise ValueError("model has no EOS ID; specify --eos-token-id or --ignore-eos")
    request = dict(token_ids=list(ids), max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, seed=seed, eos_token_ids=list(dict.fromkeys(eos)))
    validate_request(request, config["vocab_size"])
    return request


def parse_report(text):
    """允许 NCCL 前置日志，但不允许跳过损坏的 JSON 而误识别嵌套 PASS。"""
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("{"):
            result = json.loads(text[offset + len(line) - len(stripped):])
            if not isinstance(result, dict):
                raise ValueError("generation report must be an object")
            return result
        offset += len(line)
    raise ValueError("no generation JSON report")


def run_native(binary, worker, model, request, devices, output, timeout=1800):
    """运行同一个正式生成程序；文件记录输出，超时/中断终止整个进程组。"""
    (output / "request.json").write_text(json.dumps(request, ensure_ascii=False, allow_nan=False))
    command = [str(binary), "--worker", str(worker), "--model", str(model),
               "--request", str(output / "request.json"), "--devices", devices]
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    with (output / "native.stdout.log").open("wb") as stdout, (output / "native.stderr.log").open("wb") as stderr:
        with isolated_process(command, stdout=stdout, stderr=stderr) as process:
            code = wait_case(process, timeout)
    report = parse_report((output / "native.stdout.log").read_text())
    if code != 0 or report.get("status") != "PASS" or report.get("backend") != BACKEND:
        raise RuntimeError(f"generation exit={code}: {report.get('error', 'invalid backend/status')}; logs: {output}")
    if report.get("request") != request or report.get("shutdown_complete") is not True:
        raise RuntimeError("generation request mismatch or incomplete shutdown")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build-process-tp-cu126/fastllm_tp_generate")
    parser.add_argument("--worker", type=Path, default=ROOT / "build-process-tp-cu126/fastllm_tp_worker")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument("--eos-token-id", type=int, action="append")
    stop.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("generation-results"), help="日志根目录；每次创建独立子目录")
    args = parser.parse_args()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    result = dict(status="FAIL", command=shlex.join(sys.argv), prompt=args.prompt,
                  raw=args.raw, thinking=args.thinking, model=str(args.model.resolve()))
    try:
        # 只需要 tokenizer；禁止由此加载 torch 模型或下载远程代码。
        os.environ.setdefault("USE_TORCH", "0")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True, trust_remote_code=False)
        request = prepare_request(args.model.resolve(), tokenizer, args.prompt, raw=args.raw,
            thinking=args.thinking, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, seed=args.seed, eos_ids=args.eos_token_id, ignore_eos=args.ignore_eos)
        report = run_native(args.binary.resolve(), args.worker.resolve(), args.model.resolve(), request, args.devices, output)
        ids = report["token_ids"]
        if (not isinstance(ids, list) or not 1 <= len(ids) <= request["max_new_tokens"] or
            any(type(token) is not int or token < 0 for token in ids)):
            detail = repr(ids)[:256]
            types = [type(token).__name__ for token in ids[:8]] if isinstance(ids, list) else type(ids).__name__
            raise RuntimeError(f"invalid generated token IDs: values={detail}, types={types}")
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result.update(status="PASS", text=text, native=report)
        print(text)
        print(f"finish={report['finish_reason']}, tokens={len(ids)}, logs={output}", file=sys.stderr)
    except Exception as error:
        result["status"] = "FAIL"
        result["error"] = f"{type(error).__name__}: {error}"
        print(f"FAIL: {result['error']}\nlogs={output}", file=sys.stderr)
    finally:
        (output / "response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
